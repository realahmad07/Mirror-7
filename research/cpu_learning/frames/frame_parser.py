"""Learner-facing L0/L2 raw UTF-8 instruction -> typed candidate role-frame parsers.

STRICT ISOLATION (PDF pp. 4-6, 10, 14):
- Operates ONLY on raw UTF-8 instruction bytes and the learner's public ``TypedStateFrame``.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Retains exact UTF-8 byte offsets (not char offsets) and preserves multiple candidate
  parses whenever an instruction is syntactically or epistemically ambiguous.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

from mirror_crystal.cortical_active_inference import BilingualCCGHDCParser

from .state_frame import (
    CandidateActionFrame,
    FactEpistemicStatus,
    FrameParseResult,
    TypedStateFrame,
)

_TOKEN_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", flags=re.UNICODE)

_DETERMINERS: frozenset[str] = frozenset({"the", "a", "an"})
_POLITE_PREFIXES: frozenset[str] = frozenset({"please", "kindly"})
_CAUSATIVE_BARE: frozenset[str] = frozenset({"have", "let", "make"})
_CAUSATIVE_TO: frozenset[str] = frozenset({"ask", "tell", "instruct", "order"})
_PASSIVE_AUX_SINGLE: frozenset[str] = frozenset({"is", "was", "gets", "got"})
_PASSIVE_AUX_MODAL: frozenset[str] = frozenset({"should", "must", "will", "can", "may"})
_QUESTION_LEADS: frozenset[str] = frozenset({"what", "where", "who", "whom", "whose", "why", "how", "when", "which"})

# Linguistic verb lemma mapping -> (canonical_action_id, syntactic_frame_family).
# Pure lexical/syntactic surface-to-frame mapping (zero transition rules or preconditions).
_ACTION_VERB_MAP: dict[str, tuple[str, str]] = {
    "move": ("move", "transfer_loc"),
    "moves": ("move", "transfer_loc"),
    "moved": ("move", "transfer_loc"),
    "relocate": ("move", "transfer_loc"),
    "relocates": ("move", "transfer_loc"),
    "relocated": ("move", "transfer_loc"),
    "place": ("move", "transfer_loc"),
    "places": ("move", "transfer_loc"),
    "placed": ("move", "transfer_loc"),
    "put": ("move", "transfer_loc"),
    "puts": ("move", "transfer_loc"),
    "shift": ("move", "transfer_loc"),
    "shifts": ("move", "transfer_loc"),
    "shifted": ("move", "transfer_loc"),
    "carry": ("move", "transfer_loc"),
    "carries": ("move", "transfer_loc"),
    "carried": ("move", "transfer_loc"),
    "reposition": ("move", "transfer_loc"),
    "repositions": ("move", "transfer_loc"),
    "repositioned": ("move", "transfer_loc"),
    "give": ("give", "transfer_own"),
    "gives": ("give", "transfer_own"),
    "given": ("give", "transfer_own"),
    "hand": ("give", "transfer_own"),
    "hands": ("give", "transfer_own"),
    "handed": ("give", "transfer_own"),
    "pass": ("give", "transfer_own"),
    "passes": ("give", "transfer_own"),
    "passed": ("give", "transfer_own"),
    "deliver": ("give", "transfer_own"),
    "delivers": ("give", "transfer_own"),
    "delivered": ("give", "transfer_own"),
    "present": ("give", "transfer_own"),
    "presents": ("give", "transfer_own"),
    "presented": ("give", "transfer_own"),
    "offer": ("give", "transfer_own"),
    "offers": ("give", "transfer_own"),
    "offered": ("give", "transfer_own"),
    "open": ("open", "unary_target"),
    "opens": ("open", "unary_target"),
    "opened": ("open", "unary_target"),
    "unseal": ("open", "unary_target"),
    "unseals": ("open", "unary_target"),
    "unsealed": ("open", "unary_target"),
    "uncover": ("open", "unary_target"),
    "uncovers": ("open", "unary_target"),
    "uncovered": ("open", "unary_target"),
    "lock": ("lock", "unary_target"),
    "locks": ("lock", "unary_target"),
    "locked": ("lock", "unary_target"),
    "secure": ("lock", "unary_target"),
    "secures": ("lock", "unary_target"),
    "secured": ("lock", "unary_target"),
    "fasten": ("lock", "unary_target"),
    "fastens": ("lock", "unary_target"),
    "fastened": ("lock", "unary_target"),
    "latch": ("lock", "unary_target"),
    "latches": ("lock", "unary_target"),
    "latched": ("lock", "unary_target"),
    "bolt": ("lock", "unary_target"),
    "bolts": ("lock", "unary_target"),
    "bolted": ("lock", "unary_target"),
    "transfer": ("transfer", "polysemous_transfer"),
    "transfers": ("transfer", "polysemous_transfer"),
    "transferred": ("transfer", "polysemous_transfer"),
    "send": ("transfer", "polysemous_transfer"),
    "sends": ("transfer", "polysemous_transfer"),
    "sent": ("transfer", "polysemous_transfer"),
}

_PAST_PARTICIPLES: frozenset[str] = frozenset(
    {
        "moved", "relocated", "placed", "put", "shifted", "carried", "repositioned",
        "given", "handed", "passed", "delivered", "presented", "offered",
        "opened", "unsealed", "uncovered",
        "locked", "secured", "fastened", "latched", "bolted",
        "transferred", "sent",
    }
)

_SRC_PREPS_2: frozenset[tuple[str, str]] = frozenset({("out", "of"), ("off", "of")})
_SRC_PREPS_1: frozenset[str] = frozenset({"from", "off"})
_DST_PREPS_2: frozenset[tuple[str, str]] = frozenset({("over", "to"), ("on", "to"), ("in", "to")})
_DST_PREPS_1: frozenset[str] = frozenset({"to", "onto", "into", "unto"})


@dataclass(frozen=True)
class ByteToken:
    """Single lexical token with exact UTF-8 byte-span offsets in raw_instruction_bytes."""

    text: str
    lower: str
    byte_start: int
    byte_end: int

    @property
    def span(self) -> tuple[int, int]:
        return (self.byte_start, self.byte_end)


def tokenize_utf8_bytes(raw_bytes: bytes) -> tuple[ByteToken, ...]:
    """Tokenize UTF-8 bytes into ByteTokens with exact UTF-8 byte offsets (multi-byte safe)."""
    text = raw_bytes.decode("utf-8")
    char_to_byte = [0] * (len(text) + 1)
    bpos = 0
    for idx, ch in enumerate(text):
        bpos += len(ch.encode("utf-8"))
        char_to_byte[idx + 1] = bpos

    out: list[ByteToken] = []
    for match in _TOKEN_RE.finditer(text):
        tok_text = match.group(0)
        b_start = char_to_byte[match.start()]
        b_end = char_to_byte[match.end()]
        out.append(
            ByteToken(
                text=tok_text,
                lower=tok_text.casefold(),
                byte_start=b_start,
                byte_end=b_end,
            )
        )
    return tuple(out)


class BaseFrameParser(ABC):
    """Interface for standalone raw-UTF-8-to-candidate-role-frame parsers."""

    name: str = "base_frame_parser"
    version: str = "v1"

    @abstractmethod
    def parse(
        self,
        raw_instruction_bytes: bytes,
        before_frame: TypedStateFrame | None = None,
    ) -> FrameParseResult:
        """Extract typed candidate action frame(s) from raw UTF-8 instruction bytes."""


class BilingualCCGFrameAdapter(BaseFrameParser):
    """Read-only S1 adapter wrapping Mirror 7's existing BilingualCCGHDCParser.

    Faithfully exposes its (who, act, targ) output as a CandidateActionFrame so
    parser-only benchmarks can quantify the exact gap versus typed multi-role parsing.
    """

    name = "existing_mirror_bilingual_ccg"
    version = "mirror7-ccg-v1"

    def __init__(self) -> None:
        self._parser = BilingualCCGHDCParser()

    @staticmethod
    def _find_token_span(tokens: Sequence[ByteToken], target_lower: str) -> tuple[int, int] | None:
        if not target_lower:
            return None
        parts = target_lower.split()
        if not parts:
            return None
        n = len(parts)
        for i in range(len(tokens) - n + 1):
            if all(tokens[i + j].lower == parts[j] for j in range(n)):
                return (tokens[i].byte_start, tokens[i + n - 1].byte_end)
        return None

    def parse(
        self,
        raw_instruction_bytes: bytes,
        before_frame: TypedStateFrame | None = None,
    ) -> FrameParseResult:
        raw = bytes(raw_instruction_bytes)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return FrameParseResult(
                raw_instruction_bytes=raw,
                parser_name=self.name,
                parser_version=self.version,
                candidate_frames=(),
                is_ambiguous=False,
                abstained=True,
                abstention_reason="invalid_utf8",
            )
        if not text.strip():
            return FrameParseResult(
                raw_instruction_bytes=raw,
                parser_name=self.name,
                parser_version=self.version,
                candidate_frames=(),
                is_ambiguous=False,
                abstained=True,
                abstention_reason="empty_instruction",
            )

        tokens = tokenize_utf8_bytes(raw)
        edge = self._parser.parse_clause(text)
        if not edge.act or not edge.who or not edge.targ:
            return FrameParseResult(
                raw_instruction_bytes=raw,
                parser_name=self.name,
                parser_version=self.version,
                candidate_frames=(),
                is_ambiguous=False,
                abstained=True,
                abstention_reason="ccg_incomplete_roles",
            )

        actor_val = edge.who
        targ_val = edge.targ
        if before_frame is not None:
            resolved_a = before_frame.resolve_entity_id(actor_val)
            if resolved_a is not None:
                actor_val = resolved_a
            resolved_t = before_frame.resolve_entity_id(targ_val)
            if resolved_t is not None:
                targ_val = resolved_t

        roles = (("actor", actor_val), ("target", targ_val))
        spans: list[tuple[str, int, int]] = []
        act_span = self._find_token_span(tokens, edge.act)
        if act_span is not None:
            spans.append(("action", act_span[0], act_span[1]))
        who_span = self._find_token_span(tokens, edge.who)
        if who_span is not None:
            spans.append(("actor", who_span[0], who_span[1]))
        targ_span = self._find_token_span(tokens, edge.targ)
        if targ_span is not None:
            spans.append(("target", targ_span[0], targ_span[1]))

        role_types: list[tuple[str, str]] = []
        if before_frame is not None:
            for rk, rv in roles:
                rt = before_frame.entity_type_of(rv)
                if rt is not None:
                    role_types.append((rk, rt))

        frame = CandidateActionFrame(
            action_id=edge.act,
            roles=roles,
            byte_spans=tuple(spans),
            role_types=tuple(role_types),
            confidence=1.0,
            parse_rule="mirror_bilingual_ccg_3slot",
        )
        return FrameParseResult(
            raw_instruction_bytes=raw,
            parser_name=self.name,
            parser_version=self.version,
            candidate_frames=(frame,),
            is_ambiguous=False,
            abstained=False,
        )


def _match_prep(tokens: Sequence[ByteToken], idx: int) -> tuple[str | None, int]:
    """Match a directional/agentive preposition at tokens[idx]. Returns (kind, consumed_tokens)."""
    n = len(tokens)
    if idx + 1 < n:
        pair = (tokens[idx].lower, tokens[idx + 1].lower)
        if pair in _SRC_PREPS_2:
            return ("source", 2)
        if pair in _DST_PREPS_2:
            return ("destination", 2)
    tok = tokens[idx].lower
    if tok in _SRC_PREPS_1:
        return ("source", 1)
    if tok in _DST_PREPS_1:
        return ("destination", 1)
    if tok == "by":
        return ("by", 1)
    return (None, 0)


def _skip_det(tokens: Sequence[ByteToken], idx: int) -> int:
    while idx < len(tokens) and tokens[idx].lower in _DETERMINERS:
        idx += 1
    return idx

class TypedRoleFrameParser(BaseFrameParser):
    """S1 learner-facing typed, byte-span-exact, ambiguity-preserving role-frame parser.

    Supports an explicit ``use_state_types`` ablation flag to measure the contribution
    of L2 ``TypedStateFrame`` entity typing & observed locations on ambiguous constructions
    (ditransitives, polysemous verbs, preposition-free orderings, and omitted actors).
    """

    version = "s1-typed-frame-v1"

    def __init__(self, *, use_state_types: bool = True) -> None:
        self.use_state_types = bool(use_state_types)
        self.name = (
            "s1_typed_frame_parser_full"
            if self.use_state_types
            else "s1_typed_frame_parser_no_types"
        )

    def _abstain(self, raw: bytes, reason: str) -> FrameParseResult:
        return FrameParseResult(
            raw_instruction_bytes=raw,
            parser_name=self.name,
            parser_version=self.version,
            candidate_frames=(),
            is_ambiguous=False,
            abstained=True,
            abstention_reason=reason,
        )

    def _resolve(self, tok: ByteToken, before_frame: TypedStateFrame | None) -> str:
        if before_frame is not None:
            resolved = before_frame.resolve_entity_id(tok.text)
            if resolved is not None:
                return resolved
        return tok.text

    def _entity_type(self, entity_id: str, before_frame: TypedStateFrame | None) -> str | None:
        if (not self.use_state_types) or before_frame is None:
            return None
        ef = before_frame.get_entity(entity_id)
        if ef is None or not ef.attributes_observed:
            return None
        return ef.entity_type

    def _build_frame(
        self,
        action_id: str,
        action_tok: ByteToken,
        role_bindings: Mapping[str, tuple[str, ByteToken | None]],
        before_frame: TypedStateFrame | None,
        *,
        confidence: float,
        parse_rule: str,
        ambiguity_reasons: Sequence[str] = (),
    ) -> CandidateActionFrame:
        roles: list[tuple[str, str]] = []
        spans: list[tuple[str, int, int]] = [("action", action_tok.byte_start, action_tok.byte_end)]
        role_types: list[tuple[str, str]] = []

        for rk, (val, tok) in sorted(role_bindings.items()):
            roles.append((rk, val))
            if tok is not None:
                spans.append((rk, tok.byte_start, tok.byte_end))
            etype = self._entity_type(val, before_frame)
            if etype is not None:
                role_types.append((rk, etype))
            elif before_frame is not None:
                ef = before_frame.get_entity(val)
                if ef is not None:
                    role_types.append((rk, ef.entity_type))

        return CandidateActionFrame(
            action_id=action_id,
            roles=tuple(roles),
            byte_spans=tuple(spans),
            role_types=tuple(role_types),
            confidence=confidence,
            parse_rule=parse_rule,
            ambiguity_reasons=tuple(ambiguity_reasons),
        )

    def _expand_implicit_actor(
        self,
        action_id: str,
        action_tok: ByteToken,
        non_actor_roles: Mapping[str, tuple[str, ByteToken | None]],
        before_frame: TypedStateFrame | None,
        *,
        parse_rule: str,
        extra_reasons: Sequence[str] = (),
    ) -> tuple[list[CandidateActionFrame], bool]:
        if self.use_state_types and before_frame is not None:
            agents = before_frame.entities_by_type("agent")
            if len(agents) == 1:
                bindings = dict(non_actor_roles)
                bindings["actor"] = (agents[0], None)
                f = self._build_frame(
                    action_id,
                    action_tok,
                    bindings,
                    before_frame,
                    confidence=0.85,
                    parse_rule=f"{parse_rule}_implicit_single_agent",
                    ambiguity_reasons=tuple(extra_reasons) + ("implicit_actor_single_agent",),
                )
                return ([f], bool(extra_reasons))
            if len(agents) >= 2:
                conf = round(1.0 / len(agents), 6)
                frames = []
                for ag in agents:
                    bindings = dict(non_actor_roles)
                    bindings["actor"] = (ag, None)
                    frames.append(
                        self._build_frame(
                            action_id,
                            action_tok,
                            bindings,
                            before_frame,
                            confidence=conf,
                            parse_rule=f"{parse_rule}_implicit_actor_candidate",
                            ambiguity_reasons=tuple(extra_reasons) + ("implicit_actor_ambiguous",),
                        )
                    )
                return (frames, True)

        f = self._build_frame(
            action_id,
            action_tok,
            non_actor_roles,
            before_frame,
            confidence=0.5,
            parse_rule=f"{parse_rule}_missing_actor",
            ambiguity_reasons=tuple(extra_reasons) + ("missing_actor",),
        )
        return ([f], True)

    def _parse_tail_args(
        self,
        tokens: Sequence[ByteToken],
        start: int,
        *,
        allow_by: bool,
    ) -> tuple[list[ByteToken], list[ByteToken], list[ByteToken], list[ByteToken]] | None:
        """Parse post-verb tokens into (bare_args, src_toks, dst_toks, by_toks). Returns None on malformed syntax."""
        bare_args: list[ByteToken] = []
        src_toks: list[ByteToken] = []
        dst_toks: list[ByteToken] = []
        by_toks: list[ByteToken] = []
        idx = start
        n = len(tokens)
        while idx < n:
            if tokens[idx].lower in _DETERMINERS:
                idx += 1
                continue
            prep_kind, consumed = _match_prep(tokens, idx)
            if prep_kind is not None:
                if prep_kind == "by" and not allow_by:
                    return None
                idx = _skip_det(tokens, idx + consumed)
                if idx >= n:
                    return None
                next_prep, _ = _match_prep(tokens, idx)
                if next_prep is not None:
                    return None
                filler = tokens[idx]
                if prep_kind == "source":
                    src_toks.append(filler)
                elif prep_kind == "destination":
                    dst_toks.append(filler)
                else:
                    by_toks.append(filler)
                idx += 1
                continue
            bare_args.append(tokens[idx])
            idx += 1
        return (bare_args, src_toks, dst_toks, by_toks)
    def _dispatch_family(
        self,
        *,
        raw: bytes,
        canonical_action: str,
        family: str,
        verb_tok: ByteToken,
        actor_tok: ByteToken | None,
        bare_args: list[ByteToken],
        src_toks: list[ByteToken],
        dst_toks: list[ByteToken],
        before_frame: TypedStateFrame | None,
        voice_tag: str,
    ) -> FrameParseResult:
        actor_binding: tuple[str, ByteToken | None] | None = (
            (self._resolve(actor_tok, before_frame), actor_tok) if actor_tok is not None else None
        )

        # --- Family 1: unary_target (open / lock) ---
        if family == "unary_target":
            if len(bare_args) != 1 or src_toks or dst_toks:
                return self._abstain(raw, f"invalid_{canonical_action}_arity")
            tgt_tok = bare_args[0]
            tgt_val = self._resolve(tgt_tok, before_frame)
            if actor_binding is None:
                cands, is_amb = self._expand_implicit_actor(
                    canonical_action,
                    verb_tok,
                    {"target": (tgt_val, tgt_tok)},
                    before_frame,
                    parse_rule=f"{voice_tag}_unary",
                )
                return FrameParseResult(
                    raw_instruction_bytes=raw,
                    parser_name=self.name,
                    parser_version=self.version,
                    candidate_frames=tuple(cands),
                    is_ambiguous=is_amb,
                    abstained=False,
                )
            frame = self._build_frame(
                canonical_action,
                verb_tok,
                {"actor": actor_binding, "target": (tgt_val, tgt_tok)},
                before_frame,
                confidence=1.0,
                parse_rule=f"{voice_tag}_unary_typed",
            )
            return FrameParseResult(
                raw_instruction_bytes=raw,
                parser_name=self.name,
                parser_version=self.version,
                candidate_frames=(frame,),
                is_ambiguous=False,
                abstained=False,
            )

        # --- Family 2: transfer_own (give) ---
        if family == "transfer_own":
            if src_toks:
                return self._abstain(raw, "unexpected_source_for_give")
            if len(bare_args) == 1 and len(dst_toks) == 1:
                obj_tok, rec_tok = bare_args[0], dst_toks[0]
                obj_val = self._resolve(obj_tok, before_frame)
                rec_val = self._resolve(rec_tok, before_frame)
                non_actor = {"object": (obj_val, obj_tok), "recipient": (rec_val, rec_tok)}
                if actor_binding is None:
                    cands, is_amb = self._expand_implicit_actor(
                        canonical_action,
                        verb_tok,
                        non_actor,
                        before_frame,
                        parse_rule=f"{voice_tag}_give_pp",
                    )
                    return FrameParseResult(
                        raw_instruction_bytes=raw,
                        parser_name=self.name,
                        parser_version=self.version,
                        candidate_frames=tuple(cands),
                        is_ambiguous=is_amb,
                        abstained=False,
                    )
                bindings = dict(non_actor)
                bindings["actor"] = actor_binding
                frame = self._build_frame(
                    canonical_action,
                    verb_tok,
                    bindings,
                    before_frame,
                    confidence=1.0,
                    parse_rule=f"{voice_tag}_give_pp_typed",
                )
                return FrameParseResult(
                    raw_instruction_bytes=raw,
                    parser_name=self.name,
                    parser_version=self.version,
                    candidate_frames=(frame,),
                    is_ambiguous=False,
                    abstained=False,
                )

            if len(bare_args) == 2 and not dst_toks and voice_tag != "passive":
                tok1, tok2 = bare_args[0], bare_args[1]
                val1 = self._resolve(tok1, before_frame)
                val2 = self._resolve(tok2, before_frame)
                t1 = self._entity_type(val1, before_frame)
                t2 = self._entity_type(val2, before_frame)

                io_do = {"recipient": (val1, tok1), "object": (val2, tok2)}
                do_io = {"object": (val1, tok1), "recipient": (val2, tok2)}

                if t1 == "agent" and t2 == "object":
                    chosen = io_do
                    rule = f"{voice_tag}_ditransitive_typed_disambiguated"
                    if actor_binding is None:
                        cands, is_amb = self._expand_implicit_actor(
                            canonical_action, verb_tok, chosen, before_frame, parse_rule=rule
                        )
                        return FrameParseResult(
                            raw_instruction_bytes=raw,
                            parser_name=self.name,
                            parser_version=self.version,
                            candidate_frames=tuple(cands),
                            is_ambiguous=is_amb,
                            abstained=False,
                        )
                    chosen["actor"] = actor_binding
                    f = self._build_frame(
                        canonical_action, verb_tok, chosen, before_frame, confidence=0.95, parse_rule=rule
                    )
                    return FrameParseResult(
                        raw_instruction_bytes=raw,
                        parser_name=self.name,
                        parser_version=self.version,
                        candidate_frames=(f,),
                        is_ambiguous=False,
                        abstained=False,
                    )

                if t1 == "object" and t2 == "agent":
                    chosen = do_io
                    rule = f"{voice_tag}_ditransitive_typed_reversed"
                    if actor_binding is None:
                        cands, is_amb = self._expand_implicit_actor(
                            canonical_action, verb_tok, chosen, before_frame, parse_rule=rule
                        )
                        return FrameParseResult(
                            raw_instruction_bytes=raw,
                            parser_name=self.name,
                            parser_version=self.version,
                            candidate_frames=tuple(cands),
                            is_ambiguous=is_amb,
                            abstained=False,
                        )
                    chosen["actor"] = actor_binding
                    f = self._build_frame(
                        canonical_action, verb_tok, chosen, before_frame, confidence=0.90, parse_rule=rule
                    )
                    return FrameParseResult(
                        raw_instruction_bytes=raw,
                        parser_name=self.name,
                        parser_version=self.version,
                        candidate_frames=(f,),
                        is_ambiguous=False,
                        abstained=False,
                    )

                cands_all: list[CandidateActionFrame] = []
                if actor_binding is None:
                    c1, _ = self._expand_implicit_actor(
                        canonical_action,
                        verb_tok,
                        io_do,
                        before_frame,
                        parse_rule=f"{voice_tag}_ditransitive_io_do",
                        extra_reasons=("ambiguous_double_object_roles",),
                    )
                    c2, _ = self._expand_implicit_actor(
                        canonical_action,
                        verb_tok,
                        do_io,
                        before_frame,
                        parse_rule=f"{voice_tag}_ditransitive_do_io",
                        extra_reasons=("ambiguous_double_object_roles",),
                    )
                    cands_all.extend(c1)
                    cands_all.extend(c2)
                else:
                    b1 = dict(io_do)
                    b1["actor"] = actor_binding
                    b2 = dict(do_io)
                    b2["actor"] = actor_binding
                    cands_all.append(
                        self._build_frame(
                            canonical_action,
                            verb_tok,
                            b1,
                            before_frame,
                            confidence=0.55,
                            parse_rule=f"{voice_tag}_ditransitive_ambiguous_io_do",
                            ambiguity_reasons=("ambiguous_double_object_roles",),
                        )
                    )
                    cands_all.append(
                        self._build_frame(
                            canonical_action,
                            verb_tok,
                            b2,
                            before_frame,
                            confidence=0.45,
                            parse_rule=f"{voice_tag}_ditransitive_ambiguous_do_io",
                            ambiguity_reasons=("ambiguous_double_object_roles",),
                        )
                    )
                return FrameParseResult(
                    raw_instruction_bytes=raw,
                    parser_name=self.name,
                    parser_version=self.version,
                    candidate_frames=tuple(cands_all),
                    is_ambiguous=True,
                    abstained=False,
                )

            return self._abstain(raw, "invalid_give_arity")
        # --- Family 3: transfer_loc (move) ---
        if family == "transfer_loc":
            if len(bare_args) == 1 and len(src_toks) == 1 and len(dst_toks) == 1:
                obj_tok, src_tok, dst_tok = bare_args[0], src_toks[0], dst_toks[0]
                non_actor = {
                    "object": (self._resolve(obj_tok, before_frame), obj_tok),
                    "source": (self._resolve(src_tok, before_frame), src_tok),
                    "destination": (self._resolve(dst_tok, before_frame), dst_tok),
                }
                if actor_binding is None:
                    cands, is_amb = self._expand_implicit_actor(
                        canonical_action,
                        verb_tok,
                        non_actor,
                        before_frame,
                        parse_rule=f"{voice_tag}_move_pp",
                    )
                    return FrameParseResult(
                        raw_instruction_bytes=raw,
                        parser_name=self.name,
                        parser_version=self.version,
                        candidate_frames=tuple(cands),
                        is_ambiguous=is_amb,
                        abstained=False,
                    )
                bindings = dict(non_actor)
                bindings["actor"] = actor_binding
                frame = self._build_frame(
                    canonical_action,
                    verb_tok,
                    bindings,
                    before_frame,
                    confidence=1.0,
                    parse_rule=f"{voice_tag}_move_pp_typed",
                )
                return FrameParseResult(
                    raw_instruction_bytes=raw,
                    parser_name=self.name,
                    parser_version=self.version,
                    candidate_frames=(frame,),
                    is_ambiguous=False,
                    abstained=False,
                )

            if len(bare_args) == 3 and not src_toks and not dst_toks and voice_tag != "passive":
                obj_tok, loc1_tok, loc2_tok = bare_args
                obj_val = self._resolve(obj_tok, before_frame)
                loc1_val = self._resolve(loc1_tok, before_frame)
                loc2_val = self._resolve(loc2_tok, before_frame)

                score_a = 0.50
                score_b = 0.50
                if self.use_state_types and before_frame is not None:
                    at_loc1 = (
                        before_frame.query_fact_status((obj_val, "located_on", loc1_val))
                        == FactEpistemicStatus.OBSERVED_PRESENT
                    )
                    at_loc2 = (
                        before_frame.query_fact_status((obj_val, "located_on", loc2_val))
                        == FactEpistemicStatus.OBSERVED_PRESENT
                    )
                    if at_loc1 and not at_loc2:
                        score_a, score_b = 0.70, 0.30
                    elif at_loc2 and not at_loc1:
                        score_a, score_b = 0.30, 0.70

                opt_a = {
                    "object": (obj_val, obj_tok),
                    "source": (loc1_val, loc1_tok),
                    "destination": (loc2_val, loc2_tok),
                }
                opt_b = {
                    "object": (obj_val, obj_tok),
                    "source": (loc2_val, loc2_tok),
                    "destination": (loc1_val, loc1_tok),
                }
                ordered = [(opt_a, score_a, "src_dst")] if score_a >= score_b else [(opt_b, score_b, "dst_src")]
                ordered.append(
                    (opt_b, score_b, "dst_src") if score_a >= score_b else (opt_a, score_a, "src_dst")
                )

                cands_move: list[CandidateActionFrame] = []
                for opt_roles, conf, suffix in ordered:
                    if actor_binding is None:
                        sub_cands, _ = self._expand_implicit_actor(
                            canonical_action,
                            verb_tok,
                            opt_roles,
                            before_frame,
                            parse_rule=f"{voice_tag}_move_unmarked_{suffix}",
                            extra_reasons=("unmarked_source_destination_order",),
                        )
                        cands_move.extend(sub_cands)
                    else:
                        b = dict(opt_roles)
                        b["actor"] = actor_binding
                        cands_move.append(
                            self._build_frame(
                                canonical_action,
                                verb_tok,
                                b,
                                before_frame,
                                confidence=conf,
                                parse_rule=f"{voice_tag}_move_unmarked_{suffix}",
                                ambiguity_reasons=("unmarked_source_destination_order",),
                            )
                        )
                return FrameParseResult(
                    raw_instruction_bytes=raw,
                    parser_name=self.name,
                    parser_version=self.version,
                    candidate_frames=tuple(cands_move),
                    is_ambiguous=True,
                    abstained=False,
                )

            if len(bare_args) == 1 and not src_toks and len(dst_toks) == 1:
                obj_tok, dst_tok = bare_args[0], dst_toks[0]
                obj_val = self._resolve(obj_tok, before_frame)
                dst_val = self._resolve(dst_tok, before_frame)
                non_actor_partial: dict[str, tuple[str, ByteToken | None]] = {
                    "object": (obj_val, obj_tok),
                    "destination": (dst_val, dst_tok),
                }
                inferred_src: str | None = None
                if self.use_state_types and before_frame is not None:
                    locs = [o for _s, _p, o in before_frame.facts_for(obj_val, "located_on")]
                    if len(locs) == 1:
                        inferred_src = locs[0]
                if inferred_src is not None:
                    non_actor_partial["source"] = (inferred_src, None)
                    reasons = ("implicit_source_from_state",)
                    is_amb_src = False
                    conf_val = 0.85
                else:
                    reasons = ("missing_source",)
                    is_amb_src = True
                    conf_val = 0.50

                if actor_binding is None:
                    cands, _ = self._expand_implicit_actor(
                        canonical_action,
                        verb_tok,
                        non_actor_partial,
                        before_frame,
                        parse_rule=f"{voice_tag}_move_dst_only",
                        extra_reasons=reasons,
                    )
                    return FrameParseResult(
                        raw_instruction_bytes=raw,
                        parser_name=self.name,
                        parser_version=self.version,
                        candidate_frames=tuple(cands),
                        is_ambiguous=True,
                        abstained=False,
                    )
                bindings = dict(non_actor_partial)
                bindings["actor"] = actor_binding
                f = self._build_frame(
                    canonical_action,
                    verb_tok,
                    bindings,
                    before_frame,
                    confidence=conf_val,
                    parse_rule=f"{voice_tag}_move_dst_only",
                    ambiguity_reasons=reasons,
                )
                return FrameParseResult(
                    raw_instruction_bytes=raw,
                    parser_name=self.name,
                    parser_version=self.version,
                    candidate_frames=(f,),
                    is_ambiguous=is_amb_src,
                    abstained=False,
                )

            return self._abstain(raw, "invalid_move_arity")

        # --- Family 4: polysemous_transfer (transfer / send) ---
        if family == "polysemous_transfer":
            if len(bare_args) == 1 and len(src_toks) == 1 and len(dst_toks) == 1:
                return self._dispatch_family(
                    raw=raw,
                    canonical_action="move",
                    family="transfer_loc",
                    verb_tok=verb_tok,
                    actor_tok=actor_tok,
                    bare_args=bare_args,
                    src_toks=src_toks,
                    dst_toks=dst_toks,
                    before_frame=before_frame,
                    voice_tag=f"{voice_tag}_polysemous_src_pp",
                )
            if len(bare_args) == 1 and not src_toks and len(dst_toks) == 1:
                obj_tok, dst_tok = bare_args[0], dst_toks[0]
                obj_val = self._resolve(obj_tok, before_frame)
                dst_val = self._resolve(dst_tok, before_frame)
                dst_type = self._entity_type(dst_val, before_frame)

                if dst_type == "agent":
                    return self._dispatch_family(
                        raw=raw,
                        canonical_action="give",
                        family="transfer_own",
                        verb_tok=verb_tok,
                        actor_tok=actor_tok,
                        bare_args=bare_args,
                        src_toks=[],
                        dst_toks=dst_toks,
                        before_frame=before_frame,
                        voice_tag=f"{voice_tag}_polysemous_agent",
                    )
                if dst_type in ("surface", "container"):
                    return self._dispatch_family(
                        raw=raw,
                        canonical_action="move",
                        family="transfer_loc",
                        verb_tok=verb_tok,
                        actor_tok=actor_tok,
                        bare_args=bare_args,
                        src_toks=[],
                        dst_toks=dst_toks,
                        before_frame=before_frame,
                        voice_tag=f"{voice_tag}_polysemous_location",
                    )

                give_roles: dict[str, tuple[str, ByteToken | None]] = {
                    "object": (obj_val, obj_tok),
                    "recipient": (dst_val, dst_tok),
                }
                move_roles: dict[str, tuple[str, ByteToken | None]] = {
                    "object": (obj_val, obj_tok),
                    "destination": (dst_val, dst_tok),
                }
                cands_poly: list[CandidateActionFrame] = []
                if actor_binding is None:
                    c1, _ = self._expand_implicit_actor(
                        "give",
                        verb_tok,
                        give_roles,
                        before_frame,
                        parse_rule=f"{voice_tag}_polysemous_give",
                        extra_reasons=("polysemous_verb_occluded_target_type",),
                    )
                    c2, _ = self._expand_implicit_actor(
                        "move",
                        verb_tok,
                        move_roles,
                        before_frame,
                        parse_rule=f"{voice_tag}_polysemous_move",
                        extra_reasons=("polysemous_verb_occluded_target_type",),
                    )
                    cands_poly.extend(c1)
                    cands_poly.extend(c2)
                else:
                    g_b = dict(give_roles)
                    g_b["actor"] = actor_binding
                    m_b = dict(move_roles)
                    m_b["actor"] = actor_binding
                    cands_poly.append(
                        self._build_frame(
                            "give",
                            verb_tok,
                            g_b,
                            before_frame,
                            confidence=0.50,
                            parse_rule=f"{voice_tag}_polysemous_ambiguous_give",
                            ambiguity_reasons=("polysemous_verb_occluded_target_type",),
                        )
                    )
                    cands_poly.append(
                        self._build_frame(
                            "move",
                            verb_tok,
                            m_b,
                            before_frame,
                            confidence=0.50,
                            parse_rule=f"{voice_tag}_polysemous_ambiguous_move",
                            ambiguity_reasons=("polysemous_verb_occluded_target_type",),
                        )
                    )
                return FrameParseResult(
                    raw_instruction_bytes=raw,
                    parser_name=self.name,
                    parser_version=self.version,
                    candidate_frames=tuple(cands_poly),
                    is_ambiguous=True,
                    abstained=False,
                )

            return self._abstain(raw, "invalid_polysemous_transfer_arity")

        return self._abstain(raw, f"unsupported_family:{family}")

    def parse(
        self,
        raw_instruction_bytes: bytes,
        before_frame: TypedStateFrame | None = None,
    ) -> FrameParseResult:
        raw = bytes(raw_instruction_bytes)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return self._abstain(raw, "invalid_utf8")

        stripped = text.strip()
        if not stripped:
            return self._abstain(raw, "empty_instruction")

        tokens = tokenize_utf8_bytes(raw)
        if len(tokens) < 2:
            return self._abstain(raw, "insufficient_tokens")

        if stripped.endswith("?") and tokens[0].lower in _QUESTION_LEADS:
            return self._abstain(raw, "interrogative_non_action")

        idx = 0
        n = len(tokens)
        fronted_src: list[ByteToken] = []
        fronted_dst: list[ByteToken] = []

        while idx < n and tokens[idx].lower in _POLITE_PREFIXES:
            idx += 1

        while idx < n:
            prep_kind, consumed = _match_prep(tokens, idx)
            if prep_kind not in ("source", "destination"):
                break
            val_idx = _skip_det(tokens, idx + consumed)
            if val_idx >= n:
                return self._abstain(raw, "dangling_fronted_preposition")
            if prep_kind == "source":
                fronted_src.append(tokens[val_idx])
            else:
                fronted_dst.append(tokens[val_idx])
            idx = val_idx + 1

        while idx < n and tokens[idx].lower in _POLITE_PREFIXES:
            idx += 1

        if idx >= n:
            return self._abstain(raw, "missing_clause_after_fronted_pp")

        # Passive voice: "[det] <patient> (is|was|gets|should be|must be) <past_participle> ... [by <actor>]"
        p_idx = _skip_det(tokens, idx)
        if p_idx + 2 < n:
            aux_idx = p_idx + 1
            verb_cand_idx = -1
            if tokens[aux_idx].lower in _PASSIVE_AUX_SINGLE:
                verb_cand_idx = aux_idx + 1
            elif (
                aux_idx + 2 < n
                and tokens[aux_idx].lower in _PASSIVE_AUX_MODAL
                and tokens[aux_idx + 1].lower == "be"
            ):
                verb_cand_idx = aux_idx + 2
            if (
                verb_cand_idx != -1
                and verb_cand_idx < n
                and tokens[verb_cand_idx].lower in _PAST_PARTICIPLES
                and tokens[verb_cand_idx].lower in _ACTION_VERB_MAP
            ):
                patient_tok = tokens[p_idx]
                verb_tok = tokens[verb_cand_idx]
                canonical_action, family = _ACTION_VERB_MAP[verb_tok.lower]
                parsed_tail = self._parse_tail_args(tokens, verb_cand_idx + 1, allow_by=True)
                if parsed_tail is None:
                    return self._abstain(raw, "malformed_passive_tail")
                extra_bare, src_toks, dst_toks, by_toks = parsed_tail
                if extra_bare or len(by_toks) != 1:
                    return self._abstain(raw, "stative_or_malformed_passive")
                actor_tok = by_toks[0]
                return self._dispatch_family(
                    raw=raw,
                    canonical_action=canonical_action,
                    family=family,
                    verb_tok=verb_tok,
                    actor_tok=actor_tok,
                    bare_args=[patient_tok],
                    src_toks=fronted_src + src_toks,
                    dst_toks=fronted_dst + dst_toks,
                    before_frame=before_frame,
                    voice_tag="passive",
                )

        # Active / causative / imperative voice
        actor_tok: ByteToken | None = None
        verb_idx: int = -1
        voice_tag = "active"

        lead_lower = tokens[idx].lower
        if lead_lower in _CAUSATIVE_BARE:
            a_idx = _skip_det(tokens, idx + 1)
            if a_idx + 1 < n and tokens[a_idx + 1].lower in _ACTION_VERB_MAP:
                actor_tok = tokens[a_idx]
                verb_idx = a_idx + 1
                voice_tag = "causative"
        elif lead_lower in _CAUSATIVE_TO:
            a_idx = _skip_det(tokens, idx + 1)
            if (
                a_idx + 2 < n
                and tokens[a_idx + 1].lower == "to"
                and tokens[a_idx + 2].lower in _ACTION_VERB_MAP
            ):
                actor_tok = tokens[a_idx]
                verb_idx = a_idx + 2
                voice_tag = "causative_to"

        if verb_idx == -1:
            a_idx = _skip_det(tokens, idx)
            if a_idx + 1 < n and tokens[a_idx + 1].lower in _ACTION_VERB_MAP:
                actor_tok = tokens[a_idx]
                verb_idx = a_idx + 1
                voice_tag = "active"
            elif a_idx < n and tokens[a_idx].lower in _ACTION_VERB_MAP:
                actor_tok = None
                verb_idx = a_idx
                voice_tag = "imperative"
            else:
                return self._abstain(raw, "no_recognized_action_frame")

        verb_tok = tokens[verb_idx]
        canonical_action, family = _ACTION_VERB_MAP[verb_tok.lower]
        parsed_tail = self._parse_tail_args(tokens, verb_idx + 1, allow_by=False)
        if parsed_tail is None:
            return self._abstain(raw, "malformed_active_tail")
        bare_args, src_toks, dst_toks, _by_toks = parsed_tail

        return self._dispatch_family(
            raw=raw,
            canonical_action=canonical_action,
            family=family,
            verb_tok=verb_tok,
            actor_tok=actor_tok,
            bare_args=bare_args,
            src_toks=fronted_src + src_toks,
            dst_toks=fronted_dst + dst_toks,
            before_frame=before_frame,
            voice_tag=voice_tag,
        )
