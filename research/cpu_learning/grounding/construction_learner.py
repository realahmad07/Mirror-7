"""Stage S7 Raw-Byte Construction Grammar Induction & Compositional Clause Executor.

STRICT ISOLATION & SCIENTIFIC PRINCIPLES (PDF pp. 12, 13, 17):
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Learns ``ConstructionCard``s, ``LearnedNegationRule``s, ``LearnedConditionRule``s, and
  multi-step ``LearnedClauseConnector``s strictly from unlabelled ``(raw_utf8_bytes, before_state, after_state)``
  episodes by pairing ``SHARDProposalEngine`` transition recovery with UTF-8 byte-span abstraction.
- **Ambiguity Preservation & Minimal-Contrast Disambiguation (PDF p. 12):**
  Hearing an ungrounded surface pattern in a scene (``observe_utterance_in_scene``) or observing
  an ambiguous transition preserves ALL type-compatible ``CandidateActionFrame``s (``is_ambiguous=True``);
  subsequent minimal before/after contrast episodes refute inconsistent role mappings and promote
  the verified ``ConstructionCard`` (``version + 1``).
- **Epistemic Partial-Mask Safety:** Conditional clauses query ``TypedStateFrame.query_fact_status``
  and abstain honestly whenever the condition fact is ``FactEpistemicStatus.UNOBSERVED``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.frames.state_frame import (
    CandidateActionFrame,
    FactEpistemicStatus,
    FrameParseResult,
    TypedStateFrame,
)
from research.cpu_learning.grounding.byte_aligner import (
    EntityByteMention,
    UTF8TokenSpan,
    align_unlabelled_transition,
    decode_and_tokenize_utf8,
    locate_entity_mentions,
)
from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.shard.proposer import SHARDProposalEngine
from research.cpu_learning.simulator.state_graph import (
    ObservationMask,
    SparseDelta,
    StateGraph,
    canonical_json_bytes,
    compressed_bit_length,
    compute_sparse_delta,
)

CONSTRUCTION_LEARNER_SCHEMA_VERSION: str = "s7-construction-learner-v1"
_TRAILING_SENTENCE_PUNCT: frozenset[str] = frozenset({".", "!"})


class ConstructionLearnerFrozenError(RuntimeError):
    """Raised if fitting or contrast ingestion is attempted after ``freeze()``."""


@dataclass(frozen=True)
class UnlabelledGroundingEpisode:
    """Raw UTF-8 utterance paired ONLY with ``(before_state, after_state)`` (zero action/role labels)."""

    episode_id: str
    raw_instruction_bytes: bytes
    before_state: StateGraph
    after_state: StateGraph
    observation_mask: ObservationMask = ObservationMask(fully_observed=True)


@dataclass(frozen=True)
class ConstructionCard:
    """Learned mapping between an abstract lexical-slot pattern and an ``OperatorCard``'s roles."""

    construction_id: str
    action_id: str
    pattern_tokens: tuple[str, ...]
    slot_to_role: tuple[tuple[int, str], ...]
    role_allowed_types: tuple[tuple[str, frozenset[str]], ...]
    support_count: int = 1
    version: int = 1
    card_state: str = "promoted"
    provenance_episode_ids: tuple[str, ...] = ()

    def slot_map(self) -> dict[int, str]:
        return dict(self.slot_to_role)

    def allowed_types_map(self) -> dict[str, frozenset[str]]:
        return dict(self.role_allowed_types)

    def compressed_bits(self) -> int:
        return compressed_bit_length(canonical_json_bytes(self.to_executable_dict()))

    def to_executable_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "pattern_tokens": list(self.pattern_tokens),
            "slot_to_role": [[idx, r] for idx, r in self.slot_to_role],
            "role_allowed_types": {
                r: sorted(ts) for r, ts in sorted(self.role_allowed_types)
            },
        }

    def to_dict(self) -> dict[str, Any]:
        out = self.to_executable_dict()
        out["construction_id"] = self.construction_id
        out["support_count"] = self.support_count
        out["version"] = self.version
        out["card_state"] = self.card_state
        out["compressed_bits"] = self.compressed_bits()
        return out


@dataclass(frozen=True)
class LearnedNegationRule:
    """Prefix token sequence learned from suppressed no-op transitions."""

    prefix_tokens: tuple[str, ...]
    support_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_tokens": list(self.prefix_tokens),
            "support_count": self.support_count,
        }


@dataclass(frozen=True)
class LearnedConditionRule:
    """Conditional clause schema ``<intro_tokens> <ENT> <post_tokens> , <clause>`` learned by
    correlating ``TypedStateFrame`` literal facts across executed vs suppressed transitions."""

    intro_tokens: tuple[str, ...]
    post_tokens: tuple[str, ...]
    predicate: str
    expected_value: str
    support_executed: int
    support_suppressed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "intro_tokens": list(self.intro_tokens),
            "post_tokens": list(self.post_tokens),
            "predicate": self.predicate,
            "expected_value": self.expected_value,
            "support_executed": self.support_executed,
            "support_suppressed": self.support_suppressed,
        }


@dataclass(frozen=True)
class ClauseExecutionTrace:
    """Auditable trace of a single parsed & executed sub-clause."""

    clause_index: int
    raw_clause_utf8: str
    outcome: str
    action_id: str | None
    roles: dict[str, str]
    byte_spans: dict[str, tuple[int, int]]
    detail_reason: str
    added_facts: tuple[tuple[str, str, str], ...] = ()
    deleted_facts: tuple[tuple[str, str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_index": self.clause_index,
            "raw_clause_utf8": self.raw_clause_utf8,
            "outcome": self.outcome,
            "action_id": self.action_id,
            "roles": dict(self.roles),
            "byte_spans": {k: [s, e] for k, (s, e) in sorted(self.byte_spans.items())},
            "detail_reason": self.detail_reason,
            "added_facts": [list(f) for f in self.added_facts],
            "deleted_facts": [list(f) for f in self.deleted_facts],
        }


@dataclass(frozen=True)
class GroundedExecutionResult:
    """End-to-end result of executing a raw UTF-8 instruction against a ``StateGraph``."""

    final_state: StateGraph | None
    abstained: bool
    is_ambiguous: bool
    abstention_reason: str
    clause_traces: tuple[ClauseExecutionTrace, ...]
    intermediate_states: tuple[StateGraph, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstained": self.abstained,
            "is_ambiguous": self.is_ambiguous,
            "abstention_reason": self.abstention_reason,
            "clause_count": len(self.clause_traces),
            "clause_traces": [t.to_dict() for t in self.clause_traces],
        }


def _strip_edge_punct(tokens: Sequence[UTF8TokenSpan]) -> tuple[UTF8TokenSpan, ...]:
    start = 0
    end = len(tokens)
    while start < end and tokens[start].text in _TRAILING_SENTENCE_PUNCT:
        start += 1
    while end > start and tokens[end - 1].text in _TRAILING_SENTENCE_PUNCT:
        end -= 1
    return tuple(tokens[start:end])


def _abstract_tokens_to_pattern(
    tokens: Sequence[UTF8TokenSpan],
    entity_types_by_id: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[EntityByteMention, ...]]:
    """Replace observed entity mentions in ``tokens`` with ``$SLOT:0 .. $SLOT:k-1``."""
    trimmed = _strip_edge_punct(tokens)
    mentions = locate_entity_mentions(trimmed, entity_types_by_id)
    mention_by_tidx = {m.token_index: (slot_i, m) for slot_i, m in enumerate(mentions)}

    pattern: list[str] = []
    ordered_mentions: list[EntityByteMention] = []
    for tidx, tok in enumerate(trimmed):
        hit = mention_by_tidx.get(tidx)
        if hit is not None:
            slot_i, m = hit
            pattern.append(f"$SLOT:{slot_i}")
            ordered_mentions.append(m)
        else:
            pattern.append(tok.lower)
    return (tuple(pattern), tuple(ordered_mentions))


class RawByteConstructionLearner:
    """Learns construction schemas, negation prefixes, conditional guards, and multi-clause
    connectors directly from raw UTF-8 bytes + ``(before_state, after_state)`` transitions."""

    def __init__(
        self,
        operators: Mapping[str, OperatorCard],
        *,
        proposer: SHARDProposalEngine | None = None,
        min_negation_support: int = 2,
        min_condition_support: int = 2,
    ) -> None:
        self._operators: dict[str, OperatorCard] = {
            k: operators[k] for k in sorted(operators)
        }
        self._proposer = proposer if proposer is not None else SHARDProposalEngine(self._operators)
        self.min_negation_support = int(min_negation_support)
        self.min_condition_support = int(min_condition_support)

        # pattern_tokens -> list of active/shadow ConstructionCards
        self._constructions_by_pattern: dict[tuple[str, ...], list[ConstructionCard]] = {}
        self._history: list[ConstructionCard] = []
        self._negation_rules: dict[tuple[str, ...], LearnedNegationRule] = {}
        self._condition_rules: dict[tuple[tuple[str, ...], tuple[str, ...]], LearnedConditionRule] = {}
        self._clause_connectors: list[tuple[str, ...]] = []
        self._frozen = False
        self._seq = 0

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True
        self._proposer.freeze()

    @property
    def operators(self) -> dict[str, OperatorCard]:
        return dict(self._operators)

    @property
    def promoted_constructions(self) -> tuple[ConstructionCard, ...]:
        out: list[ConstructionCard] = []
        for cards in self._constructions_by_pattern.values():
            for c in cards:
                if c.card_state == "promoted":
                    out.append(c)
        return tuple(sorted(out, key=lambda c: c.construction_id))

    @property
    def negation_rules(self) -> tuple[LearnedNegationRule, ...]:
        return tuple(
            sorted(self._negation_rules.values(), key=lambda r: (-len(r.prefix_tokens), r.prefix_tokens))
        )

    @property
    def condition_rules(self) -> tuple[LearnedConditionRule, ...]:
        return tuple(
            sorted(
                self._condition_rules.values(),
                key=lambda r: (r.intro_tokens, r.post_tokens),
            )
        )

    @property
    def clause_connectors(self) -> tuple[tuple[str, ...], ...]:
        # Longest connectors first so "; then" matches before ";"
        return tuple(sorted(self._clause_connectors, key=lambda c: (-len(c), c)))

    def _next_cid(self, action_id: str) -> str:
        self._seq += 1
        return f"cx_{action_id}_{self._seq:04d}"

    def observe_utterance_in_scene(
        self,
        raw_bytes: bytes,
        scene_state: StateGraph,
        *,
        mask: ObservationMask | None = None,
    ) -> tuple[ConstructionCard, ...]:
        """Register all type-compatible ``shadow`` construction hypotheses for an ungrounded
        utterance heard in ``scene_state`` BEFORE outcome contrast is observed (PDF p. 12).

        Preserves every competing ``(action_id, slot_to_role)`` hypothesis consistent with
        observed entity types so ``parse_clause`` reports ``is_ambiguous=True`` until minimal
        contrast episodes disambiguate it.
        """
        if self._frozen:
            raise ConstructionLearnerFrozenError("Learner is frozen")
        valid_utf8, tokens = decode_and_tokenize_utf8(raw_bytes)
        if not valid_utf8 or not tokens:
            return ()
        obs = mask or ObservationMask(fully_observed=True)
        ent_types = {
            eid: ent.entity_type
            for eid, ent in scene_state.entities
            if obs.is_entity_observed(eid)
        }
        pattern, mentions = _abstract_tokens_to_pattern(tokens, ent_types)
        if not mentions:
            return ()

        # Do not overwrite already-promoted disambiguated constructions
        existing = self._constructions_by_pattern.get(pattern, [])
        if any(c.card_state == "promoted" for c in existing):
            return tuple(existing)

        mention_types = [m.entity_type for m in mentions]
        shadow_cards: list[ConstructionCard] = []
        seen_sigs: set[tuple[str, tuple[tuple[int, str], ...]]] = {
            (c.action_id, c.slot_to_role) for c in existing
        }

        for aid, op_card in sorted(self._operators.items()):
            r_vars = op_card.role_variables
            if len(r_vars) != len(mentions):
                continue
            allowed = op_card.allowed_types_map()
            for perm in itertools.permutations(r_vars):
                if all(
                    (not allowed.get(perm[i])) or (mention_types[i] in allowed[perm[i]])
                    for i in range(len(perm))
                ):
                    s2r = tuple((i, perm[i]) for i in range(len(perm)))
                    sig = (aid, s2r)
                    if sig not in seen_sigs:
                        seen_sigs.add(sig)
                        sc = ConstructionCard(
                            construction_id=self._next_cid(aid),
                            action_id=aid,
                            pattern_tokens=pattern,
                            slot_to_role=s2r,
                            role_allowed_types=op_card.required_types,
                            support_count=0,
                            version=1,
                            card_state="shadow",
                        )
                        shadow_cards.append(sc)
                        self._history.append(sc)

        if shadow_cards:
            self._constructions_by_pattern.setdefault(pattern, []).extend(shadow_cards)
        return tuple(self._constructions_by_pattern.get(pattern, ()))

    def _record_verified_alignment(
        self,
        pattern: tuple[str, ...],
        action_id: str,
        slot_to_role: tuple[tuple[int, str], ...],
        episode_id: str,
    ) -> ConstructionCard:
        op_card = self._operators[action_id]
        existing = self._constructions_by_pattern.get(pattern, [])

        # Retire any competing shadow hypotheses on this exact surface pattern!
        kept: list[ConstructionCard] = []
        matched_prev: ConstructionCard | None = None
        had_shadow = any(c.card_state == "shadow" for c in existing)

        for c in existing:
            if c.action_id == action_id and c.slot_to_role == slot_to_role:
                matched_prev = c
            elif c.card_state == "promoted":
                kept.append(c)
            else:
                # Competing shadow hypothesis refuted by verified transition contrast; retain in _history
                retired = ConstructionCard(
                    construction_id=c.construction_id,
                    action_id=c.action_id,
                    pattern_tokens=c.pattern_tokens,
                    slot_to_role=c.slot_to_role,
                    role_allowed_types=c.role_allowed_types,
                    support_count=c.support_count,
                    version=c.version,
                    card_state="retired",
                    provenance_episode_ids=c.provenance_episode_ids,
                )
                self._history.append(retired)

        if matched_prev is not None:
            new_ver = matched_prev.version + 1 if (had_shadow and matched_prev.card_state == "shadow") else matched_prev.version
            prov = tuple(dict.fromkeys(list(matched_prev.provenance_episode_ids) + [episode_id]))
            updated = ConstructionCard(
                construction_id=matched_prev.construction_id,
                action_id=action_id,
                pattern_tokens=pattern,
                slot_to_role=slot_to_role,
                role_allowed_types=op_card.required_types,
                support_count=matched_prev.support_count + 1,
                version=new_ver,
                card_state="promoted",
                provenance_episode_ids=prov,
            )
        else:
            ver = 2 if had_shadow else 1
            updated = ConstructionCard(
                construction_id=self._next_cid(action_id),
                action_id=action_id,
                pattern_tokens=pattern,
                slot_to_role=slot_to_role,
                role_allowed_types=op_card.required_types,
                support_count=1,
                version=ver,
                card_state="promoted",
                provenance_episode_ids=(episode_id,),
            )

        kept.append(updated)
        self._constructions_by_pattern[pattern] = kept
        self._history.append(updated)
        return updated

    def fit_episodes(
        self,
        episodes: Sequence[UnlabelledGroundingEpisode],
    ) -> dict[str, Any]:
        """Induce positive constructions (Pass 1), then negation prefixes, conditional guards,
        and sequential clause connectors (Pass 2) strictly from unlabelled transitions."""
        if self._frozen:
            raise ConstructionLearnerFrozenError("Cannot call fit_episodes() on a frozen learner")

        # ------------------------------------------------------------------
        # Pass 1: Positive Atomic Construction Induction
        # ------------------------------------------------------------------
        remaining_for_pass2: list[UnlabelledGroundingEpisode] = []

        for ep in episodes:
            aligned = align_unlabelled_transition(
                ep.raw_instruction_bytes,
                ep.before_state,
                ep.after_state,
                self._proposer,
                mask=ep.observation_mask,
            )
            if not aligned.valid_utf8 or not aligned.tokens:
                continue

            # Check if comma-prefixed conditional or multi-clause (contains ";" or ",")
            tok_lowers = [t.lower for t in _strip_edge_punct(aligned.tokens)]
            if "," in tok_lowers or ";" in tok_lowers or aligned.is_noop_transition:
                remaining_for_pass2.append(ep)
                if "," in tok_lowers or ";" in tok_lowers or aligned.is_noop_transition:
                    continue

            ent_types = {
                eid: ent.entity_type
                for eid, ent in ep.before_state.entities
                if ep.observation_mask.is_entity_observed(eid)
            }
            pattern, mentions = _abstract_tokens_to_pattern(aligned.tokens, ent_types)
            if not mentions or not aligned.candidates:
                remaining_for_pass2.append(ep)
                continue

            if len(aligned.candidates) == 1:
                cand = aligned.candidates[0]
                role_by_eid = {eid: rname for rname, eid in cand.roles}
                if all(m.entity_id in role_by_eid for m in mentions) and len(mentions) == len(cand.roles):
                    s2r = tuple((idx, role_by_eid[m.entity_id]) for idx, m in enumerate(mentions))
                    self._record_verified_alignment(pattern, cand.action_id, s2r, ep.episode_id)
                else:
                    remaining_for_pass2.append(ep)
            else:
                # Multiple transitions explain this episode -> check if an already-promoted card resolves it,
                # otherwise register all competing alignments as shadow until disambiguated!
                cur = self._constructions_by_pattern.setdefault(pattern, [])
                cand_s2rs: list[tuple[str, tuple[tuple[int, str], ...]]] = []
                for cand in aligned.candidates:
                    role_by_eid = {eid: rname for rname, eid in cand.roles}
                    if all(m.entity_id in role_by_eid for m in mentions) and len(mentions) == len(cand.roles):
                        s2r = tuple((idx, role_by_eid[m.entity_id]) for idx, m in enumerate(mentions))
                        cand_s2rs.append((cand.action_id, s2r))
                if not cand_s2rs:
                    remaining_for_pass2.append(ep)
                    continue
                promoted_matches = [
                    (aid, s2r)
                    for aid, s2r in cand_s2rs
                    if any(x.card_state == "promoted" and x.action_id == aid and x.slot_to_role == s2r for x in cur)
                ]
                if len(promoted_matches) == 1:
                    aid, s2r = promoted_matches[0]
                    self._record_verified_alignment(pattern, aid, s2r, ep.episode_id)
                else:
                    for aid, s2r in cand_s2rs:
                        if not any(x.action_id == aid and x.slot_to_role == s2r for x in cur):
                            sc = ConstructionCard(
                                construction_id=self._next_cid(aid),
                                action_id=aid,
                                pattern_tokens=pattern,
                                slot_to_role=s2r,
                                role_allowed_types=self._operators[aid].required_types,
                                support_count=1,
                                version=1,
                                card_state="shadow",
                                provenance_episode_ids=(ep.episode_id,),
                            )
                            cur.append(sc)
                            self._history.append(sc)

        # ------------------------------------------------------------------
        # Pass 2: Negation Prefixes, Conditional Clauses & Sequential Connectors
        # ------------------------------------------------------------------
        neg_counts: dict[tuple[str, ...], int] = {}
        # (intro_tokens, post_tokens) -> {"exec": [frozenset_facts], "supp": [frozenset_facts]}
        cond_evidence: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            dict[str, list[frozenset[tuple[str, str]]]],
        ] = {}
        connector_counts: dict[tuple[str, ...], int] = {}

        for ep in remaining_for_pass2:
            valid_utf8, raw_tokens = decode_and_tokenize_utf8(ep.raw_instruction_bytes)
            if not valid_utf8:
                continue
            tokens = _strip_edge_punct(raw_tokens)
            if not tokens:
                continue
            frame = TypedStateFrame.from_state_graph(ep.before_state, mask=ep.observation_mask)
            ent_types = {eid: ef.entity_type for eid, ef in frame.entities}
            obs_delta = compute_sparse_delta(ep.before_state, ep.after_state, mask=ep.observation_mask)

            # 2A. Conditional pattern: <prefix with 1 entity> "," <body_tokens>
            comma_indices = [i for i, t in enumerate(tokens) if t.text == ","]
            if len(comma_indices) == 1:
                c_idx = comma_indices[0]
                head_toks = tokens[:c_idx]
                body_toks = tokens[c_idx + 1 :]
                head_mentions = locate_entity_mentions(head_toks, ent_types)
                if len(head_mentions) == 1 and body_toks:
                    hm = head_mentions[0]
                    # Locate hm inside head_toks
                    hm_pos = hm.token_index
                    intro = tuple(t.lower for t in head_toks[:hm_pos])
                    post = tuple(t.lower for t in head_toks[hm_pos + 1 :])
                    body_bytes = ep.raw_instruction_bytes[body_toks[0].byte_start : body_toks[-1].byte_end]
                    body_parse = self.parse_atomic_clause(body_bytes, frame)
                    unamb = body_parse.unambiguous_frame
                    if intro and post and unamb is not None:
                        ent_facts = frozenset(
                            (p, v)
                            for _s, p, v in frame.facts_for(hm.entity_id)
                            if v in {"true", "false"}
                        )
                        if obs_delta.is_noop:
                            bucket = cond_evidence.setdefault((intro, post), {"exec": [], "supp": []})
                            bucket["supp"].append(ent_facts)
                            continue
                        op_card = self._operators[unamb.action_id]
                        sim_res = op_card.apply(ep.before_state, unamb.role_dict(), mask=ep.observation_mask)
                        if sim_res.applied and sim_res.after_state == ep.after_state:
                            bucket = cond_evidence.setdefault((intro, post), {"exec": [], "supp": []})
                            bucket["exec"].append(ent_facts)
                            continue

            # 2B. Negation prefix: obs_delta is no-op, prefix has 0 entities, suffix parses to an applicable action!
            if obs_delta.is_noop:
                matched_neg = False
                for split_k in range(1, min(4, len(tokens))):
                    pref_toks = tokens[:split_k]
                    suff_toks = tokens[split_k:]
                    if locate_entity_mentions(pref_toks, ent_types):
                        break
                    suff_bytes = ep.raw_instruction_bytes[suff_toks[0].byte_start : suff_toks[-1].byte_end]
                    suff_parse = self.parse_atomic_clause(suff_bytes, frame)
                    unamb = suff_parse.unambiguous_frame
                    if unamb is not None:
                        op_card = self._operators[unamb.action_id]
                        sim_res = op_card.apply(ep.before_state, unamb.role_dict(), mask=ep.observation_mask)
                        if sim_res.applied:
                            pref_key = tuple(t.lower for t in pref_toks)
                            neg_counts[pref_key] = neg_counts.get(pref_key, 0) + 1
                            matched_neg = True
                            break
                if matched_neg:
                    continue

            # 2C. Two-clause sequential connector discovery: <clause1> <conn_tokens> <clause2>
            for i in range(1, len(tokens) - 1):
                for conn_len in (1, 2, 3):
                    j = i + conn_len
                    if j >= len(tokens):
                        continue
                    conn_slice = tokens[i:j]
                    if locate_entity_mentions(conn_slice, ent_types):
                        continue
                    c1_toks = tokens[:i]
                    c2_toks = tokens[j:]
                    if not c1_toks or not c2_toks:
                        continue
                    b1 = ep.raw_instruction_bytes[c1_toks[0].byte_start : c1_toks[-1].byte_end]
                    p1 = self.parse_atomic_clause(b1, frame)
                    u1 = p1.unambiguous_frame
                    if u1 is None:
                        continue
                    r1 = self._operators[u1.action_id].apply(
                        ep.before_state, u1.role_dict(), mask=ep.observation_mask
                    )
                    if r1.abstained or r1.after_state is None:
                        continue
                    frame2 = TypedStateFrame.from_state_graph(r1.after_state, mask=ep.observation_mask)
                    b2 = ep.raw_instruction_bytes[c2_toks[0].byte_start : c2_toks[-1].byte_end]
                    p2 = self.parse_atomic_clause(b2, frame2)
                    u2 = p2.unambiguous_frame
                    if u2 is None:
                        continue
                    r2 = self._operators[u2.action_id].apply(
                        r1.after_state, u2.role_dict(), mask=ep.observation_mask
                    )
                    if (not r2.abstained) and r2.after_state == ep.after_state:
                        ckey = tuple(t.lower for t in conn_slice)
                        connector_counts[ckey] = connector_counts.get(ckey, 0) + 1

        for pref_key, cnt in sorted(neg_counts.items()):
            if cnt >= self.min_negation_support:
                prev_n = self._negation_rules.get(pref_key)
                tot_n = cnt + (prev_n.support_count if prev_n is not None else 0)
                self._negation_rules[pref_key] = LearnedNegationRule(pref_key, tot_n)

        for (intro, post), ev in sorted(cond_evidence.items()):
            exec_sets = ev["exec"]
            supp_sets = ev["supp"]
            if len(exec_sets) >= 1 and len(supp_sets) >= 1 and (len(exec_sets) + len(supp_sets)) >= self.min_condition_support:
                common_exec = set(exec_sets[0])
                for s in exec_sets[1:]:
                    common_exec &= s
                discriminating = [
                    (pred, val)
                    for pred, val in sorted(common_exec)
                    if all((pred, val) not in s_set for s_set in supp_sets)
                ]
                if len(discriminating) == 1:
                    pred, val = discriminating[0]
                    self._condition_rules[(intro, post)] = LearnedConditionRule(
                        intro_tokens=intro,
                        post_tokens=post,
                        predicate=pred,
                        expected_value=val,
                        support_executed=len(exec_sets),
                        support_suppressed=len(supp_sets),
                    )

        for ckey, cnt in sorted(connector_counts.items()):
            if cnt >= 1 and ckey not in self._clause_connectors:
                self._clause_connectors.append(ckey)

        return {
            "promoted_constructions": len(self.promoted_constructions),
            "negation_rules": len(self._negation_rules),
            "condition_rules": len(self._condition_rules),
            "clause_connectors": len(self._clause_connectors),
        }

    def parse_atomic_clause(
        self,
        raw_clause_bytes: bytes,
        state_frame: TypedStateFrame,
    ) -> FrameParseResult:
        """Parse a single atomic UTF-8 clause into 0, 1, or multiple ``CandidateActionFrame``s."""
        valid_utf8, tokens = decode_and_tokenize_utf8(raw_clause_bytes)
        if not valid_utf8:
            return FrameParseResult(
                raw_instruction_bytes=raw_clause_bytes,
                parser_name="raw_byte_construction_learner",
                parser_version=CONSTRUCTION_LEARNER_SCHEMA_VERSION,
                candidate_frames=(),
                is_ambiguous=False,
                abstained=True,
                abstention_reason="invalid_utf8_bytes",
            )

        trimmed = _strip_edge_punct(tokens)
        if not trimmed:
            return FrameParseResult(
                raw_instruction_bytes=raw_clause_bytes,
                parser_name="raw_byte_construction_learner",
                parser_version=CONSTRUCTION_LEARNER_SCHEMA_VERSION,
                candidate_frames=(),
                is_ambiguous=False,
                abstained=True,
                abstention_reason="empty_clause",
            )

        # Reject clauses mentioning occluded entity IDs directly (epistemic honesty)
        for tok in trimmed:
            if tok.text in state_frame.occluded_entity_ids or tok.lower in {
                e.casefold() for e in state_frame.occluded_entity_ids
            }:
                return FrameParseResult(
                    raw_instruction_bytes=raw_clause_bytes,
                    parser_name="raw_byte_construction_learner",
                    parser_version=CONSTRUCTION_LEARNER_SCHEMA_VERSION,
                    candidate_frames=(),
                    is_ambiguous=False,
                    abstained=True,
                    abstention_reason="mentioned_occluded_entity",
                )

        ent_types = {eid: ef.entity_type for eid, ef in state_frame.entities}
        pattern, mentions = _abstract_tokens_to_pattern(trimmed, ent_types)
        if not mentions:
            return FrameParseResult(
                raw_instruction_bytes=raw_clause_bytes,
                parser_name="raw_byte_construction_learner",
                parser_version=CONSTRUCTION_LEARNER_SCHEMA_VERSION,
                candidate_frames=(),
                is_ambiguous=False,
                abstained=True,
                abstention_reason="no_grounded_entities_in_clause",
            )

        matching_cards = [
            c
            for c in self._constructions_by_pattern.get(pattern, ())
            if c.card_state in {"promoted", "shadow"}
        ]
        # Prefer promoted cards over shadow cards if promoted exist
        promoted_here = [c for c in matching_cards if c.card_state == "promoted"]
        active_pool = promoted_here if promoted_here else matching_cards

        cand_frames: list[CandidateActionFrame] = []
        for cx in active_pool:
            smap = cx.slot_map()
            allowed_map = cx.allowed_types_map()
            if len(smap) != len(mentions):
                continue
            type_ok = True
            roles_list: list[tuple[str, str]] = []
            spans_list: list[tuple[str, int, int]] = []
            rtypes_list: list[tuple[str, str]] = []
            for s_idx, m in enumerate(mentions):
                rname = smap.get(s_idx)
                if rname is None:
                    type_ok = False
                    break
                allowed_ts = allowed_map.get(rname, frozenset())
                if allowed_ts and m.entity_type not in allowed_ts:
                    type_ok = False
                    break
                roles_list.append((rname, m.entity_id))
                spans_list.append((rname, m.byte_start, m.byte_end))
                rtypes_list.append((rname, m.entity_type))
            if not type_ok:
                continue
            cand_frames.append(
                CandidateActionFrame(
                    action_id=cx.action_id,
                    roles=tuple(roles_list),
                    byte_spans=tuple(spans_list),
                    role_types=tuple(rtypes_list),
                    confidence=1.0 if cx.card_state == "promoted" else 0.5,
                    parse_rule=cx.construction_id,
                    ambiguity_reasons=("shadow_uncontrasted_construction",) if cx.card_state == "shadow" else (),
                )
            )

        if not cand_frames:
            return FrameParseResult(
                raw_instruction_bytes=raw_clause_bytes,
                parser_name="raw_byte_construction_learner",
                parser_version=CONSTRUCTION_LEARNER_SCHEMA_VERSION,
                candidate_frames=(),
                is_ambiguous=False,
                abstained=True,
                abstention_reason="ungrounded_construction_pattern",
            )

        # Deduplicate identical (action_id, roles) predictions across synonymous rules
        dedup: dict[tuple[str, tuple[tuple[str, str], ...]], CandidateActionFrame] = {}
        for cf in cand_frames:
            key = (cf.action_id, cf.roles)
            if key not in dedup:
                dedup[key] = cf
        unique_cands = list(dedup.values())

        is_ambig = len(unique_cands) > 1 or any(c.confidence < 1.0 for c in unique_cands)
        if is_ambig and len(unique_cands) > 1:
            n_c = len(unique_cands)
            unique_cands = [
                CandidateActionFrame(
                    action_id=c.action_id,
                    roles=c.roles,
                    byte_spans=c.byte_spans,
                    role_types=c.role_types,
                    confidence=round(1.0 / float(n_c), 6),
                    parse_rule=c.parse_rule,
                    ambiguity_reasons=tuple(set(c.ambiguity_reasons) | {"multiple_competing_constructions"}),
                )
                for c in unique_cands
            ]

        return FrameParseResult(
            raw_instruction_bytes=raw_clause_bytes,
            parser_name="raw_byte_construction_learner",
            parser_version=CONSTRUCTION_LEARNER_SCHEMA_VERSION,
            candidate_frames=tuple(unique_cands),
            is_ambiguous=is_ambig,
            abstained=False,
        )

    def compressed_grammar_bits(self) -> int:
        payload = {
            "constructions": [c.to_executable_dict() for c in self.promoted_constructions],
            "negations": [r.to_dict() for r in self.negation_rules],
            "conditions": [r.to_dict() for r in self.condition_rules],
            "connectors": [list(c) for c in self.clause_connectors],
        }
        return compressed_bit_length(canonical_json_bytes(payload))


class GroundedInstructionExecutor:
    """Executes single- and multi-step raw UTF-8 instructions (active, passive, SOV, negated,
    conditional, and chained depth 1..6) strictly through learned constructions & ``OperatorCard``s."""

    def __init__(self, construction_learner: RawByteConstructionLearner) -> None:
        self.learner = construction_learner

    def _split_clauses(
        self,
        raw_bytes: bytes,
        tokens: Sequence[UTF8TokenSpan],
    ) -> list[tuple[bytes, int]]:
        """Split ``raw_bytes`` on learned sequential connectors while preserving global byte offsets."""
        connectors = self.learner.clause_connectors
        if not connectors:
            return [(raw_bytes, 0)]

        trimmed = _strip_edge_punct(tokens)
        if not trimmed:
            return [(raw_bytes, 0)]

        clauses: list[tuple[bytes, int]] = []
        seg_start_idx = 0
        idx = 0
        n = len(trimmed)

        while idx < n:
            matched_conn_len = 0
            for conn in connectors:
                clen = len(conn)
                if idx + clen <= n and tuple(t.lower for t in trimmed[idx : idx + clen]) == conn:
                    matched_conn_len = clen
                    break
            if matched_conn_len > 0:
                if idx > seg_start_idx:
                    b_start = trimmed[seg_start_idx].byte_start
                    b_end = trimmed[idx - 1].byte_end
                    clauses.append((raw_bytes[b_start:b_end], b_start))
                idx += matched_conn_len
                seg_start_idx = idx
            else:
                idx += 1

        if seg_start_idx < n:
            b_start = trimmed[seg_start_idx].byte_start
            b_end = trimmed[n - 1].byte_end
            clauses.append((raw_bytes[b_start:b_end], b_start))

        return clauses if clauses else [(raw_bytes, 0)]

    def execute_raw_utf8(
        self,
        raw_instruction_bytes: bytes,
        initial_state: StateGraph,
        *,
        mask: ObservationMask | None = None,
    ) -> GroundedExecutionResult:
        """Parse and execute ``raw_instruction_bytes`` end-to-end against ``initial_state``."""
        obs = mask or ObservationMask(fully_observed=True)
        valid_utf8, all_tokens = decode_and_tokenize_utf8(raw_instruction_bytes)
        if not valid_utf8 or not _strip_edge_punct(all_tokens):
            return GroundedExecutionResult(
                final_state=None,
                abstained=True,
                is_ambiguous=False,
                abstention_reason="invalid_or_empty_utf8",
                clause_traces=(),
                intermediate_states=(),
            )

        clause_slices = self._split_clauses(raw_instruction_bytes, all_tokens)
        cur_state = initial_state.copy()
        traces: list[ClauseExecutionTrace] = []
        intermediates: list[StateGraph] = []

        for c_idx, (c_bytes, g_offset) in enumerate(clause_slices):
            c_text = c_bytes.decode("utf-8")
            _, c_toks_raw = decode_and_tokenize_utf8(c_bytes)
            c_toks = _strip_edge_punct(c_toks_raw)
            frame = TypedStateFrame.from_state_graph(cur_state, mask=obs)
            ent_types = {eid: ef.entity_type for eid, ef in frame.entities}

            # 1. Check Conditional Guard: <intro> <ENT> <post> "," <body>
            comma_pos = [i for i, t in enumerate(c_toks) if t.text == ","]
            target_body_bytes = c_bytes
            target_body_offset = g_offset
            cond_false_detail: str | None = None

            if len(comma_pos) == 1:
                cp = comma_pos[0]
                head_toks = c_toks[:cp]
                body_toks = c_toks[cp + 1 :]
                if not body_toks:
                    return GroundedExecutionResult(
                        final_state=None,
                        abstained=True,
                        is_ambiguous=False,
                        abstention_reason="empty_conditional_body",
                        clause_traces=tuple(traces),
                        intermediate_states=tuple(intermediates),
                    )
                # Check if head mentions an occluded entity!
                for ht in head_toks:
                    if ht.text in frame.occluded_entity_ids or ht.lower in {
                        e.casefold() for e in frame.occluded_entity_ids
                    }:
                        return GroundedExecutionResult(
                            final_state=None,
                            abstained=True,
                            is_ambiguous=False,
                            abstention_reason="unobserved_condition_entity_under_mask",
                            clause_traces=tuple(traces),
                            intermediate_states=tuple(intermediates),
                        )
                head_mentions = locate_entity_mentions(head_toks, ent_types)
                if len(head_mentions) != 1:
                    return GroundedExecutionResult(
                        final_state=None,
                        abstained=True,
                        is_ambiguous=False,
                        abstention_reason="ungrounded_conditional_entity",
                        clause_traces=tuple(traces),
                        intermediate_states=tuple(intermediates),
                    )
                hm = head_mentions[0]
                intro_k = tuple(t.lower for t in head_toks[: hm.token_index])
                post_k = tuple(t.lower for t in head_toks[hm.token_index + 1 :])
                cond_rule = self.learner._condition_rules.get((intro_k, post_k))
                if cond_rule is None:
                    return GroundedExecutionResult(
                        final_state=None,
                        abstained=True,
                        is_ambiguous=False,
                        abstention_reason="ungrounded_condition_cue",
                        clause_traces=tuple(traces),
                        intermediate_states=tuple(intermediates),
                    )

                # Epistemic 3-valued check on condition fact under ObservationMask!
                ep_val = frame.query_fact_status(
                    (hm.entity_id, cond_rule.predicate, cond_rule.expected_value)
                )
                if ep_val == FactEpistemicStatus.UNOBSERVED:
                    return GroundedExecutionResult(
                        final_state=None,
                        abstained=True,
                        is_ambiguous=False,
                        abstention_reason="unobserved_condition_fact_under_mask",
                        clause_traces=tuple(traces),
                        intermediate_states=tuple(intermediates),
                    )

                b_rel_start = body_toks[0].byte_start
                b_rel_end = body_toks[-1].byte_end
                target_body_bytes = c_bytes[b_rel_start:b_rel_end]
                target_body_offset = g_offset + b_rel_start
                if ep_val != FactEpistemicStatus.OBSERVED_PRESENT:
                    cond_false_detail = f"condition_not_met:{hm.entity_id}.{cond_rule.predicate}"

            # 2. Check Learned Negation Prefix on target_body_bytes (symmetric across true & false conditions!)
            _, b_toks_raw = decode_and_tokenize_utf8(target_body_bytes)
            b_toks = _strip_edge_punct(b_toks_raw)
            negated = False
            for neg_rule in self.learner.negation_rules:
                plen = len(neg_rule.prefix_tokens)
                if len(b_toks) > plen and tuple(t.lower for t in b_toks[:plen]) == neg_rule.prefix_tokens:
                    rem_start = b_toks[plen].byte_start
                    rem_end = b_toks[-1].byte_end
                    target_body_bytes = target_body_bytes[rem_start:rem_end]
                    target_body_offset += rem_start
                    negated = True
                    break

            # 3. Parse atomic action clause (verifies body is grounded even when condition_false or negated!)
            parsed = self.learner.parse_atomic_clause(target_body_bytes, frame)
            if parsed.abstained or parsed.is_ambiguous or parsed.unambiguous_frame is None:
                fallback_reason = (
                    "ambiguous_conditional_body"
                    if cond_false_detail is not None
                    else "ambiguous_clause_parse"
                )
                return GroundedExecutionResult(
                    final_state=None,
                    abstained=True,
                    is_ambiguous=parsed.is_ambiguous,
                    abstention_reason=parsed.abstention_reason or fallback_reason,
                    clause_traces=tuple(traces),
                    intermediate_states=tuple(intermediates),
                )

            unamb = parsed.unambiguous_frame
            global_spans = {
                k: (s + target_body_offset, e + target_body_offset)
                for k, (s, e) in unamb.span_dict().items()
            }

            if cond_false_detail is not None:
                traces.append(
                    ClauseExecutionTrace(
                        clause_index=c_idx,
                        raw_clause_utf8=c_text,
                        outcome="condition_false_noop",
                        action_id=unamb.action_id,
                        roles=unamb.role_dict(),
                        byte_spans=global_spans,
                        detail_reason=cond_false_detail,
                    )
                )
                intermediates.append(cur_state.copy())
                continue

            if negated:
                traces.append(
                    ClauseExecutionTrace(
                        clause_index=c_idx,
                        raw_clause_utf8=c_text,
                        outcome="negated_noop",
                        action_id=unamb.action_id,
                        roles=unamb.role_dict(),
                        byte_spans=global_spans,
                        detail_reason="suppressed_by_learned_negation",
                    )
                )
                intermediates.append(cur_state.copy())
                continue

            op_card = self.learner.operators.get(unamb.action_id)
            if op_card is None:
                return GroundedExecutionResult(
                    final_state=None,
                    abstained=True,
                    is_ambiguous=False,
                    abstention_reason="unknown_operator_card",
                    clause_traces=tuple(traces),
                    intermediate_states=tuple(intermediates),
                )

            app = op_card.apply(cur_state, unamb.role_dict(), mask=obs)
            if app.abstained or app.after_state is None:
                return GroundedExecutionResult(
                    final_state=None,
                    abstained=True,
                    is_ambiguous=False,
                    abstention_reason=app.reason,
                    clause_traces=tuple(traces),
                    intermediate_states=tuple(intermediates),
                )

            step_delta: SparseDelta = app.delta or compute_sparse_delta(cur_state, app.after_state, mask=obs)
            cur_state = app.after_state
            traces.append(
                ClauseExecutionTrace(
                    clause_index=c_idx,
                    raw_clause_utf8=c_text,
                    outcome="applied" if app.applied else "blocked",
                    action_id=unamb.action_id,
                    roles=unamb.role_dict(),
                    byte_spans=global_spans,
                    detail_reason=app.reason,
                    added_facts=step_delta.added_facts,
                    deleted_facts=step_delta.deleted_facts,
                )
            )
            intermediates.append(cur_state.copy())

        return GroundedExecutionResult(
            final_state=cur_state,
            abstained=False,
            is_ambiguous=False,
            abstention_reason="",
            clause_traces=tuple(traces),
            intermediate_states=tuple(intermediates),
        )
