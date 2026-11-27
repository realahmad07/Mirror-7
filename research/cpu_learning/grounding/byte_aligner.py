"""Stage S7 Raw UTF-8 Byte-Span Tokenizer & Unlabelled Transition-to-Utterance Aligner.

STRICT ISOLATION (PDF pp. 12, 13, 17):
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Operates directly on raw UTF-8 ``bytes``, computing exact multi-byte-safe ``[start, end)``
  byte offsets and recovering ``(action_id, role_bindings)`` strictly from unlabelled
  ``(before_state, after_state)`` transitions via ``SHARDProposalEngine`` + ``OperatorCard``s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.shard.proposer import (
    SHARDProposalEngine,
    _StateBindingIndex,
    enumerate_compatible_bindings,
)
from research.cpu_learning.simulator.state_graph import (
    ObservationMask,
    SparseDelta,
    StateGraph,
    compute_sparse_delta,
)

BYTE_ALIGNER_SCHEMA_VERSION: str = "s7-byte-aligner-v1"

_PUNCTUATION_CHARS: frozenset[str] = frozenset({",", ";", ".", ":", "!", "?", "(", ")", "[", "]"})


@dataclass(frozen=True)
class UTF8TokenSpan:
    """Single token extracted from raw UTF-8 bytes with exact byte boundaries."""

    text: str
    lower: str
    byte_start: int
    byte_end: int
    is_punct: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "lower": self.lower,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "is_punct": self.is_punct,
        }


@dataclass(frozen=True)
class EntityByteMention:
    """Exact occurrence of a world entity identifier inside raw UTF-8 instruction bytes."""

    entity_id: str
    entity_type: str
    token_index: int
    byte_start: int
    byte_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "token_index": self.token_index,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
        }


@dataclass(frozen=True)
class TransitionAlignmentCandidate:
    """One verified ``(action_id, roles)`` hypothesis explaining ``(before_state -> after_state)``."""

    action_id: str
    roles: tuple[tuple[str, str], ...]
    role_byte_spans: tuple[tuple[str, int, int], ...]

    def role_dict(self) -> dict[str, str]:
        return dict(self.roles)

    def span_dict(self) -> dict[str, tuple[int, int]]:
        return {k: (s, e) for k, s, e in self.role_byte_spans}


@dataclass(frozen=True)
class AlignedUtteranceObservation:
    """Result of aligning raw UTF-8 bytes against an unlabelled ``(before_state, after_state)`` pair."""

    raw_bytes: bytes
    valid_utf8: bool
    tokens: tuple[UTF8TokenSpan, ...]
    mentions: tuple[EntityByteMention, ...]
    observed_delta: SparseDelta
    candidates: tuple[TransitionAlignmentCandidate, ...]

    @property
    def is_noop_transition(self) -> bool:
        return self.observed_delta.is_noop

    @property
    def unique_alignment(self) -> TransitionAlignmentCandidate | None:
        if len(self.candidates) == 1:
            return self.candidates[0]
        return None


def decode_and_tokenize_utf8(raw_bytes: bytes) -> tuple[bool, tuple[UTF8TokenSpan, ...]]:
    """Decode ``raw_bytes`` strictly as UTF-8 and segment into exact multi-byte-safe ``UTF8TokenSpan``s."""
    data = bytes(raw_bytes)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return (False, ())

    tokens: list[UTF8TokenSpan] = []
    cur_chars: list[str] = []
    cur_start = 0
    byte_pos = 0

    def _flush(end_pos: int) -> None:
        if cur_chars:
            tok = "".join(cur_chars)
            tokens.append(
                UTF8TokenSpan(
                    text=tok,
                    lower=tok.casefold(),
                    byte_start=cur_start,
                    byte_end=end_pos,
                    is_punct=False,
                )
            )
            cur_chars.clear()

    for ch in text:
        ch_len = len(ch.encode("utf-8"))
        if ch.isspace():
            _flush(byte_pos)
        elif ch in _PUNCTUATION_CHARS:
            _flush(byte_pos)
            tokens.append(
                UTF8TokenSpan(
                    text=ch,
                    lower=ch,
                    byte_start=byte_pos,
                    byte_end=byte_pos + ch_len,
                    is_punct=True,
                )
            )
        else:
            if not cur_chars:
                cur_start = byte_pos
            cur_chars.append(ch)
        byte_pos += ch_len

    _flush(byte_pos)
    return (True, tuple(tokens))


def locate_entity_mentions(
    tokens: Sequence[UTF8TokenSpan],
    entity_types_by_id: Mapping[str, str],
) -> tuple[EntityByteMention, ...]:
    """Locate all token occurrences matching observed entity IDs in ``entity_types_by_id``."""
    lower_to_id: dict[str, str] = {}
    ambig_lower: set[str] = set()
    for eid in sorted(entity_types_by_id.keys()):
        low = eid.casefold()
        if low in lower_to_id and lower_to_id[low] != eid:
            ambig_lower.add(low)
        else:
            lower_to_id[low] = eid

    mentions: list[EntityByteMention] = []
    for idx, tok in enumerate(tokens):
        if tok.is_punct:
            continue
        if tok.text in entity_types_by_id:
            eid = tok.text
        elif tok.lower not in ambig_lower and tok.lower in lower_to_id:
            eid = lower_to_id[tok.lower]
        else:
            continue
        mentions.append(
            EntityByteMention(
                entity_id=eid,
                entity_type=entity_types_by_id[eid],
                token_index=idx,
                byte_start=tok.byte_start,
                byte_end=tok.byte_end,
            )
        )
    return tuple(mentions)


def align_unlabelled_transition(
    raw_bytes: bytes,
    before_state: StateGraph,
    after_state: StateGraph,
    proposer: SHARDProposalEngine,
    *,
    mask: ObservationMask | None = None,
) -> AlignedUtteranceObservation:
    """Align ``raw_bytes`` against ``(before_state -> after_state)`` WITHOUT action/role labels.

    Uses ``SHARDProposalEngine`` first for fast 10,240-bit BSC transition recovery, then checks
    whether any additional ``(action_id, roles)`` over the mentioned entities also yields the
    identical ``SparseDelta`` so genuine ambiguity is preserved rather than hidden.
    """
    valid_utf8, tokens = decode_and_tokenize_utf8(raw_bytes)
    obs = mask or ObservationMask(fully_observed=True)
    ent_types = {
        eid: ent.entity_type
        for eid, ent in before_state.entities
        if obs.is_entity_observed(eid)
    }
    mentions = locate_entity_mentions(tokens, ent_types) if valid_utf8 else ()
    delta = compute_sparse_delta(before_state, after_state, mask=obs)

    if not valid_utf8 or delta.is_noop or not mentions:
        return AlignedUtteranceObservation(
            raw_bytes=bytes(raw_bytes),
            valid_utf8=valid_utf8,
            tokens=tokens,
            mentions=mentions,
            observed_delta=delta,
            candidates=(),
        )

    mention_span_by_eid: dict[str, tuple[int, int]] = {}
    for m in mentions:
        if m.entity_id not in mention_span_by_eid:
            mention_span_by_eid[m.entity_id] = (m.byte_start, m.byte_end)
    mentioned_set = set(mention_span_by_eid.keys())

    # Fast SHARD 10,240-bit BSC recovery + completeness check over mentioned entities
    shard_rec = proposer.recover_transition_shard_guided(before_state, delta, mask=obs)
    matched: list[TransitionAlignmentCandidate] = []
    seen_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    def _try_register(aid: str, roles_map: Mapping[str, str]) -> None:
        if not set(roles_map.values()).issubset(mentioned_set):
            return
        r_tup = tuple(sorted((str(k), str(v)) for k, v in roles_map.items()))
        key = (aid, r_tup)
        if key in seen_keys:
            return
        seen_keys.add(key)
        spans = tuple(
            sorted(
                (rname, mention_span_by_eid[eid][0], mention_span_by_eid[eid][1])
                for rname, eid in r_tup
            )
        )
        matched.append(
            TransitionAlignmentCandidate(
                action_id=aid,
                roles=r_tup,
                role_byte_spans=spans,
            )
        )

    if shard_rec.solved and shard_rec.action_id is not None and shard_rec.roles is not None:
        _try_register(shard_rec.action_id, shard_rec.roles)

    # Also verify over mentioned entities across operators in case SHARD found a non-mentioned
    # equivalent binding or multiple operators tie on this transition!
    sidx = _StateBindingIndex(before_state)
    restricted_order_by_card: dict[str, dict[str, list[str]]] = {}
    for aid, card in sorted(proposer.operators.items()):
        allowed_map = card.allowed_types_map()
        r_ord: dict[str, list[str]] = {}
        possible = True
        for rname in card.role_variables:
            allowed_ts = allowed_map.get(rname, frozenset())
            cands_here = [
                eid
                for eid in sorted(mentioned_set)
                if not allowed_ts or ent_types.get(eid) in allowed_ts
            ]
            if not cands_here:
                possible = False
                break
            r_ord[rname] = cands_here
        if possible:
            restricted_order_by_card[aid] = r_ord

    for aid, r_ord in sorted(restricted_order_by_card.items()):
        card: OperatorCard = proposer.operators[aid]
        for roles_map in enumerate_compatible_bindings(card, sidx, role_entity_order=r_ord):
            if not set(roles_map.values()).issubset(mentioned_set):
                continue
            res = card.apply(before_state, roles_map, mask=obs)
            if res.applied and res.delta == delta:
                _try_register(aid, roles_map)

    return AlignedUtteranceObservation(
        raw_bytes=bytes(raw_bytes),
        valid_utf8=valid_utf8,
        tokens=tokens,
        mentions=mentions,
        observed_delta=delta,
        candidates=tuple(matched),
    )
