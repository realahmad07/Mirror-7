"""Stage S11.1 Stage 4: Bi-Temporal Relation, Supersession & Contradiction Graph.

STRICT ISOLATION & NON-DESTRUCTIVE BELIEF REVISION:
- Pure Python 3.11 stdlib + Mirror 7's ``TropicalJTMS`` (``mirror_crystal.quotient_egraph_resonator``).
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
- ZERO silent overwrites or deletions of prior evidence!
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Sequence

from mirror_crystal.quotient_egraph_resonator import DefeasibleRule, TropicalJTMS
from research.open_world.contracts import content_tokens, normalize_text, tokenize
from research.open_world.learner.evidence_ledger import (
    RawEvidenceUnit,
    parse_timestamp_date,
)
from research.open_world.learner.extraction import (
    CandidateEvent,
    detect_polarity_and_modality,
    extract_candidate_events,
    morph_stem,
)

UPDATE_REVISION_CUES: frozenset[str] = frozenset(
    {
        "actually", "now", "currently", "updated", "update", "changed", "switched",
        "moved", "replaced", "instead", "anymore", "latest", "recent", "recently",
        "new", "newly", "bought", "painted", "upgraded", "started", "stopped", "quit",
        "decided", "just", "finally", "nowadays", "lately",
    }
)
HISTORICAL_CUES: frozenset[str] = frozenset(
    {
        "initially", "originally", "previously", "formerly", "before", "earlier",
        "used", "old", "previous", "first", "prior", "past",
    }
)
_WORD_NUMS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "couple": 2, "few": 3, "several": 4,
}
_AGO_RE = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple(?:\s+of)?|few)\s+"
    r"(days?|weeks?|months?|years?)\s+ago\b",
    re.I,
)
_COPULA_REL_RE = re.compile(
    r"^(.{2,60}?)\s+\b(is|are|was|were|became|has|have|had|lives?\s+in|kept|keep|keeps|moved\s+to|painted|switched\s+to|changed\s+to|works?\s+(?:at|in|as)|plays?|costs?|spent|added|owns?)\b\s+(.{1,90})$",
    re.I,
)


def resolve_event_date(text: str, obs_timestamp: str) -> date | None:
    """Resolve event date ``t_event`` from explicit dates or relative offsets against ``t_obs``."""
    explicit = parse_timestamp_date(text)
    if explicit is not None:
        return explicit

    base = parse_timestamp_date(obs_timestamp)
    if base is None:
        return None

    tl = text.casefold()
    if "day before yesterday" in tl:
        return base - timedelta(days=2)
    if "yesterday" in tl:
        return base - timedelta(days=1)
    if "last week" in tl:
        return base - timedelta(days=7)
    if "last month" in tl:
        return base - timedelta(days=30)
    if "last year" in tl:
        return base - timedelta(days=365)

    m = _AGO_RE.search(tl)
    if m:
        raw_n = m.group(1).split()[0]
        n = int(raw_n) if raw_n.isdigit() else _WORD_NUMS.get(raw_n, 1)
        unit = m.group(2).casefold()
        if unit.startswith("day"):
            return base - timedelta(days=n)
        if unit.startswith("week"):
            return base - timedelta(days=7 * n)
        if unit.startswith("month"):
            return base - timedelta(days=30 * n)
        if unit.startswith("year"):
            return base - timedelta(days=365 * n)

    return base


@dataclass(frozen=True)
class SurfaceRelation:
    """Open-vocabulary evidence-grounded relation with bi-temporal coordinates (Stage 4)."""

    relation_id: str
    arg_left: str
    rel_surface: str
    arg_right: str
    entity_key: str
    value_norm: str
    polarity: str  # AFFIRMED | NEGATED | UNCERTAIN
    t_obs: int
    t_event: int
    evidence_id: str
    source_id: str
    speaker_role: str
    revision_cue: bool
    historical_cue: bool
    raw_text: str


@dataclass(frozen=True)
class BeliefEdge:
    """Non-destructive provenance link between relations: SUPERSEDES, CONTESTED, or CORROBORATES."""

    edge_kind: str  # SUPERSEDES | CONTESTED | CORROBORATES
    from_rel_id: str
    to_rel_id: str
    reason: str
    confidence: float


class BiTemporalGraph:
    """Bi-temporal relation, supersession, and contradiction graph (Stage 4)."""

    def __init__(self, *, enable_bitemporal_supersession: bool = True) -> None:
        self.enable_bitemporal_supersession = bool(enable_bitemporal_supersession)
        self.relations: dict[str, SurfaceRelation] = {}
        self.events: list[CandidateEvent] = []
        self.edges: list[BeliefEdge] = []
        self.supersedes_map: dict[str, list[str]] = defaultdict(list)
        self.superseded_by: dict[str, list[str]] = defaultdict(list)
        self.contested_pairs: set[tuple[str, str]] = set()
        self.by_entity_key: dict[str, list[str]] = defaultdict(list)
        self.by_source_id: dict[str, list[str]] = defaultdict(list)
        self.jtms = TropicalJTMS()
        self._rel_seq: int = 0

    def ingest_units(self, units: Sequence[RawEvidenceUnit]) -> None:
        """Extract open-vocabulary events and surface relations from raw units and detect conflicts/supersessions."""
        for u in units:
            evs = extract_candidate_events(u)
            self.events.extend(evs)
            self._extract_relations_from_unit(u, evs)
        self._resolve_supersession_and_conflicts()

    def _extract_relations_from_unit(
        self,
        unit: RawEvidenceUnit,
        events: Sequence[CandidateEvent],
    ) -> None:
        t_obs = unit.observed_ordinal if self.enable_bitemporal_supersession else 0
        ev_date = resolve_event_date(unit.text, unit.observed_timestamp) if self.enable_bitemporal_supersession else None
        t_event = ev_date.toordinal() if ev_date is not None else 0

        raw_toks = set(tokenize(unit.text))
        rev_cue = bool(raw_toks & UPDATE_REVISION_CUES)
        hist_cue = bool(raw_toks & HISTORICAL_CUES) or ("used to" in unit.text.casefold())

        m = _COPULA_REL_RE.match(unit.text.strip().rstrip(".!?"))
        if m:
            left = m.group(1).strip()
            rel = m.group(2).strip()
            right = m.group(3).strip()
            pol, _mod = detect_polarity_and_modality(unit.text)
            self._add_relation(
                arg_left=left,
                rel_surface=rel,
                arg_right=right,
                polarity=pol,
                t_obs=t_obs,
                t_event=t_event,
                unit=unit,
                revision_cue=rev_cue,
                historical_cue=hist_cue,
            )
            return

        for ev in events:
            if not ev.arg_spans:
                continue
            left = ev.arg_spans[0]
            right = ev.arg_spans[1] if len(ev.arg_spans) >= 2 else ev.clause_text
            self._add_relation(
                arg_left=left,
                rel_surface=ev.trigger_span,
                arg_right=right,
                polarity=ev.polarity,
                t_obs=t_obs,
                t_event=t_event,
                unit=unit,
                revision_cue=rev_cue,
                historical_cue=hist_cue,
            )

    def _add_relation(
        self,
        *,
        arg_left: str,
        rel_surface: str,
        arg_right: str,
        polarity: str,
        t_obs: int,
        t_event: int,
        unit: RawEvidenceUnit,
        revision_cue: bool,
        historical_cue: bool,
    ) -> None:
        left_stems = [morph_stem(t) for t in content_tokens(arg_left)][:4]
        rel_stems = [morph_stem(t) for t in content_tokens(rel_surface)][:2]
        if not left_stems:
            left_stems = [morph_stem(t) for t in content_tokens(unit.text)][:3]
        if not left_stems:
            return
        ekey = "|".join(sorted(set(left_stems[:3]))) + "::" + ("|".join(rel_stems) if rel_stems else "state")
        vnorm = normalize_text(arg_right)
        rid = f"rel_{self._rel_seq}"
        self._rel_seq += 1

        rel = SurfaceRelation(
            relation_id=rid,
            arg_left=arg_left,
            rel_surface=rel_surface,
            arg_right=arg_right,
            entity_key=ekey,
            value_norm=vnorm,
            polarity=polarity,
            t_obs=t_obs,
            t_event=t_event,
            evidence_id=unit.evidence_id,
            source_id=unit.source_id,
            speaker_role=unit.speaker_role,
            revision_cue=revision_cue,
            historical_cue=historical_cue,
            raw_text=unit.text,
        )
        self.relations[rid] = rel
        self.by_entity_key[ekey].append(rid)
        self.by_source_id[unit.source_id].append(rid)

    def _resolve_supersession_and_conflicts(self) -> None:
        """Emit non-destructive SUPERSEDES, CONTESTED, and CORROBORATES edges across entity clusters."""
        entity_buckets: dict[str, list[SurfaceRelation]] = defaultdict(list)
        cue_stems = {morph_stem(w) for w in (UPDATE_REVISION_CUES | HISTORICAL_CUES)}
        for rel in self.relations.values():
            entity_buckets[rel.entity_key].append(rel)
            left_head = rel.entity_key.split("::", 1)[0]
            if left_head and len(left_head) >= 3:
                entity_buckets[f"ent::{left_head}"].append(rel)
            # Also bucket by shared entity stems across full clause (handles "I moved my telescope..." vs "I stored my telescope...")
            all_stems = [
                morph_stem(t) for t in content_tokens(rel.raw_text)
                if len(t) >= 4 and morph_stem(t) not in cue_stems
            ]
            uniq_st = sorted(set(all_stems))
            for st in uniq_st:
                if len(st) >= 5:
                    entity_buckets[f"ent::{st}"].append(rel)
            for i_s in range(len(uniq_st)):
                for j_s in range(i_s + 1, min(len(uniq_st), i_s + 5)):
                    entity_buckets[f"ent::{uniq_st[i_s]}|{uniq_st[j_s]}"].append(rel)

        seen_pairs: set[tuple[str, str]] = set()
        for bucket_key, rels in entity_buckets.items():
            if len(rels) < 2 or len(rels) > 25:
                continue
            is_coarse = bucket_key.startswith("ent::")
            ordered = sorted(rels, key=lambda r: (r.t_obs, r.t_event, r.relation_id))
            for i in range(len(ordered)):
                r1 = ordered[i]
                for j in range(i + 1, len(ordered)):
                    r2 = ordered[j]
                    if r1.relation_id == r2.relation_id:
                        continue
                    pair = (r1.relation_id, r2.relation_id)
                    if pair in seen_pairs:
                        continue
                    if is_coarse and r1.source_id == r2.source_id:
                        continue
                    if is_coarse and not (r2.revision_cue or r1.historical_cue or r1.polarity != r2.polarity):
                        continue
                    seen_pairs.add(pair)

                    same_val = (
                        r1.value_norm == r2.value_norm
                        and r1.polarity == r2.polarity
                    )
                    if same_val:
                        if r1.source_id != r2.source_id:
                            self.edges.append(
                                BeliefEdge("CORROBORATES", r2.relation_id, r1.relation_id, "multi_source_agreement", 0.9)
                            )
                        continue

                    t1 = r1.t_obs or r1.t_event
                    t2 = r2.t_obs or r2.t_event
                    if (
                        self.enable_bitemporal_supersession
                        and t1 > 0
                        and t2 > t1
                        and (r1.speaker_role == r2.speaker_role or r2.revision_cue or r1.historical_cue)
                    ):
                        self.edges.append(
                            BeliefEdge(
                                "SUPERSEDES",
                                r2.relation_id,
                                r1.relation_id,
                                f"bitemporal_revision:t2={t2}>t1={t1}",
                                0.88 if r2.revision_cue else 0.78,
                            )
                        )
                        self.supersedes_map[r2.relation_id].append(r1.relation_id)
                        self.superseded_by[r1.relation_id].append(r2.relation_id)
                        self.jtms.assert_fact(f"obs:{r1.relation_id}", True)
                        self.jtms.assert_fact(f"obs:{r2.relation_id}", True)
                        self.jtms.add_rule(
                            DefeasibleRule(
                                rule_id=f"old:{r1.relation_id}",
                                antecedents=(f"obs:{r1.relation_id}",),
                                consequent=f"active:{r1.entity_key}",
                                negated=False,
                                specificity_rank=1,
                                tropical_cost=10,
                            )
                        )
                        self.jtms.add_rule(
                            DefeasibleRule(
                                rule_id=f"new:{r2.relation_id}",
                                antecedents=(f"obs:{r2.relation_id}",),
                                consequent=f"active:{r1.entity_key}",
                                negated=False,
                                specificity_rank=2,
                                tropical_cost=5,
                            )
                        )
                    else:
                        self.edges.append(
                            BeliefEdge(
                                "CONTESTED",
                                r2.relation_id,
                                r1.relation_id,
                                "unresolved_contradictory_observation",
                                0.5,
                            )
                        )
                        self.contested_pairs.add((r1.relation_id, r2.relation_id))
                        self.contested_pairs.add((r2.relation_id, r1.relation_id))

    def is_contested(self, relation_id: str) -> bool:
        return any(a == relation_id for a, _ in self.contested_pairs)

    def has_later_supersession(self, relation_id: str) -> bool:
        return bool(self.superseded_by.get(relation_id))

    def get_supersession_chain(self, relation_id: str) -> tuple[SurfaceRelation, ...]:
        """Return full chronologically ordered chain of relations connected by SUPERSEDES edges."""
        visited: set[str] = {relation_id}
        queue = [relation_id]
        while queue:
            cur = queue.pop(0)
            for nb in list(self.supersedes_map.get(cur, ())) + list(self.superseded_by.get(cur, ())):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        chain = [self.relations[rid] for rid in visited if rid in self.relations]
        chain.sort(key=lambda r: (r.t_obs, r.t_event, r.relation_id))
        return tuple(chain)

    def resolve_query_state(
        self,
        candidate_rel_ids: Sequence[str],
        *,
        temporal_mode: str = "latest",
    ) -> dict[str, object]:
        """Resolve latest vs historical state over ``SUPERSEDES`` chains while surfacing conflicts."""
        rels = [self.relations[rid] for rid in candidate_rel_ids if rid in self.relations]
        if not rels:
            return {"winner": None, "chain_sources": (), "contested": False, "superseded_count": 0}

        if not self.enable_bitemporal_supersession:
            winner = rels[0]
            contested = any(self.is_contested(r.relation_id) for r in rels)
            return {
                "winner": winner,
                "chain_sources": (winner.source_id,),
                "contested": contested,
                "superseded_count": 0,
            }

        expanded: dict[str, SurfaceRelation] = {}
        for r in rels:
            for cr in self.get_supersession_chain(r.relation_id):
                expanded[cr.relation_id] = cr
        ordered = sorted(expanded.values(), key=lambda r: (r.t_obs, r.t_event, r.relation_id))

        if temporal_mode == "earliest":
            winner = ordered[0]
        else:
            tips = [r for r in ordered if not self.has_later_supersession(r.relation_id)]
            winner = tips[-1] if tips else ordered[-1]

        contested = self.is_contested(winner.relation_id)
        chain_sources = tuple(dict.fromkeys(r.source_id for r in ordered))
        return {
            "winner": winner,
            "chain_sources": chain_sources,
            "contested": contested,
            "superseded_count": sum(1 for r in ordered if self.has_later_supersession(r.relation_id)),
        }

    def stats(self) -> dict[str, int]:
        return {
            "num_relations": len(self.relations),
            "num_events": len(self.events),
            "num_supersedes_edges": sum(1 for e in self.edges if e.edge_kind == "SUPERSEDES"),
            "num_contested_edges": sum(1 for e in self.edges if e.edge_kind == "CONTESTED"),
            "num_corroborates_edges": sum(1 for e in self.edges if e.edge_kind == "CORROBORATES"),
        }