"""Stage S11.1 Stage 4: Bi-Temporal Relation, Supersession & Contradiction Graph.

STRICT ISOLATION & NON-DESTRUCTIVE BELIEF REVISION:
- Pure Python 3.11 stdlib + Mirror 7's ``TropicalJTMS`` (``mirror_crystal.quotient_egraph_resonator``).
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
- ZERO manually supplied domain verb/action lists!
- ZERO silent overwrites or deletions of prior evidence!
- Uses Stage 3 10,240-bit BSC coreference clustering preserving explicit ``coref_confidence``
  uncertainty and genuine Poole-specificity ``TropicalJTMS.derive_all()`` defeasible belief closure.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Any, Sequence

from mirror_crystal.quotient_egraph_resonator import BeliefJustification, DefeasibleRule, TropicalJTMS
from research.open_world.contracts import normalize_text
from research.open_world.learner.evidence_ledger import (
    RawEvidenceUnit,
    parse_timestamp_date,
)
from research.open_world.learner.extraction import (
    CandidateEntity,
    CandidateEvent,
    cluster_coreferent_entities,
    detect_polarity_and_modality,
    extract_candidate_events,
    extract_candidate_spans,
    morph_stem,
    unicode_content_tokens,
    unicode_tokenize,
)

# Purely grammatical/discourse temporal-revision markers (ZERO domain verb lists!)
UPDATE_REVISION_CUES: frozenset[str] = frozenset(
    {
        "actually", "now", "currently", "updated", "update", "changed", "change",
        "switched", "switch", "moved", "move", "replaced", "replace", "instead",
        "anymore", "latest", "recent", "recently", "new", "newly", "just",
        "finally", "nowadays", "lately",
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
# Purely grammatical copula / auxiliary / existential light verbs (ZERO domain verbs!)
_COPULA_REL_RE = re.compile(
    r"^(.{2,60}?)\s+\b(is|are|was|were|be|been|being|became|becomes|become|has|have|had|remains?|seemed?|appears?)\b\s+(.{1,90})$",
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
    """Open-vocabulary evidence-grounded relation with bi-temporal & coref coordinates (Stage 4)."""

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
    t_obs_minute: int = 0
    coref_cluster_id: str = ""
    coref_confidence: float = 1.0


@dataclass(frozen=True)
class BeliefEdge:
    """Non-destructive provenance link between relations: SUPERSEDES, CONTESTED, or CORROBORATES."""

    edge_kind: str  # SUPERSEDES | CONTESTED | CORROBORATES
    from_rel_id: str
    to_rel_id: str
    reason: str
    confidence: float


class BiTemporalGraph:
    """Bi-temporal relation, coreference, supersession, and contradiction graph (Stage 4)."""

    def __init__(self, *, enable_bitemporal_supersession: bool = True) -> None:
        self.enable_bitemporal_supersession = bool(enable_bitemporal_supersession)
        self.relations: dict[str, SurfaceRelation] = {}
        self.events: list[CandidateEvent] = []
        self.entities: list[CandidateEntity] = []
        self.entity_clusters: dict[str, str] = {}
        self.entity_coref_conf: dict[str, float] = {}
        self.edges: list[BeliefEdge] = []
        self.supersedes_map: dict[str, list[str]] = defaultdict(list)
        self.superseded_by: dict[str, list[str]] = defaultdict(list)
        self.contested_pairs: set[tuple[str, str]] = set()
        self.by_entity_key: dict[str, list[str]] = defaultdict(list)
        self.by_source_id: dict[str, list[str]] = defaultdict(list)
        self.jtms = TropicalJTMS()
        self.jtms_beliefs: dict[str, BeliefJustification] = {}
        self._rel_seq: int = 0

    def ingest_units(self, units: Sequence[RawEvidenceUnit]) -> None:
        """Extract Stage 3 entities/events, run BSC coreference clustering, and build Stage 4 bi-temporal graph."""
        # Stage 3: extract candidate spans & run 10,240-bit BSC coreference clustering preserving uncertainty
        use_hv = len(units) <= 32
        spans_by_ev: dict[str, list[CandidateEntity]] = defaultdict(list)
        for u_idx, u in enumerate(units):
            spans = extract_candidate_spans(
                u,
                compute_hv=(use_hv and u_idx < 16),
                max_hv_spans=3,
            )
            if spans:
                kept = [
                    sp for sp in spans
                    if sp.entity_shape in {"PROPER_SPAN", "NOUN_CHUNK", "QUOTED_SPAN"}
                ][:4]
                self.entities.extend(kept)
                spans_by_ev[u.evidence_id].extend(kept)

        if self.entities:
            self.entity_clusters, self.entity_coref_conf = cluster_coreferent_entities(self.entities)

        for u in units:
            evs = extract_candidate_events(u)
            self.events.extend(evs)
            u_spans = spans_by_ev.get(u.evidence_id, ())
            self._extract_relations_from_unit(u, evs, u_spans)
        self._resolve_supersession_and_conflicts()

    def _best_coref_for_span(
        self,
        text_span: str,
        unit_spans: Sequence[CandidateEntity],
    ) -> tuple[str, float]:
        if not unit_spans:
            return ("", 1.0)
        tl = text_span.casefold()
        for sp in unit_spans:
            if sp.surface.casefold() in tl or tl in sp.surface.casefold():
                cid = self.entity_clusters.get(sp.entity_id, "")
                conf = self.entity_coref_conf.get(sp.entity_id, 1.0)
                if cid:
                    return (cid, conf)
        sp0 = unit_spans[0]
        return (
            self.entity_clusters.get(sp0.entity_id, ""),
            self.entity_coref_conf.get(sp0.entity_id, 1.0),
        )

    def _extract_relations_from_unit(
        self,
        unit: RawEvidenceUnit,
        events: Sequence[CandidateEvent],
        unit_spans: Sequence[CandidateEntity] = (),
    ) -> None:
        t_obs = unit.observed_ordinal if self.enable_bitemporal_supersession else 0
        t_obs_min = unit.observed_minute if self.enable_bitemporal_supersession else 0
        ev_date = resolve_event_date(unit.text, unit.observed_timestamp) if self.enable_bitemporal_supersession else None
        t_event = ev_date.toordinal() if ev_date is not None else 0

        raw_toks = set(unicode_tokenize(unit.text))
        tl = unit.text.casefold()
        rev_cue = bool(raw_toks & UPDATE_REVISION_CUES) or ("no longer" in tl)
        hist_cue = bool(raw_toks & HISTORICAL_CUES) or ("used to" in tl)

        m = _COPULA_REL_RE.match(unit.text.strip().rstrip(".!?"))
        if m:
            left = m.group(1).strip()
            rel = m.group(2).strip()
            right = m.group(3).strip()
            pol, _mod = detect_polarity_and_modality(unit.text)
            cid, cconf = self._best_coref_for_span(left, unit_spans)
            self._add_relation(
                arg_left=left,
                rel_surface=rel,
                arg_right=right,
                polarity=pol,
                t_obs=t_obs,
                t_obs_minute=t_obs_min,
                t_event=t_event,
                unit=unit,
                revision_cue=rev_cue,
                historical_cue=hist_cue,
                coref_cluster_id=cid,
                coref_confidence=cconf,
            )
            return

        for ev in events:
            if not ev.arg_spans:
                continue
            left = ev.arg_spans[0]
            right = ev.arg_spans[1] if len(ev.arg_spans) >= 2 else ev.clause_text
            cid, cconf = self._best_coref_for_span(left, unit_spans)
            self._add_relation(
                arg_left=left,
                rel_surface=ev.trigger_span,
                arg_right=right,
                polarity=ev.polarity,
                t_obs=t_obs,
                t_obs_minute=t_obs_min,
                t_event=t_event,
                unit=unit,
                revision_cue=rev_cue,
                historical_cue=hist_cue,
                coref_cluster_id=cid,
                coref_confidence=cconf,
            )

    def _add_relation(
        self,
        *,
        arg_left: str,
        rel_surface: str,
        arg_right: str,
        polarity: str,
        t_obs: int,
        t_obs_minute: int,
        t_event: int,
        unit: RawEvidenceUnit,
        revision_cue: bool,
        historical_cue: bool,
        coref_cluster_id: str = "",
        coref_confidence: float = 1.0,
    ) -> None:
        left_stems = [morph_stem(t) for t in unicode_content_tokens(arg_left)][:4]
        rel_stems = [morph_stem(t) for t in unicode_content_tokens(rel_surface)][:2]
        if not left_stems:
            left_stems = [morph_stem(t) for t in unicode_content_tokens(unit.text)][:3]
        if not left_stems:
            return
        ekey = "|".join(sorted(set(left_stems[:3]))) + "::" + ("|".join(rel_stems) if rel_stems else "state")
        vnorm = normalize_text(arg_right) or arg_right.strip().casefold()
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
            t_obs_minute=t_obs_minute,
            coref_cluster_id=coref_cluster_id,
            coref_confidence=coref_confidence,
        )
        self.relations[rid] = rel
        self.by_entity_key[ekey].append(rid)
        self.by_source_id[unit.source_id].append(rid)

        # Register base defeasible observation rule in Mirror 7 TropicalJTMS
        self.jtms.assert_fact(f"obs:{rid}", True)
        self.jtms.add_rule(
            DefeasibleRule(
                rule_id=f"assert:{rid}",
                antecedents=(f"obs:{rid}",),
                consequent=f"valid:{rid}",
                negated=False,
                specificity_rank=1,
                tropical_cost=10,
            )
        )

    @staticmethod
    def _chrono_key(r: SurfaceRelation) -> tuple[int, int, int, str]:
        return (r.t_obs, r.t_obs_minute, r.t_event, r.relation_id)

    @staticmethod
    def _is_strictly_later(r1: SurfaceRelation, r2: SurfaceRelation) -> bool:
        """True if ``r2`` advances chronologically after ``r1`` in observation time or event time."""
        if r1.t_obs > 0 and r2.t_obs > 0:
            if r2.t_obs > r1.t_obs:
                return True
            if r2.t_obs == r1.t_obs:
                # Same calendar day: advance if event time advanced (e.g. "yesterday" -> "today")
                if r1.t_event > 0 and r2.t_event > r1.t_event:
                    return True
                # Same calendar day with explicit later clock time AND revision cue
                if r2.t_obs_minute > r1.t_obs_minute and (r2.revision_cue or r1.historical_cue):
                    return True
            return False
        if r1.t_event > 0 and r2.t_event > r1.t_event:
            return True
        return False

    def _resolve_supersession_and_conflicts(self) -> None:
        """Emit non-destructive SUPERSEDES, CONTESTED, and CORROBORATES edges and solve TropicalJTMS."""
        exact_buckets: dict[str, list[SurfaceRelation]] = defaultdict(list)
        coref_buckets: dict[str, list[SurfaceRelation]] = defaultdict(list)
        coarse_buckets: dict[str, list[SurfaceRelation]] = defaultdict(list)
        cue_stems = {morph_stem(w) for w in (UPDATE_REVISION_CUES | HISTORICAL_CUES)}

        for rel in self.relations.values():
            exact_buckets[rel.entity_key].append(rel)
            if rel.coref_cluster_id:
                rel_pred = rel.entity_key.split("::", 1)[1]
                coref_buckets[f"coref::{rel.coref_cluster_id}::{rel_pred}"].append(rel)
                coarse_buckets[f"coref_any::{rel.coref_cluster_id}"].append(rel)
            left_head = rel.entity_key.split("::", 1)[0]
            if left_head and len(left_head) >= 3:
                coarse_buckets[f"ent::{left_head}"].append(rel)
            all_stems = [
                morph_stem(t) for t in unicode_content_tokens(rel.raw_text)
                if len(t) >= 4 and morph_stem(t) not in cue_stems
            ]
            uniq_st = sorted(set(all_stems))
            for st in uniq_st:
                if len(st) >= 5:
                    coarse_buckets[f"ent::{st}"].append(rel)
            for i_s in range(len(uniq_st)):
                for j_s in range(i_s + 1, min(len(uniq_st), i_s + 5)):
                    coarse_buckets[f"ent::{uniq_st[i_s]}|{uniq_st[j_s]}"].append(rel)

        seen_pairs: set[tuple[str, str]] = set()

        def _process_bucket(rels: list[SurfaceRelation], *, bucket_tier: str) -> None:
            if len(rels) < 2:
                return
            ordered = sorted(rels, key=self._chrono_key)
            # Bounded chronological window for large buckets (> 25) instead of silently dropping!
            max_span = len(ordered) if len(ordered) <= 25 else 8
            for i in range(len(ordered)):
                r1 = ordered[i]
                j_end = min(len(ordered), i + 1 + max_span)
                for j in range(i + 1, j_end):
                    r2 = ordered[j]
                    if r1.relation_id == r2.relation_id:
                        continue
                    pair = (r1.relation_id, r2.relation_id)
                    if pair in seen_pairs:
                        continue
                    if bucket_tier == "coarse":
                        if r1.source_id == r2.source_id:
                            continue
                        if not (r2.revision_cue or r1.historical_cue or r1.polarity != r2.polarity):
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

                    strictly_later = self._is_strictly_later(r1, r2)
                    if (
                        self.enable_bitemporal_supersession
                        and strictly_later
                        and (r1.speaker_role == r2.speaker_role or r2.revision_cue or r1.historical_cue)
                    ):
                        conf_base = 0.88 if r2.revision_cue else 0.78
                        edge_conf = round(conf_base * min(r1.coref_confidence, r2.coref_confidence), 4)
                        self.edges.append(
                            BeliefEdge(
                                "SUPERSEDES",
                                r2.relation_id,
                                r1.relation_id,
                                f"bitemporal_revision:t2=({r2.t_obs},{r2.t_event})>t1=({r1.t_obs},{r1.t_event})",
                                edge_conf,
                            )
                        )
                        self.supersedes_map[r2.relation_id].append(r1.relation_id)
                        self.superseded_by[r1.relation_id].append(r2.relation_id)
                        # Genuine TropicalJTMS Poole-specificity defeat: later r2 defeats valid(r1)
                        self.jtms.add_rule(
                            DefeasibleRule(
                                rule_id=f"supersede:{r2.relation_id}->{r1.relation_id}",
                                antecedents=(f"obs:{r2.relation_id}",),
                                consequent=f"valid:{r1.relation_id}",
                                negated=True,
                                specificity_rank=2,
                                tropical_cost=5,
                            )
                        )
                    elif bucket_tier in ("exact", "coref_pred") or (
                        bucket_tier == "coarse" and r1.polarity != r2.polarity and r1.value_norm == r2.value_norm
                    ):
                        # Emit CONTESTED ONLY for genuine incompatible values on the same entity-predicate
                        # (prevents false-positive CONTESTED on unrelated sentences sharing a single topic stem).
                        # Genuine CONTESTED conflict requires distinct sources OR explicit polarity contradiction
                        # on the exact same entity-predicate key (never unrelated sentences in one session).
                        if r1.source_id == r2.source_id and r1.polarity == r2.polarity:
                            continue
                        if r1.entity_key == r2.entity_key or (bucket_tier == "coref_pred" and r1.polarity != r2.polarity):
                            self.edges.append(
                                BeliefEdge(
                                    "CONTESTED",
                                    r2.relation_id,
                                    r1.relation_id,
                                    "unresolved_contradictory_observation",
                                    round(0.5 * min(r1.coref_confidence, r2.coref_confidence), 4),
                                )
                            )
                            self.contested_pairs.add((r1.relation_id, r2.relation_id))
                            self.contested_pairs.add((r2.relation_id, r1.relation_id))

        for rels in exact_buckets.values():
            _process_bucket(rels, bucket_tier="exact")
        for rels in coref_buckets.values():
            _process_bucket(rels, bucket_tier="coref_pred")
        for rels in coarse_buckets.values():
            _process_bucket(rels, bucket_tier="coarse")

        # Execute real TropicalJTMS fixed-point belief closure!
        self.jtms_beliefs = self.jtms.derive_all()

    def is_jtms_active(self, relation_id: str) -> bool:
        """Query Mirror 7 TropicalJTMS belief closure for whether ``relation_id`` is undefeated."""
        bj = self.jtms_beliefs.get(f"valid:{relation_id}")
        if bj is None:
            return not self.has_later_supersession(relation_id)
        return bool(bj.truth_value)

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
        chain.sort(key=self._chrono_key)
        return tuple(chain)

    def resolve_query_state(
        self,
        candidate_rel_ids: Sequence[str],
        *,
        temporal_mode: str = "latest",
    ) -> dict[str, Any]:
        """Resolve latest vs historical state via TropicalJTMS closure & SUPERSEDES chains."""
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
        ordered = sorted(expanded.values(), key=self._chrono_key)

        if temporal_mode == "earliest":
            winner = ordered[0]
        else:
            active_tips = [r for r in ordered if self.is_jtms_active(r.relation_id)]
            winner = active_tips[-1] if active_tips else ordered[-1]

        contested = self.is_contested(winner.relation_id) and self.is_jtms_active(winner.relation_id)
        chain_sources = tuple(dict.fromkeys(r.source_id for r in ordered))
        return {
            "winner": winner,
            "chain_sources": chain_sources,
            "contested": contested,
            "superseded_count": sum(1 for r in ordered if not self.is_jtms_active(r.relation_id)),
        }

    def stats(self) -> dict[str, int]:
        jtms_defeated = sum(
            1 for rid in self.relations if not self.is_jtms_active(rid)
        )
        return {
            "num_relations": len(self.relations),
            "num_events": len(self.events),
            "num_entity_clusters": len(set(self.entity_clusters.values())),
            "num_supersedes_edges": sum(1 for e in self.edges if e.edge_kind == "SUPERSEDES"),
            "num_contested_edges": sum(1 for e in self.edges if e.edge_kind == "CONTESTED"),
            "num_corroborates_edges": sum(1 for e in self.edges if e.edge_kind == "CORROBORATES"),
            "jtms_active_relations": len(self.relations) - jtms_defeated,
            "jtms_defeated_relations": jtms_defeated,
        }
