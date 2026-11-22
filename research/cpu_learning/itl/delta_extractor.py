"""Public EpisodeLedger observation extraction and empirical corpus statistics for Stage S2 ITL.

STRICT ISOLATION:
- Reads exclusively from S1's append-only ``EpisodeLedger`` (public observations only).
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Separates effect-bearing vs no-op episodes purely from observed ``SparseDelta.is_noop``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from research.cpu_learning.frames.state_frame import TypedEntityFrame, TypedStateFrame
from research.cpu_learning.itl.role_unifier import extract_entity_literal_facts
from research.cpu_learning.ledger.episode_ledger import EpisodeLedger, LedgerEpisodeRecord
from research.cpu_learning.simulator.state_graph import SparseDelta


@dataclass(frozen=True)
class ExtractedEpisodeObservation:
    """Learner-facing public episode observation pulled from S1 EpisodeLedger."""

    episode_id: str
    record_sha256: str
    action_id: str
    roles: dict[str, str]
    before_frame: TypedStateFrame
    after_frame: TypedStateFrame
    sparse_delta: SparseDelta
    has_effect: bool


@dataclass
class CorpusStatistics:
    """Empirical attribute and predicate distributions across training before-states.

    Used strictly for domain-agnostic MDL redundancy pruning (distinguishing constant
    background attributes from varying properties and detecting functionally entailed
    summary attributes). Contains zero hardcoded domain knowledge.
    """

    attr_domains_by_type: dict[tuple[str, str], set[Any]] = field(default_factory=dict)
    literal_fact_domains_by_type: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    global_literal_fact_domains: dict[str, set[str]] = field(default_factory=dict)
    entity_instances_by_type: dict[
        str, list[tuple[TypedEntityFrame, frozenset[tuple[str, str]]]]
    ] = field(default_factory=dict)
    edited_attrs_by_type: dict[str, set[str]] = field(default_factory=dict)
    varying_pos_role_attrs_by_type: dict[tuple[str, str], set[tuple[str, str]]] = field(
        default_factory=dict
    )

    def attr_domain_for_types(self, entity_types: frozenset[str], attr_name: str) -> set[Any]:
        vals: set[Any] = set()
        for et in entity_types:
            vals.update(self.attr_domains_by_type.get((et, attr_name), ()))
        return vals

    def is_attr_edited_on_types(self, entity_types: frozenset[str], attr_name: str) -> bool:
        return any(attr_name in self.edited_attrs_by_type.get(et, ()) for et in entity_types)

    def other_varying_pos_role_slots(
        self,
        entity_types: frozenset[str],
        attr_name: str,
        *,
        exclude_action: str,
        exclude_role: str,
    ) -> set[tuple[str, str]]:
        slots: set[tuple[str, str]] = set()
        for et in entity_types:
            slots.update(self.varying_pos_role_attrs_by_type.get((et, attr_name), ()))
        slots.discard((exclude_action, exclude_role))
        return slots

    def literal_fact_domain_for_type(self, entity_type: str, predicate: str) -> set[str]:
        return set(self.literal_fact_domains_by_type.get((entity_type, predicate), ()))

    def global_literal_fact_domain(self, predicate: str) -> set[str]:
        return set(self.global_literal_fact_domains.get(predicate, ()))


def record_to_observation(rec: LedgerEpisodeRecord) -> ExtractedEpisodeObservation | None:
    """Convert an unambiguous EpisodeLedger record into an ExtractedEpisodeObservation."""
    if not rec.selected_action_id or rec.selected_roles is None or rec.is_action_ambiguous:
        return None
    return ExtractedEpisodeObservation(
        episode_id=rec.episode_id,
        record_sha256=rec.record_sha256,
        action_id=rec.selected_action_id,
        roles=dict(rec.selected_roles),
        before_frame=rec.before_frame,
        after_frame=rec.after_frame,
        sparse_delta=rec.sparse_delta,
        has_effect=not rec.sparse_delta.is_noop,
    )


def extract_observations_from_ledger(
    ledger: EpisodeLedger,
    *,
    split_id: str | None = None,
    seed: int | None = None,
    caller_tag: str = "itl_delta_extractor",
) -> dict[str, list[ExtractedEpisodeObservation]]:
    """Read public records from ``EpisodeLedger`` and group by ``selected_action_id``."""
    if split_id is not None:
        records = ledger.query_by_split(split_id, seed=seed, caller_tag=caller_tag)
    else:
        records = ledger.iter_records(caller_tag=caller_tag)
        if seed is not None:
            records = tuple(r for r in records if r.seed == int(seed))

    grouped: dict[str, list[ExtractedEpisodeObservation]] = {}
    for rec in records:
        obs = record_to_observation(rec)
        if obs is None:
            continue
        grouped.setdefault(obs.action_id, []).append(obs)
    return grouped


def partition_positive_negative(
    observations: Sequence[ExtractedEpisodeObservation],
) -> tuple[list[ExtractedEpisodeObservation], list[ExtractedEpisodeObservation]]:
    """Partition demonstrations into effect-bearing vs zero-delta purely by observed SparseDelta."""
    pos = [o for o in observations if o.has_effect]
    neg = [o for o in observations if not o.has_effect]
    return (pos, neg)


def collect_corpus_statistics(
    grouped_observations: dict[str, list[ExtractedEpisodeObservation]],
) -> CorpusStatistics:
    """Compute empirical attribute/fact distributions and cross-operator variance statistics."""
    stats = CorpusStatistics()
    for action_id, obs_list in grouped_observations.items():
        pos_role_attr_vals: dict[tuple[str, str, str], set[Any]] = {}
        for obs in obs_list:
            frame = obs.before_frame
            for eid, ef in frame.entities:
                if not ef.attributes_observed:
                    continue
                etype = ef.entity_type
                lit_facts = extract_entity_literal_facts(frame, eid)
                stats.entity_instances_by_type.setdefault(etype, []).append((ef, lit_facts))
                for ak, av in ef.observed_attributes:
                    stats.attr_domains_by_type.setdefault((etype, ak), set()).add(av)
                for pred, val in lit_facts:
                    stats.literal_fact_domains_by_type.setdefault((etype, pred), set()).add(val)
                    stats.global_literal_fact_domains.setdefault(pred, set()).add(val)

            # Track attributes genuinely mutated by transitions (causally coupled attributes)
            for eid, attr_name, _old_v, _new_v in obs.sparse_delta.attribute_edits:
                ef = frame.get_entity(eid)
                if ef is not None and ef.attributes_observed:
                    stats.edited_attrs_by_type.setdefault(ef.entity_type, set()).add(attr_name)

            # Track per-(role, entity_type, attr) variation across positive demonstrations
            if obs.has_effect:
                for rname, eid in obs.roles.items():
                    ef = frame.get_entity(eid)
                    if ef is None or not ef.attributes_observed:
                        continue
                    for ak, av in ef.observed_attributes:
                        pos_role_attr_vals.setdefault((rname, ef.entity_type, ak), set()).add(av)

        for (rname, etype, ak), vals in pos_role_attr_vals.items():
            if len(vals) >= 2:
                stats.varying_pos_role_attrs_by_type.setdefault((etype, ak), set()).add(
                    (action_id, rname)
                )

    return stats
