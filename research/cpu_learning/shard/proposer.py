"""Stage S4 SHARD-Guided Candidate Proposal Engine (Tasks A, B, and C).

STRICT ISOLATION & CORE PRINCIPLE (PDF pp. 9, 11, 14, 15):
- **Similarity is NOT proof:** SHARD 10,240-bit BSC hypervectors and multi-table LSH
  indexes ONLY propose and rank candidate ``(OperatorCard, role_binding)`` transitions.
  Every accepted transition or multi-step plan MUST be strictly verified by exact
  ``OperatorCard.apply()`` execution and exact state/delta equality.
- ZERO GPU, ZERO backpropagation, ZERO dense neural nets, ZERO hardcoded domain literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
"""
from __future__ import annotations

import heapq
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from mirror_crystal.quotient_egraph_resonator import (
    D_UNIFIED,
    hamming_sim10240,
    hv10240_of,
)
from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.shard.encoders import (
    cleanup_entity_ranking,
    effect_slot_key_hv,
    encode_delta_entity_slots_hv,
    encode_sequence_hv,
    encode_sparse_delta_hv,
    entity_symbol_hv,
    unbind_role_slot_hv,
    unbind_step_from_sequence,
)
from research.cpu_learning.shard.episodic_index import (
    SHARDEpisodicIndex,
    SHARDOperatorIndex,
)
from research.cpu_learning.simulator.state_graph import (
    ObservationMask,
    SparseDelta,
    StateGraph,
    canonical_json_bytes,
    compressed_bit_length,
    compute_sparse_delta,
)

SHARD_PROPOSER_SCHEMA_VERSION: str = "s4-shard-proposer-v1"


@dataclass(frozen=True)
class TaskARecoveryResult:
    """Result of unlabelled (before_state, observed_delta) -> (operator, roles) recovery."""

    method: str
    solved: bool
    action_id: str | None
    roles: dict[str, str] | None
    predicted_after_state: StateGraph | None
    predicted_delta: SparseDelta | None
    proposals_evaluated: int
    elapsed_seconds: float
    memory_accesses: int
    itl_verified: bool
    top_similarity: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "solved": self.solved,
            "action_id": self.action_id,
            "roles": dict(self.roles) if self.roles is not None else None,
            "proposals_evaluated": self.proposals_evaluated,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "memory_accesses": self.memory_accesses,
            "itl_verified": self.itl_verified,
            "top_similarity": self.top_similarity,
        }


@dataclass(frozen=True)
class TaskBPlanResult:
    """Result of goal-directed multi-step plan search from initial_state to target_state."""

    method: str
    solved: bool
    plan: tuple[tuple[str, dict[str, str]], ...]
    final_state: StateGraph | None
    proposals_evaluated: int
    nodes_expanded: int
    elapsed_seconds: float
    memory_accesses: int
    plan_length: int
    itl_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "solved": self.solved,
            "plan_actions": [aid for aid, _ in self.plan],
            "proposals_evaluated": self.proposals_evaluated,
            "nodes_expanded": self.nodes_expanded,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "memory_accesses": self.memory_accesses,
            "plan_length": self.plan_length,
            "itl_verified": self.itl_verified,
        }


@dataclass(frozen=True)
class BindDepthProbeResult:
    """Bind-depth geometry stress probe result at a single sequence depth."""

    depth: int
    trials: int
    codebook_size: int
    permuted_cleanup_accuracy: float
    naive_superposition_accuracy: float
    permuted_mean_sim: float
    naive_mean_sim: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "trials": self.trials,
            "codebook_size": self.codebook_size,
            "permuted_cleanup_accuracy": round(self.permuted_cleanup_accuracy, 6),
            "naive_superposition_accuracy": round(self.naive_superposition_accuracy, 6),
            "permuted_mean_sim": round(self.permuted_mean_sim, 2),
            "naive_mean_sim": round(self.naive_mean_sim, 2),
        }


class _StateBindingIndex:
    """Fast per-state structural lookup tables shared identically across ALL search methods."""

    __slots__ = ("state", "ent_map", "by_type", "facts_by_sp", "facts_by_op")

    def __init__(self, state: StateGraph) -> None:
        self.state = state
        self.ent_map = state.entity_map()
        by_type: dict[str, list[str]] = {}
        for eid, ent in state.entities:
            by_type.setdefault(ent.entity_type, []).append(eid)
        self.by_type = by_type

        facts_by_sp: dict[tuple[str, str], list[str]] = {}
        facts_by_op: dict[tuple[str, str], list[str]] = {}
        for s, p, o in sorted(state.facts):
            facts_by_sp.setdefault((s, p), []).append(o)
            facts_by_op.setdefault((o, p), []).append(s)
        self.facts_by_sp = facts_by_sp
        self.facts_by_op = facts_by_op


def _card_effect_slot_keys(card: OperatorCard) -> dict[str, tuple[int, ...]]:
    """Precompute 10,240-bit effect-slot keys per role variable of ``card`` (domain-agnostic)."""
    slots: dict[str, list[int]] = {}
    for s_t, pred, o_t in card.add_facts:
        if s_t.startswith("?"):
            slots.setdefault(s_t[1:], []).append(effect_slot_key_hv("add_subj", pred))
        if o_t.startswith("?"):
            slots.setdefault(o_t[1:], []).append(effect_slot_key_hv("add_obj", pred))
    for s_t, pred, o_t in card.delete_facts:
        if s_t.startswith("?"):
            slots.setdefault(s_t[1:], []).append(effect_slot_key_hv("del_subj", pred))
        if o_t.startswith("?"):
            slots.setdefault(o_t[1:], []).append(effect_slot_key_hv("del_obj", pred))
    for edit in card.attribute_edits:
        slots.setdefault(edit.role, []).append(effect_slot_key_hv("edit_ent", edit.attribute))
    return {r: tuple(ks) for r, ks in slots.items()}


def _unary_candidates_for_role(
    card: OperatorCard,
    role: str,
    sidx: _StateBindingIndex,
) -> list[str]:
    """Filter world entities for ``role`` by unary type, attribute, literal, and conditional guards."""
    allowed = card.allowed_types_map().get(role, frozenset())
    if allowed:
        cands: list[str] = []
        for et in sorted(allowed):
            cands.extend(sidx.by_type.get(et, ()))
        cands.sort()
    else:
        cands = sorted(sidx.ent_map.keys())

    if not cands:
        return []

    attr_reqs = [ag for ag in card.attribute_guards if ag.role == role]
    lit_reqs = [
        (pred, obj)
        for subj, pred, obj in card.required_facts
        if subj == f"?{role}" and not obj.startswith("?")
    ]
    cond_reqs = [cg for cg in card.conditional_fact_guards if cg.role == role]

    filtered: list[str] = []
    state = sidx.state
    ent_map = sidx.ent_map
    for eid in cands:
        ent = ent_map[eid]
        ok = True
        for ag in attr_reqs:
            if ent.get(ag.attribute) != ag.expected_value:
                ok = False
                break
        if not ok:
            continue
        for pred, val in lit_reqs:
            if not state.has_fact(eid, pred, val):
                ok = False
                break
        if not ok:
            continue
        for cg in cond_reqs:
            if ent.entity_type == cg.when_entity_type and not state.has_fact(
                eid, cg.predicate, cg.value
            ):
                ok = False
                break
        if ok:
            filtered.append(eid)
    return filtered


def enumerate_compatible_bindings(
    card: OperatorCard,
    sidx: _StateBindingIndex,
    *,
    role_entity_order: Mapping[str, Sequence[str]] | None = None,
    rng: random.Random | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, str]]:
    """Yield typed, relational-CSP-consistent role bindings for ``card`` on ``sidx``.

    Shared identically across SHARD-guided, unguided exhaustive, and random proposers so
    search comparisons are 100% fair (differing strictly in operator & slot ordering).
    """
    role_vars = list(card.role_variables)
    if not role_vars:
        return

    domains: dict[str, list[str]] = {}
    for r in role_vars:
        base_cands = _unary_candidates_for_role(card, r, sidx)
        if not base_cands:
            return
        if role_entity_order is not None and r in role_entity_order:
            allowed_set = set(base_cands)
            ordered = [e for e in role_entity_order[r] if e in allowed_set]
            ordered_set = set(ordered)
            for e in base_cands:
                if e not in ordered_set:
                    ordered.append(e)
            domains[r] = ordered
        elif rng is not None:
            shuffled = list(base_cands)
            rng.shuffle(shuffled)
            domains[r] = shuffled
        else:
            domains[r] = base_cands

    # Binary constraints between roles: inequality guards and role-to-role required_facts
    ineq_pairs: dict[str, set[str]] = {r: set() for r in role_vars}
    for ig in card.inequality_guards:
        if ig.role_a in ineq_pairs and ig.role_b in ineq_pairs:
            ineq_pairs[ig.role_a].add(ig.role_b)
            ineq_pairs[ig.role_b].add(ig.role_a)

    rel_edges: list[tuple[str, str, str]] = [
        (s[1:], p, o[1:])
        for s, p, o in card.required_facts
        if s.startswith("?") and o.startswith("?")
    ]
    # Order roles: smallest domain first (or effect-ranked roles first when SHARD order is supplied)
    if role_entity_order is not None:
        ordered_roles = sorted(
            role_vars,
            key=lambda r: (0 if r in role_entity_order else 1, len(domains[r]), r),
        )
    else:
        ordered_roles = sorted(role_vars, key=lambda r: (len(domains[r]), r))

    assigned: dict[str, str] = {}
    yielded = 0
    state = sidx.state

    def _backtrack(pos: int) -> Iterator[dict[str, str]]:
        nonlocal yielded
        if limit is not None and yielded >= limit:
            return
        if pos == len(ordered_roles):
            yielded += 1
            yield dict(assigned)
            return

        r = ordered_roles[pos]
        # Narrow candidate list using any already-assigned relational edges
        cands = domains[r]
        for rs, pred, ro in rel_edges:
            if rs == r and ro in assigned:
                allowed_s = set(sidx.facts_by_op.get((assigned[ro], pred), ()))
                cands = [e for e in cands if e in allowed_s]
            elif ro == r and rs in assigned:
                allowed_o = set(sidx.facts_by_sp.get((assigned[rs], pred), ()))
                cands = [e for e in cands if e in allowed_o]
            if not cands:
                return

        forbidden = {assigned[other] for other in ineq_pairs[r] if other in assigned}
        for eid in cands:
            if eid in forbidden:
                continue
            # Double-check self-edge if any
            valid = True
            for rs, pred, ro in rel_edges:
                if rs == r and ro == r and not state.has_fact(eid, pred, eid):
                    valid = False
                    break
            if not valid:
                continue
            assigned[r] = eid
            yield from _backtrack(pos + 1)
            del assigned[r]
            if limit is not None and yielded >= limit:
                return

    yield from _backtrack(0)


def _rank_role_entities_via_hdc(
    card: OperatorCard,
    slot_keys_map: Mapping[str, tuple[int, ...]],
    delta_slot_hv: int,
    sidx: _StateBindingIndex,
) -> dict[str, list[str]]:
    """Rank concrete entity candidates for each effect-participating role via 10,240-bit XOR unbinding."""
    ranking: dict[str, list[str]] = {}
    allowed_map = card.allowed_types_map()
    for rname, slot_keys in slot_keys_map.items():
        if not slot_keys:
            continue
        allowed_ts = allowed_map.get(rname, frozenset())
        cands: list[str] = []
        if allowed_ts:
            for et in sorted(allowed_ts):
                cands.extend(sidx.by_type.get(et, ()))
        else:
            cands = list(sidx.ent_map.keys())
        if len(cands) <= 1:
            ranking[rname] = cands
            continue

        if len(slot_keys) == 1:
            unbound = unbind_role_slot_hv(delta_slot_hv, slot_keys[0])
            ranking[rname] = [eid for eid, _sim in cleanup_entity_ranking(unbound, cands)]
        else:
            unbound_vecs = [unbind_role_slot_hv(delta_slot_hv, sk) for sk in slot_keys]
            scored = [
                (
                    eid,
                    sum(hamming_sim10240(uv, entity_symbol_hv(eid)) for uv in unbound_vecs),
                )
                for eid in cands
            ]
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            ranking[rname] = [eid for eid, _sim in scored]
    return ranking


def _state_diff_count(cur: StateGraph, target: StateGraph) -> int:
    """Exact symmetric-difference count of facts and entity attributes between two states."""
    diff = len(cur.facts ^ target.facts)
    cur_ents = cur.entity_map()
    tgt_ents = target.entity_map()
    for eid, t_ent in tgt_ents.items():
        c_ent = cur_ents.get(eid)
        if c_ent is None:
            diff += 1 + len(t_ent.attributes)
        elif c_ent.attributes != t_ent.attributes:
            c_a = c_ent.attr_dict()
            t_a = t_ent.attr_dict()
            for k in set(c_a) | set(t_a):
                if c_a.get(k) != t_a.get(k):
                    diff += 1
    return diff


class SHARDProposalEngine:
    """Warm-started SHARD 10,240-bit BSC proposal engine backed by strict ITL verification."""

    def __init__(
        self,
        operators: Mapping[str, OperatorCard],
        *,
        episodic_index: SHARDEpisodicIndex | None = None,
        num_lsh_tables: int = 12,
        bits_per_table: int = 4,
        seed_tag: str = "shard_proposer_v1",
    ) -> None:
        self._operators: dict[str, OperatorCard] = {
            aid: operators[aid] for aid in sorted(operators)
        }
        self._action_ids: tuple[str, ...] = tuple(self._operators.keys())
        self._slot_keys: dict[str, dict[str, tuple[int, ...]]] = {
            aid: _card_effect_slot_keys(card) for aid, card in self._operators.items()
        }
        self.operator_index = SHARDOperatorIndex(
            num_tables=num_lsh_tables,
            bits_per_table=bits_per_table,
            seed_tag=f"{seed_tag}:ops",
        )
        self.operator_index.index_library(self._operators)
        self.episodic_index = episodic_index

    @property
    def operators(self) -> dict[str, OperatorCard]:
        return dict(self._operators)

    @property
    def is_frozen(self) -> bool:
        ep_ok = True if self.episodic_index is None else self.episodic_index.is_frozen
        return self.operator_index.is_frozen and ep_ok

    def freeze(self) -> None:
        self.operator_index.freeze()
        if self.episodic_index is not None:
            self.episodic_index.freeze()

    def _rank_operators_with_shard(
        self,
        query_hv: int,
        *,
        top_k: int | None = None,
        include_fallback: bool = False,
    ) -> tuple[list[tuple[OperatorCard, int]], int]:
        """Rank operators via LSH + exact 10,240-bit POPCNT similarity (+ optional episodic warm-start)."""
        mem_before = self.operator_index.memory_accesses + (
            self.episodic_index.memory_accesses if self.episodic_index is not None else 0
        )
        op_ranked = self.operator_index.query_by_hv(
            query_hv, top_k=top_k, include_fallback=include_fallback
        )

        # Episodic LSH warm-start boost (similarity is advisory only; ITL still verifies!)
        if self.episodic_index is not None and len(self.episodic_index) > 0:
            ep_hits = self.episodic_index.query_by_delta_hv(query_hv, top_k=3, include_fallback=False)
            boost: dict[str, int] = {}
            for entry, sim in ep_hits:
                if sim > 0:
                    prev = boost.get(entry.action_id, 0)
                    if sim > prev:
                        boost[entry.action_id] = sim
            if boost:
                op_ranked = sorted(
                    ((card, sim + (boost.get(card.action_id, 0) // 4)) for card, sim in op_ranked),
                    key=lambda pair: (-pair[1], pair[0].action_id),
                )

        mem_after = self.operator_index.memory_accesses + (
            self.episodic_index.memory_accesses if self.episodic_index is not None else 0
        )
        return (op_ranked, mem_after - mem_before)

    # ------------------------------------------------------------------
    # Task A: Unlabelled Transition Operator + Role-Slot Recovery
    # ------------------------------------------------------------------

    def recover_transition_shard_guided(
        self,
        before_state: StateGraph,
        observed_delta: SparseDelta,
        *,
        mask: ObservationMask | None = None,
    ) -> TaskARecoveryResult:
        """SHARD LSH/Hamming operator ranking + HDC slot unbinding + strict ITL verification."""
        t0 = time.perf_counter()
        q_hv = encode_sparse_delta_hv(observed_delta, before_state)
        slot_hv = encode_delta_entity_slots_hv(observed_delta, before_state)
        # Stage 1: Sub-linear LSH bucket shortlist reranked by exact 10,240-bit POPCNT
        op_ranked, mem_acc = self._rank_operators_with_shard(q_hv, include_fallback=False)
        top_sim = op_ranked[0][1] if op_ranked else 0

        sidx = _StateBindingIndex(before_state)
        proposals = 0
        tried_ids: set[str] = set()

        for card, _sim in op_ranked:
            tried_ids.add(card.action_id)
            role_order = _rank_role_entities_via_hdc(
                card, self._slot_keys[card.action_id], slot_hv, sidx
            )
            for roles in enumerate_compatible_bindings(card, sidx, role_entity_order=role_order):
                proposals += 1
                res = card.apply(before_state, roles, mask=mask)
                if res.applied and res.delta == observed_delta and res.after_state is not None:
                    return TaskARecoveryResult(
                        method="shard_guided_proposer",
                        solved=True,
                        action_id=card.action_id,
                        roles=roles,
                        predicted_after_state=res.after_state,
                        predicted_delta=res.delta,
                        proposals_evaluated=proposals,
                        elapsed_seconds=time.perf_counter() - t0,
                        memory_accesses=mem_acc,
                        itl_verified=True,
                        top_similarity=top_sim,
                    )

        # Stage 2: Exact completeness fallback ONLY if LSH bucket shortlist missed the true operator
        if len(tried_ids) < len(self._operators):
            full_ranked, extra_mem = self._rank_operators_with_shard(q_hv, include_fallback=True)
            mem_acc += extra_mem
            for card, _sim in full_ranked:
                if card.action_id in tried_ids:
                    continue
                role_order = _rank_role_entities_via_hdc(
                    card, self._slot_keys[card.action_id], slot_hv, sidx
                )
                for roles in enumerate_compatible_bindings(card, sidx, role_entity_order=role_order):
                    proposals += 1
                    res = card.apply(before_state, roles, mask=mask)
                    if res.applied and res.delta == observed_delta and res.after_state is not None:
                        return TaskARecoveryResult(
                            method="shard_guided_proposer",
                            solved=True,
                            action_id=card.action_id,
                            roles=roles,
                            predicted_after_state=res.after_state,
                            predicted_delta=res.delta,
                            proposals_evaluated=proposals,
                            elapsed_seconds=time.perf_counter() - t0,
                            memory_accesses=mem_acc,
                            itl_verified=True,
                            top_similarity=top_sim,
                        )

        return TaskARecoveryResult(
            method="shard_guided_proposer",
            solved=False,
            action_id=None,
            roles=None,
            predicted_after_state=None,
            predicted_delta=None,
            proposals_evaluated=proposals,
            elapsed_seconds=time.perf_counter() - t0,
            memory_accesses=mem_acc,
            itl_verified=True,
            top_similarity=top_sim,
        )

    def recover_transition_unguided(
        self,
        before_state: StateGraph,
        observed_delta: SparseDelta,
        *,
        mask: ObservationMask | None = None,
    ) -> TaskARecoveryResult:
        """Canonical lexicographic operator + typed CSP enumeration + strict ITL verification."""
        t0 = time.perf_counter()
        sidx = _StateBindingIndex(before_state)
        proposals = 0

        for aid in self._action_ids:
            card = self._operators[aid]
            for roles in enumerate_compatible_bindings(card, sidx):
                proposals += 1
                res = card.apply(before_state, roles, mask=mask)
                if res.applied and res.delta == observed_delta and res.after_state is not None:
                    return TaskARecoveryResult(
                        method="unguided_exhaustive_proposer",
                        solved=True,
                        action_id=card.action_id,
                        roles=roles,
                        predicted_after_state=res.after_state,
                        predicted_delta=res.delta,
                        proposals_evaluated=proposals,
                        elapsed_seconds=time.perf_counter() - t0,
                        memory_accesses=0,
                        itl_verified=True,
                    )

        return TaskARecoveryResult(
            method="unguided_exhaustive_proposer",
            solved=False,
            action_id=None,
            roles=None,
            predicted_after_state=None,
            predicted_delta=None,
            proposals_evaluated=proposals,
            elapsed_seconds=time.perf_counter() - t0,
            memory_accesses=0,
            itl_verified=True,
        )

    def recover_transition_random(
        self,
        before_state: StateGraph,
        observed_delta: SparseDelta,
        *,
        seed: int = 0,
        mask: ObservationMask | None = None,
    ) -> TaskARecoveryResult:
        """Seeded random permutation without replacement over operators & typed CSP bindings + ITL."""
        t0 = time.perf_counter()
        rng = random.Random(seed)
        sidx = _StateBindingIndex(before_state)
        op_order = list(self._action_ids)
        rng.shuffle(op_order)
        proposals = 0

        for aid in op_order:
            card = self._operators[aid]
            for roles in enumerate_compatible_bindings(card, sidx, rng=rng):
                proposals += 1
                res = card.apply(before_state, roles, mask=mask)
                if res.applied and res.delta == observed_delta and res.after_state is not None:
                    return TaskARecoveryResult(
                        method="random_proposer",
                        solved=True,
                        action_id=card.action_id,
                        roles=roles,
                        predicted_after_state=res.after_state,
                        predicted_delta=res.delta,
                        proposals_evaluated=proposals,
                        elapsed_seconds=time.perf_counter() - t0,
                        memory_accesses=0,
                        itl_verified=True,
                    )

        return TaskARecoveryResult(
            method="random_proposer",
            solved=False,
            action_id=None,
            roles=None,
            predicted_after_state=None,
            predicted_delta=None,
            proposals_evaluated=proposals,
            elapsed_seconds=time.perf_counter() - t0,
            memory_accesses=0,
            itl_verified=True,
        )

    def recover_transition_retrieval_only_no_itl(
        self,
        before_state: StateGraph,
        observed_delta: SparseDelta,
    ) -> TaskARecoveryResult:
        """Similarity-only baseline WITHOUT ITL verification (PDF p. 15: similarity is NOT proof).

        Retrieves the nearest indexed training episode via 10,240-bit LSH/Hamming similarity
        and returns its raw stored action/roles/delta directly without verifying against ITL.
        """
        t0 = time.perf_counter()
        if self.episodic_index is None or len(self.episodic_index) == 0:
            return TaskARecoveryResult(
                method="shard_retrieval_only_no_itl",
                solved=False,
                action_id=None,
                roles=None,
                predicted_after_state=None,
                predicted_delta=None,
                proposals_evaluated=1,
                elapsed_seconds=time.perf_counter() - t0,
                memory_accesses=0,
                itl_verified=False,
            )

        mem_before = self.episodic_index.memory_accesses
        hits = self.episodic_index.query_by_delta(observed_delta, before_state, top_k=1)
        mem_acc = self.episodic_index.memory_accesses - mem_before
        if not hits:
            return TaskARecoveryResult(
                method="shard_retrieval_only_no_itl",
                solved=False,
                action_id=None,
                roles=None,
                predicted_after_state=None,
                predicted_delta=None,
                proposals_evaluated=1,
                elapsed_seconds=time.perf_counter() - t0,
                memory_accesses=mem_acc,
                itl_verified=False,
            )
        entry, sim = hits[0]
        pred_after = entry.sparse_delta.apply_to(before_state)
        exact_match = entry.sparse_delta == observed_delta and pred_after == observed_delta.apply_to(
            before_state
        )
        return TaskARecoveryResult(
            method="shard_retrieval_only_no_itl",
            solved=exact_match,
            action_id=entry.action_id,
            roles=entry.role_dict(),
            predicted_after_state=pred_after,
            predicted_delta=entry.sparse_delta,
            proposals_evaluated=1,
            elapsed_seconds=time.perf_counter() - t0,
            memory_accesses=mem_acc,
            itl_verified=False,
            top_similarity=sim,
        )

    # ------------------------------------------------------------------
    # Task B: Warm-Started Multi-Step Goal-Directed Plan Search
    # ------------------------------------------------------------------

    def plan_goal_shard_guided(
        self,
        initial_state: StateGraph,
        target_state: StateGraph,
        *,
        max_depth: int = 6,
        max_proposals: int = 1200,
        top_k_ops: int = 5,
        max_bindings_per_op: int = 6,
        min_op_similarity: int = 900,
    ) -> TaskBPlanResult:
        """SHARD goal-residual BSC best-first search with strict ``OperatorCard.apply()`` verification."""
        t0 = time.perf_counter()
        if initial_state == target_state:
            return TaskBPlanResult(
                method="shard_guided_planner",
                solved=True,
                plan=(),
                final_state=initial_state.copy(),
                proposals_evaluated=0,
                nodes_expanded=0,
                elapsed_seconds=time.perf_counter() - t0,
                memory_accesses=0,
                plan_length=0,
                itl_verified=True,
            )

        init_h = _state_diff_count(initial_state, target_state)
        init_fp = initial_state.fingerprint()
        # Heap entries: (phase, h_residual, depth, tie_counter, state, plan)
        # phase=0 expands SHARD positive-similarity shortlist; phase=1 is completeness fallback.
        heap: list[tuple[int, int, int, int, StateGraph, tuple[tuple[str, dict[str, str]], ...]]] = [
            (0, init_h, 0, 0, initial_state, ())
        ]
        visited_best_depth: dict[str, int] = {init_fp: 0}
        tie = 1
        proposals = 0
        nodes_expanded = 0
        total_mem = 0

        while heap and proposals < max_proposals:
            phase, cur_h, depth, _t, cur_state, plan = heapq.heappop(heap)
            if depth >= max_depth:
                continue

            nodes_expanded += 1
            sidx = _StateBindingIndex(cur_state)

            if phase == 0:
                res_delta = compute_sparse_delta(cur_state, target_state)
                res_hv = encode_sparse_delta_hv(res_delta, cur_state)
                slot_hv = encode_delta_entity_slots_hv(res_delta, cur_state)
                op_ranked, mem_acc = self._rank_operators_with_shard(
                    res_hv, top_k=top_k_ops, include_fallback=False
                )
                total_mem += mem_acc

                selected_ops = [
                    (card, sim) for card, sim in op_ranked if sim >= min_op_similarity
                ][:top_k_ops]
                if not selected_ops:
                    op_ranked_full, extra_mem = self._rank_operators_with_shard(
                        res_hv, top_k=top_k_ops, include_fallback=True
                    )
                    total_mem += extra_mem
                    selected_ops = op_ranked_full[:top_k_ops]

                # Queue phase=1 completeness fallback for this state in case shortlist stalls
                if len(selected_ops) < len(self._operators):
                    heapq.heappush(heap, (1, cur_h + 100, depth, tie, cur_state, plan))
                    tie += 1

                for card, _sim in selected_ops:
                    role_order = _rank_role_entities_via_hdc(
                        card, self._slot_keys[card.action_id], slot_hv, sidx
                    )
                    for roles in enumerate_compatible_bindings(
                        card,
                        sidx,
                        role_entity_order=role_order,
                        limit=max_bindings_per_op,
                    ):
                        if proposals >= max_proposals:
                            break
                        proposals += 1
                        res = card.apply(cur_state, roles)
                        if not res.applied or res.after_state is None:
                            continue
                        nxt = res.after_state
                        new_plan = plan + ((card.action_id, roles),)
                        if nxt == target_state:
                            return TaskBPlanResult(
                                method="shard_guided_planner",
                                solved=True,
                                plan=new_plan,
                                final_state=nxt,
                                proposals_evaluated=proposals,
                                nodes_expanded=nodes_expanded,
                                elapsed_seconds=time.perf_counter() - t0,
                                memory_accesses=total_mem,
                                plan_length=len(new_plan),
                                itl_verified=True,
                            )
                        nxt_fp = nxt.fingerprint()
                        nxt_d = depth + 1
                        if nxt_d < max_depth and visited_best_depth.get(nxt_fp, max_depth + 1) > nxt_d:
                            visited_best_depth[nxt_fp] = nxt_d
                            nxt_h = _state_diff_count(nxt, target_state)
                            heapq.heappush(heap, (0, nxt_h, nxt_d, tie, nxt, new_plan))
                            tie += 1
            else:
                # Completeness fallback across remaining operators
                for aid in self._action_ids:
                    card = self._operators[aid]
                    for roles in enumerate_compatible_bindings(card, sidx):
                        if proposals >= max_proposals:
                            break
                        proposals += 1
                        res = card.apply(cur_state, roles)
                        if not res.applied or res.after_state is None:
                            continue
                        nxt = res.after_state
                        new_plan = plan + ((card.action_id, roles),)
                        if nxt == target_state:
                            return TaskBPlanResult(
                                method="shard_guided_planner",
                                solved=True,
                                plan=new_plan,
                                final_state=nxt,
                                proposals_evaluated=proposals,
                                nodes_expanded=nodes_expanded,
                                elapsed_seconds=time.perf_counter() - t0,
                                memory_accesses=total_mem,
                                plan_length=len(new_plan),
                                itl_verified=True,
                            )
                        nxt_fp = nxt.fingerprint()
                        nxt_d = depth + 1
                        if nxt_d < max_depth and visited_best_depth.get(nxt_fp, max_depth + 1) > nxt_d:
                            visited_best_depth[nxt_fp] = nxt_d
                            nxt_h = _state_diff_count(nxt, target_state)
                            heapq.heappush(heap, (0, nxt_h, nxt_d, tie, nxt, new_plan))
                            tie += 1

        return TaskBPlanResult(
            method="shard_guided_planner",
            solved=False,
            plan=(),
            final_state=None,
            proposals_evaluated=proposals,
            nodes_expanded=nodes_expanded,
            elapsed_seconds=time.perf_counter() - t0,
            memory_accesses=total_mem,
            plan_length=0,
            itl_verified=True,
        )

    def plan_goal_unguided_bfs(
        self,
        initial_state: StateGraph,
        target_state: StateGraph,
        *,
        max_depth: int = 6,
        max_proposals: int = 1200,
    ) -> TaskBPlanResult:
        """Unguided Breadth-First Search over the verified operator library (exact ITL verification)."""
        t0 = time.perf_counter()
        if initial_state == target_state:
            return TaskBPlanResult(
                method="unguided_bfs_planner",
                solved=True,
                plan=(),
                final_state=initial_state.copy(),
                proposals_evaluated=0,
                nodes_expanded=0,
                elapsed_seconds=time.perf_counter() - t0,
                memory_accesses=0,
                plan_length=0,
                itl_verified=True,
            )

        queue: deque[tuple[StateGraph, tuple[tuple[str, dict[str, str]], ...]]] = deque(
            [(initial_state, ())]
        )
        visited: set[str] = {initial_state.fingerprint()}
        proposals = 0
        nodes_expanded = 0

        while queue and proposals < max_proposals:
            cur_state, plan = queue.popleft()
            if len(plan) >= max_depth:
                continue
            nodes_expanded += 1
            sidx = _StateBindingIndex(cur_state)

            for aid in self._action_ids:
                card = self._operators[aid]
                for roles in enumerate_compatible_bindings(card, sidx):
                    if proposals >= max_proposals:
                        break
                    proposals += 1
                    res = card.apply(cur_state, roles)
                    if not res.applied or res.after_state is None:
                        continue
                    nxt = res.after_state
                    new_plan = plan + ((card.action_id, roles),)
                    if nxt == target_state:
                        return TaskBPlanResult(
                            method="unguided_bfs_planner",
                            solved=True,
                            plan=new_plan,
                            final_state=nxt,
                            proposals_evaluated=proposals,
                            nodes_expanded=nodes_expanded,
                            elapsed_seconds=time.perf_counter() - t0,
                            memory_accesses=0,
                            plan_length=len(new_plan),
                            itl_verified=True,
                        )
                    nxt_fp = nxt.fingerprint()
                    if len(new_plan) < max_depth and nxt_fp not in visited:
                        visited.add(nxt_fp)
                        queue.append((nxt, new_plan))

        return TaskBPlanResult(
            method="unguided_bfs_planner",
            solved=False,
            plan=(),
            final_state=None,
            proposals_evaluated=proposals,
            nodes_expanded=nodes_expanded,
            elapsed_seconds=time.perf_counter() - t0,
            memory_accesses=0,
            plan_length=0,
            itl_verified=True,
        )

    def plan_goal_random_walk(
        self,
        initial_state: StateGraph,
        target_state: StateGraph,
        *,
        seed: int = 0,
        max_depth: int = 6,
        max_proposals: int = 1200,
    ) -> TaskBPlanResult:
        """Seeded random-restart walk over the verified operator library (exact ITL verification)."""
        t0 = time.perf_counter()
        if initial_state == target_state:
            return TaskBPlanResult(
                method="random_walk_planner",
                solved=True,
                plan=(),
                final_state=initial_state.copy(),
                proposals_evaluated=0,
                nodes_expanded=0,
                elapsed_seconds=time.perf_counter() - t0,
                memory_accesses=0,
                plan_length=0,
                itl_verified=True,
            )

        rng = random.Random(seed)
        proposals = 0
        nodes_expanded = 0
        op_list = list(self._action_ids)
        init_sidx = _StateBindingIndex(initial_state)

        while proposals < max_proposals:
            cur_state = initial_state
            cur_sidx = init_sidx
            plan: list[tuple[str, dict[str, str]]] = []
            for _step in range(max_depth):
                if proposals >= max_proposals:
                    break
                nodes_expanded += 1
                rng.shuffle(op_list)
                moved = False
                for aid in op_list:
                    card = self._operators[aid]
                    for roles in enumerate_compatible_bindings(card, cur_sidx, rng=rng, limit=3):
                        if proposals >= max_proposals:
                            break
                        proposals += 1
                        res = card.apply(cur_state, roles)
                        if not res.applied or res.after_state is None:
                            continue
                        nxt = res.after_state
                        plan.append((card.action_id, roles))
                        if nxt == target_state:
                            return TaskBPlanResult(
                                method="random_walk_planner",
                                solved=True,
                                plan=tuple(plan),
                                final_state=nxt,
                                proposals_evaluated=proposals,
                                nodes_expanded=nodes_expanded,
                                elapsed_seconds=time.perf_counter() - t0,
                                memory_accesses=0,
                                plan_length=len(plan),
                                itl_verified=True,
                            )
                        cur_state = nxt
                        cur_sidx = _StateBindingIndex(cur_state)
                        moved = True
                        break
                    if moved or proposals >= max_proposals:
                        break
                if not moved:
                    break

        return TaskBPlanResult(
            method="random_walk_planner",
            solved=False,
            plan=(),
            final_state=None,
            proposals_evaluated=proposals,
            nodes_expanded=nodes_expanded,
            elapsed_seconds=time.perf_counter() - t0,
            memory_accesses=0,
            plan_length=0,
            itl_verified=True,
        )

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> dict[str, Any]:
        """Report exact Stage S4 bit accounting (PDF p. 15) across operator cards and SHARD indexes."""
        exec_payload = canonical_json_bytes(
            {aid: c.to_executable_dict() for aid, c in sorted(self._operators.items())}
        )
        op_raw_bits = len(exec_payload) * 8
        op_mdl_bits = compressed_bit_length(exec_payload)
        active_bundle_bits = self.operator_index.index_bits()
        ep_bits = self.episodic_index.index_bits() if self.episodic_index is not None else 0
        total_mdl_bits = op_mdl_bits + self.operator_index.compressed_index_bits()
        return {
            "raw_training_data_bits": raw_training_bits,
            "compressed_training_data_bits": compressed_training_bits,
            "episode_store_bits": ep_bits,
            "active_bundle_bits": active_bundle_bits,
            "operator_library_bits": op_raw_bits,
            "compressed_operator_mdl_bits": op_mdl_bits,
            "compressed_model_mdl_bits": total_mdl_bits,
            "compression_ratio_vs_raw": round(total_mdl_bits / max(1, raw_training_bits), 6),
        }


def run_bind_depth_geometry_probe(
    *,
    depths: Sequence[int] = (1, 2, 3, 4, 5, 6),
    trials_per_depth: int = 25,
    codebook_size: int = 64,
    seed: int = 101,
) -> list[BindDepthProbeResult]:
    """Task C (PDF pp. 9 & 14): Stress-test bind-depth clean-up unbinding across depths 1..6.

    Compares role-specific coprime permutation + clean-up memory vs naive unordered
    superposition across ``trials_per_depth`` random symbol sequences (allowing repeats)
    over a ``codebook_size``-symbol 10,240-bit BSC vocabulary.
    """
    rng = random.Random(seed * 7919 + 42)
    labels = [f"sym_{i:03d}" for i in range(codebook_size)]
    codebook = {lbl: hv10240_of(f"shard:bind_probe:{seed}:{lbl}") for lbl in labels}

    results: list[BindDepthProbeResult] = []
    for d in depths:
        depth = int(d)
        perm_hits = 0
        naive_hits = 0
        perm_sim_sum = 0
        naive_sim_sum = 0
        total_queries = trials_per_depth * depth

        for _t in range(trials_per_depth):
            seq_labels = [rng.choice(labels) for _ in range(depth)]
            seq_hvs = [codebook[lbl] for lbl in seq_labels]
            perm_bundle = encode_sequence_hv(seq_hvs, use_role_permutation=True)
            naive_bundle = encode_sequence_hv(seq_hvs, use_role_permutation=False)

            for pos in range(depth):
                pred_p, sim_p = unbind_step_from_sequence(
                    perm_bundle, pos, codebook, use_role_permutation=True
                )
                pred_n, sim_n = unbind_step_from_sequence(
                    naive_bundle, pos, codebook, use_role_permutation=False
                )
                if pred_p == seq_labels[pos]:
                    perm_hits += 1
                if pred_n == seq_labels[pos]:
                    naive_hits += 1
                perm_sim_sum += sim_p
                naive_sim_sum += sim_n

        results.append(
            BindDepthProbeResult(
                depth=depth,
                trials=trials_per_depth,
                codebook_size=codebook_size,
                permuted_cleanup_accuracy=perm_hits / float(max(1, total_queries)),
                naive_superposition_accuracy=naive_hits / float(max(1, total_queries)),
                permuted_mean_sim=perm_sim_sum / float(max(1, total_queries)),
                naive_mean_sim=naive_sim_sum / float(max(1, total_queries)),
            )
        )
    return results
