"""Stage S9 Self-Curriculum Scheduler, Anomaly-Triggered Budgeting & Fair Curriculum Controls.

STRICT ISOLATION & PYTHON 3.11 COMPATIBILITY (PDF pp. 4–5, 7, 10, 12–14, 17):
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Uses pure Python 3.11 standard library ONLY (zero Python 3.12+ APIs).
- Implements ``SelfCurriculumScheduler`` (anomaly-triggered information-gain-per-ms budgeting
  across heterogeneous relational, raw-UTF-8 grounding, and functional synthesis frontiers)
  alongside two strictly fair controls under identical query and CPU budgets:
  1. ``PassiveStreamController``: passive sequential stream ingestion with zero active query
     selection, zero boundary-dream refutation, and zero anomaly prioritization.
  2. ``RandomCurriculumController``: spends the exact same query and CPU budget selecting
     unqueried frontier probes uniformly at random without replacement.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

from research.cpu_learning.continual.replay_manager import (
    CandidateUpdateOutcome,
    ContinualKnowledgeBase,
)
from research.cpu_learning.grounding.byte_aligner import decode_and_tokenize_utf8
from research.cpu_learning.grounding.construction_learner import (
    ConstructionCard,
    GroundedInstructionExecutor,
    RawByteConstructionLearner,
    UnlabelledGroundingEpisode,
    _abstract_tokens_to_pattern,
)
from research.cpu_learning.itl.contrast_harness import TransitionEnvironment
from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.rdc.consolidation import ConsolidationResult, RDCConsolidator
from research.cpu_learning.rdc.mdl_judge import MDLExample
from research.cpu_learning.rdc.refuter import score_probe_disagreement
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    ObservationMask,
    SparseDelta,
    StateGraph,
    canonical_json_bytes,
    sha256_hex,
)
from research.cpu_learning.transfer.transfer_inducer import (
    ExprNode,
    FunctionalOperatorCard,
    RestrictedExecutionSandbox,
    SlotTransitionEpisode,
    SynthesisDiagnostics,
    _COMBINATOR_FAMILY_FEATURES,
    _FAMILY_PROTOTYPE_HVS,
    _generate_family_candidates,
    encode_io_behavior_hv,
    hamming_sim10240,
    normalize_slot_value,
    synthesize_typed_boundary_inputs,
)

CURRICULUM_SCHEMA_VERSION: str = "s9-self-curriculum-v1"

AnomalyKind = Literal[
    "prediction_mismatch",
    "epistemic_abstention",
    "hypothesis_ambiguity",
    "unresolved_guard_uncertainty",
    "library_bloat_high",
]

CurriculumActionKind = Literal[
    "max_disagreement_query",
    "contrast_probe",
    "dream_refutation",
    "induce_or_repair",
    "consolidate_mdl",
    "regression_replay",
    "idle_noop",
]

_VALID_ANOMALY_KINDS: frozenset[str] = frozenset(
    {
        "prediction_mismatch",
        "epistemic_abstention",
        "hypothesis_ambiguity",
        "unresolved_guard_uncertainty",
        "library_bloat_high",
    }
)

_VALID_CURRICULUM_ACTIONS: frozenset[str] = frozenset(
    {
        "max_disagreement_query",
        "contrast_probe",
        "dream_refutation",
        "induce_or_repair",
        "consolidate_mdl",
        "regression_replay",
        "idle_noop",
    }
)


class ControllerFrozenError(RuntimeError):
    """Raised when a frozen controller attempts to mutate state or query an environment."""


CurriculumFrozenError = ControllerFrozenError


# ---------------------------------------------------------------------------
# 1. Core Anomaly, Budget & Curriculum Action Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalySignal:
    """Domain-agnostic epistemic anomaly detected on an active learning frontier (PDF p. 14)."""

    anomaly_id: str
    frontier_id: str
    anomaly_kind: AnomalyKind
    severity: float
    candidate_count: int
    estimated_cost_ms: float
    target_skill_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.anomaly_kind not in _VALID_ANOMALY_KINDS:
            raise ValueError(f"Unsupported anomaly_kind: {self.anomaly_kind!r}")
        if float(self.severity) < 0.0:
            raise ValueError(f"Anomaly severity must be >= 0.0, got {self.severity}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "frontier_id": self.frontier_id,
            "anomaly_kind": self.anomaly_kind,
            "severity": round(float(self.severity), 6),
            "candidate_count": int(self.candidate_count),
            "estimated_cost_ms": round(float(self.estimated_cost_ms), 4),
            "target_skill_id": self.target_skill_id,
            "payload": copy.deepcopy(dict(self.payload)),
        }


@dataclass
class CurriculumBudget:
    """Shared active-query and CPU-time budget governing autonomous and control controllers."""

    max_env_queries: int
    max_total_cpu_ms: float
    learning_slice_ms: float
    min_info_gain_threshold: float = 1e-4
    queries_spent: int = 0
    cpu_ms_spent: float = 0.0
    slices_executed: int = 0

    @property
    def queries_remaining(self) -> int:
        return max(0, int(self.max_env_queries) - int(self.queries_spent))

    @property
    def cpu_ms_remaining(self) -> float:
        return max(0.0, float(self.max_total_cpu_ms) - float(self.cpu_ms_spent))

    @property
    def exhausted(self) -> bool:
        return (
            int(self.queries_spent) >= int(self.max_env_queries)
            or float(self.cpu_ms_spent) >= float(self.max_total_cpu_ms)
        )

    def can_spend_query(self) -> bool:
        return (
            int(self.queries_spent) < int(self.max_env_queries)
            and float(self.cpu_ms_spent) < float(self.max_total_cpu_ms)
        )

    def can_run_slice(self) -> bool:
        return float(self.cpu_ms_spent) < float(self.max_total_cpu_ms)

    def record_query(self, count: int = 1) -> None:
        self.queries_spent += int(count)

    def record_cpu_ms(self, elapsed_ms: float) -> None:
        self.cpu_ms_spent += max(0.0, float(elapsed_ms))
        self.slices_executed += 1

    def clone_fresh(self) -> CurriculumBudget:
        return CurriculumBudget(
            max_env_queries=int(self.max_env_queries),
            max_total_cpu_ms=float(self.max_total_cpu_ms),
            learning_slice_ms=float(self.learning_slice_ms),
            min_info_gain_threshold=float(self.min_info_gain_threshold),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_env_queries": int(self.max_env_queries),
            "max_total_cpu_ms": round(float(self.max_total_cpu_ms), 4),
            "learning_slice_ms": round(float(self.learning_slice_ms), 4),
            "min_info_gain_threshold": float(self.min_info_gain_threshold),
            "queries_spent": int(self.queries_spent),
            "queries_remaining": self.queries_remaining,
            "cpu_ms_spent": round(float(self.cpu_ms_spent), 4),
            "cpu_ms_remaining": round(self.cpu_ms_remaining, 4),
            "slices_executed": int(self.slices_executed),
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class CurriculumAction:
    """Ranked autonomous curriculum decision selected for a single bounded learning slice."""

    action_kind: CurriculumActionKind
    frontier_id: str
    target_skill_id: str | None
    expected_info_gain: float
    estimated_ms: float
    score_ig_per_ms: float
    probe_index: int | None = None
    probe_payload: Any = None
    trigger_anomaly_kind: str = "none"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.action_kind not in _VALID_CURRICULUM_ACTIONS:
            raise ValueError(f"Unsupported curriculum action_kind: {self.action_kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_kind": self.action_kind,
            "frontier_id": self.frontier_id,
            "target_skill_id": self.target_skill_id,
            "expected_info_gain": round(float(self.expected_info_gain), 6),
            "estimated_ms": round(float(self.estimated_ms), 4),
            "score_ig_per_ms": round(float(self.score_ig_per_ms), 6),
            "probe_index": self.probe_index,
            "trigger_anomaly_kind": self.trigger_anomaly_kind,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# 2. Domain-Agnostic Active Learning Frontier States
# ---------------------------------------------------------------------------


def _card_mdl_sort_key(card: OperatorCard) -> tuple[int, str]:
    total_guards = (
        len(card.required_facts)
        + len(card.conditional_fact_guards)
        + len(card.indirect_fact_guards)
        + len(card.attribute_guards)
        + len(card.inequality_guards)
    )
    return (total_guards, card.fingerprint())


class RelationalFrontierState:
    """Domain-agnostic relational operator version-space & consolidation frontier."""

    def __init__(
        self,
        *,
        frontier_id: str,
        domain_id: str,
        hypotheses_by_skill: Mapping[str, Sequence[OperatorCard]],
        candidate_probes: Sequence[tuple[StateGraph, ActionInvocation, str]],
        pending_mismatches: Sequence[tuple[StateGraph, ActionInvocation, StateGraph, SparseDelta]] = (),
        duplicate_candidates: Sequence[tuple[str, str]] = (),
        equivalence_probes_by_pair: Mapping[tuple[str, str], Sequence[tuple[StateGraph, Mapping[str, str]]]] | None = None,
        mdl_examples: Sequence[MDLExample] = (),
    ) -> None:
        self.frontier_id = str(frontier_id)
        self.domain_id = str(domain_id)
        self.hypotheses_by_skill: dict[str, list[OperatorCard]] = {
            str(k): list(v) for k, v in sorted(hypotheses_by_skill.items())
        }
        self._sort_key_cache: dict[int, tuple[int, str]] = {
            id(c): _card_mdl_sort_key(c)
            for cards in self.hypotheses_by_skill.values()
            for c in cards
        }
        self.candidate_probes: list[tuple[StateGraph, ActionInvocation, str]] = list(candidate_probes)
        self.queried_indices: set[int] = set()
        self.zero_disagreement_indices: set[int] = set()
        self.active_ig_indices: set[int] = set()
        self.scan_cursor: int = 0
        self.active_ig_indices: set[int] = set()
        self.scan_cursor: int = 0
        self.pending_mismatches: list[tuple[StateGraph, ActionInvocation, StateGraph, SparseDelta]] = list(
            pending_mismatches
        )
        self.duplicate_candidates: list[tuple[str, str]] = list(duplicate_candidates)
        self.equivalence_probes_by_pair: dict[tuple[str, str], list[tuple[StateGraph, dict[str, str]]]] = {
            (str(k[0]), str(k[1])): [(st, dict(rm)) for st, rm in v]
            for k, v in (equivalence_probes_by_pair or {}).items()
        }
        self.mdl_examples: list[MDLExample] = list(mdl_examples)
        self.alias_map: dict[str, str] = {}
        self.queries_spent_on_frontier: int = 0

    def representative_card(self, skill_id: str) -> OperatorCard:
        hyps = self.hypotheses_by_skill[skill_id]
        return min(
            hyps,
            key=lambda c: self._sort_key_cache.setdefault(id(c), _card_mdl_sort_key(c)),
        )

    def unqueried_probe_indices(self) -> list[int]:
        return [
            idx
            for idx in range(len(self.candidate_probes))
            if idx not in self.queried_indices
        ]

    def total_unresolved_hypotheses(self) -> int:
        return sum(max(0, len(hs) - 1) for hs in self.hypotheses_by_skill.values())

    def is_converged(self) -> bool:
        return (
            not self.pending_mismatches
            and not self.duplicate_candidates
            and all(len(hs) <= 1 for hs in self.hypotheses_by_skill.values())
        )

    def filter_with_verified_transition(
        self,
        state: StateGraph,
        action: ActionInvocation,
        env_after: StateGraph,
        env_delta: SparseDelta,
    ) -> tuple[int, int, OperatorCard | None]:
        """Filter ``hypotheses_by_skill[action.action_id]`` against an external verified transition."""
        aid = action.action_id
        hyps = self.hypotheses_by_skill.get(aid)
        if not hyps:
            return (0, 0, None)
        prev_n = len(hyps)
        roles_q = action.role_dict()
        env_applied = not env_delta.is_noop
        effect_match_cache: dict[tuple[Any, Any, Any], bool] = {}
        survivors: list[OperatorCard] = []

        for h in hyps:
            ok, _reason, unknown = h.check_preconditions(state, roles_q)
            if unknown:
                continue
            if not ok:
                if (not env_applied) and state == env_after:
                    survivors.append(h)
                continue
            ekey = (h.add_facts, h.delete_facts, h.attribute_edits)
            matched = effect_match_cache.get(ekey)
            if matched is None:
                res = h.apply(state, roles_q)
                matched = (
                    (not res.abstained)
                    and res.applied == env_applied
                    and res.after_state == env_after
                )
                effect_match_cache[ekey] = matched
            if matched:
                survivors.append(h)

        if survivors:
            self.hypotheses_by_skill[aid] = survivors
        rep = self.representative_card(aid)
        return (prev_n, len(self.hypotheses_by_skill[aid]), rep)


class GroundingFrontierState:
    """Domain-agnostic raw UTF-8 grounding ambiguity & abstention frontier."""

    def __init__(
        self,
        *,
        frontier_id: str,
        domain_id: str,
        operators: Mapping[str, OperatorCard],
        initial_episodes: Sequence[UnlabelledGroundingEpisode],
        ambiguous_utterances: Sequence[tuple[bytes, StateGraph, ObservationMask]] = (),
        candidate_probes: Sequence[tuple[str, bytes, StateGraph, ObservationMask]] = (),
        min_negation_support: int = 2,
        min_condition_support: int = 2,
    ) -> None:
        self.frontier_id = str(frontier_id)
        self.domain_id = str(domain_id)
        self._operators: dict[str, OperatorCard] = {
            str(k): v for k, v in sorted(operators.items())
        }
        self.min_negation_support = int(min_negation_support)
        self.min_condition_support = int(min_condition_support)
        self.observed_episodes: list[UnlabelledGroundingEpisode] = list(initial_episodes)
        self._pass2_episodes: list[UnlabelledGroundingEpisode] = [
            ep for ep in initial_episodes if self._is_pass2_episode(ep)
        ]
        self._ambiguous_seeds: list[tuple[bytes, StateGraph, ObservationMask]] = list(
            ambiguous_utterances
        )
        self.candidate_probes: list[tuple[str, bytes, StateGraph, ObservationMask]] = list(
            candidate_probes
        )
        self.queried_indices: set[int] = set()
        self.already_grounded_indices: set[int] = set()
        self.active_ig_indices: set[int] = set()
        self.scan_cursor: int = 0
        self.pending_episodes: list[UnlabelledGroundingEpisode] = []
        self.queries_spent_on_frontier: int = 0

        self.learner = self._rebuild_learner()

    @staticmethod
    def _is_pass2_episode(ep: UnlabelledGroundingEpisode) -> bool:
        if ep.before_state == ep.after_state:
            return True
        raw = ep.raw_instruction_bytes
        return (b"," in raw) or (b";" in raw)

    def _rebuild_learner(self) -> RawByteConstructionLearner:
        learner = RawByteConstructionLearner(
            self._operators,
            min_negation_support=self.min_negation_support,
            min_condition_support=self.min_condition_support,
        )
        for raw_b, scene_st, mask in self._ambiguous_seeds:
            learner.observe_utterance_in_scene(raw_b, scene_st, mask=mask)
        if self.observed_episodes:
            learner.fit_episodes(self.observed_episodes)
        return learner

    def update_operators(self, new_operators: Mapping[str, OperatorCard]) -> None:
        new_map = {str(k): v for k, v in sorted(new_operators.items())}
        if set(new_map.keys()) == set(self._operators.keys()) and all(
            new_map[k].to_executable_dict() == self._operators[k].to_executable_dict()
            for k in new_map
        ):
            return
        prev_keys = set(self._operators.keys())
        self._operators = new_map
        if set(new_map.keys()) != prev_keys:
            self.learner = self._rebuild_learner()
            self.already_grounded_indices.clear()
        else:
            # Guard-only refinement on existing action schema: update in O(1) without full rebuild
            self.learner._operators = dict(new_map)
            self.learner._proposer._operators = dict(new_map)

    def ingest_verified_episode(self, ep: UnlabelledGroundingEpisode) -> RawByteConstructionLearner:
        self.observed_episodes.append(ep)
        if self._is_pass2_episode(ep):
            self._pass2_episodes.append(ep)
            self.learner.fit_episodes(self._pass2_episodes)
        else:
            self.learner.fit_episodes([ep])
            if self._pass2_episodes:
                self.learner.fit_episodes(self._pass2_episodes)
        return self.learner

    def unqueried_probe_indices(self) -> list[int]:
        return [
            idx
            for idx in range(len(self.candidate_probes))
            if idx not in self.queried_indices
        ]

    def unresolved_shadow_patterns(self) -> dict[tuple[str, ...], list[ConstructionCard]]:
        out: dict[tuple[str, ...], list[ConstructionCard]] = {}
        for pat, cards in self.learner._constructions_by_pattern.items():
            if any(c.card_state == "promoted" for c in cards):
                continue
            shadows = [c for c in cards if c.card_state == "shadow"]
            if len(shadows) > 1:
                out[pat] = shadows
        return out

    def score_probe_information_gain(
        self,
        probe_idx: int,
        executor: GroundedInstructionExecutor | None = None,
    ) -> tuple[float, AnomalyKind | None, str]:
        """Compute expected information gain (in bits) of querying grounding ``probe_idx``."""
        if probe_idx in self.queried_indices or probe_idx in self.already_grounded_indices:
            return (0.0, None, "already_grounded_unambiguous")
        _pid, raw_bytes, scene_state, mask = self.candidate_probes[probe_idx]
        exec_engine = executor if executor is not None else GroundedInstructionExecutor(self.learner)
        exec_res = exec_engine.execute_raw_utf8(raw_bytes, scene_state, mask=mask)
        if not exec_res.abstained:
            self.already_grounded_indices.add(probe_idx)
            return (0.0, None, "already_grounded_unambiguous")

        valid_utf8, tokens = decode_and_tokenize_utf8(raw_bytes)
        if not valid_utf8 or not tokens:
            return (0.0, None, "invalid_or_empty_utf8")

        ent_types = {
            eid: ent.entity_type
            for eid, ent in scene_state.entities
            if mask.is_entity_observed(eid)
        }
        pattern, mentions = _abstract_tokens_to_pattern(tokens, ent_types)
        tok_lowers = [t.lower for t in tokens]

        # Check shadow construction ambiguity first (exact Shannon split entropy across shadow cards)
        cards = self.learner._constructions_by_pattern.get(pattern, [])
        shadows = [c for c in cards if c.card_state == "shadow"]
        if len(shadows) > 1 and len(mentions) >= 1 and not any(c.card_state == "promoted" for c in cards):
            outcome_counts: dict[str, int] = {}
            for sc in shadows:
                op_card = self._operators.get(sc.action_id)
                if op_card is None or len(sc.slot_to_role) != len(mentions):
                    continue
                roles = {rname: mentions[s_idx].entity_id for s_idx, rname in sc.slot_to_role}
                app = op_card.apply(scene_state, roles)
                sig = app.after_state.fingerprint() if app.after_state is not None else "NOOP"
                outcome_counts[sig] = outcome_counts.get(sig, 0) + 1
            total_s = sum(outcome_counts.values())
            if total_s <= 1 or len(outcome_counts) <= 1:
                return (0.0, None, "non_disambiguating_symmetric_scene")
            ent = 0.0
            for cnt in outcome_counts.values():
                p = cnt / float(total_s)
                ent -= p * math.log2(p)
            return (ent + 0.25, "hypothesis_ambiguity", f"shadow_ambiguity:{len(shadows)}")

        # Unlearned compositional / syntactic construct (passive/SOV, negation, conditional, connector)
        if "," in tok_lowers or ";" in tok_lowers:
            return (0.95, "epistemic_abstention", "ungrounded_clause_or_condition_construct")
        return (1.10, "epistemic_abstention", "ungrounded_surface_construction")


def enumerate_consistent_slot_programs(
    arg_types: Sequence[str],
    out_type: str,
    param_names: Sequence[str],
    io_pairs: Sequence[tuple[tuple[Any, ...], Any]],
    *,
    max_candidates_per_family: int = 240,
) -> list[ExprNode]:
    """Enumerate all boundary-safe typed ``ExprNode`` candidates across SHARD-ranked families
    consistent with ``io_pairs`` (100% domain-agnostic, zero task names)."""
    norm_pairs = [
        (tuple(normalize_slot_value(x) for x in xs), normalize_slot_value(y))
        for xs, y in io_pairs
    ]
    inferred_consts: list[int] = []
    if out_type == "int" and norm_pairs:
        y_ints = [int(y) for _xs, y in norm_pairs]
        max_y = max(y_ints)
        min_y = min(y_ints)
        if min_y >= 0 and 2 <= max_y <= 64:
            inferred_consts.append(max_y + 1)

    canonical_families = list(_COMBINATOR_FAMILY_FEATURES.keys())
    query_hv = encode_io_behavior_hv(arg_types, out_type, norm_pairs)
    ranked_families = sorted(
        canonical_families,
        key=lambda fam: (
            -hamming_sim10240(query_hv, _FAMILY_PROTOTYPE_HVS[fam]),
            canonical_families.index(fam),
        ),
    )

    envs = [{p: xs[i] for i, p in enumerate(param_names)} for xs, _y in norm_pairs]
    gold_outs = tuple(y for _xs, y in norm_pairs)
    obs_inputs = [xs for xs, _y in norm_pairs]
    boundary_inputs = synthesize_typed_boundary_inputs(arg_types, obs_inputs)
    boundary_envs = [{p: bxs[i] for i, p in enumerate(param_names)} for bxs in boundary_inputs]

    seen_exprs: set[tuple[Any, ...]] = set()
    survivors_by_boundary_sig: dict[tuple[Any, ...], ExprNode] = {}

    for fam in ranked_families:
        fam_cands = _generate_family_candidates(
            fam, param_names, arg_types, out_type, inferred_consts
        )
        evaluated_in_fam = 0
        for cand in fam_cands:
            if evaluated_in_fam >= max_candidates_per_family:
                break
            ckey = cand.to_canonical_tuple()
            if ckey in seen_exprs:
                continue
            seen_exprs.add(ckey)
            evaluated_in_fam += 1

            preds: list[Any] = []
            failed = False
            for env in envs:
                try:
                    val = normalize_slot_value(cand.eval_env(env, step_limit=500))
                except Exception:
                    failed = True
                    break
                preds.append(val)
            if failed:
                continue
            obs_sig = tuple(preds)
            if obs_sig != gold_outs or not all(
                type(p) is type(g) for p, g in zip(obs_sig, gold_outs)
            ):
                continue

            # RDC synthetic boundary safety check (refutes unsafe empty-slice / div-by-zero programs)
            b_preds: list[Any] = []
            b_failed = False
            for benv in boundary_envs:
                try:
                    bval = normalize_slot_value(cand.eval_env(benv, step_limit=500))
                except Exception:
                    b_failed = True
                    break
                b_preds.append(bval)
            if b_failed:
                continue

            bsig = tuple(b_preds)
            prev = survivors_by_boundary_sig.get(bsig)
            if prev is None or (cand.node_count(), cand.compressed_bits()) < (
                prev.node_count(),
                prev.compressed_bits(),
            ):
                survivors_by_boundary_sig[bsig] = cand

    return sorted(
        survivors_by_boundary_sig.values(),
        key=lambda c: (c.node_count(), c.compressed_bits(), c.to_python_expr()),
    )


class FunctionalSkillState:
    """Version-space state for a single functional slot-transition skill."""

    def __init__(
        self,
        *,
        skill_id: str,
        input_slots: Sequence[tuple[str, str]],
        output_slots: Sequence[tuple[str, str]],
        initial_episodes: Sequence[SlotTransitionEpisode],
        candidate_probes: Sequence[Mapping[str, Any]],
    ) -> None:
        self.skill_id = str(skill_id)
        self.input_slots: tuple[tuple[str, str], ...] = tuple(
            (str(k), str(t)) for k, t in input_slots
        )
        self.output_slots: tuple[tuple[str, str], ...] = tuple(
            (str(k), str(t)) for k, t in output_slots
        )
        self.observed_episodes: list[SlotTransitionEpisode] = list(initial_episodes)
        self.candidate_probes: list[dict[str, Any]] = [
            {str(k): normalize_slot_value(v) for k, v in p.items()}
            for p in candidate_probes
        ]
        self.queried_indices: set[int] = set()
        self.zero_disagreement_indices: set[int] = set()
        self.active_ig_indices: set[int] = set()
        self.scan_cursor: int = 0
        self.surviving_by_out_slot: dict[str, list[ExprNode]] = {}
        self._recompute_version_space()

    def _recompute_version_space(self) -> None:
        p_names = [k for k, _t in self.input_slots]
        arg_types = [t for _k, t in self.input_slots]
        for out_slot, out_type in self.output_slots:
            io_pairs = [
                (
                    tuple(ep.before_dict()[p] for p in p_names),
                    ep.after_dict()[out_slot],
                )
                for ep in self.observed_episodes
            ]
            self.surviving_by_out_slot[out_slot] = enumerate_consistent_slot_programs(
                arg_types, out_type, p_names, io_pairs
            )

    def ingest_verified_episode(self, ep: SlotTransitionEpisode) -> None:
        self.observed_episodes.append(ep)
        p_names = [k for k, _t in self.input_slots]
        b_map = ep.before_dict()
        a_map = ep.after_dict()
        env = {p: normalize_slot_value(b_map[p]) for p in p_names}
        need_reenum = False
        for out_slot, _out_type in self.output_slots:
            gold_val = normalize_slot_value(a_map[out_slot])
            filtered: list[ExprNode] = []
            for cand in self.surviving_by_out_slot.get(out_slot, []):
                try:
                    pred = normalize_slot_value(cand.eval_env(env, step_limit=500))
                except Exception:
                    continue
                if pred == gold_val and type(pred) is type(gold_val):
                    filtered.append(cand)
            if filtered:
                self.surviving_by_out_slot[out_slot] = filtered
            else:
                need_reenum = True
        if need_reenum:
            self._recompute_version_space()
            self.zero_disagreement_indices.clear()

    def ambiguity_count(self) -> int:
        return max((len(v) for v in self.surviving_by_out_slot.values()), default=0)

    def is_converged(self) -> bool:
        return all(len(v) == 1 for v in self.surviving_by_out_slot.values())

    def score_probe_disagreement(self, probe_idx: int) -> float:
        if probe_idx in self.queried_indices or probe_idx in self.zero_disagreement_indices:
            return 0.0
        if self.is_converged():
            return 0.0
        p_names = [k for k, _t in self.input_slots]
        before = self.candidate_probes[probe_idx]
        env = {p: before[p] for p in p_names}
        total_ent = 0.0
        for out_slot, _out_t in self.output_slots:
            cands = self.surviving_by_out_slot.get(out_slot, [])
            if len(cands) <= 1:
                continue
            counts: dict[Any, int] = {}
            valid_n = 0
            for cand in cands:
                try:
                    val = normalize_slot_value(cand.eval_env(env, step_limit=500))
                except Exception:
                    continue
                counts[val] = counts.get(val, 0) + 1
                valid_n += 1
            if valid_n > 1 and len(counts) > 1:
                for cnt in counts.values():
                    p = cnt / float(valid_n)
                    total_ent -= p * math.log2(p)
        if total_ent <= 0.0:
            self.zero_disagreement_indices.add(probe_idx)
        return total_ent

    def build_verified_card(self, *, version: int = 1) -> FunctionalOperatorCard:
        """Compile MDL-minimal surviving expressions and verify in ``RestrictedExecutionSandbox``."""
        p_names = [k for k, _t in self.input_slots]
        slot_progs: list[tuple[str, ExprNode]] = []
        py_sources: list[tuple[str, str]] = []
        diags: list[SynthesisDiagnostics] = []
        all_solved = True

        for out_slot, _out_t in self.output_slots:
            cands = self.surviving_by_out_slot.get(out_slot, [])
            if not cands:
                all_solved = False
                continue
            best_expr = min(
                cands,
                key=lambda c: (c.node_count(), c.compressed_bits(), c.to_python_expr()),
            )
            fn_name = f"_slot_{out_slot}"
            py_src = best_expr.compile_to_python_function(fn_name, p_names)
            fn_obj, reason = RestrictedExecutionSandbox.compile_verified_function(py_src, fn_name)
            if fn_obj is None:
                raise RuntimeError(
                    f"Sandbox safety rejection for {self.skill_id}.{out_slot}: {reason}"
                )
            slot_progs.append((out_slot, best_expr))
            py_sources.append((out_slot, py_src))
            diags.append(
                SynthesisDiagnostics(
                    target_slot=out_slot,
                    solved=True,
                    use_shard_guidance=True,
                    candidates_evaluated=len(cands),
                    observational_classes=len(cands),
                    consistent_survivors=len(cands),
                    boundary_refuted_candidates=0,
                    ranked_families=tuple(_COMBINATOR_FAMILY_FEATURES.keys())[:3],
                    selected_program_bits=best_expr.compressed_bits(),
                    selected_node_count=best_expr.node_count(),
                )
            )

        card = FunctionalOperatorCard(
            card_id=f"fn_{self.skill_id}_v{version}",
            action_id=self.skill_id,
            input_slots=self.input_slots,
            slot_programs=tuple(slot_progs),
            python_sources=tuple(py_sources),
            version=int(version),
            support_episodes=len(self.observed_episodes),
            all_slots_solved=all_solved and len(slot_progs) == len(self.output_slots),
            diagnostics=tuple(diags),
        )
        # Verify exact replay on all observed training/active episodes via sandbox execution
        for ep in self.observed_episodes:
            pred_after = card.apply_slots(ep.before_dict(), use_sandbox=True)
            a_map = ep.after_dict()
            for out_slot, _ot in self.output_slots:
                if pred_after[out_slot] != normalize_slot_value(a_map[out_slot]):
                    raise RuntimeError(
                        f"Observed episode replay mismatch on {self.skill_id}.{out_slot}"
                    )
        return card


class FunctionalFrontierState:
    """Domain-agnostic functional program synthesis & active version-space frontier."""

    def __init__(
        self,
        *,
        frontier_id: str,
        domain_id: str,
        skills: Sequence[FunctionalSkillState],
    ) -> None:
        self.frontier_id = str(frontier_id)
        self.domain_id = str(domain_id)
        self.skills_by_id: dict[str, FunctionalSkillState] = {
            s.skill_id: s for s in skills
        }
        self.queries_spent_on_frontier: int = 0

    def is_converged(self) -> bool:
        return all(s.is_converged() for s in self.skills_by_id.values())

    def unqueried_skill_probes(self) -> list[tuple[str, int]]:
        pairs: list[tuple[str, int]] = []
        for sid, sstate in sorted(self.skills_by_id.items()):
            for idx in range(len(sstate.candidate_probes)):
                if idx not in sstate.queried_indices:
                    pairs.append((sid, idx))
        return pairs


# ---------------------------------------------------------------------------
# 3. SelfCurriculumScheduler (Anomaly-Triggered Information-Gain-Per-Ms)
# ---------------------------------------------------------------------------


class SelfCurriculumScheduler:
    """Domain-agnostic autonomous curriculum scheduler (PDF pp. 4–5, 7, 12, 14).

    Inspects multi-frontier epistemic anomalies, ranks candidate actions by expected
    information gain per CPU millisecond, and stops querying frontiers once verified.
    """

    def __init__(
        self,
        *,
        relational_query_cost_ms: float = 0.65,
        grounding_query_cost_ms: float = 0.70,
        functional_query_cost_ms: float = 0.55,
        consolidation_cost_ms: float = 0.80,
        repair_cost_ms: float = 0.50,
        max_probe_scans_per_tick: int = 12,
    ) -> None:
        self.relational_query_cost_ms = float(relational_query_cost_ms)
        self.grounding_query_cost_ms = float(grounding_query_cost_ms)
        self.functional_query_cost_ms = float(functional_query_cost_ms)
        self.consolidation_cost_ms = float(consolidation_cost_ms)
        self.repair_cost_ms = float(repair_cost_ms)
        self.max_probe_scans_per_tick = max(8, int(max_probe_scans_per_tick))
        self._anomaly_seq: int = 0

    def _next_aid(self, prefix: str) -> str:
        self._anomaly_seq += 1
        return f"anom_{prefix}_{self._anomaly_seq:05d}"

    def inspect_anomalies(
        self,
        *,
        relational_frontiers: Sequence[RelationalFrontierState] = (),
        grounding_frontiers: Sequence[GroundingFrontierState] = (),
        functional_frontiers: Sequence[FunctionalFrontierState] = (),
        max_scan_ms: float | None = None,
    ) -> list[AnomalySignal]:
        """Scan all registered frontiers and return active ``AnomalySignal``s ranked by severity."""
        t_anom0 = time.perf_counter()
        signals: list[AnomalySignal] = []

        for rf in relational_frontiers:
            if rf.pending_mismatches:
                signals.append(
                    AnomalySignal(
                        anomaly_id=self._next_aid("mismatch"),
                        frontier_id=rf.frontier_id,
                        anomaly_kind="prediction_mismatch",
                        severity=2.5 * len(rf.pending_mismatches),
                        candidate_count=len(rf.pending_mismatches),
                        estimated_cost_ms=self.repair_cost_ms,
                        target_skill_id=rf.pending_mismatches[0][1].action_id,
                        payload={"pending_count": len(rf.pending_mismatches)},
                    )
                )
            for sid, hyps in sorted(rf.hypotheses_by_skill.items()):
                if len(hyps) > 1:
                    signals.append(
                        AnomalySignal(
                            anomaly_id=self._next_aid("guard"),
                            frontier_id=rf.frontier_id,
                            anomaly_kind="unresolved_guard_uncertainty",
                            severity=math.log2(len(hyps)),
                            candidate_count=len(hyps),
                            estimated_cost_ms=self.relational_query_cost_ms,
                            target_skill_id=sid,
                            payload={"surviving_hypotheses": len(hyps)},
                        )
                    )
            if rf.duplicate_candidates:
                canon_id, dup_id = rf.duplicate_candidates[0]
                signals.append(
                    AnomalySignal(
                        anomaly_id=self._next_aid("bloat"),
                        frontier_id=rf.frontier_id,
                        anomaly_kind="library_bloat_high",
                        severity=1.35 * len(rf.duplicate_candidates),
                        candidate_count=len(rf.duplicate_candidates),
                        estimated_cost_ms=self.consolidation_cost_ms,
                        target_skill_id=dup_id,
                        payload={"canonical_id": canon_id, "duplicate_id": dup_id},
                    )
                )

        for gf in grounding_frontiers:
            if gf.pending_episodes:
                signals.append(
                    AnomalySignal(
                        anomaly_id=self._next_aid("g_repair"),
                        frontier_id=gf.frontier_id,
                        anomaly_kind="prediction_mismatch",
                        severity=2.2 * len(gf.pending_episodes),
                        candidate_count=len(gf.pending_episodes),
                        estimated_cost_ms=self.repair_cost_ms,
                    )
                )
            shadows = gf.unresolved_shadow_patterns()
            if shadows:
                max_k = max(len(v) for v in shadows.values())
                signals.append(
                    AnomalySignal(
                        anomaly_id=self._next_aid("g_ambig"),
                        frontier_id=gf.frontier_id,
                        anomaly_kind="hypothesis_ambiguity",
                        severity=math.log2(max_k) * len(shadows),
                        candidate_count=sum(len(v) for v in shadows.values()),
                        estimated_cost_ms=self.grounding_query_cost_ms,
                        payload={"ambiguous_patterns": len(shadows)},
                    )
                )
            else:
                # Fast bounded check: skip expensive UTF-8 execution scans when slice scan budget < 2.0 ms
                abstaining = len(gf.active_ig_indices - gf.queried_indices)
                if abstaining == 0 and (
                    max_scan_ms is None
                    or (
                        max_scan_ms >= 2.0
                        and ((time.perf_counter() - t_anom0) * 1000.0) + 1.2 < max_scan_ms
                    )
                ):
                    exec_engine = GroundedInstructionExecutor(gf.learner)
                    checked_a = 0
                    for idx in gf.unqueried_probe_indices()[gf.scan_cursor:]:
                        if (
                            max_scan_ms is not None
                            and ((time.perf_counter() - t_anom0) * 1000.0) >= max_scan_ms
                        ):
                            break
                        if idx in gf.queried_indices or idx in gf.already_grounded_indices:
                            continue
                        checked_a += 1
                        ig, akind, _rsn = gf.score_probe_information_gain(idx, executor=exec_engine)
                        if ig > 0.0 and akind == "epistemic_abstention":
                            gf.active_ig_indices.add(idx)
                            abstaining += 1
                            break
                        if checked_a >= 8:
                            break
                if abstaining > 0:
                    signals.append(
                        AnomalySignal(
                            anomaly_id=self._next_aid("g_abstain"),
                            frontier_id=gf.frontier_id,
                            anomaly_kind="epistemic_abstention",
                            severity=1.05 * min(abstaining, 4),
                            candidate_count=abstaining,
                            estimated_cost_ms=self.grounding_query_cost_ms,
                            payload={"abstaining_probes": abstaining},
                        )
                    )

        for ff in functional_frontiers:
            for sid, sstate in sorted(ff.skills_by_id.items()):
                k_amb = sstate.ambiguity_count()
                if k_amb > 1:
                    signals.append(
                        AnomalySignal(
                            anomaly_id=self._next_aid("fn_ambig"),
                            frontier_id=ff.frontier_id,
                            anomaly_kind="hypothesis_ambiguity",
                            severity=math.log2(k_amb),
                            candidate_count=k_amb,
                            estimated_cost_ms=self.functional_query_cost_ms,
                            target_skill_id=sid,
                            payload={"surviving_programs": k_amb},
                        )
                    )

        signals.sort(key=lambda s: (-s.severity, s.frontier_id, s.anomaly_id))
        return signals

    def select_next_action(
        self,
        budget: CurriculumBudget,
        *,
        relational_frontiers: Sequence[RelationalFrontierState] = (),
        grounding_frontiers: Sequence[GroundingFrontierState] = (),
        functional_frontiers: Sequence[FunctionalFrontierState] = (),
        pending_replay_audit: bool = False,
        max_selection_ms: float | None = None,
    ) -> CurriculumAction:
        """Rank all candidate curriculum actions by ``expected_info_gain / estimated_ms``."""
        if not budget.can_run_slice():
            return CurriculumAction(
                action_kind="idle_noop",
                frontier_id="none",
                target_skill_id=None,
                expected_info_gain=0.0,
                estimated_ms=0.01,
                score_ig_per_ms=0.0,
                reason="cpu_budget_exhausted",
            )

        if pending_replay_audit:
            ig = 3.0
            ms = 0.40
            return CurriculumAction(
                action_kind="regression_replay",
                frontier_id="continual_kb",
                target_skill_id=None,
                expected_info_gain=ig,
                estimated_ms=ms,
                score_ig_per_ms=ig / ms,
                trigger_anomaly_kind="prediction_mismatch",
                reason="mandatory_post_update_regression_audit",
            )

        candidates: list[CurriculumAction] = []

        # 1. Zero-env-query internal repairs & MDL consolidations
        for rf in relational_frontiers:
            if rf.pending_mismatches:
                ig = 2.8
                ms = self.repair_cost_ms
                candidates.append(
                    CurriculumAction(
                        action_kind="induce_or_repair",
                        frontier_id=rf.frontier_id,
                        target_skill_id=rf.pending_mismatches[0][1].action_id,
                        expected_info_gain=ig,
                        estimated_ms=ms,
                        score_ig_per_ms=ig / max(0.05, ms),
                        trigger_anomaly_kind="prediction_mismatch",
                        reason="observed_stream_prediction_mismatch",
                    )
                )

        for gf in grounding_frontiers:
            if gf.pending_episodes:
                ig = 2.5
                ms = self.repair_cost_ms
                candidates.append(
                    CurriculumAction(
                        action_kind="induce_or_repair",
                        frontier_id=gf.frontier_id,
                        target_skill_id=None,
                        expected_info_gain=ig,
                        estimated_ms=ms,
                        score_ig_per_ms=ig / max(0.05, ms),
                        trigger_anomaly_kind="prediction_mismatch",
                        reason="fit_buffered_unlabelled_grounding_episodes",
                    )
                )

        # 2. Active environment queries (fair per-frontier slice allocation + strict wall-clock deadline!)
        t_sel0 = time.perf_counter()
        raw_cap_ms = (
            min(float(budget.cpu_ms_remaining), min(22.0, 0.25 * float(budget.learning_slice_ms)))
            if max_selection_ms is None
            else min(float(budget.cpu_ms_remaining), float(max_selection_ms))
        )
        max_sel_ms = max(0.0, raw_cap_ms)
        n_groups = (
            (1 if relational_frontiers else 0)
            + (1 if grounding_frontiers else 0)
            + (1 if functional_frontiers else 0)
        )
        per_group_ms = max_sel_ms / float(max(1, n_groups))
        rf_has_positive_ig: dict[str, bool] = {rf.frontier_id: False for rf in relational_frontiers}
        if budget.can_spend_query() and max_sel_ms > 0.0:
            # 2a. Relational version-space max-disagreement / boundary-dream probes
            t_g_rel = time.perf_counter()
            rel_cap_ms = min(per_group_ms, max(0.0, max_sel_ms - ((t_g_rel - t_sel0) * 1000.0)))
            for rf in relational_frontiers:
                if ((time.perf_counter() - t_g_rel) * 1000.0) >= rel_cap_ms:
                    break
                best_idx: int | None = None
                best_ig = 0.0
                best_tag = "pool_probe"
                best_aid: str | None = None
                n_pool = len(rf.candidate_probes)

                check_active: list[int] = sorted(rf.active_ig_indices - rf.queried_indices)
                check_new: list[int] = []
                r_cur = rf.scan_cursor
                while (
                    r_cur < n_pool
                    and len(check_new) < self.max_probe_scans_per_tick
                    and ((time.perf_counter() - t_g_rel) * 1000.0) < rel_cap_ms
                ):
                    if (
                        r_cur not in rf.queried_indices
                        and r_cur not in rf.zero_disagreement_indices
                        and r_cur not in rf.active_ig_indices
                    ):
                        check_new.append(r_cur)
                    r_cur += 1
                check_order: list[int] = check_active + check_new
                r_broke = False
                r_scored_to = rf.scan_cursor

                for idx_pos, idx in enumerate(check_order):
                    st, act, ptag = rf.candidate_probes[idx]
                    hyps = rf.hypotheses_by_skill.get(act.action_id, [])
                    if len(hyps) <= 1:
                        rf.zero_disagreement_indices.add(idx)
                        rf.active_ig_indices.discard(idx)
                        if idx >= r_scored_to:
                            r_scored_to = idx + 1
                        continue
                    est_eval_ms = 0.015 * len(hyps)
                    if (((time.perf_counter() - t_g_rel) * 1000.0) + est_eval_ms) > rel_cap_ms:
                        if idx_pos > 0 or max_sel_ms < 2.5:
                            r_broke = True
                            break
                    sc = score_probe_disagreement(st, act, hyps)
                    if idx >= r_scored_to:
                        r_scored_to = idx + 1
                    if sc <= 0.0:
                        rf.zero_disagreement_indices.add(idx)
                        rf.active_ig_indices.discard(idx)
                        continue
                    rf.active_ig_indices.add(idx)
                    if sc > best_ig:
                        best_ig = sc
                        best_idx = idx
                        best_tag = ptag
                        best_aid = act.action_id
                rf.scan_cursor = max(rf.scan_cursor, r_scored_to if r_broke else r_cur)
                if best_idx is not None and best_ig >= budget.min_info_gain_threshold:
                    rf_has_positive_ig[rf.frontier_id] = True
                    akind: CurriculumActionKind = (
                        "dream_refutation" if best_tag == "boundary_dream" else "max_disagreement_query"
                    )
                    ms = self.relational_query_cost_ms
                    candidates.append(
                        CurriculumAction(
                            action_kind=akind,
                            frontier_id=rf.frontier_id,
                            target_skill_id=best_aid,
                            expected_info_gain=best_ig,
                            estimated_ms=ms,
                            score_ig_per_ms=best_ig / max(0.05, ms),
                            probe_index=best_idx,
                            trigger_anomaly_kind="unresolved_guard_uncertainty",
                            reason=f"relational_split_entropy:{best_ig:.4f}",
                        )
                    )

            # 2b. Grounding minimal-contrast / disambiguation probes
            t_g_gnd = time.perf_counter()
            gnd_cap_ms = min(per_group_ms, max(0.0, max_sel_ms - ((t_g_gnd - t_sel0) * 1000.0)))
            for gf in grounding_frontiers:
                if (((time.perf_counter() - t_g_gnd) * 1000.0) + 1.2) >= gnd_cap_ms and max_sel_ms < 2.5:
                    break
                if ((time.perf_counter() - t_g_gnd) * 1000.0) >= gnd_cap_ms:
                    break
                exec_engine = GroundedInstructionExecutor(gf.learner)
                best_g_idx: int | None = None
                best_g_ig = 0.0
                best_g_anom: str = "epistemic_abstention"
                best_g_reason = ""
                n_gpool = len(gf.candidate_probes)

                g_active: list[int] = sorted(gf.active_ig_indices - gf.queried_indices)
                g_new: list[int] = []
                g_cur = gf.scan_cursor
                while (
                    g_cur < n_gpool
                    and len(g_new) < self.max_probe_scans_per_tick
                    and ((time.perf_counter() - t_g_gnd) * 1000.0) < gnd_cap_ms
                ):
                    if (
                        g_cur not in gf.queried_indices
                        and g_cur not in gf.already_grounded_indices
                        and g_cur not in gf.active_ig_indices
                    ):
                        g_new.append(g_cur)
                    g_cur += 1
                g_order: list[int] = g_active + g_new
                g_broke = False
                g_scored_to = gf.scan_cursor

                for idx_pos, idx in enumerate(g_order):
                    if idx_pos > 0 and ((time.perf_counter() - t_g_gnd) * 1000.0) >= gnd_cap_ms:
                        g_broke = True
                        break
                    ig, ak, rsn = gf.score_probe_information_gain(idx, executor=exec_engine)
                    if idx >= g_scored_to:
                        g_scored_to = idx + 1
                    if ig <= 0.0:
                        gf.active_ig_indices.discard(idx)
                        continue
                    gf.active_ig_indices.add(idx)
                    if ig > best_g_ig:
                        best_g_ig = ig
                        best_g_idx = idx
                        best_g_anom = ak or "epistemic_abstention"
                        best_g_reason = rsn
                gf.scan_cursor = max(gf.scan_cursor, g_scored_to if g_broke else g_cur)
                if best_g_idx is not None and best_g_ig >= budget.min_info_gain_threshold:
                    ms = self.grounding_query_cost_ms
                    candidates.append(
                        CurriculumAction(
                            action_kind="contrast_probe",
                            frontier_id=gf.frontier_id,
                            target_skill_id=gf.candidate_probes[best_g_idx][0],
                            expected_info_gain=best_g_ig,
                            estimated_ms=ms,
                            score_ig_per_ms=best_g_ig / max(0.05, ms),
                            probe_index=best_g_idx,
                            trigger_anomaly_kind=best_g_anom,
                            reason=best_g_reason,
                        )
                    )

            # 2c. Functional synthesis max-disagreement I/O probes
            for ff in functional_frontiers:
                if ((time.perf_counter() - t_sel0) * 1000.0) >= max_sel_ms:
                    break
                best_f_sid: str | None = None
                best_f_idx: int | None = None
                best_f_ig = 0.0
                per_skill_cap = max(8, self.max_probe_scans_per_tick // max(1, len(ff.skills_by_id)))
                for sid, sstate in sorted(ff.skills_by_id.items()):
                    if ((time.perf_counter() - t_sel0) * 1000.0) >= max_sel_ms:
                        break
                    if sstate.is_converged():
                        continue
                    f_active: list[int] = sorted(sstate.active_ig_indices - sstate.queried_indices)
                    f_new: list[int] = []
                    f_cur = sstate.scan_cursor
                    while (
                        f_cur < len(sstate.candidate_probes)
                        and len(f_new) < per_skill_cap
                        and ((time.perf_counter() - t_sel0) * 1000.0) < max_sel_ms
                    ):
                        if (
                            f_cur not in sstate.queried_indices
                            and f_cur not in sstate.zero_disagreement_indices
                            and f_cur not in sstate.active_ig_indices
                        ):
                            f_new.append(f_cur)
                        f_cur += 1
                    f_order: list[int] = f_active + f_new
                    f_broke = False
                    f_scored_to = sstate.scan_cursor
                    for idx_pos, idx in enumerate(f_order):
                        if idx_pos > 0 and ((time.perf_counter() - t_sel0) * 1000.0) >= max_sel_ms:
                            f_broke = True
                            break
                        ig = sstate.score_probe_disagreement(idx)
                        if idx >= f_scored_to:
                            f_scored_to = idx + 1
                        if ig <= 0.0:
                            sstate.active_ig_indices.discard(idx)
                            continue
                        sstate.active_ig_indices.add(idx)
                        if ig > best_f_ig:
                            best_f_ig = ig
                            best_f_sid = sid
                            best_f_idx = idx
                    sstate.scan_cursor = max(sstate.scan_cursor, f_scored_to if f_broke else f_cur)
                if (
                    best_f_sid is not None
                    and best_f_idx is not None
                    and best_f_ig >= budget.min_info_gain_threshold
                ):
                    ms = self.functional_query_cost_ms
                    candidates.append(
                        CurriculumAction(
                            action_kind="max_disagreement_query",
                            frontier_id=ff.frontier_id,
                            target_skill_id=best_f_sid,
                            expected_info_gain=best_f_ig,
                            estimated_ms=ms,
                            score_ig_per_ms=best_f_ig / max(0.05, ms),
                            probe_index=best_f_idx,
                            trigger_anomaly_kind="hypothesis_ambiguity",
                            reason=f"functional_split_entropy:{best_f_ig:.4f}",
                        )
                    )

        # 3. Trigger MDL Consolidation once a relational frontier has scanned its pool & converged on behavior
        for rf in relational_frontiers:
            if (
                rf.duplicate_candidates
                and rf.scan_cursor >= len(rf.candidate_probes)
                and not rf_has_positive_ig.get(rf.frontier_id, False)
            ):
                canon_id, dup_id = rf.duplicate_candidates[0]
                ig = 1.6
                ms = self.consolidation_cost_ms
                candidates.append(
                    CurriculumAction(
                        action_kind="consolidate_mdl",
                        frontier_id=rf.frontier_id,
                        target_skill_id=dup_id,
                        expected_info_gain=ig,
                        estimated_ms=ms,
                        score_ig_per_ms=ig / max(0.05, ms),
                        probe_payload=(canon_id, dup_id),
                        trigger_anomaly_kind="library_bloat_high",
                        reason=f"consolidate_duplicate:{dup_id}->{canon_id}",
                    )
                )

        if not candidates:
            return CurriculumAction(
                action_kind="idle_noop",
                frontier_id="none",
                target_skill_id=None,
                expected_info_gain=0.0,
                estimated_ms=0.02,
                score_ig_per_ms=0.0,
                reason="all_frontiers_verified_or_query_budget_reached",
            )

        candidates.sort(
            key=lambda a: (
                -a.score_ig_per_ms,
                -a.expected_info_gain,
                a.frontier_id,
                a.action_kind,
            )
        )
        return candidates[0]


# ---------------------------------------------------------------------------
# 4. Shared Frontier Update Helpers & Fair Curriculum Control Baselines
# ---------------------------------------------------------------------------


def sync_grounding_learner_to_kb(
    kb: ContinualKnowledgeBase,
    gf: GroundingFrontierState,
    *,
    phase_index: int,
) -> int:
    """Promote verified ``ConstructionCard``s and learned syntax rules from ``gf.learner`` into ``kb``."""
    promoted_count = 0
    for ccard in gf.learner.promoted_constructions:
        existing = kb.get_active_skill(gf.domain_id, ccard.construction_id)
        exec_d = ccard.to_executable_dict()
        sha = sha256_hex(canonical_json_bytes(exec_d))
        if existing is not None and existing.canonical_sha256 == sha:
            continue
        out: CandidateUpdateOutcome = kb.apply_candidate_update_with_replay(
            domain_id=gf.domain_id,
            skill_id=ccard.construction_id,
            kind="utf8_construction",
            executable_dict=exec_d,
            executable_object=ccard,
            phase_index=phase_index,
            notes="verified_grounding_promotion",
        )
        if out.accepted:
            promoted_count += 1
    kb.set_domain_aux(gf.domain_id, "grounding_learner", gf.learner)
    kb.set_domain_aux(gf.domain_id, "negation_rules", gf.learner.negation_rules)
    kb.set_domain_aux(gf.domain_id, "condition_rules", gf.learner.condition_rules)
    kb.set_domain_aux(gf.domain_id, "clause_connectors", gf.learner.clause_connectors)
    return promoted_count


class PassiveStreamController:
    """Fair Control 1: Passive sequential stream ingestion under identical CPU budget.

    Observes ONLY the passive stream (zero active queries, zero boundary-dream refutation,
    zero anomaly-triggered disambiguation), updates ``ContinualKnowledgeBase`` with full
    regression replay, and strictly enforces ``.freeze()``.
    """

    comparison_only: bool = False
    uses_backprop: bool = False

    def __init__(
        self,
        *,
        kb: ContinualKnowledgeBase,
        budget: CurriculumBudget,
        relational_frontiers: Sequence[RelationalFrontierState] = (),
        grounding_frontiers: Sequence[GroundingFrontierState] = (),
        functional_frontiers: Sequence[FunctionalFrontierState] = (),
    ) -> None:
        self.kb = kb
        self.budget = budget
        self.relational_frontiers = list(relational_frontiers)
        self.grounding_frontiers = list(grounding_frontiers)
        self.functional_frontiers = list(functional_frontiers)
        self._frozen = False
        self.post_freeze_env_queries: int = 0
        self.self_certified_promotions: int = 0

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True
        for gf in self.grounding_frontiers:
            if not gf.learner.is_frozen:
                gf.learner.freeze()
        if not self.kb.is_frozen:
            self.kb.freeze()

    def ingest_passive_stream(self, *, phase_index: int = 1) -> dict[str, Any]:
        """Promote cold-start / passive-stream hypotheses without any active environment queries."""
        if self._frozen:
            raise CurriculumFrozenError("PassiveStreamController is frozen")
        t0 = time.perf_counter()
        promoted = 0

        for rf in self.relational_frontiers:
            while rf.pending_mismatches:
                st, act, after_st, delta = rf.pending_mismatches.pop(0)
                rf.filter_with_verified_transition(st, act, after_st, delta)
            active_ops: dict[str, OperatorCard] = {}
            for sid in sorted(rf.hypotheses_by_skill.keys()):
                rep = rf.representative_card(sid)
                out = self.kb.apply_candidate_update_with_replay(
                    domain_id=rf.domain_id,
                    skill_id=sid,
                    kind="relational_operator",
                    executable_dict=rep.to_executable_dict(),
                    executable_object=rep,
                    phase_index=phase_index,
                    notes="passive_stream_promotion",
                )
                if out.accepted:
                    promoted += 1
                    active_ops[sid] = rep
            for gf in self.grounding_frontiers:
                if active_ops:
                    gf.update_operators(active_ops)

        for gf in self.grounding_frontiers:
            while gf.pending_episodes:
                ep = gf.pending_episodes.pop(0)
                gf.ingest_verified_episode(ep)
            promoted += sync_grounding_learner_to_kb(self.kb, gf, phase_index=phase_index)

        for ff in self.functional_frontiers:
            for sid, sstate in sorted(ff.skills_by_id.items()):
                fcard = sstate.build_verified_card(version=1)
                out = self.kb.apply_candidate_update_with_replay(
                    domain_id=ff.domain_id,
                    skill_id=sid,
                    kind="functional_program",
                    executable_dict=fcard.to_executable_dict(),
                    executable_object=fcard,
                    phase_index=phase_index,
                    notes="passive_stream_functional_promotion",
                )
                if out.accepted:
                    promoted += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.budget.record_cpu_ms(elapsed_ms)
        return {
            "controller": "passive_stream_controller",
            "queries_spent": self.budget.queries_spent,
            "cpu_ms_spent": round(self.budget.cpu_ms_spent, 4),
            "promoted_skills": promoted,
        }


class RandomCurriculumController:
    """Fair Control 2: Spends the EXACT SAME active query & CPU budget uniformly at random.

    Draws unqueried probes uniformly at random across all frontiers from the exact same
    candidate pools, runs the exact same black-box verifiers and version-space/synthesis
    updates, and promotes through ``ContinualKnowledgeBase`` with full regression replay.
    """

    comparison_only: bool = False
    uses_backprop: bool = False

    def __init__(
        self,
        *,
        kb: ContinualKnowledgeBase,
        budget: CurriculumBudget,
        seed: int = 0,
        relational_frontiers: Sequence[RelationalFrontierState] = (),
        grounding_frontiers: Sequence[GroundingFrontierState] = (),
        functional_frontiers: Sequence[FunctionalFrontierState] = (),
        relational_envs: Mapping[str, TransitionEnvironment] | None = None,
        grounding_envs: Mapping[str, Callable[[bytes, StateGraph, ObservationMask], tuple[StateGraph, SparseDelta]]] | None = None,
        functional_envs: Mapping[tuple[str, str], Callable[[Mapping[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        self.kb = kb
        self.budget = budget
        self.rng = random.Random(int(seed))
        self.relational_frontiers = list(relational_frontiers)
        self.grounding_frontiers = list(grounding_frontiers)
        self.functional_frontiers = list(functional_frontiers)
        self._rel_envs = dict(relational_envs or {})
        self._gnd_envs = dict(grounding_envs or {})
        self._fn_envs = dict(functional_envs or {})
        self._frozen = False
        self.post_freeze_env_queries: int = 0
        self.blocked_post_freeze_attempts: int = 0
        self.self_certified_promotions: int = 0
        self.query_history: list[dict[str, Any]] = []

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True
        for gf in self.grounding_frontiers:
            if not gf.learner.is_frozen:
                gf.learner.freeze()
        if not self.kb.is_frozen:
            self.kb.freeze()

    def run_random_curriculum(
        self,
        *,
        phase_index: int = 1,
        step_accuracy_hook: Callable[[ContinualKnowledgeBase, int], float] | None = None,
        target_accuracy: float = 0.95,
    ) -> dict[str, Any]:
        """Spend the full query budget selecting uniformly at random across all unqueried probes."""
        if self._frozen:
            self.blocked_post_freeze_attempts += 1
            raise ControllerFrozenError("RandomCurriculumController is frozen")

        # First register cold-start passive state identically to PassiveStreamController
        init_ctrl = PassiveStreamController(
            kb=self.kb,
            budget=self.budget,
            relational_frontiers=self.relational_frontiers,
            grounding_frontiers=self.grounding_frontiers,
            functional_frontiers=self.functional_frontiers,
        )
        init_ctrl.ingest_passive_stream(phase_index=phase_index)

        # Build flat global pool of all unqueried probes across all 3 frontiers
        global_pool: list[tuple[str, str, str, int]] = []
        for rf in self.relational_frontiers:
            for idx in rf.unqueried_probe_indices():
                global_pool.append(("relational", rf.frontier_id, rf.candidate_probes[idx][1].action_id, idx))
        for gf in self.grounding_frontiers:
            for idx in gf.unqueried_probe_indices():
                global_pool.append(("grounding", gf.frontier_id, gf.candidate_probes[idx][0], idx))
        for ff in self.functional_frontiers:
            for sid, idx in ff.unqueried_skill_probes():
                global_pool.append(("functional", ff.frontier_id, sid, idx))

        self.rng.shuffle(global_pool)
        rf_map = {f.frontier_id: f for f in self.relational_frontiers}
        gf_map = {f.frontier_id: f for f in self.grounding_frontiers}
        ff_map = {f.frontier_id: f for f in self.functional_frontiers}

        queries_to_target: int | None = None
        if step_accuracy_hook is not None:
            if step_accuracy_hook(self.kb, 0) >= target_accuracy:
                queries_to_target = 0

        cursor = 0
        while self.budget.can_spend_query() and cursor < len(global_pool):
            t0 = time.perf_counter()
            f_type, fid, sid, p_idx = global_pool[cursor]
            cursor += 1
            kb_updated = False

            if f_type == "relational":
                rf = rf_map[fid]
                if p_idx in rf.queried_indices:
                    continue
                rf.queried_indices.add(p_idx)
                st, act, _ptag = rf.candidate_probes[p_idx]
                env = self._rel_envs[fid]
                after_st, delta = env.step(st, act)
                self.budget.record_query(1)
                rf.queries_spent_on_frontier += 1
                prev_n, new_n, rep = rf.filter_with_verified_transition(st, act, after_st, delta)
                if rep is not None and new_n < prev_n:
                    out_r = self.kb.apply_candidate_update_with_replay(
                        domain_id=rf.domain_id,
                        skill_id=act.action_id,
                        kind="relational_operator",
                        executable_dict=rep.to_executable_dict(),
                        executable_object=rep,
                        phase_index=phase_index,
                        notes="random_curriculum_relational_update",
                    )
                    kb_updated = out_r.accepted
                    active_ops = {
                        k: v
                        for k, v in self.kb.get_active_domain_objects(
                            rf.domain_id, kind="relational_operator"
                        ).items()
                        if isinstance(v, OperatorCard)
                    }
                    for gf in self.grounding_frontiers:
                        if active_ops:
                            gf.update_operators(active_ops)

            elif f_type == "grounding":
                gf = gf_map[fid]
                if p_idx in gf.queried_indices:
                    continue
                gf.queried_indices.add(p_idx)
                pid, raw_b, scene_st, mask = gf.candidate_probes[p_idx]
                g_env = self._gnd_envs[fid]
                after_st, _delta = g_env(raw_b, scene_st, mask)
                self.budget.record_query(1)
                gf.queries_spent_on_frontier += 1
                ep = UnlabelledGroundingEpisode(
                    episode_id=f"rand_gnd_{pid}_{self.budget.queries_spent}",
                    raw_instruction_bytes=raw_b,
                    before_state=scene_st,
                    after_state=after_st,
                    observation_mask=mask,
                )
                prev_prom = len(gf.learner.promoted_constructions)
                prev_rules = (
                    len(gf.learner.negation_rules)
                    + len(gf.learner.condition_rules)
                    + len(gf.learner.clause_connectors)
                )
                gf.ingest_verified_episode(ep)
                n_added = sync_grounding_learner_to_kb(self.kb, gf, phase_index=phase_index)
                new_rules = (
                    len(gf.learner.negation_rules)
                    + len(gf.learner.condition_rules)
                    + len(gf.learner.clause_connectors)
                )
                kb_updated = (n_added > 0) or (len(gf.learner.promoted_constructions) > prev_prom) or (new_rules > prev_rules)

            elif f_type == "functional":
                ff = ff_map[fid]
                sstate = ff.skills_by_id[sid]
                if p_idx in sstate.queried_indices:
                    continue
                sstate.queried_indices.add(p_idx)
                before_slots = sstate.candidate_probes[p_idx]
                fn_env = self._fn_envs[(fid, sid)]
                after_slots = fn_env(before_slots)
                self.budget.record_query(1)
                ff.queries_spent_on_frontier += 1
                ep_f = SlotTransitionEpisode(
                    episode_id=f"rand_fn_{sid}_{self.budget.queries_spent}",
                    action_id=sid,
                    before_slots=tuple(before_slots.items()),
                    after_slots=tuple(after_slots.items()),
                )
                prev_amb = sstate.ambiguity_count()
                sstate.ingest_verified_episode(ep_f)
                if sstate.ambiguity_count() < prev_amb:
                    ver = len(self.kb.get_skill_history(ff.domain_id, sid)) + 1
                    fcard = sstate.build_verified_card(version=ver)
                    out_f = self.kb.apply_candidate_update_with_replay(
                        domain_id=ff.domain_id,
                        skill_id=sid,
                        kind="functional_program",
                        executable_dict=fcard.to_executable_dict(),
                        executable_object=fcard,
                        phase_index=phase_index,
                        notes="random_curriculum_functional_update",
                    )
                    kb_updated = out_f.accepted

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.budget.record_cpu_ms(elapsed_ms)
            self.query_history.append(
                {
                    "query_index": self.budget.queries_spent,
                    "frontier_type": f_type,
                    "frontier_id": fid,
                    "skill_id": sid,
                    "probe_index": p_idx,
                }
            )
            if kb_updated and queries_to_target is None and step_accuracy_hook is not None:
                if step_accuracy_hook(self.kb, self.budget.queries_spent) >= target_accuracy:
                    queries_to_target = self.budget.queries_spent

        return {
            "controller": "random_curriculum_controller",
            "queries_spent": self.budget.queries_spent,
            "cpu_ms_spent": round(self.budget.cpu_ms_spent, 4),
            "queries_to_target": queries_to_target,
        }


__all__ = [
    "AnomalyKind",
    "AnomalySignal",
    "CURRICULUM_SCHEMA_VERSION",
    "ControllerFrozenError",
    "CurriculumAction",
    "CurriculumActionKind",
    "CurriculumBudget",
    "CurriculumFrozenError",
    "FunctionalFrontierState",
    "FunctionalSkillState",
    "GroundingFrontierState",
    "PassiveStreamController",
    "RandomCurriculumController",
    "RelationalFrontierState",
    "SelfCurriculumScheduler",
    "enumerate_consistent_slot_programs",
    "sync_grounding_learner_to_kb",
]
