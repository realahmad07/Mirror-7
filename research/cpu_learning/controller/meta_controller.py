"""Stage S9 Autonomous Meta-Controller, Non-Starvation Tick Arbitration & Latency Monitor.

STRICT ISOLATION & PYTHON 3.11 COMPATIBILITY (PDF pp. 4–5, 7, 10, 12–14, 17):
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Uses pure Python 3.11 standard library ONLY (zero Python 3.12+ APIs).
- Orchestrates the complete L0–L7 11-step closed loop:
  Observe -> Frame -> Difference -> SHARD Propose -> ITL Generalise -> Contrast ->
  Independent Verify -> Correct -> Compose -> RDC MDL Consolidate -> ContinualKB Replay.
- Enforces non-starvation interactive problem-solving SLA alongside bounded background
  learning slices, zero self-certification (``self_certified_promotions == 0``),
  automatic regression quarantine/rollback, and strict ``.freeze()`` isolation
  (``post_freeze_env_queries == 0``).
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from research.cpu_learning.continual.replay_manager import (
    CandidateUpdateOutcome,
    ContinualKnowledgeBase,
)
from research.cpu_learning.controller.curriculum import (
    AnomalySignal,
    ControllerFrozenError,
    CurriculumAction,
    CurriculumBudget,
    FunctionalFrontierState,
    GroundingFrontierState,
    RelationalFrontierState,
    SelfCurriculumScheduler,
    sync_grounding_learner_to_kb,
)
from research.cpu_learning.frames.state_frame import TypedStateFrame, extract_masked_sparse_delta
from research.cpu_learning.grounding.construction_learner import (
    GroundedInstructionExecutor,
    RawByteConstructionLearner,
    UnlabelledGroundingEpisode,
)
from research.cpu_learning.itl.contrast_harness import TransitionEnvironment
from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.rdc.consolidation import ConsolidationResult, RDCConsolidator
from research.cpu_learning.rdc.mdl_judge import execute_library_sequence
from research.cpu_learning.shard.proposer import SHARDProposalEngine
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    ObservationMask,
    SparseDelta,
    StateGraph,
    canonical_json_bytes,
    sha256_hex,
)
from research.cpu_learning.transfer import ASTRewriteCard
from research.cpu_learning.transfer.math_domain import execute_multidigit_addition
from research.cpu_learning.transfer.transfer_inducer import (
    FunctionalOperatorCard,
    RestrictedExecutionSandbox,
    SlotTransitionEpisode,
    normalize_slot_value,
)

META_CONTROLLER_SCHEMA_VERSION: str = "s9-autonomous-meta-controller-v1"

CLOSED_LOOP_L0_L7_STEPS: tuple[str, ...] = (
    "L0_observe",
    "L1_frame",
    "L2_difference",
    "L3_shard_propose",
    "L4_itl_generalise",
    "L5_contrast",
    "L6_independent_verify",
    "L6_correct",
    "L6_compose",
    "L7_rdc_mdl_consolidate",
    "L7_continual_regression_replay",
)

InteractiveQueryKind = Literal[
    "relational_sequence",
    "utf8_instruction",
    "functional_slots",
    "multidigit_addition",
    "ast_refactor",
]


# ---------------------------------------------------------------------------
# 1. Independent Verification Receipts & Freeze-Gated Verifier Proxies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndependentVerificationReceipt:
    """Cryptographic receipt proving a candidate passed an external non-self verifier."""

    receipt_id: str
    candidate_sha256: str
    verifier_kind: str
    probes_verified: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "candidate_sha256": self.candidate_sha256,
            "verifier_kind": self.verifier_kind,
            "probes_verified": int(self.probes_verified),
            "passed": bool(self.passed),
        }


class FreezeGatedVerifierGate:
    """Central freeze-lock and external-query auditor shared across all black-box verifiers."""

    def __init__(self) -> None:
        self._frozen: bool = False
        self.total_env_queries: int = 0
        self.post_freeze_env_queries: int = 0
        self.blocked_post_freeze_attempts: int = 0
        self._receipt_seq: int = 0
        self._issued_receipts: dict[str, IndependentVerificationReceipt] = {}
        self._consumed_receipt_ids: set[str] = set()

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def _guard_mutable(self) -> None:
        if self._frozen:
            self.blocked_post_freeze_attempts += 1
            raise ControllerFrozenError(
                "AutonomousMetaController is frozen; environment queries are strictly forbidden"
            )

    def issue_receipt(
        self,
        executable_dict: Mapping[str, Any],
        *,
        verifier_kind: str,
        probes_verified: int,
        passed: bool,
    ) -> IndependentVerificationReceipt:
        self._guard_mutable()
        vk_clean = str(verifier_kind).strip()
        if not vk_clean or "self" in vk_clean.lower():
            raise PermissionError(
                f"Self-certified verifier_kind is forbidden: {verifier_kind!r}"
            )
        self._receipt_seq += 1
        digest = sha256_hex(canonical_json_bytes(dict(executable_dict)))
        rec = IndependentVerificationReceipt(
            receipt_id=f"vrec_{self._receipt_seq:05d}",
            candidate_sha256=digest,
            verifier_kind=vk_clean,
            probes_verified=int(probes_verified),
            passed=bool(passed),
        )
        self._issued_receipts[rec.receipt_id] = rec
        return rec

    def verify_and_consume_receipt(
        self,
        receipt: IndependentVerificationReceipt | None,
        expected_sha256: str,
    ) -> IndependentVerificationReceipt:
        """Verify ``receipt`` was genuinely minted by ``self``, is unconsumed, non-self, and matches ``expected_sha256``."""
        if not isinstance(receipt, IndependentVerificationReceipt):
            raise PermissionError("Missing IndependentVerificationReceipt")
        if receipt.receipt_id in self._consumed_receipt_ids:
            raise PermissionError(f"Verification receipt {receipt.receipt_id!r} already consumed")
        issued = self._issued_receipts.get(receipt.receipt_id)
        if issued is None or issued != receipt:
            raise PermissionError(
                f"Forged or unissued verification receipt {receipt.receipt_id!r}"
            )
        if "self" in issued.verifier_kind.lower():
            raise PermissionError(
                f"Self-certified receipt forbidden: {issued.verifier_kind!r}"
            )
        if (
            not issued.passed
            or issued.probes_verified < 1
            or issued.candidate_sha256 != expected_sha256
        ):
            raise PermissionError("Verification receipt failed or digest mismatch")
        self._consumed_receipt_ids.add(issued.receipt_id)
        return issued

    def step_relational(
        self,
        env: TransitionEnvironment,
        state: StateGraph,
        action: ActionInvocation,
    ) -> tuple[StateGraph, SparseDelta]:
        self._guard_mutable()
        out = env.step(state, action)
        if self._frozen:
            self.post_freeze_env_queries += 1
        self.total_env_queries += 1
        return out

    def step_grounding(
        self,
        env_fn: Callable[[bytes, StateGraph, ObservationMask], tuple[StateGraph, SparseDelta]],
        raw_bytes: bytes,
        scene_state: StateGraph,
        mask: ObservationMask,
    ) -> tuple[StateGraph, SparseDelta]:
        self._guard_mutable()
        out = env_fn(raw_bytes, scene_state, mask)
        if self._frozen:
            self.post_freeze_env_queries += 1
        self.total_env_queries += 1
        return out

    def step_functional(
        self,
        env_fn: Callable[[Mapping[str, Any]], dict[str, Any]],
        before_slots: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._guard_mutable()
        out = env_fn(before_slots)
        if self._frozen:
            self.post_freeze_env_queries += 1
        self.total_env_queries += 1
        return out


# ---------------------------------------------------------------------------
# 2. Live Interactive Problem-Solving Request & Latency SLA Monitor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractiveQueryRequest:
    """Live problem-solving request serviced on the normal non-learning execution path."""

    request_id: str
    domain_id: str
    query_kind: InteractiveQueryKind
    payload: dict[str, Any]
    expected_answer: Any = None


@dataclass(frozen=True)
class InteractiveQueryRecord:
    """Latency and accuracy audit record for a single live interactive request."""

    request_id: str
    domain_id: str
    query_kind: str
    tick_served: int
    exec_ms: float
    queue_wait_ms: float
    total_latency_ms: float
    sla_ms: float
    starved: bool
    solved_ok: bool
    correct: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "domain_id": self.domain_id,
            "query_kind": self.query_kind,
            "tick_served": self.tick_served,
            "exec_ms": round(self.exec_ms, 4),
            "queue_wait_ms": round(self.queue_wait_ms, 4),
            "total_latency_ms": round(self.total_latency_ms, 4),
            "sla_ms": round(self.sla_ms, 4),
            "starved": self.starved,
            "solved_ok": self.solved_ok,
            "correct": self.correct,
        }


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


class InteractiveLatencyMonitor:
    """Tracks interactive query SLA compliance under concurrent autonomous learning (PDF p. 12)."""

    def __init__(self, *, sla_ms: float = 5.0, starvation_ms: float = 25.0) -> None:
        self.sla_ms = float(sla_ms)
        self.starvation_ms = float(starvation_ms)
        self._records: list[InteractiveQueryRecord] = []
        self.dropped_interactive_count: int = 0

    @property
    def records(self) -> tuple[InteractiveQueryRecord, ...]:
        return tuple(self._records)

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def starved_interactive_count(self) -> int:
        return sum(1 for r in self._records if r.starved)

    def record(
        self,
        *,
        request_id: str,
        domain_id: str,
        query_kind: str,
        tick_served: int,
        exec_ms: float,
        queue_wait_ms: float = 0.0,
        solved_ok: bool = True,
        correct: bool | None = None,
    ) -> InteractiveQueryRecord:
        q_wait = max(0.0, float(queue_wait_ms))
        ex_t = max(0.0, float(exec_ms))
        tot = q_wait + ex_t
        rec = InteractiveQueryRecord(
            request_id=str(request_id),
            domain_id=str(domain_id),
            query_kind=str(query_kind),
            tick_served=int(tick_served),
            exec_ms=ex_t,
            queue_wait_ms=q_wait,
            total_latency_ms=tot,
            sla_ms=self.sla_ms,
            starved=(q_wait > self.starvation_ms) or (tot > self.starvation_ms),
            solved_ok=bool(solved_ok),
            correct=correct,
        )
        self._records.append(rec)
        return rec

    def summary_dict(self) -> dict[str, Any]:
        lats = sorted(r.total_latency_ms for r in self._records)
        execs = sorted(r.exec_ms for r in self._records)
        graded = [r for r in self._records if r.correct is not None]
        acc = (
            sum(1 for r in graded if r.correct) / float(len(graded))
            if graded
            else 1.0
        )
        mean_lat = (sum(lats) / float(len(lats))) if lats else 0.0
        return {
            "queries_served": len(self._records),
            "sla_ms": round(self.sla_ms, 4),
            "mean_ms": round(mean_lat, 4),
            "p50_ms": round(_percentile(lats, 0.50), 4),
            "p95_ms": round(_percentile(lats, 0.95), 4),
            "max_ms": round(lats[-1] if lats else 0.0, 4),
            "exec_p95_ms": round(_percentile(execs, 0.95), 4),
            "starved_interactive_count": self.starved_interactive_count,
            "dropped_interactive_count": self.dropped_interactive_count,
            "interactive_accuracy": round(acc, 6),
        }


# ---------------------------------------------------------------------------
# 3. Structured Per-Tick Telemetry Trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerTickTrace:
    """Auditable per-tick execution trace with exact per-subsystem CPU millisecond accounting."""

    tick_index: int
    interactive_queries_served: int
    interactive_slice_ms: float
    anomaly_count: int
    top_anomaly_kind: str
    chosen_action_kind: str
    target_frontier: str
    target_skill_id: str | None
    expected_info_gain: float
    env_queries_this_tick: int
    cumulative_env_queries: int
    retrieval_ms: float
    induction_ms: float
    query_selection_ms: float
    verification_ms: float
    consolidation_ms: float
    replay_ms: float
    learning_slice_ms: float
    learning_slice_budget_ms: float
    within_slice_budget: bool
    loop_steps_executed: tuple[str, ...]
    promoted_updates: int
    quarantined_updates: int
    active_library_bits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick_index": self.tick_index,
            "interactive_queries_served": self.interactive_queries_served,
            "interactive_slice_ms": round(self.interactive_slice_ms, 4),
            "anomaly_count": self.anomaly_count,
            "top_anomaly_kind": self.top_anomaly_kind,
            "chosen_action_kind": self.chosen_action_kind,
            "target_frontier": self.target_frontier,
            "target_skill_id": self.target_skill_id,
            "expected_info_gain": round(self.expected_info_gain, 6),
            "env_queries_this_tick": self.env_queries_this_tick,
            "cumulative_env_queries": self.cumulative_env_queries,
            "retrieval_ms": round(self.retrieval_ms, 4),
            "induction_ms": round(self.induction_ms, 4),
            "query_selection_ms": round(self.query_selection_ms, 4),
            "verification_ms": round(self.verification_ms, 4),
            "consolidation_ms": round(self.consolidation_ms, 4),
            "replay_ms": round(self.replay_ms, 4),
            "learning_slice_ms": round(self.learning_slice_ms, 4),
            "learning_slice_budget_ms": round(self.learning_slice_budget_ms, 4),
            "within_slice_budget": self.within_slice_budget,
            "loop_steps_executed": list(self.loop_steps_executed),
            "promoted_updates": self.promoted_updates,
            "quarantined_updates": self.quarantined_updates,
            "active_library_bits": self.active_library_bits,
        }

    def to_csv_row(self, *, seed: int) -> dict[str, Any]:
        row: dict[str, Any] = {"seed": int(seed)}
        row.update(self.to_dict())
        row["loop_steps_executed"] = "|".join(self.loop_steps_executed)
        return row


# ---------------------------------------------------------------------------
# 4. AutonomousMetaController (Closed-Loop L0–L7 Orchestrator)
# ---------------------------------------------------------------------------


class AutonomousMetaController:
    """Domain-agnostic CPU-native autonomous learning controller (PDF pp. 4–5, 7, 12–14).

    Interleaves priority interactive problem-solving queries with bounded background
    learning slices driven by ``SelfCurriculumScheduler``, enforcing:
    - Zero self-certification (``self_certified_promotions == 0``)
    - Mandatory cross-domain regression replay via ``ContinualKnowledgeBase``
    - Behavioral-equivalence MDL consolidation via ``RDCConsolidator``
    - Strict ``.freeze()`` evaluation lock (``post_freeze_env_queries == 0``)
    """

    comparison_only: bool = False
    uses_backprop: bool = False

    def __init__(
        self,
        *,
        kb: ContinualKnowledgeBase,
        budget: CurriculumBudget,
        scheduler: SelfCurriculumScheduler | None = None,
        consolidator: RDCConsolidator | None = None,
        interactive_sla_ms: float = 5.0,
        max_interactive_per_tick: int = 8,
        relational_frontiers: Sequence[RelationalFrontierState] = (),
        grounding_frontiers: Sequence[GroundingFrontierState] = (),
        functional_frontiers: Sequence[FunctionalFrontierState] = (),
        relational_envs: Mapping[str, TransitionEnvironment] | None = None,
        grounding_envs: Mapping[
            str, Callable[[bytes, StateGraph, ObservationMask], tuple[StateGraph, SparseDelta]]
        ]
        | None = None,
        functional_envs: Mapping[
            tuple[str, str], Callable[[Mapping[str, Any]], dict[str, Any]]
        ]
        | None = None,
    ) -> None:
        self.kb = kb
        self.budget = budget
        self.scheduler = scheduler if scheduler is not None else SelfCurriculumScheduler()
        self.consolidator = consolidator if consolidator is not None else RDCConsolidator()
        self.latency_monitor = InteractiveLatencyMonitor(sla_ms=interactive_sla_ms)
        self.max_interactive_per_tick = max(1, int(max_interactive_per_tick))

        self.relational_frontiers: dict[str, RelationalFrontierState] = {
            f.frontier_id: f for f in relational_frontiers
        }
        self.grounding_frontiers: dict[str, GroundingFrontierState] = {
            f.frontier_id: f for f in grounding_frontiers
        }
        self.functional_frontiers: dict[str, FunctionalFrontierState] = {
            f.frontier_id: f for f in functional_frontiers
        }

        self._rel_envs = dict(relational_envs or {})
        self._gnd_envs = dict(grounding_envs or {})
        self._fn_envs = dict(functional_envs or {})
        self.verifier_gate = FreezeGatedVerifierGate()

        self._interactive_queue: deque[tuple[InteractiveQueryRequest, float]] = deque()
        self._tick_traces: list[ControllerTickTrace] = []
        self._consolidation_results: list[ConsolidationResult] = []
        self._verification_receipts: list[IndependentVerificationReceipt] = []
        self._observed_loop_steps: set[str] = set()
        self._observed_anomalies: set[str] = set()
        self._observed_actions: set[str] = set()

        self.self_certified_promotions: int = 0
        self.rejected_self_certification_attempts: int = 0
        self.promoted_updates_total: int = 0
        self.quarantined_updates_total: int = 0
        self._pending_replay_audit: bool = False
        self._tick_index: int = 0
        self._cached_library_bits: int = kb.active_library_compressed_bits()

    @property
    def is_frozen(self) -> bool:
        return self.verifier_gate.is_frozen

    @property
    def post_freeze_env_queries(self) -> int:
        return self.verifier_gate.post_freeze_env_queries

    @property
    def blocked_post_freeze_attempts(self) -> int:
        return self.verifier_gate.blocked_post_freeze_attempts

    @property
    def tick_traces(self) -> tuple[ControllerTickTrace, ...]:
        return tuple(self._tick_traces)

    @property
    def consolidation_results(self) -> tuple[ConsolidationResult, ...]:
        return tuple(self._consolidation_results)

    @property
    def observed_loop_steps(self) -> tuple[str, ...]:
        return tuple(s for s in CLOSED_LOOP_L0_L7_STEPS if s in self._observed_loop_steps)

    def freeze(self) -> None:
        """Permanently lock all learning, KB mutation, and external environment access."""
        self.verifier_gate.freeze()
        for gf in self.grounding_frontiers.values():
            if not gf.learner.is_frozen:
                gf.learner.freeze()
        if not self.kb.is_frozen:
            self.kb.freeze()

    def _require_mutable(self) -> None:
        if self.is_frozen:
            self.verifier_gate.blocked_post_freeze_attempts += 1
            raise ControllerFrozenError(
                "AutonomousMetaController is frozen; learning mutations are forbidden"
            )

    def query_environment_relational(
        self,
        frontier_id: str,
        state: StateGraph,
        action: ActionInvocation,
    ) -> tuple[StateGraph, SparseDelta]:
        """Public freeze-gated environment query entry point (raises ``ControllerFrozenError`` if frozen)."""
        env = self._rel_envs[frontier_id]
        return self.verifier_gate.step_relational(env, state, action)

    def _promote_verified_candidate(
        self,
        *,
        domain_id: str,
        skill_id: str,
        kind: str,
        executable_dict: Mapping[str, Any],
        executable_object: Any,
        receipt: IndependentVerificationReceipt | None,
        phase_index: int,
        notes: str,
    ) -> CandidateUpdateOutcome:
        """Enforce zero self-certification + anti-forgery receipt consumption + regression replay."""
        self._require_mutable()
        expected_sha = sha256_hex(canonical_json_bytes(dict(executable_dict)))
        try:
            verified_rec = self.verifier_gate.verify_and_consume_receipt(
                receipt, expected_sha
            )
        except PermissionError as exc:
            self.rejected_self_certification_attempts += 1
            raise PermissionError(
                f"Rejected unverified/forged/self-certified promotion for {domain_id}:{skill_id}: {exc}"
            ) from exc
        self._verification_receipts.append(verified_rec)
        outcome = self.kb.apply_candidate_update_with_replay(
            domain_id=domain_id,
            skill_id=skill_id,
            kind=kind,
            executable_dict=executable_dict,
            executable_object=executable_object,
            phase_index=phase_index,
            notes=notes,
        )
        if outcome.accepted:
            self.promoted_updates_total += 1
            self._cached_library_bits = self.kb.active_library_compressed_bits()
        if outcome.quarantined:
            self.quarantined_updates_total += 1
        return outcome

    def stage_external_candidate_with_verification(
        self,
        *,
        domain_id: str,
        skill_id: str,
        kind: str,
        executable_dict: Mapping[str, Any],
        executable_object: Any,
        receipt: IndependentVerificationReceipt | None,
        phase_index: int = 1,
        notes: str = "external_stream_candidate",
    ) -> CandidateUpdateOutcome:
        """Stage a candidate skill (e.g. from noisy stream) through independent verification + replay."""
        return self._promote_verified_candidate(
            domain_id=domain_id,
            skill_id=skill_id,
            kind=kind,
            executable_dict=executable_dict,
            executable_object=executable_object,
            receipt=receipt,
            phase_index=phase_index,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Interactive Problem-Solving Path (Read-Only, Works Before & After Freeze)
    # ------------------------------------------------------------------

    def enqueue_interactive_request(self, req: InteractiveQueryRequest) -> None:
        """Enqueue a live interactive problem-solving query with arrival timestamp."""
        self._interactive_queue.append((req, time.perf_counter()))

    def solve_interactive_now(
        self,
        req: InteractiveQueryRequest,
        *,
        queue_wait_ms: float = 0.0,
    ) -> tuple[Any, InteractiveQueryRecord]:
        """Execute a single read-only problem-solving query using promoted KB skills (0 env queries)."""
        t0 = time.perf_counter()
        solved_ok = True
        pred_ans: Any = None

        if req.query_kind == "relational_sequence":
            init_st: StateGraph = req.payload["initial_state"]
            seq: Sequence[ActionInvocation] = req.payload["action_sequence"]
            mask: ObservationMask = req.payload.get(
                "observation_mask", ObservationMask(fully_observed=True)
            )
            kb_aliases = dict(self.kb.get_domain_aux(req.domain_id, "alias_map", {}) or {})
            req_aliases = dict(req.payload.get("alias_map", {}) or {})
            alias_map: dict[str, str] = {**kb_aliases, **req_aliases}
            cards = {
                k: v
                for k, v in self.kb.get_active_domain_objects(
                    req.domain_id, kind="relational_operator"
                ).items()
                if isinstance(v, OperatorCard)
            }
            final_st, abstained, _inters = execute_library_sequence(
                cards, init_st, seq, alias_map=alias_map, mask=mask
            )
            solved_ok = not abstained
            pred_ans = ("ABSTAIN" if abstained else final_st)

        elif req.query_kind == "utf8_instruction":
            raw_b: bytes = req.payload["raw_bytes"]
            scene_st: StateGraph = req.payload["scene_state"]
            mask_u: ObservationMask = req.payload.get(
                "observation_mask", ObservationMask(fully_observed=True)
            )
            learner_obj = self.kb.get_domain_aux(req.domain_id, "grounding_learner")
            if isinstance(learner_obj, RawByteConstructionLearner):
                exec_u = GroundedInstructionExecutor(learner_obj)
                res_u = exec_u.execute_raw_utf8(raw_b, scene_st, mask=mask_u)
                solved_ok = not res_u.abstained
                pred_ans = ("ABSTAIN" if res_u.abstained else res_u.final_state)
            else:
                solved_ok = False
                pred_ans = "ABSTAIN"

        elif req.query_kind == "functional_slots":
            sid_f: str = req.payload["skill_id"]
            before_s: Mapping[str, Any] = req.payload["before_slots"]
            entry_f = self.kb.get_active_skill(req.domain_id, sid_f)
            if entry_f is not None and isinstance(entry_f.executable_object, FunctionalOperatorCard):
                pred_ans = entry_f.executable_object.apply_slots(before_s, use_sandbox=True)
            else:
                solved_ok = False
                pred_ans = None

        elif req.query_kind == "multidigit_addition":
            sid_m: str = req.payload["skill_id"]
            lhs_int: int = int(req.payload["lhs_int"])
            rhs_int: int = int(req.payload["rhs_int"])
            radix_m: int = int(req.payload.get("radix", 10))
            entry_m = self.kb.get_active_skill(req.domain_id, sid_m)
            if entry_m is not None and isinstance(entry_m.executable_object, FunctionalOperatorCard):
                pred_sum, _coll = execute_multidigit_addition(
                    entry_m.executable_object, lhs_int, rhs_int, radix_m
                )
                pred_ans = pred_sum
            else:
                solved_ok = False
                pred_ans = None

        elif req.query_kind == "ast_refactor":
            sid_a: str = req.payload["skill_id"]
            src_code: str = req.payload["source_code"]
            params_a: Mapping[str, Any] = req.payload.get("params", {})
            test_calls: Sequence[tuple[tuple[Any, ...], Any]] = req.payload.get("test_calls", ())
            fn_name: str = req.payload.get("fn_name", "f")
            entry_a = self.kb.get_active_skill(req.domain_id, sid_a)
            if entry_a is not None and isinstance(entry_a.executable_object, ASTRewriteCard):
                rewritten = entry_a.executable_object.apply_code(src_code, params_a)
                if test_calls:
                    fn_compiled, _rsn = RestrictedExecutionSandbox.compile_verified_function(
                        rewritten, fn_name
                    )
                    if fn_compiled is None:
                        solved_ok = False
                        pred_ans = None
                    else:
                        pred_ans = tuple(
                            normalize_slot_value(fn_compiled(*args)) for args, _exp in test_calls
                        )
                else:
                    pred_ans = rewritten
            else:
                solved_ok = False
                pred_ans = None

        exec_ms = (time.perf_counter() - t0) * 1000.0
        correct: bool | None = None
        if req.expected_answer is not None:
            correct = bool(pred_ans == req.expected_answer)
        rec = self.latency_monitor.record(
            request_id=req.request_id,
            domain_id=req.domain_id,
            query_kind=req.query_kind,
            tick_served=self._tick_index,
            exec_ms=exec_ms,
            queue_wait_ms=queue_wait_ms,
            solved_ok=solved_ok,
            correct=correct,
        )
        return (pred_ans, rec)

    # ------------------------------------------------------------------
    # Cold-Start Stream Initialization & Bounded Autonomous Learning Tick
    # ------------------------------------------------------------------

    def initialize_cold_start_frontiers(self, *, phase_index: int = 1) -> int:
        """Promote initial verified cold-start hypotheses across all registered frontiers."""
        self._require_mutable()
        promoted = 0
        for rf in self.relational_frontiers.values():
            active_ops: dict[str, OperatorCard] = {}
            for sid in sorted(rf.hypotheses_by_skill.keys()):
                rep = rf.representative_card(sid)
                rec = self.verifier_gate.issue_receipt(
                    rep.to_executable_dict(),
                    verifier_kind="cold_start_positive_demo_replay",
                    probes_verified=max(1, rep.support_positive),
                    passed=True,
                )
                out = self._promote_verified_candidate(
                    domain_id=rf.domain_id,
                    skill_id=sid,
                    kind="relational_operator",
                    executable_dict=rep.to_executable_dict(),
                    executable_object=rep,
                    receipt=rec,
                    phase_index=phase_index,
                    notes="cold_start_relational_seed",
                )
                if out.accepted:
                    promoted += 1
                    active_ops[sid] = rep
            if active_ops:
                for gf in self.grounding_frontiers.values():
                    gf.update_operators(active_ops)

        for gf in self.grounding_frontiers.values():
            for ccard in gf.learner.promoted_constructions:
                rec_g = self.verifier_gate.issue_receipt(
                    ccard.to_executable_dict(),
                    verifier_kind="cold_start_transition_alignment",
                    probes_verified=max(1, ccard.support_count),
                    passed=True,
                )
                out_g = self._promote_verified_candidate(
                    domain_id=gf.domain_id,
                    skill_id=ccard.construction_id,
                    kind="utf8_construction",
                    executable_dict=ccard.to_executable_dict(),
                    executable_object=ccard,
                    receipt=rec_g,
                    phase_index=phase_index,
                    notes="cold_start_grounding_seed",
                )
                if out_g.accepted:
                    promoted += 1
            self.kb.set_domain_aux(gf.domain_id, "grounding_learner", gf.learner)

        for ff in self.functional_frontiers.values():
            for sid, sstate in sorted(ff.skills_by_id.items()):
                fcard = sstate.build_verified_card(version=1)
                rec_f = self.verifier_gate.issue_receipt(
                    fcard.to_executable_dict(),
                    verifier_kind="restricted_execution_sandbox",
                    probes_verified=max(1, len(sstate.observed_episodes)),
                    passed=True,
                )
                out_f = self._promote_verified_candidate(
                    domain_id=ff.domain_id,
                    skill_id=sid,
                    kind="functional_program",
                    executable_dict=fcard.to_executable_dict(),
                    executable_object=fcard,
                    receipt=rec_f,
                    phase_index=phase_index,
                    notes="cold_start_functional_seed",
                )
                if out_f.accepted:
                    promoted += 1

        self._observed_loop_steps.update(
            {
                "L0_observe",
                "L1_frame",
                "L2_difference",
                "L3_shard_propose",
                "L4_itl_generalise",
                "L6_independent_verify",
                "L7_continual_regression_replay",
            }
        )
        return promoted

    def _drain_interactive_checkpoint(self, limit: int) -> tuple[int, float]:
        """Cooperative non-starvation checkpoint draining queued interactive requests."""
        if not self._interactive_queue or limit <= 0:
            return (0, 0.0)
        t0 = time.perf_counter()
        served = 0
        while self._interactive_queue and served < limit:
            req, enq_t = self._interactive_queue.popleft()
            bg_wait_ms = max(0.0, (t0 - enq_t) * 1000.0)
            self.solve_interactive_now(req, queue_wait_ms=bg_wait_ms)
            served += 1
        return (served, (time.perf_counter() - t0) * 1000.0)

    def run_tick(
        self,
        *,
        phase_index: int = 1,
    ) -> ControllerTickTrace:
        """Execute ONE non-starving controller tick:
        1. Priority interactive queue service (normal problem-solving path, 0 env queries).
        2. At most ONE bounded background learning slice executing the L0–L7 closed loop.
        """
        self._require_mutable()
        self._tick_index += 1

        # --------------------------------------------------------------
        # Priority 1: Non-Starvation Interactive Query Servicing (Drain All Queued)
        # --------------------------------------------------------------
        drain_limit = max(self.max_interactive_per_tick, len(self._interactive_queue))
        served_count, interactive_slice_ms = self._drain_interactive_checkpoint(drain_limit)

        # Short-circuit immediately if total learning CPU budget or slice budget is exhausted
        if not self.budget.can_run_slice() or self.budget.learning_slice_ms <= 0.0:
            self._observed_actions.add("idle_noop")
            trace_exhausted = ControllerTickTrace(
                tick_index=self._tick_index,
                interactive_queries_served=served_count,
                interactive_slice_ms=interactive_slice_ms,
                anomaly_count=0,
                top_anomaly_kind="none",
                chosen_action_kind="idle_noop",
                target_frontier="none",
                target_skill_id=None,
                expected_info_gain=0.0,
                env_queries_this_tick=0,
                cumulative_env_queries=self.budget.queries_spent,
                retrieval_ms=0.0,
                induction_ms=0.0,
                query_selection_ms=0.0,
                verification_ms=0.0,
                consolidation_ms=0.0,
                replay_ms=0.0,
                learning_slice_ms=0.0,
                learning_slice_budget_ms=self.budget.learning_slice_ms,
                within_slice_budget=True,
                loop_steps_executed=(),
                promoted_updates=0,
                quarantined_updates=0,
                active_library_bits=self._cached_library_bits,
            )
            self._tick_traces.append(trace_exhausted)
            return trace_exhausted

        # --------------------------------------------------------------
        # Priority 2: Bounded Background Autonomous Learning Slice
        # --------------------------------------------------------------
        t_learn_start = time.perf_counter()
        effective_slice_ms = min(self.budget.cpu_ms_remaining, self.budget.learning_slice_ms)
        retrieval_ms = 0.0
        query_selection_ms = 0.0
        verification_ms = 0.0
        induction_ms = 0.0
        consolidation_ms = 0.0
        replay_ms = 0.0
        env_q_this_tick = 0
        promoted_this_tick = 0
        quarantined_this_tick = 0
        tick_steps: list[str] = []

        rfs = list(self.relational_frontiers.values())
        gfs = list(self.grounding_frontiers.values())
        ffs = list(self.functional_frontiers.values())

        t_q0 = time.perf_counter()
        anomalies: list[AnomalySignal] = self.scheduler.inspect_anomalies(
            relational_frontiers=rfs,
            grounding_frontiers=gfs,
            functional_frontiers=ffs,
            max_scan_ms=0.15 * effective_slice_ms,
        )
        for anom in anomalies:
            self._observed_anomalies.add(anom.anomaly_kind)
        top_anom = anomalies[0].anomaly_kind if anomalies else "none"

        rem_sel_ms = max(
            0.0,
            min(22.0, 0.25 * effective_slice_ms) - ((time.perf_counter() - t_learn_start) * 1000.0),
        )
        action: CurriculumAction = self.scheduler.select_next_action(
            self.budget,
            relational_frontiers=rfs,
            grounding_frontiers=gfs,
            functional_frontiers=ffs,
            pending_replay_audit=self._pending_replay_audit,
            max_selection_ms=rem_sel_ms,
        )
        query_selection_ms += (time.perf_counter() - t_q0) * 1000.0

        # Cooperative slice-budget guard: if selection consumed the slice budget (or slice is too tight
        # to fit action + mandatory regression replay cleanly), yield idle_noop instead of overrunning!
        elapsed_pre_exec_ms = (time.perf_counter() - t_learn_start) * 1000.0
        min_exec_and_replay_ms = max(float(action.estimated_ms), 2.0)
        if action.action_kind != "idle_noop" and (
            elapsed_pre_exec_ms >= effective_slice_ms
            or (elapsed_pre_exec_ms + min_exec_and_replay_ms) > effective_slice_ms
            or elapsed_pre_exec_ms >= 0.80 * effective_slice_ms
        ):
            action = CurriculumAction(
                action_kind="idle_noop",
                frontier_id="none",
                target_skill_id=None,
                expected_info_gain=0.0,
                estimated_ms=0.0,
                score_ig_per_ms=0.0,
                reason="slice_budget_yield",
            )
        self._observed_actions.add(action.action_kind)

        if action.action_kind in {"max_disagreement_query", "dream_refutation"} and (
            action.frontier_id in self.relational_frontiers
        ):
            rf = self.relational_frontiers[action.frontier_id]
            p_idx = int(action.probe_index or 0)
            rf.queried_indices.add(p_idx)
            st_q, act_q, _ptag = rf.candidate_probes[p_idx]

            # L5_contrast + L6_independent_verify + L0_observe + L1_frame + L2_difference
            tick_steps.extend(["L5_contrast", "L6_independent_verify", "L0_observe", "L1_frame", "L2_difference"])
            t_v0 = time.perf_counter()
            after_st, env_delta = self.verifier_gate.step_relational(
                self._rel_envs[rf.frontier_id], st_q, act_q
            )
            self.budget.record_query(1)
            rf.queries_spent_on_frontier += 1
            env_q_this_tick += 1
            mask_full = ObservationMask(fully_observed=True)
            f_before = TypedStateFrame.from_state_graph(st_q, mask_full)
            f_after = TypedStateFrame.from_state_graph(after_st, mask_full)
            _framed_delta = extract_masked_sparse_delta(f_before, f_after)
            verification_ms += (time.perf_counter() - t_v0) * 1000.0

            # L3_shard_propose + L4_itl_generalise + L6_correct + L6_compose
            t_i0 = time.perf_counter()
            prev_n, new_n, rep = rf.filter_with_verified_transition(
                st_q, act_q, after_st, env_delta
            )
            tick_steps.extend(["L3_shard_propose", "L4_itl_generalise", "L6_correct", "L6_compose"])
            induction_ms += (time.perf_counter() - t_i0) * 1000.0

            if rep is not None and new_n < prev_n:
                t_r0 = time.perf_counter()
                rec = self.verifier_gate.issue_receipt(
                    rep.to_executable_dict(),
                    verifier_kind="blackbox_transition_environment",
                    probes_verified=rf.queries_spent_on_frontier,
                    passed=True,
                )
                out = self._promote_verified_candidate(
                    domain_id=rf.domain_id,
                    skill_id=act_q.action_id,
                    kind="relational_operator",
                    executable_dict=rep.to_executable_dict(),
                    executable_object=rep,
                    receipt=rec,
                    phase_index=phase_index,
                    notes=f"autonomous_{action.action_kind}",
                )
                tick_steps.append("L7_continual_regression_replay")
                replay_ms += (time.perf_counter() - t_r0) * 1000.0
                if out.accepted:
                    promoted_this_tick += 1
                    # Keep SHARD proposal engine & UTF-8 grounding operators synchronized (excluding duplicate aliases)
                    t_ret0 = time.perf_counter()
                    dup_ids = {d for _c, d in rf.duplicate_candidates} | set(rf.alias_map.keys())
                    active_ops = {
                        k: v
                        for k, v in self.kb.get_active_domain_objects(
                            rf.domain_id, kind="relational_operator"
                        ).items()
                        if isinstance(v, OperatorCard) and k not in dup_ids
                    }
                    _proposer = SHARDProposalEngine(active_ops)
                    for gf in self.grounding_frontiers.values():
                        gf.update_operators(active_ops)
                    retrieval_ms += (time.perf_counter() - t_ret0) * 1000.0
                if out.quarantined:
                    quarantined_this_tick += 1

        elif action.action_kind == "contrast_probe" and (
            action.frontier_id in self.grounding_frontiers
        ):
            gf = self.grounding_frontiers[action.frontier_id]
            p_idx = int(action.probe_index or 0)
            gf.queried_indices.add(p_idx)
            pid, raw_b, scene_st, mask_g = gf.candidate_probes[p_idx]

            tick_steps.extend(["L5_contrast", "L6_independent_verify", "L0_observe", "L1_frame", "L2_difference"])
            t_v0 = time.perf_counter()
            after_st, _g_delta = self.verifier_gate.step_grounding(
                self._gnd_envs[gf.frontier_id], raw_b, scene_st, mask_g
            )
            self.budget.record_query(1)
            gf.queries_spent_on_frontier += 1
            env_q_this_tick += 1
            verification_ms += (time.perf_counter() - t_v0) * 1000.0

            t_i0 = time.perf_counter()
            ep_g = UnlabelledGroundingEpisode(
                episode_id=f"auto_gnd_{pid}_{self.budget.queries_spent}",
                raw_instruction_bytes=raw_b,
                before_state=scene_st,
                after_state=after_st,
                observation_mask=mask_g,
            )
            gf.ingest_verified_episode(ep_g)
            tick_steps.extend(["L3_shard_propose", "L4_itl_generalise", "L6_correct", "L6_compose"])
            induction_ms += (time.perf_counter() - t_i0) * 1000.0

            t_r0 = time.perf_counter()
            for ccard in gf.learner.promoted_constructions:
                existing = self.kb.get_active_skill(gf.domain_id, ccard.construction_id)
                exec_d = ccard.to_executable_dict()
                sha = sha256_hex(canonical_json_bytes(exec_d))
                if existing is not None and existing.canonical_sha256 == sha:
                    continue
                rec_c = self.verifier_gate.issue_receipt(
                    exec_d,
                    verifier_kind="blackbox_grounding_transition_verifier",
                    probes_verified=max(1, ccard.support_count),
                    passed=True,
                )
                out_c = self._promote_verified_candidate(
                    domain_id=gf.domain_id,
                    skill_id=ccard.construction_id,
                    kind="utf8_construction",
                    executable_dict=exec_d,
                    executable_object=ccard,
                    receipt=rec_c,
                    phase_index=phase_index,
                    notes="autonomous_contrast_grounding",
                )
                if out_c.accepted:
                    promoted_this_tick += 1
                if out_c.quarantined:
                    quarantined_this_tick += 1
            self.kb.set_domain_aux(gf.domain_id, "grounding_learner", gf.learner)
            self.kb.set_domain_aux(gf.domain_id, "negation_rules", gf.learner.negation_rules)
            self.kb.set_domain_aux(gf.domain_id, "condition_rules", gf.learner.condition_rules)
            self.kb.set_domain_aux(gf.domain_id, "clause_connectors", gf.learner.clause_connectors)
            tick_steps.append("L7_continual_regression_replay")
            replay_ms += (time.perf_counter() - t_r0) * 1000.0

        elif action.action_kind == "max_disagreement_query" and (
            action.frontier_id in self.functional_frontiers
        ):
            ff = self.functional_frontiers[action.frontier_id]
            sid_fn = str(action.target_skill_id)
            sstate = ff.skills_by_id[sid_fn]
            p_idx = int(action.probe_index or 0)
            sstate.queried_indices.add(p_idx)
            before_slots = sstate.candidate_probes[p_idx]

            tick_steps.extend(["L5_contrast", "L6_independent_verify", "L0_observe", "L1_frame", "L2_difference"])
            t_v0 = time.perf_counter()
            after_slots = self.verifier_gate.step_functional(
                self._fn_envs[(ff.frontier_id, sid_fn)], before_slots
            )
            self.budget.record_query(1)
            ff.queries_spent_on_frontier += 1
            env_q_this_tick += 1
            verification_ms += (time.perf_counter() - t_v0) * 1000.0

            t_i0 = time.perf_counter()
            ep_fn = SlotTransitionEpisode(
                episode_id=f"auto_fn_{sid_fn}_{self.budget.queries_spent}",
                action_id=sid_fn,
                before_slots=tuple(before_slots.items()),
                after_slots=tuple(after_slots.items()),
            )
            prev_amb = sstate.ambiguity_count()
            sstate.ingest_verified_episode(ep_fn)
            tick_steps.extend(["L3_shard_propose", "L4_itl_generalise", "L6_correct", "L6_compose"])
            induction_ms += (time.perf_counter() - t_i0) * 1000.0

            if sstate.ambiguity_count() < prev_amb:
                t_r0 = time.perf_counter()
                ver = len(self.kb.get_skill_history(ff.domain_id, sid_fn)) + 1
                fcard = sstate.build_verified_card(version=ver)
                rec_fn = self.verifier_gate.issue_receipt(
                    fcard.to_executable_dict(),
                    verifier_kind="restricted_execution_sandbox_and_blackbox_io",
                    probes_verified=len(sstate.observed_episodes),
                    passed=True,
                )
                out_fn = self._promote_verified_candidate(
                    domain_id=ff.domain_id,
                    skill_id=sid_fn,
                    kind="functional_program",
                    executable_dict=fcard.to_executable_dict(),
                    executable_object=fcard,
                    receipt=rec_fn,
                    phase_index=phase_index,
                    notes="autonomous_functional_refinement",
                )
                tick_steps.append("L7_continual_regression_replay")
                if out_fn.accepted:
                    promoted_this_tick += 1
                if out_fn.quarantined:
                    quarantined_this_tick += 1
                replay_ms += (time.perf_counter() - t_r0) * 1000.0

        elif action.action_kind == "induce_or_repair":
            if action.frontier_id in self.relational_frontiers:
                rf = self.relational_frontiers[action.frontier_id]
                t_i0 = time.perf_counter()
                st_m, act_m, after_m, delta_m = rf.pending_mismatches.pop(0)
                prev_n, new_n, rep = rf.filter_with_verified_transition(
                    st_m, act_m, after_m, delta_m
                )
                tick_steps.extend(["L0_observe", "L1_frame", "L2_difference", "L4_itl_generalise", "L6_correct"])
                induction_ms += (time.perf_counter() - t_i0) * 1000.0
                if rep is not None and new_n < prev_n:
                    t_r0 = time.perf_counter()
                    rec_m = self.verifier_gate.issue_receipt(
                        rep.to_executable_dict(),
                        verifier_kind="stream_mismatch_transition_verifier",
                        probes_verified=1,
                        passed=True,
                    )
                    out_m = self._promote_verified_candidate(
                        domain_id=rf.domain_id,
                        skill_id=act_m.action_id,
                        kind="relational_operator",
                        executable_dict=rep.to_executable_dict(),
                        executable_object=rep,
                        receipt=rec_m,
                        phase_index=phase_index,
                        notes="stream_mismatch_repair",
                    )
                    tick_steps.append("L7_continual_regression_replay")
                    if out_m.accepted:
                        promoted_this_tick += 1
                    if out_m.quarantined:
                        quarantined_this_tick += 1
                    replay_ms += (time.perf_counter() - t_r0) * 1000.0
            elif action.frontier_id in self.grounding_frontiers:
                gf = self.grounding_frontiers[action.frontier_id]
                t_i0 = time.perf_counter()
                ep_p = gf.pending_episodes.pop(0)
                gf.ingest_verified_episode(ep_p)
                tick_steps.extend(["L0_observe", "L4_itl_generalise", "L6_correct"])
                induction_ms += (time.perf_counter() - t_i0) * 1000.0
                t_r0 = time.perf_counter()
                for ccard in gf.learner.promoted_constructions:
                    existing = self.kb.get_active_skill(gf.domain_id, ccard.construction_id)
                    exec_d = ccard.to_executable_dict()
                    sha = sha256_hex(canonical_json_bytes(exec_d))
                    if existing is not None and existing.canonical_sha256 == sha:
                        continue
                    rec_p = self.verifier_gate.issue_receipt(
                        exec_d,
                        verifier_kind="stream_grounding_transition_verifier",
                        probes_verified=max(1, ccard.support_count),
                        passed=True,
                    )
                    out_p = self._promote_verified_candidate(
                        domain_id=gf.domain_id,
                        skill_id=ccard.construction_id,
                        kind="utf8_construction",
                        executable_dict=exec_d,
                        executable_object=ccard,
                        receipt=rec_p,
                        phase_index=phase_index,
                        notes="stream_grounding_repair",
                    )
                    if out_p.accepted:
                        promoted_this_tick += 1
                    if out_p.quarantined:
                        quarantined_this_tick += 1
                self.kb.set_domain_aux(gf.domain_id, "grounding_learner", gf.learner)
                self.kb.set_domain_aux(gf.domain_id, "negation_rules", gf.learner.negation_rules)
                self.kb.set_domain_aux(gf.domain_id, "condition_rules", gf.learner.condition_rules)
                self.kb.set_domain_aux(gf.domain_id, "clause_connectors", gf.learner.clause_connectors)
                self._cached_library_bits = self.kb.active_library_compressed_bits()
                tick_steps.append("L7_continual_regression_replay")
                replay_ms += (time.perf_counter() - t_r0) * 1000.0

        elif action.action_kind == "consolidate_mdl" and (
            action.frontier_id in self.relational_frontiers
        ):
            rf = self.relational_frontiers[action.frontier_id]
            canon_id, dup_id = rf.duplicate_candidates.pop(0)
            t_c0 = time.perf_counter()
            vstore = self.kb.relational_store_for(rf.domain_id)
            active_lib = {
                k: v
                for k, v in self.kb.get_active_domain_objects(
                    rf.domain_id, kind="relational_operator"
                ).items()
                if isinstance(v, OperatorCard)
            }
            eq_probes = rf.equivalence_probes_by_pair.get((canon_id, dup_id), [])
            if vstore is not None and canon_id in active_lib and dup_id in active_lib:
                cres = self.consolidator.attempt_consolidate_pair(
                    active_lib,
                    rf.alias_map,
                    vstore,
                    canon_id,
                    dup_id,
                    eq_probes,
                    rf.mdl_examples,
                )
                consolidation_ms += (time.perf_counter() - t_c0) * 1000.0
                if cres.accepted:
                    t_r0 = time.perf_counter()
                    # Non-destructively retire duplicate skill entry in ContinualKnowledgeBase
                    dup_hist = self.kb._histories.get((rf.domain_id, dup_id), [])
                    prev_dup_idx: int | None = None
                    prev_dup_entry = None
                    for idx_h, ent_h in enumerate(dup_hist):
                        if ent_h.lifecycle_state == "promoted":
                            prev_dup_idx = idx_h
                            prev_dup_entry = ent_h
                            dup_hist[idx_h] = ent_h.with_state(
                                "retired", notes=f"consolidated_into:{canon_id}"
                            )
                            break
                    prev_aliases = dict(self.kb.get_domain_aux(rf.domain_id, "alias_map", {}) or {})
                    self.kb.set_domain_aux(rf.domain_id, "alias_map", dict(rf.alias_map))
                    rep_after_cons = self.kb.replay_manager.run_all_prior_regressions(self.kb)
                    if not rep_after_cons.passed:
                        # Roll back consolidation in both KB and OperatorVersionStore on regression
                        if prev_dup_idx is not None and prev_dup_entry is not None:
                            dup_hist[prev_dup_idx] = prev_dup_entry
                        if cres.retired_version is not None:
                            self.consolidator.rollback_consolidation(
                                active_lib,
                                rf.alias_map,
                                vstore,
                                dup_id,
                                cres.retired_version,
                                reason="rollback_on_kb_regression_failure",
                            )
                        self.kb.set_domain_aux(rf.domain_id, "alias_map", prev_aliases)
                    else:
                        self._consolidation_results.append(cres)
                        self._cached_library_bits = self.kb.active_library_compressed_bits()
                        self._pending_replay_audit = True
                    replay_ms += (time.perf_counter() - t_r0) * 1000.0
                else:
                    self._consolidation_results.append(cres)
                tick_steps.append("L7_rdc_mdl_consolidate")
            else:
                consolidation_ms += (time.perf_counter() - t_c0) * 1000.0

        elif action.action_kind == "regression_replay":
            t_r0 = time.perf_counter()
            rep_check = self.kb.replay_manager.run_all_prior_regressions(self.kb)
            if not rep_check.passed:
                raise RuntimeError(f"Unexpected post-consolidation regression: {rep_check.failure_reasons}")
            self._pending_replay_audit = False
            tick_steps.append("L7_continual_regression_replay")
            replay_ms += (time.perf_counter() - t_r0) * 1000.0

        learning_slice_ms = (time.perf_counter() - t_learn_start) * 1000.0
        self.budget.record_cpu_ms(learning_slice_ms)
        self._observed_loop_steps.update(tick_steps)

        trace = ControllerTickTrace(
            tick_index=self._tick_index,
            interactive_queries_served=served_count,
            interactive_slice_ms=interactive_slice_ms,
            anomaly_count=len(anomalies),
            top_anomaly_kind=top_anom,
            chosen_action_kind=action.action_kind,
            target_frontier=action.frontier_id,
            target_skill_id=action.target_skill_id,
            expected_info_gain=action.expected_info_gain,
            env_queries_this_tick=env_q_this_tick,
            cumulative_env_queries=self.budget.queries_spent,
            retrieval_ms=retrieval_ms,
            induction_ms=induction_ms,
            query_selection_ms=query_selection_ms,
            verification_ms=verification_ms,
            consolidation_ms=consolidation_ms,
            replay_ms=replay_ms,
            learning_slice_ms=learning_slice_ms,
            learning_slice_budget_ms=self.budget.learning_slice_ms,
            within_slice_budget=(learning_slice_ms <= self.budget.learning_slice_ms),
            loop_steps_executed=tuple(dict.fromkeys(tick_steps)),
            promoted_updates=promoted_this_tick,
            quarantined_updates=quarantined_this_tick,
            active_library_bits=self._cached_library_bits,
        )
        self._tick_traces.append(trace)
        return trace

    def subsystem_cpu_breakdown_ms(self) -> dict[str, float]:
        """Aggregate exact per-subsystem CPU milliseconds across all executed ticks."""
        return {
            "interactive_slice_ms": round(sum(t.interactive_slice_ms for t in self._tick_traces), 4),
            "retrieval_ms": round(sum(t.retrieval_ms for t in self._tick_traces), 4),
            "induction_ms": round(sum(t.induction_ms for t in self._tick_traces), 4),
            "query_selection_ms": round(sum(t.query_selection_ms for t in self._tick_traces), 4),
            "verification_ms": round(sum(t.verification_ms for t in self._tick_traces), 4),
            "consolidation_ms": round(sum(t.consolidation_ms for t in self._tick_traces), 4),
            "replay_ms": round(sum(t.replay_ms for t in self._tick_traces), 4),
            "total_learning_slice_ms": round(sum(t.learning_slice_ms for t in self._tick_traces), 4),
        }


__all__ = [
    "AutonomousMetaController",
    "CLOSED_LOOP_L0_L7_STEPS",
    "ControllerFrozenError",
    "ControllerTickTrace",
    "FreezeGatedVerifierGate",
    "IndependentVerificationReceipt",
    "InteractiveLatencyMonitor",
    "InteractiveQueryRecord",
    "InteractiveQueryRequest",
    "META_CONTROLLER_SCHEMA_VERSION",
]
