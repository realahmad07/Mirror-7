"""Stage S5 RDC Mechanism 3: Interventional Leave-One-Operator-Out Causal Blame Attribution.

STRICT ISOLATION & AMBIGUITY SAFEGUARD (PDF p. 8):
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Identifies faulty operators in multi-step (depth 2..6) failed trajectories via
  counterfactual leave-one-operator-out replay (either against an independent black-box
  ``TransitionEnvironment`` or against candidate historical/shadow operator versions).
- **Ambiguity Safeguard:** Whenever multiple operators tie in causal rescue OR no single
  operator intervention suffices to repair the trajectory (e.g., multi-operator fault),
  returns ``is_ambiguous=True`` and preserves ALL competing suspects instead of forcing
  a spurious single guess.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.itl.contrast_harness import TransitionEnvironment
from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.rdc.mdl_judge import execute_library_sequence
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    ObservationMask,
    StateGraph,
)

RDC_BLAME_SCHEMA_VERSION: str = "s5-rdc-blame-v1"
_MISSING_ATTR = object()


@dataclass(frozen=True)
class BlameAttributionReport:
    """Structured causal blame attribution for one multi-step trajectory."""

    method: str
    trajectory_failed: bool
    active_operators_in_trace: tuple[str, ...]
    blame_scores: dict[str, float]
    blame_distribution: dict[str, float]
    exact_repair_operators: tuple[str, ...]
    is_ambiguous: bool
    unique_culprit: str | None
    suspected_operators: tuple[str, ...]
    env_interventions_used: int = 0
    schema_version: str = RDC_BLAME_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "trajectory_failed": self.trajectory_failed,
            "active_operators_in_trace": list(self.active_operators_in_trace),
            "blame_scores": {k: round(v, 6) for k, v in sorted(self.blame_scores.items())},
            "blame_distribution": {
                k: round(v, 6) for k, v in sorted(self.blame_distribution.items())
            },
            "exact_repair_operators": list(self.exact_repair_operators),
            "is_ambiguous": self.is_ambiguous,
            "unique_culprit": self.unique_culprit,
            "suspected_operators": list(self.suspected_operators),
            "env_interventions_used": self.env_interventions_used,
        }


def _state_mismatch_distance(a: StateGraph, b: StateGraph) -> int:
    diff = len(a.facts ^ b.facts)
    am = a.entity_map()
    bm = b.entity_map()
    for eid in set(am) | set(bm):
        ea = am.get(eid)
        eb = bm.get(eid)
        if ea is None or eb is None:
            diff += 2
            continue
        if ea.entity_type != eb.entity_type:
            diff += 1
        aa = ea.attr_dict()
        ba = eb.attr_dict()
        for ak in set(aa) | set(ba):
            if aa.get(ak, _MISSING_ATTR) != ba.get(ak, _MISSING_ATTR):
                diff += 1
    return diff


def _replay_with_env_intervention(
    library: Mapping[str, OperatorCard],
    initial_state: StateGraph,
    actions: Sequence[ActionInvocation],
    intervene_action_id: str,
    env: TransitionEnvironment,
    *,
    mask: ObservationMask | None = None,
) -> tuple[StateGraph | None, bool, int]:
    """Replay ``actions`` replacing ONLY ``intervene_action_id`` with independent ``env.step``."""
    cur = initial_state.copy()
    env_calls = 0
    for act in actions:
        if act.action_id == intervene_action_id:
            cur, _delta = env.step(cur, act, mask=mask)
            env_calls += 1
        else:
            card = library.get(act.action_id)
            if card is None:
                return (None, True, env_calls)
            res = card.apply(cur, act.role_dict(), mask=mask)
            if res.abstained or res.after_state is None:
                return (None, True, env_calls)
            cur = res.after_state
    return (cur, False, env_calls)


def _finalize_blame_report(
    method: str,
    active_ops: tuple[str, ...],
    blame_scores: dict[str, float],
    exact_repair_ops: tuple[str, ...],
    env_calls: int,
) -> BlameAttributionReport:
    if len(exact_repair_ops) == 1:
        culprit = exact_repair_ops[0]
        dist = {op: (1.0 if op == culprit else 0.0) for op in active_ops}
        return BlameAttributionReport(
            method=method,
            trajectory_failed=True,
            active_operators_in_trace=active_ops,
            blame_scores=blame_scores,
            blame_distribution=dist,
            exact_repair_operators=exact_repair_ops,
            is_ambiguous=False,
            unique_culprit=culprit,
            suspected_operators=(culprit,),
            env_interventions_used=env_calls,
        )

    if len(exact_repair_ops) > 1:
        suspects = tuple(sorted(exact_repair_ops))
        w = 1.0 / float(len(suspects))
        dist = {op: (w if op in suspects else 0.0) for op in active_ops}
        return BlameAttributionReport(
            method=method,
            trajectory_failed=True,
            active_operators_in_trace=active_ops,
            blame_scores=blame_scores,
            blame_distribution=dist,
            exact_repair_operators=suspects,
            is_ambiguous=True,
            unique_culprit=None,
            suspected_operators=suspects,
            env_interventions_used=env_calls,
        )

    # Zero single-operator exact repairs -> genuinely multi-fault / ambiguous! Never force 1 guess!
    positive_ops = sorted(
        (op for op in active_ops if blame_scores.get(op, 0.0) > 1e-9),
        key=lambda op: (-blame_scores[op], op),
    )
    if len(positive_ops) >= 2:
        suspects = tuple(positive_ops)
        tot_pos = sum(blame_scores[op] for op in positive_ops)
        dist = {
            op: (blame_scores[op] / tot_pos if op in positive_ops else 0.0)
            for op in active_ops
        }
    else:
        # When 0 or 1 operator shows direct single-op improvement, an upstream fault may
        # mask a downstream fault (or faults cancel). Keep all active operators in the
        # suspect set so masked culprits are never prematurely discarded.
        suspects = tuple(
            sorted(active_ops, key=lambda op: (-blame_scores.get(op, 0.0), op))
        )
        raw_weights = {op: blame_scores.get(op, 0.0) + 0.1 for op in active_ops}
        tot_w = sum(raw_weights.values())
        dist = {op: raw_weights[op] / float(tot_w) for op in active_ops} if tot_w > 0 else {}

    return BlameAttributionReport(
        method=method,
        trajectory_failed=True,
        active_operators_in_trace=active_ops,
        blame_scores=blame_scores,
        blame_distribution=dist,
        exact_repair_operators=(),
        is_ambiguous=True,
        unique_culprit=None,
        suspected_operators=suspects,
        env_interventions_used=env_calls,
    )


class InterventionalBlameEngine:
    """Mechanism 3: Leave-One-Operator-Out Causal Blame Attribution with Ambiguity Honesty."""

    def attribute_blame_via_env(
        self,
        library: Mapping[str, OperatorCard],
        initial_state: StateGraph,
        actions: Sequence[ActionInvocation],
        observed_final_state: StateGraph,
        env: TransitionEnvironment,
        *,
        mask: ObservationMask | None = None,
    ) -> BlameAttributionReport:
        """Compute leave-one-operator-out causal blame via black-box counterfactual replay."""
        active_ops = tuple(sorted({act.action_id for act in actions}))
        pred_final, abstained, _inters = execute_library_sequence(
            library, initial_state, actions, mask=mask
        )
        if (not abstained) and pred_final == observed_final_state:
            return BlameAttributionReport(
                method="interventional_blame",
                trajectory_failed=False,
                active_operators_in_trace=active_ops,
                blame_scores={op: 0.0 for op in active_ops},
                blame_distribution={op: 0.0 for op in active_ops},
                exact_repair_operators=(),
                is_ambiguous=False,
                unique_culprit=None,
                suspected_operators=(),
                env_interventions_used=0,
            )

        base_err = (
            _state_mismatch_distance(initial_state, observed_final_state) + 10
            if (abstained or pred_final is None)
            else max(1, _state_mismatch_distance(pred_final, observed_final_state))
        )

        blame_scores: dict[str, float] = {}
        exact_repairs: list[str] = []
        total_env_calls = 0

        for op_id in active_ops:
            cf_final, cf_abst, calls = _replay_with_env_intervention(
                library, initial_state, actions, op_id, env, mask=mask
            )
            total_env_calls += calls
            if (not cf_abst) and cf_final == observed_final_state:
                exact_repairs.append(op_id)
                blame_scores[op_id] = 2.0
            elif cf_abst or cf_final is None:
                blame_scores[op_id] = 0.0
            else:
                cf_err = _state_mismatch_distance(cf_final, observed_final_state)
                blame_scores[op_id] = max(0.0, (base_err - cf_err) / float(base_err))

        return _finalize_blame_report(
            "interventional_blame",
            active_ops,
            blame_scores,
            tuple(sorted(exact_repairs)),
            total_env_calls,
        )

    def attribute_blame_via_alternatives(
        self,
        library: Mapping[str, OperatorCard],
        initial_state: StateGraph,
        actions: Sequence[ActionInvocation],
        observed_final_state: StateGraph,
        alternatives_by_action: Mapping[str, Sequence[OperatorCard]],
        *,
        mask: ObservationMask | None = None,
    ) -> BlameAttributionReport:
        """Offline leave-one-operator-out blame substituting historical/shadow candidate cards."""
        active_ops = tuple(sorted({act.action_id for act in actions}))
        pred_final, abstained, _inters = execute_library_sequence(
            library, initial_state, actions, mask=mask
        )
        if (not abstained) and pred_final == observed_final_state:
            return BlameAttributionReport(
                method="interventional_blame_offline",
                trajectory_failed=False,
                active_operators_in_trace=active_ops,
                blame_scores={op: 0.0 for op in active_ops},
                blame_distribution={op: 0.0 for op in active_ops},
                exact_repair_operators=(),
                is_ambiguous=False,
                unique_culprit=None,
                suspected_operators=(),
                env_interventions_used=0,
            )

        base_err = (
            _state_mismatch_distance(initial_state, observed_final_state) + 10
            if (abstained or pred_final is None)
            else max(1, _state_mismatch_distance(pred_final, observed_final_state))
        )

        blame_scores: dict[str, float] = {}
        exact_repairs: list[str] = []

        for op_id in active_ops:
            best_score = 0.0
            repaired = False
            for alt_card in alternatives_by_action.get(op_id, ()):
                trial_lib = dict(library)
                trial_lib[op_id] = alt_card
                cf_final, cf_abst, _ = execute_library_sequence(
                    trial_lib, initial_state, actions, mask=mask
                )
                if (not cf_abst) and cf_final == observed_final_state:
                    repaired = True
                    best_score = 2.0
                    break
                if (not cf_abst) and cf_final is not None:
                    cf_err = _state_mismatch_distance(cf_final, observed_final_state)
                    sc = max(0.0, (base_err - cf_err) / float(base_err))
                    if sc > best_score:
                        best_score = sc
            if repaired:
                exact_repairs.append(op_id)
            blame_scores[op_id] = best_score

        return _finalize_blame_report(
            "interventional_blame_offline",
            active_ops,
            blame_scores,
            tuple(sorted(exact_repairs)),
            0,
        )

    def attribute_blame_random(
        self,
        actions: Sequence[ActionInvocation],
        *,
        seed: int = 0,
    ) -> BlameAttributionReport:
        """Control baseline: assigns full blame to a random active operator from the trace."""
        active_ops = tuple(sorted({act.action_id for act in actions}))
        if not active_ops:
            return BlameAttributionReport(
                method="random_blame",
                trajectory_failed=True,
                active_operators_in_trace=(),
                blame_scores={},
                blame_distribution={},
                exact_repair_operators=(),
                is_ambiguous=True,
                unique_culprit=None,
                suspected_operators=(),
            )
        chosen = random.Random(seed).choice(active_ops)
        scores = {op: (1.0 if op == chosen else 0.0) for op in active_ops}
        return BlameAttributionReport(
            method="random_blame",
            trajectory_failed=True,
            active_operators_in_trace=active_ops,
            blame_scores=scores,
            blame_distribution=dict(scores),
            exact_repair_operators=(),
            is_ambiguous=False,
            unique_culprit=chosen,
            suspected_operators=(chosen,),
        )

    def attribute_blame_uniform(
        self,
        actions: Sequence[ActionInvocation],
    ) -> BlameAttributionReport:
        """Control baseline: assigns uniform 1/k blame to every active operator in the trace."""
        active_ops = tuple(sorted({act.action_id for act in actions}))
        k = len(active_ops)
        if k == 0:
            return BlameAttributionReport(
                method="uniform_blame",
                trajectory_failed=True,
                active_operators_in_trace=(),
                blame_scores={},
                blame_distribution={},
                exact_repair_operators=(),
                is_ambiguous=True,
                unique_culprit=None,
                suspected_operators=(),
            )
        w = 1.0 / float(k)
        scores = {op: w for op in active_ops}
        return BlameAttributionReport(
            method="uniform_blame",
            trajectory_failed=True,
            active_operators_in_trace=active_ops,
            blame_scores=scores,
            blame_distribution=dict(scores),
            exact_repair_operators=(),
            is_ambiguous=(k > 1),
            unique_culprit=(active_ops[0] if k == 1 else None),
            suspected_operators=active_ops,
        )


def score_blame_attribution(
    report: BlameAttributionReport,
    true_faulty_operator: str,
) -> dict[str, float]:
    """Score one single-fault attribution report: exact top-1, MRR, and Brier score."""
    top1_exact = 1.0 if (not report.is_ambiguous and report.unique_culprit == true_faulty_operator) else 0.0
    # Mean Reciprocal Rank (for ties, use conservative expected rank = (start + end) / 2)
    ordered = sorted(
        report.blame_distribution.items(), key=lambda kv: (-kv[1], kv[0])
    )
    target_p = report.blame_distribution.get(true_faulty_operator, 0.0)
    higher = sum(1 for _op, p in ordered if p > target_p + 1e-12)
    tied = sum(1 for _op, p in ordered if abs(p - target_p) <= 1e-12)
    avg_rank = higher + (tied + 1) / 2.0 if tied > 0 else float(len(ordered) + 1)
    mrr = 1.0 / avg_rank

    brier = sum(
        (p - (1.0 if op == true_faulty_operator else 0.0)) ** 2
        for op, p in report.blame_distribution.items()
    ) / float(max(1, len(report.blame_distribution)))
    return {
        "top1_accuracy": top1_exact,
        "mrr": round(mrr, 6),
        "brier_score": round(brier, 6),
    }
