from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from phase348_semantic_planning_bridge import SemanticPlanningRequest
from phase349_semantic_constraint_planning import execute_constrained_planning


@dataclass(frozen=True)
class OperationPlanningPolicy:
    operation: str | None = None
    preferred_actions: tuple[str, ...] = ()


_OPERATION_ACTION_HINTS: dict[str, tuple[str, ...]] = {
    "create": ("create", "make", "build", "set"),
    "debug": ("debug", "fix", "repair"),
    "analyze": ("analyze", "inspect", "measure"),
    "calculate": ("calculate", "compute"),
    "predict": ("predict", "forecast"),
    "find": ("find", "search"),
}


def planning_policy(request: SemanticPlanningRequest) -> OperationPlanningPolicy:
    operations = tuple(str(x) for x in request.context.get("operations", ()))
    operation = operations[0] if operations else None
    return OperationPlanningPolicy(operation, _OPERATION_ACTION_HINTS.get(operation, ()))


def execute_operation_aware_planning(
    request: SemanticPlanningRequest,
    planner,
    *,
    max_depth: int = 12,
    max_nodes: int = 1000,
):
    """Use semantic operation only as a deterministic action-ordering hint.

    Candidate actions remain exclusively those learned by the transition model.
    """
    policy = planning_policy(request)
    if not policy.preferred_actions:
        return execute_constrained_planning(
            request, planner, max_depth=max_depth, max_nodes=max_nodes
        )

    original_actions = planner.model.actions

    def ordered_actions():
        actions = original_actions()
        preferred = tuple(a for a in actions if a in policy.preferred_actions)
        remainder = tuple(a for a in actions if a not in policy.preferred_actions)
        return preferred + remainder

    planner.model.actions = ordered_actions
    try:
        return execute_constrained_planning(
            request, planner, max_depth=max_depth, max_nodes=max_nodes
        )
    finally:
        planner.model.actions = original_actions
