from __future__ import annotations

import re
from dataclasses import dataclass

from phase348_semantic_planning_bridge import SemanticPlanningRequest, execute_planning_request as _execute


@dataclass(frozen=True)
class PlanningConstraints:
    """Action-selection limits explicitly induced from semantic constraints."""

    max_steps: int | None = None


def planning_constraints(request: SemanticPlanningRequest) -> PlanningConstraints:
    for item in request.context.get("evidence", ()):
        match = re.fullmatch(r"max_steps:(\d+)", str(item))
        if match:
            return PlanningConstraints(max_steps=int(match.group(1)))
    return PlanningConstraints()


def execute_constrained_planning(
    request: SemanticPlanningRequest,
    planner,
    *,
    max_depth: int = 12,
    max_nodes: int = 1000,
):
    """Apply only explicit semantic search limits; never alter world state or invent actions."""
    constraints = planning_constraints(request)
    effective_depth = min(max_depth, constraints.max_steps) if constraints.max_steps is not None else max_depth
    return _execute(request, planner, max_depth=effective_depth, max_nodes=max_nodes)
