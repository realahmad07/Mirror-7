from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from phase34_reasoning_planning.mirror7_phase34 import Goal, GoalPlanner, Plan
from phase347_semantic_reasoning_bridge import build_reasoning_context


@dataclass(frozen=True)
class SemanticPlanningRequest:
    """A planning request derived from semantic task state plus explicit world state."""

    context: Mapping[str, Any]
    world_state: Mapping[str, Any]
    goal: Goal


@dataclass(frozen=True)
class PlanningConstraints:
    """Action-selection limits explicitly supported by semantic task text."""

    max_steps: int | None = None


def _extract_max_steps(context: Mapping[str, Any]) -> int | None:
    if "upper_bound" not in context.get("constraints", ()):
        return None
    evidence = " ".join(str(item) for item in context.get("evidence", ()))
    match = re.search(r"\b(?:under|below|less than|at most)\s+(\d+)\s+(?:steps?|actions?)\b", evidence, re.I)
    return int(match.group(1)) if match else None


def build_planning_request(
    semantic_state,
    world_state: Mapping[str, Any],
    goal_requirements: Mapping[str, Any],
) -> SemanticPlanningRequest | None:
    context = build_reasoning_context(semantic_state)
    if context is None:
        return None
    if not isinstance(world_state, Mapping):
        raise TypeError("world_state must be a mapping")
    return SemanticPlanningRequest(
        context=context,
        world_state=dict(world_state),
        goal=Goal.from_mapping(goal_requirements),
    )


def planning_constraints(request: SemanticPlanningRequest) -> PlanningConstraints:
    return PlanningConstraints(max_steps=_extract_max_steps(request.context))


def execute_planning_request(
    request: SemanticPlanningRequest,
    planner: GoalPlanner,
    *,
    max_depth: int = 12,
    max_nodes: int = 1000,
) -> Plan | None:
    """Plan against explicit world state; semantic constraints may only tighten search limits."""
    constraints = planning_constraints(request)
    effective_depth = max_depth
    if constraints.max_steps is not None:
        effective_depth = min(effective_depth, constraints.max_steps)
    return planner.plan(
        request.world_state,
        request.goal,
        max_depth=effective_depth,
        max_nodes=max_nodes,
    )
