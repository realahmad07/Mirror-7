from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from phase34_reasoning_planning import Goal, GoalPlanner, Plan
from phase347_semantic_reasoning_bridge import build_reasoning_context


@dataclass(frozen=True)
class SemanticPlanningRequest:
    """A planning request derived from semantic task state plus explicit world state."""

    context: Mapping[str, Any]
    world_state: Mapping[str, Any]
    goal: Goal


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


def execute_planning_request(
    request: SemanticPlanningRequest,
    planner: GoalPlanner,
    *,
    max_depth: int = 12,
    max_nodes: int = 1000,
) -> Plan | None:
    """Plan against explicit world state; semantic context remains task metadata."""
    return planner.plan(
        request.world_state,
        request.goal,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
