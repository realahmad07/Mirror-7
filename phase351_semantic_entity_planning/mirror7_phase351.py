from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from phase348_semantic_planning_bridge import SemanticPlanningRequest
from phase349_semantic_constraint_planning import execute_constrained_planning


@dataclass(frozen=True)
class EntityBinding:
    entity: str
    world_key: str
    world_value: Any


def resolve_entity_binding(request: SemanticPlanningRequest) -> EntityBinding | None:
    """Resolve an induced entity only against facts explicitly present in world state."""
    entities = tuple(str(x) for x in request.context.get("entities", ()))
    if not entities:
        return None
    for entity in entities:
        for key, value in request.world_state.items():
            if isinstance(value, str) and value.casefold() == entity.casefold():
                return EntityBinding(entity, str(key), value)
    return None


def entity_target_is_consistent(request: SemanticPlanningRequest) -> bool:
    """Reject only explicit entity/goal contradictions; never fabricate missing facts."""
    binding = resolve_entity_binding(request)
    if binding is None:
        return True
    goal_map = dict(request.goal.requirements)
    if binding.world_key in goal_map:
        return goal_map[binding.world_key] == binding.world_value
    return True


def execute_entity_aware_planning(
    request: SemanticPlanningRequest,
    planner,
    *,
    max_depth: int = 12,
    max_nodes: int = 1000,
):
    """Plan only when an explicit entity/world-state binding does not contradict the goal."""
    if not entity_target_is_consistent(request):
        return None
    return execute_constrained_planning(
        request, planner, max_depth=max_depth, max_nodes=max_nodes
    )
