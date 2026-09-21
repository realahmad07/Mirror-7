from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from phase348_semantic_planning_bridge import SemanticPlanningRequest
from phase351_semantic_entity_planning import execute_entity_aware_planning


@dataclass(frozen=True)
class DesiredOutputPolicy:
    desired_output: str | None = None
    output_mode: str | None = None


@dataclass(frozen=True)
class SemanticExecutionResult:
    plan: Any
    output: DesiredOutputPolicy


_OUTPUT_MODES = {
    "explanation": "explain",
    "comparison": "compare",
    "diagnosis_or_fix": "debug",
    "artifact_or_implementation": "create",
    "summary": "summarize",
    "translation": "translate",
    "computed_result": "calculate",
    "enumeration": "list",
    "analysis": "analyze",
    "prediction": "predict",
    "retrieval": "find",
}


def desired_output_policy(request: SemanticPlanningRequest) -> DesiredOutputPolicy:
    value = request.context.get("desired_output")
    desired = str(value) if value is not None else None
    return DesiredOutputPolicy(desired_output=desired, output_mode=_OUTPUT_MODES.get(desired))


def execute_output_aware_planning(
    request: SemanticPlanningRequest,
    planner,
    *,
    max_depth: int = 12,
    max_nodes: int = 1000,
) -> SemanticExecutionResult:
    """Plan using existing world/model state and preserve semantic output intent separately."""
    plan = execute_entity_aware_planning(
        request, planner, max_depth=max_depth, max_nodes=max_nodes
    )
    return SemanticExecutionResult(plan=plan, output=desired_output_policy(request))


def output_intent_is_consistent(result: SemanticExecutionResult) -> bool:
    policy = result.output
    return policy.desired_output is None or policy.output_mode is not None
