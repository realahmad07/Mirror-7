from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

from phase352_semantic_output_execution import DesiredOutputPolicy, SemanticExecutionResult, desired_output_policy

@dataclass(frozen=True)
class ResponseRealizationRequest:
    output: DesiredOutputPolicy
    semantic_context: Mapping[str, Any]
    plan_actions: tuple[str, ...]
    observation: Any

def build_response_realization_request(
    request,
    execution: SemanticExecutionResult,
    observation: Any,
) -> ResponseRealizationRequest:
    return ResponseRealizationRequest(
        output=desired_output_policy(request),
        semantic_context=dict(request.context),
        plan_actions=tuple(execution.plan.actions) if execution.plan is not None else (),
        observation=observation,
    )

def has_realization_intent(result: ResponseRealizationRequest) -> bool:
    return result.output.desired_output is not None
