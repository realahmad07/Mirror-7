from __future__ import annotations
from dataclasses import dataclass
from phase353_response_realization_bridge import ResponseRealizationRequest

@dataclass(frozen=True)
class ResponseRealizationPolicy:
    mode: str | None
    requires_plan: bool
    include_context: bool

def derive_response_realization_policy(request: ResponseRealizationRequest) -> ResponseRealizationPolicy:
    mode=request.output.output_mode
    requires_plan=bool(request.plan_actions) and mode is not None
    return ResponseRealizationPolicy(mode=mode, requires_plan=requires_plan, include_context=True)

def policy_is_non_destructive(policy: ResponseRealizationPolicy) -> bool:
    return policy.include_context is True
