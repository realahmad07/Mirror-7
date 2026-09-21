from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

from phase354_response_realization_policy import ResponseRealizationPolicy

@dataclass(frozen=True)
class BackendRealizationContract:
    mode: str | None
    context: Mapping[str, Any]
    actions: tuple[str, ...]
    observation: Any

def make_backend_realization_contract(realization_request, policy: ResponseRealizationPolicy) -> BackendRealizationContract:
    return BackendRealizationContract(
        mode=policy.mode,
        context=dict(realization_request.semantic_context),
        actions=tuple(realization_request.plan_actions),
        observation=realization_request.observation,
    )
