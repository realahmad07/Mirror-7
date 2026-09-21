from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from phase355_backend_realization_contract import BackendRealizationContract


@dataclass(frozen=True)
class ResponseGenerationRequest:
    contract: BackendRealizationContract


def build_response_generation_request(
    contract: BackendRealizationContract | None,
) -> ResponseGenerationRequest | None:
    if contract is None:
        return None
    return ResponseGenerationRequest(contract=contract)


def generate_response(
    request: ResponseGenerationRequest | None,
    generator: Callable[[ResponseGenerationRequest], Any],
) -> Any:
    if request is None:
        return None
    if not callable(generator):
        raise TypeError("generator must be callable")
    return generator(request)
