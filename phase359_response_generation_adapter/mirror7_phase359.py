from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from phase357_response_generation_interface import (
    ResponseGenerationRequest,
    generate_response,
)


class ResponseModelAdapter(Protocol):
    """Minimal model boundary for response realization."""

    def generate(self, request: ResponseGenerationRequest) -> Any:
        ...


@dataclass(frozen=True)
class ResponseGenerationAdapter:
    """Adapts a model object with generate(request) to the backend callable contract."""

    model: ResponseModelAdapter

    def __post_init__(self) -> None:
        if not hasattr(self.model, "generate") or not callable(self.model.generate):
            raise TypeError("model must expose callable generate(request)")

    def __call__(self, request: ResponseGenerationRequest) -> Any:
        return self.model.generate(request)


def make_response_generator(
    model: ResponseModelAdapter,
) -> Callable[[ResponseGenerationRequest], Any]:
    return ResponseGenerationAdapter(model)


def run_adapter(
    request: ResponseGenerationRequest | None,
    adapter: Callable[[ResponseGenerationRequest], Any],
) -> Any:
    return generate_response(request, adapter)
