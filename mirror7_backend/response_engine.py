from __future__ import annotations

from typing import Any, Callable

from phase357_response_generation_interface import (
    build_response_generation_request,
    generate_response,
)
from phase359_response_generation_adapter import make_response_generator
from phase362_response_generation_config import ResponseGenerationConfig
from phase364_response_generation_validation import validate_generated_response


class UnifiedResponseEngine:
    """Single validated boundary from a realization contract to user-facing output.

    The cognitive runtime remains responsible for reasoning and producing the
    realization contract. This class owns model adaptation, request creation,
    generation, and output-policy validation. It deliberately does not invent
    a fallback answer when no contract or model is available.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        config: ResponseGenerationConfig | None = None,
    ) -> None:
        self.config = config or ResponseGenerationConfig()
        self._generator: Callable[[Any], Any] | None = None
        self.set_model(model)

    @property
    def enabled(self) -> bool:
        return self._generator is not None

    def set_model(self, model: Any | None) -> None:
        if model is None:
            self._generator = None
        elif callable(model):
            self._generator = model
        else:
            self._generator = make_response_generator(model)

    def set_config(self, config: ResponseGenerationConfig) -> None:
        if not isinstance(config, ResponseGenerationConfig):
            raise TypeError("config must be a ResponseGenerationConfig")
        self.config = config

    def respond(self, realization_contract: Any | None) -> Any | None:
        if realization_contract is None or self._generator is None:
            return None

        request = build_response_generation_request(realization_contract)
        if request is None:
            return None

        generated = generate_response(request, self._generator)
        return validate_generated_response(
            generated,
            mode=getattr(realization_contract, "mode", None),
            config=self.config,
        )
