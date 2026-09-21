from __future__ import annotations

from typing import Any

from phase362_response_generation_config import ResponseGenerationConfig
from phase363_response_model_loading_boundary import load_response_model
from phase364_response_generation_validation import validate_generated_response


def configure_backend_response_model(service: Any, loader: Any) -> Any:
    """Load an injected response model and register it with the backend."""
    loaded = load_response_model(loader)
    service.set_response_model(loaded.model)
    return loaded.model


def validate_backend_response(result: Any, config: ResponseGenerationConfig) -> Any:
    """Validate a backend result's generated response against its contract."""
    contract = getattr(result, "realization_contract", None)
    mode = getattr(contract, "mode", None)
    return validate_generated_response(
        getattr(result, "response", None),
        mode=mode,
        config=config,
    )
