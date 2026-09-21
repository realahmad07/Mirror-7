from __future__ import annotations

from typing import Any, Mapping

from phase362_response_generation_config import ResponseGenerationConfig


def _response_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        for key in ("text", "response", "output"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return None


def validate_generated_response(
    response: Any,
    *,
    mode: str | None,
    config: ResponseGenerationConfig,
) -> Any:
    """Validate shape/size policy without rewriting the model output."""
    if response is None:
        raise ValueError("response generation returned None")
    if config.allowed_modes and mode not in config.allowed_modes:
        raise ValueError(f"response mode is not allowed: {mode!r}")

    text = _response_text(response)
    if text is not None:
        if config.reject_empty and not text.strip():
            raise ValueError("generated response is empty")
        if len(text) > config.max_output_chars:
            raise ValueError("generated response exceeds configured character limit")
    return response
