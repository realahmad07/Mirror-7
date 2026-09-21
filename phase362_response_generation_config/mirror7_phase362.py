from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ResponseGenerationConfig:
    """Model-agnostic limits and policy for backend response realization."""

    max_output_chars: int = 8192
    reject_empty: bool = True
    allowed_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        modes = tuple(str(mode) for mode in self.allowed_modes)
        if len(set(modes)) != len(modes):
            raise ValueError("allowed_modes must not contain duplicates")
        if any(not mode for mode in modes):
            raise ValueError("allowed_modes must contain non-empty strings")
        object.__setattr__(self, "allowed_modes", modes)


def make_generation_config(
    *,
    max_output_chars: int = 8192,
    reject_empty: bool = True,
    allowed_modes: Iterable[str] = (),
) -> ResponseGenerationConfig:
    return ResponseGenerationConfig(
        max_output_chars=max_output_chars,
        reject_empty=reject_empty,
        allowed_modes=tuple(allowed_modes),
    )
