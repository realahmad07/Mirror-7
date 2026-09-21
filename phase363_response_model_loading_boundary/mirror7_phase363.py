from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ResponseModelLoader(Protocol):
    """Minimal loader boundary for a response model."""

    def load(self) -> Any:
        ...


@dataclass(frozen=True)
class LoadedResponseModel:
    model: Any

    def __post_init__(self) -> None:
        if not hasattr(self.model, "generate") or not callable(self.model.generate):
            raise TypeError("loaded model must expose callable generate(request)")


def load_response_model(loader: ResponseModelLoader) -> LoadedResponseModel:
    if not hasattr(loader, "load") or not callable(loader.load):
        raise TypeError("loader must expose callable load()")
    return LoadedResponseModel(loader.load())
