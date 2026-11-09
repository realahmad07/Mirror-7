"""Shared response contracts for Mirror Crystal inference engines."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class CrystalResponse:
    """Stable response object shared by in-memory and disk-backed engines."""
    answer: str
    confidence: float
    mode: str
    matched_rule_id: int | None
    latency_ms: float
    sandbox_verified: bool
    reasoning_trace: list[str]
    epistemic_tier: str = "high_confidence"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
