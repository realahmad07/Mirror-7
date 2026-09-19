from dataclasses import dataclass
from typing import Any, List, Mapping

@dataclass(frozen=True)
class PromotionRecord:
    version: int
    accepted: bool
    reason: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    train_gain: float
    held_out_score: float
    regression_ok: bool

class PromotionRegistry:
    """Atomically promotes verified variants and records enough state for rollback."""
    def __init__(self, initial: Mapping[str, Any]):
        self._current = dict(initial)
        self._history: List[PromotionRecord] = []
        self._version = 0

    @property
    def current(self) -> dict:
        return dict(self._current)

    @property
    def history(self) -> tuple[PromotionRecord, ...]:
        return tuple(self._history)

    def promote(self, candidate: Mapping[str, Any], *, train_gain: float, held_out_score: float, regression_ok: bool) -> PromotionRecord:
        if not regression_ok or train_gain <= 0.0 or held_out_score <= 0.0:
            return PromotionRecord(self._version, False, "rejected by promotion gate", self.current, self.current, float(train_gain), float(held_out_score), bool(regression_ok))
        before = self.current
        self._version += 1
        self._current = dict(candidate)
        record = PromotionRecord(self._version, True, "promoted", before, self.current, float(train_gain), float(held_out_score), bool(regression_ok))
        self._history.append(record)
        return record

    def rollback(self, version: int | None = None) -> PromotionRecord:
        if not self._history:
            return PromotionRecord(self._version, False, "no promoted version available", self.current, self.current, 0.0, 0.0, False)
        target = self._history[-1]
        if version is not None:
            matches = [r for r in self._history if r.version == int(version)]
            if not matches:
                return PromotionRecord(self._version, False, "unknown version", self.current, self.current, 0.0, 0.0, False)
            target = matches[0]
        before = self.current
        self._current = dict(target.before)
        self._version += 1
        record = PromotionRecord(self._version, True, f"rollback to before version {target.version}", before, self.current, 0.0, 0.0, True)
        self._history.append(record)
        return record
