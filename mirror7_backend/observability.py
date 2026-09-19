from __future__ import annotations

from threading import Lock
from typing import Any


class BackendMetrics:
    """Small dependency-free counters for backend operational visibility."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters = {
            "sessions_created": 0,
            "sessions_closed": 0,
            "steps_started": 0,
            "steps_succeeded": 0,
            "steps_failed": 0,
        }

    def increment(self, name: str) -> None:
        with self._lock:
            if name not in self._counters:
                raise KeyError(f"unknown metric: {name}")
            self._counters[name] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def reset(self) -> None:
        with self._lock:
            for name in self._counters:
                self._counters[name] = 0
