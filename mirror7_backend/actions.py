from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable
import time


class ActionDenied(PermissionError):
    pass


@dataclass(frozen=True)
class ActionPolicy:
    allowed: frozenset[str]
    max_calls: int = 32
    timeout_seconds: float = 5.0

    def __post_init__(self):
        if self.max_calls < 1:
            raise ValueError("max_calls must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class ActionGateway:
    """Explicit action boundary.

    The gateway never executes an action unless its name is allow-listed.
    Call budgets are per gateway instance. Actual wall-clock interruption of
    arbitrary Python callables is intentionally not faked; callers requiring
    hard isolation must use the production Docker sandbox.
    """

    def __init__(
        self,
        actions: dict[str, Callable[..., Any]],
        policy: ActionPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        if any(not isinstance(k, str) or not callable(v) for k, v in actions.items()):
            raise TypeError("actions must map string names to callables")
        unknown = set(actions) - set(policy.allowed)
        if unknown:
            raise ValueError(f"actions not present in policy: {sorted(unknown)}")
        self.actions = dict(actions)
        self.policy = policy
        self.clock = clock
        self.calls = 0
        self._lock = RLock()

    def execute(self, name: str, *args, **kwargs) -> Any:
        with self._lock:
            if name not in self.policy.allowed:
                raise ActionDenied(f"action not allowed: {name}")
            if name not in self.actions:
                raise ActionDenied(f"action unavailable: {name}")
            if self.calls >= self.policy.max_calls:
                raise ActionDenied("action call budget exhausted")
            self.calls += 1
        started = self.clock()
        result = self.actions[name](*args, **kwargs)
        elapsed = self.clock() - started
        if elapsed > self.policy.timeout_seconds:
            raise TimeoutError(
                f"action exceeded backend budget ({elapsed:.3f}s > "
                f"{self.policy.timeout_seconds:.3f}s)"
            )
        return result
