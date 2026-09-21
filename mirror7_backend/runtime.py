from __future__ import annotations

from dataclasses import dataclass, asdict
from threading import RLock
from time import time
from typing import Any, Callable, Mapping
import copy
import hashlib
import json

from phase353_response_realization_bridge import ResponseRealizationRequest
from phase354_response_realization_policy import derive_response_realization_policy
from phase355_backend_realization_contract import make_backend_realization_contract


@dataclass(frozen=True)
class BackendResult:
    session_id: str
    sequence: int
    observation: Any
    goal: str | None
    engine_result: Any
    state_digest: str
    realization_contract: Any | None = None


class BackendSession:
    """Thread-safe, UI-independent session boundary.

    The engine is injected so existing Mirror 7 mechanisms remain unchanged.
    A default engine is provided by the unified cognitive runtime when
    available. State is treated as opaque backend data and is never silently
    mutated by the session wrapper.
    """

    VERSION = 1

    def __init__(
        self,
        session_id: str,
        engine: Any | None = None,
        *,
        max_history: int = 512,
        clock: Callable[[], float] = time,
    ):
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id must be a non-empty string")
        if max_history < 1:
            raise ValueError("max_history must be positive")
        if engine is None:
            from phase140_unified_cognitive_runtime.mirror7_phase140 import (
                UnifiedCognitiveRuntime,
            )
            engine = UnifiedCognitiveRuntime()
        if not hasattr(engine, "step") or not callable(engine.step):
            raise TypeError("engine must expose callable step()")
        self.session_id = session_id
        self.engine = engine
        self.max_history = max_history
        self.clock = clock
        self.sequence = 0
        self.history: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}
        self._lock = RLock()

    def _build_realization_contract(self, engine_result: Any, observation: Any) -> Any | None:
        semantic_state = getattr(engine_result, "semantic_state", None)
        if semantic_state is None:
            return None
        from phase347_semantic_reasoning_bridge import build_reasoning_context
        from phase352_semantic_output_execution import DesiredOutputPolicy

        context = build_reasoning_context(semantic_state)
        if context is None:
            return None
        output = DesiredOutputPolicy(
            desired_output=semantic_state.desired_output,
            output_mode={
                "explanation": "explain",
                "comparison": "compare",
                "diagnosis_or_fix": "debug",
                "artifact_or_implementation": "create",
                "summary": "summarize",
                "translation": "translate",
                "computed_result": "calculate",
                "enumeration": "list",
                "analysis": "analyze",
                "prediction": "predict",
                "retrieval": "find",
            }.get(semantic_state.desired_output),
        )
        realization_request = ResponseRealizationRequest(
            output=output,
            semantic_context=context,
            plan_actions=(),
            observation=copy.deepcopy(observation),
        )
        policy = derive_response_realization_policy(realization_request)
        return make_backend_realization_contract(realization_request, policy)

    def step(
        self,
        observation: Any,
        *,
        goal: str | None = None,
        research_tasks: tuple[Any, ...] = (),
        views: tuple[Any, ...] = (),
    ) -> BackendResult:
        with self._lock:
            safe_observation = copy.deepcopy(observation)
            engine_result = self.engine.step(
                safe_observation,
                goal=goal,
                research_tasks=research_tasks,
                views=views,
            )
            realization_contract = self._build_realization_contract(
                engine_result, safe_observation
            )
            self.sequence += 1
            self.state = self._normalize_state(engine_result)
            event = {
                "sequence": self.sequence,
                "time": float(self.clock()),
                "observation": safe_observation,
                "goal": goal,
                "result": copy.deepcopy(engine_result),
                "state": copy.deepcopy(self.state),
                "realization_contract": copy.deepcopy(realization_contract),
            }
            self.history.append(event)
            if len(self.history) > self.max_history:
                del self.history[: len(self.history) - self.max_history]
            return BackendResult(
                self.session_id,
                self.sequence,
                safe_observation,
                goal,
                copy.deepcopy(engine_result),
                self.state_digest(),
                copy.deepcopy(realization_contract),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.VERSION,
                "session_id": self.session_id,
                "sequence": self.sequence,
                "state": copy.deepcopy(self.state),
                "history": copy.deepcopy(self.history),
            }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("version") != self.VERSION:
            raise ValueError("unsupported backend session version")
        if snapshot.get("session_id") != self.session_id:
            raise ValueError("snapshot belongs to a different session")
        sequence = snapshot.get("sequence")
        history = snapshot.get("history")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("invalid snapshot sequence")
        if not isinstance(history, list) or len(history) > self.max_history:
            raise ValueError("invalid snapshot history")
        with self._lock:
            self.sequence = sequence
            self.state = copy.deepcopy(dict(snapshot.get("state", {})))
            self.history = copy.deepcopy(history)

    def state_digest(self) -> str:
        payload = json.dumps(
            self.state, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _normalize_state(result: Any) -> dict[str, Any]:
        if hasattr(result, "__dataclass_fields__"):
            raw = asdict(result)
        elif isinstance(result, Mapping):
            raw = dict(result)
        else:
            raw = {"result": result}
        return raw
