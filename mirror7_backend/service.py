from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Mapping

from .actions import ActionGateway
from .persistence import CheckpointStore
from .runtime import BackendResult, BackendSession


class BackendService:
    """UI-independent orchestration facade for the Mirror 7 backend.

    The service owns sessions and composes the already-verified runtime,
    persistence, and action boundaries. It does not implement cognition.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        action_gateway: ActionGateway | None = None,
        max_sessions: int = 128,
    ):
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.engine_factory = engine_factory
        self.checkpoint_store = checkpoint_store
        self.action_gateway = action_gateway
        self.max_sessions = max_sessions
        self._sessions: dict[str, BackendSession] = {}
        self._lock = RLock()

    def create_session(self, session_id: str, *, max_history: int = 512) -> BackendSession:
        with self._lock:
            if session_id in self._sessions:
                raise ValueError("session already exists")
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("backend session capacity exhausted")
            engine = self.engine_factory() if self.engine_factory is not None else None
            session = BackendSession(session_id, engine=engine, max_history=max_history)
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> BackendSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    def step(
        self,
        session_id: str,
        observation: Any,
        *,
        goal: str | None = None,
        research_tasks: tuple[Any, ...] = (),
        views: tuple[Any, ...] = (),
    ) -> BackendResult:
        return self.get_session(session_id).step(
            observation, goal=goal, research_tasks=research_tasks, views=views
        )

    def save(self, session_id: str):
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        session = self.get_session(session_id)
        return self.checkpoint_store.save(session_id, session.snapshot())

    def restore(self, session_id: str, *, max_history: int = 512) -> BackendSession:
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        snapshot = self.checkpoint_store.load(session_id)
        with self._lock:
            if session_id in self._sessions:
                raise ValueError("session already exists")
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("backend session capacity exhausted")
            engine = self.engine_factory() if self.engine_factory is not None else None
            session = BackendSession(session_id, engine=engine, max_history=max_history)
            session.restore(snapshot)
            self._sessions[session_id] = session
            return session

    def execute_action(self, name: str, *args, **kwargs) -> Any:
        if self.action_gateway is None:
            raise RuntimeError("action gateway is not configured")
        return self.action_gateway.execute(name, *args, **kwargs)

    def close_session(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session: {session_id}")
            del self._sessions[session_id]

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "version": self.VERSION,
                "sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "session_ids": tuple(sorted(self._sessions)),
                "persistence": self.checkpoint_store is not None,
                "actions": self.action_gateway is not None,
            }
