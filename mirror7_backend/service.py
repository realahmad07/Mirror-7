from __future__ import annotations

from threading import Condition, RLock
from typing import Any, Callable, Mapping

from .actions import ActionGateway
from .observability import BackendMetrics
from .persistence import CheckpointStore
from .runtime import BackendResult, BackendSession


class BackendService:
    """UI-independent orchestration facade for the Mirror 7 backend."""

    VERSION = 3

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        action_gateway: ActionGateway | None = None,
        max_sessions: int = 128,
        metrics: BackendMetrics | None = None,
    ):
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.engine_factory = engine_factory
        self.checkpoint_store = checkpoint_store
        self.action_gateway = action_gateway
        self.max_sessions = max_sessions
        self.metrics = metrics or BackendMetrics()
        self._sessions: dict[str, BackendSession] = {}
        self._active: dict[str, int] = {}
        self._closing: set[str] = set()
        self._condition = Condition(RLock())

    def _register(self, session_id: str, session: BackendSession) -> BackendSession:
        if session_id in self._sessions:
            raise ValueError("session already exists")
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError("backend session capacity exhausted")
        self._sessions[session_id] = session
        self._active[session_id] = 0
        return session

    def create_session(self, session_id: str, *, max_history: int = 512) -> BackendSession:
        with self._condition:
            engine = self.engine_factory() if self.engine_factory is not None else None
            session = self._register(
                session_id,
                BackendSession(session_id, engine=engine, max_history=max_history),
            )
            self.metrics.increment("sessions_created")
            return session

    def get_session(self, session_id: str) -> BackendSession:
        with self._condition:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    def _begin_step(self, session_id: str) -> BackendSession:
        with self._condition:
            if session_id in self._closing:
                raise RuntimeError("session is closing")
            try:
                session = self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc
            self._active[session_id] += 1
            self.metrics.increment("steps_started")
            return session

    def _end_step(self, session_id: str, *, succeeded: bool) -> None:
        with self._condition:
            self._active[session_id] -= 1
            self.metrics.increment("steps_succeeded" if succeeded else "steps_failed")
            self._condition.notify_all()

    def step(
        self,
        session_id: str,
        observation: Any,
        *,
        goal: str | None = None,
        research_tasks: tuple[Any, ...] = (),
        views: tuple[Any, ...] = (),
    ) -> BackendResult:
        session = self._begin_step(session_id)
        succeeded = False
        try:
            result = session.step(
                observation, goal=goal, research_tasks=research_tasks, views=views
            )
            succeeded = True
            return result
        finally:
            self._end_step(session_id, succeeded=succeeded)

    def save(self, session_id: str):
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        session = self.get_session(session_id)
        return self.checkpoint_store.save(session_id, session.snapshot())

    def restore(self, session_id: str, *, max_history: int = 512) -> BackendSession:
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        snapshot = self.checkpoint_store.load(session_id)
        with self._condition:
            engine = self.engine_factory() if self.engine_factory is not None else None
            session = BackendSession(session_id, engine=engine, max_history=max_history)
            session.restore(snapshot)
            return self._register(session_id, session)

    def execute_action(self, name: str, *args, **kwargs) -> Any:
        if self.action_gateway is None:
            raise RuntimeError("action gateway is not configured")
        return self.action_gateway.execute(name, *args, **kwargs)

    def close_session(self, session_id: str) -> None:
        with self._condition:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session: {session_id}")
            self._closing.add(session_id)
            try:
                while self._active[session_id]:
                    self._condition.wait()
                del self._sessions[session_id]
                del self._active[session_id]
                self.metrics.increment("sessions_closed")
            finally:
                self._closing.discard(session_id)
                self._condition.notify_all()

    def list_sessions(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(sorted(self._sessions))

    def session_snapshot(self, session_id: str) -> Mapping[str, Any]:
        return self.get_session(session_id).snapshot()

    def has_checkpoint(self, session_id: str) -> bool:
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        return self.checkpoint_store.path_for(session_id).is_file()

    def delete_checkpoint(self, session_id: str) -> None:
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        self.checkpoint_store.delete(session_id)

    def status(self) -> Mapping[str, Any]:
        with self._condition:
            return {
                "version": self.VERSION,
                "sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "session_ids": tuple(sorted(self._sessions)),
                "active_steps": sum(self._active.values()),
                "closing_sessions": len(self._closing),
                "persistence": self.checkpoint_store is not None,
                "actions": self.action_gateway is not None,
                "metrics": self.metrics.snapshot(),
            }
