"""Mirror 7 backend runtime primitives.

UI-agnostic backend boundary for sessions, persistence, actions, and verified
execution. The backend delegates cognition to an injected engine; it does not
replace or weaken the research mechanisms.
"""
from .runtime import BackendSession, BackendResult
from .persistence import CheckpointStore, CheckpointError
from .actions import ActionGateway, ActionPolicy, ActionDenied

__all__ = [
    "BackendSession", "BackendResult",
    "CheckpointStore", "CheckpointError",
    "ActionGateway", "ActionPolicy", "ActionDenied",
]
