from dataclasses import dataclass
from mirror7_backend.runtime import BackendSession


@dataclass(frozen=True)
class FakeResult:
    value: int


class FakeEngine:
    def __init__(self):
        self.calls = []

    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        self.calls.append((observation, goal))
        return FakeResult(len(self.calls))


def test_backend_session_delegates_without_replacing_engine():
    engine = FakeEngine()
    session = BackendSession("s1", engine=engine)
    result = session.step({"x": 1}, goal="g")
    assert result.sequence == 1
    assert result.engine_result == FakeResult(1)
    assert engine.calls == [({"x": 1}, "g")]


def test_history_is_bounded_and_snapshot_is_json_safe():
    session = BackendSession("s2", engine=FakeEngine(), max_history=2)
    for i in range(3):
        session.step({"i": i})
    snap = session.snapshot()
    assert snap["sequence"] == 3
    assert len(snap["history"]) == 2
    assert session.state_digest()


def test_restore_rejects_other_session():
    session = BackendSession("s3", engine=FakeEngine())
    other = BackendSession("s4", engine=FakeEngine())
    snap = session.snapshot()
    try:
        other.restore(snap)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-session restore must fail")
