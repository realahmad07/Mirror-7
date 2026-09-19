from mirror7_backend.runtime import BackendSession
from mirror7_backend.persistence import CheckpointStore
from mirror7_backend.actions import ActionGateway, ActionPolicy


class Engine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        return {"observation": observation, "goal": goal, "count": len(research_tasks)}


def test_runtime_persistence_action_boundary(tmp_path):
    session = BackendSession("integration", Engine())
    result = session.step({"raw": [1, 2]}, goal="reach", research_tasks=("r1",))
    assert result.sequence == 1
    store = CheckpointStore(tmp_path)
    store.save(session.session_id, session.snapshot())
    restored = store.load("integration")
    assert restored["sequence"] == 1
    gateway = ActionGateway(
        {"record": lambda value: {"recorded": value}},
        ActionPolicy(frozenset({"record"})),
    )
    assert gateway.execute("record", result.state_digest)["recorded"] == result.state_digest
