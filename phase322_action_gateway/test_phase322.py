import time
import pytest
from mirror7_backend.actions import ActionGateway, ActionPolicy, ActionDenied


def test_allowlisted_action_executes():
    gateway = ActionGateway(
        {"add": lambda a, b: a + b},
        ActionPolicy(frozenset({"add"})),
    )
    assert gateway.execute("add", 2, 3) == 5


def test_non_allowlisted_action_is_denied():
    gateway = ActionGateway(
        {"add": lambda a, b: a + b},
        ActionPolicy(frozenset({"add"})),
    )
    with pytest.raises(ActionDenied):
        gateway.execute("delete", 1)


def test_call_budget_is_enforced():
    gateway = ActionGateway(
        {"ping": lambda: "ok"},
        ActionPolicy(frozenset({"ping"}), max_calls=1),
    )
    assert gateway.execute("ping") == "ok"
    with pytest.raises(ActionDenied):
        gateway.execute("ping")


def test_elapsed_budget_is_enforced():
    gateway = ActionGateway(
        {"slow": lambda: time.sleep(0.01)},
        ActionPolicy(frozenset({"slow"}), timeout_seconds=0.001),
    )
    with pytest.raises(TimeoutError):
        gateway.execute("slow")
