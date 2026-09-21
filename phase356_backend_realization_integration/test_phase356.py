from types import SimpleNamespace

from mirror7_backend.runtime import BackendSession
from phase343_semantic_state import SemanticStateInducer
from phase356_backend_realization_integration import backend_realization_contract_from_result


class Engine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        state = SemanticStateInducer().discover(observation) if isinstance(observation, str) else None
        return SimpleNamespace(semantic_state=state)


class PlainEngine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        return {"observation": observation}


def test_backend_exposes_realization_contract_from_verified_semantic_state():
    result = BackendSession("s", Engine()).step("Explain Python")
    contract = backend_realization_contract_from_result(result)
    assert contract is not None
    assert contract.mode == "explain"
    assert "Python" in contract.context["entities"]
    assert contract.actions == ()


def test_backend_does_not_invent_contract_without_semantic_state():
    result = BackendSession("s", PlainEngine()).step({"x": 1})
    assert backend_realization_contract_from_result(result) is None
    assert result.realization_contract is None


def test_contract_observation_is_copy_isolated():
    observation = "Explain Python"
    result = BackendSession("s", Engine()).step(observation)
    contract = result.realization_contract
    assert contract.observation == observation
    assert contract.actions == ()


def test_unknown_semantic_output_does_not_require_a_plan():
    class UnknownEngine:
        def step(self, observation, *, goal=None, research_tasks=(), views=()):
            state = SimpleNamespace(
                desired_output=None,
                entities=(),
                operations=(),
                constraints=(),
                context=(),
                evidence=(),
            )
            return SimpleNamespace(semantic_state=state)

    result = BackendSession("s", UnknownEngine()).step("x")
    assert result.realization_contract is not None
    assert result.realization_contract.mode is None
    assert result.realization_contract.actions == ()
