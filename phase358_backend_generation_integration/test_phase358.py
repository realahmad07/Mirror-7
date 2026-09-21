from types import SimpleNamespace

from mirror7_backend.service import BackendService
from mirror7_backend.runtime import BackendSession


class Engine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        return SimpleNamespace(semantic_state=None)


class SemanticEngine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        from phase343_semantic_state import SemanticStateInducer
        return SimpleNamespace(semantic_state=SemanticStateInducer().discover(observation))


def test_service_generation_is_optional():
    service = BackendService(engine_factory=SemanticEngine)
    service.create_session("s")
    result = service.step("s", "Explain Python")
    assert result.response is None


def test_service_delegates_verified_contract_to_generator():
    seen = []

    def generator(request):
        seen.append(request)
        return f"generated:{request.contract.mode}"

    service = BackendService(
        engine_factory=SemanticEngine,
        response_generator=generator,
    )
    service.create_session("s")
    result = service.step("s", "Explain Python")
    assert result.response == "generated:explain"
    assert len(seen) == 1
    assert seen[0].contract.mode == "explain"
    assert "Python" in seen[0].contract.context["entities"]


def test_service_does_not_call_generator_without_contract():
    calls = []

    def generator(request):
        calls.append(request)
        return "unexpected"

    service = BackendService(
        engine_factory=Engine,
        response_generator=generator,
    )
    service.create_session("s")
    result = service.step("s", {"x": 1})
    assert result.response is None
    assert calls == []


def test_generation_failure_is_reported_as_failed_step():
    def generator(request):
        raise RuntimeError("generator failed")

    service = BackendService(
        engine_factory=SemanticEngine,
        response_generator=generator,
    )
    service.create_session("s")
    try:
        service.step("s", "Explain Python")
    except RuntimeError as exc:
        assert str(exc) == "generator failed"
    else:
        raise AssertionError("expected generator failure")
    assert service.status()["metrics"]["steps_failed"] == 1


def test_backend_result_response_isolated_from_contract():
    def generator(request):
        return {"text": "ok", "mode": request.contract.mode}

    service = BackendService(
        engine_factory=SemanticEngine,
        response_generator=generator,
    )
    service.create_session("s")
    result = service.step("s", "Explain Python")
    assert result.response == {"text": "ok", "mode": "explain"}
    assert result.realization_contract.actions == ()
