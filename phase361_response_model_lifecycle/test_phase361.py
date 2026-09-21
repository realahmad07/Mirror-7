from mirror7_backend.service import BackendService


class Model:
    def generate(self, request):
        return {"mode": request.contract.mode, "text": "ok"}


class Engine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        from phase343_semantic_state import SemanticStateInducer
        return type("Result", (), {
            "semantic_state": SemanticStateInducer().discover(observation)
        })()


def test_runtime_model_registration_enables_generation():
    service = BackendService(engine_factory=Engine)
    service.create_session("s")
    service.set_response_model(Model())
    result = service.step("s", "Explain Python")
    assert result.response == {"mode": "explain", "text": "ok"}
    assert service.status()["response_generation"] is True


def test_runtime_model_registration_can_be_removed():
    service = BackendService(engine_factory=Engine)
    service.create_session("s")
    service.set_response_model(Model())
    service.set_response_model(None)
    result = service.step("s", "Explain Python")
    assert result.response is None
    assert service.status()["response_generation"] is False


def test_invalid_runtime_model_fails_before_generation():
    service = BackendService(engine_factory=Engine)
    service.create_session("s")
    try:
        service.set_response_model(object())
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")
