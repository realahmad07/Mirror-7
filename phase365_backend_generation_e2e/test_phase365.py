from mirror7_backend.service import BackendService
from phase365_backend_generation_e2e import configure_backend_response_model, validate_backend_response
from phase362_response_generation_config import ResponseGenerationConfig


class Model:
    def generate(self, request):
        return {"mode": request.contract.mode, "text": "Mirror response"}


class Loader:
    def load(self):
        return Model()


class Engine:
    def step(self, observation, *, goal=None, research_tasks=(), views=()):
        from phase343_semantic_state import SemanticStateInducer
        return type("Result", (), {
            "semantic_state": SemanticStateInducer().discover(observation)
        })()


def test_backend_generation_end_to_end_with_injected_model():
    service = BackendService(engine_factory=Engine)
    service.create_session("e2e")
    model = configure_backend_response_model(service, Loader())
    assert isinstance(model, Model)
    result = service.step("e2e", "Explain Python")
    assert result.response["mode"] == "explain"
    assert validate_backend_response(result, ResponseGenerationConfig()) == result.response


def test_backend_generation_validation_preserves_model_output():
    service = BackendService(engine_factory=Engine)
    service.create_session("e2e")
    configure_backend_response_model(service, Loader())
    result = service.step("e2e", "Explain Python")
    assert result.response["text"] == "Mirror response"
