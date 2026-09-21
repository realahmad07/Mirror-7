import pytest

from phase357_response_generation_interface import build_response_generation_request
from phase355_backend_realization_contract import BackendRealizationContract
from phase359_response_generation_adapter import (
    ResponseGenerationAdapter,
    make_response_generator,
    run_adapter,
)


class Model:
    def __init__(self):
        self.seen = []

    def generate(self, request):
        self.seen.append(request)
        return {"text": "ok", "mode": request.contract.mode}


def request():
    return build_response_generation_request(
        BackendRealizationContract(
            mode="explain",
            context={"entities": ("Python",)},
            actions=("inspect",),
            observation="Explain Python",
        )
    )


def test_adapter_delegates_to_model():
    model = Model()
    adapter = ResponseGenerationAdapter(model)
    req = request()
    assert adapter(req) == {"text": "ok", "mode": "explain"}
    assert model.seen == [req]


def test_factory_returns_callable_adapter():
    model = Model()
    generator = make_response_generator(model)
    assert callable(generator)
    assert generator(request())["text"] == "ok"


def test_adapter_does_not_change_verified_contract():
    model = Model()
    req = request()
    original = req.contract
    ResponseGenerationAdapter(model)(req)
    assert model.seen[0].contract is original
    assert original.actions == ("inspect",)


def test_invalid_model_rejected():
    with pytest.raises(TypeError):
        ResponseGenerationAdapter(object())


def test_missing_request_fails_closed():
    model = Model()
    adapter = ResponseGenerationAdapter(model)
    assert run_adapter(None, adapter) is None


def test_model_failure_propagates():
    class Failing:
        def generate(self, request):
            raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure"):
        run_adapter(request(), ResponseGenerationAdapter(Failing()))
