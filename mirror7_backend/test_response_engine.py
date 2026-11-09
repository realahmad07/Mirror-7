from types import SimpleNamespace

import pytest

from phase362_response_generation_config import ResponseGenerationConfig
from mirror7_backend.response_engine import UnifiedResponseEngine


def contract(mode="explain"):
    return SimpleNamespace(
        mode=mode,
        context={"question": "What is 2 + 2?"},
        actions=(),
        observation={"prompt": "What is 2 + 2?"},
    )


def test_returns_none_without_contract_or_model():
    engine = UnifiedResponseEngine()
    assert engine.respond(contract()) is None
    engine.set_model(lambda request: "4")
    assert engine.respond(None) is None


def test_callable_model_generates_validated_response():
    seen = []
    engine = UnifiedResponseEngine(lambda request: seen.append(request) or "4")
    assert engine.enabled is True
    assert engine.respond(contract()) == "4"
    assert len(seen) == 1
    assert seen[0].contract.context["question"] == "What is 2 + 2?"


def test_model_object_is_adapted():
    class Model:
        def generate(self, request):
            return {"text": "4"}

    engine = UnifiedResponseEngine(Model())
    assert engine.respond(contract()) == {"text": "4"}


def test_empty_and_oversized_outputs_fail_closed():
    engine = UnifiedResponseEngine(lambda request: "   ")
    with pytest.raises(ValueError, match="empty"):
        engine.respond(contract())

    engine = UnifiedResponseEngine(
        lambda request: "12345",
        config=ResponseGenerationConfig(max_output_chars=4),
    )
    with pytest.raises(ValueError, match="character limit"):
        engine.respond(contract())


def test_disallowed_mode_is_rejected():
    engine = UnifiedResponseEngine(
        lambda request: "answer",
        config=ResponseGenerationConfig(allowed_modes=("debug",)),
    )
    with pytest.raises(ValueError, match="not allowed"):
        engine.respond(contract("explain"))


def test_model_can_be_removed_and_config_updated():
    engine = UnifiedResponseEngine(lambda request: "answer")
    engine.set_model(None)
    assert engine.enabled is False
    engine.set_config(ResponseGenerationConfig(max_output_chars=20))
    assert engine.respond(contract()) is None
