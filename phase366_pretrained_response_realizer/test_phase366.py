from __future__ import annotations

import sys
import types

import pytest

from phase355_backend_realization_contract import BackendRealizationContract
from phase366_pretrained_response_realizer import (
    DEFAULT_MODEL_ID,
    PretrainedResponseRealizer,
    build_realizer_prompt,
    load_pretrained_realizer,
)


def contract():
    return BackendRealizationContract(
        mode="explain",
        context={"entities": ("Python",), "desired_output": "explanation"},
        actions=("explain",),
        observation={"verified": True},
    )


def test_prompt_contains_verified_contract_data():
    prompt = build_realizer_prompt(contract())
    assert "Python" in prompt
    assert "explain" in prompt
    assert "Do not invent" in prompt


def test_model_adapter_generates_text():
    class Parameter:
        device = "cpu"

    class Model:
        def parameters(self):
            return iter([Parameter()])

        def generate(self, **kwargs):
            return kwargs["input_ids"].new_tensor([[1, 2, 3, 4]])

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors):
            import torch
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, skip_special_tokens=True):
            return "A verified answer."

    result = PretrainedResponseRealizer(Model(), Tokenizer()).generate(
        types.SimpleNamespace(contract=contract())
    )
    assert result == "A verified answer."


def test_default_model_is_small_instruct_model():
    assert DEFAULT_MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"


def test_loader_reports_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="transformers is required"):
        load_pretrained_realizer()


def test_verified_fact_is_returned_directly():
    verified_contract = BackendRealizationContract(
        mode="answer",
        context={
            "task": "Tell the user whether the requested file was created.",
            "verified_fact": "The file was not created.",
        },
        actions=(),
        observation="No file creation action was executed.",
    )

    class FailingModel:
        def parameters(self):
            raise AssertionError("Model must not be called for an authoritative verified fact.")

    result = PretrainedResponseRealizer(
        FailingModel(),
        object(),
    ).generate(types.SimpleNamespace(contract=verified_contract))

    assert result == "The file was not created."

def test_backend_service_accepts_pretrained_realizer_adapter():
    from mirror7_backend.service import BackendService

    class Model:
        def generate(self, request):
            assert request.contract.mode == "explain"
            return "Mirror response"

    class Engine:
        def step(self, observation, *, goal=None, research_tasks=(), views=()):
            from phase343_semantic_state import SemanticStateInducer

            return type(
                "Result",
                (),
                {"semantic_state": SemanticStateInducer().discover(observation)},
            )()

    import torch

    class Parameter:
        device = "cpu"

    class Tokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, prompt, return_tensors):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, skip_special_tokens=True):
            return "Mirror response"

    class GenerationModel(Model):
        def parameters(self):
            return iter([Parameter()])

        def generate(self, **kwargs):
            return kwargs["input_ids"].new_tensor([[1, 2, 3]])

    service = BackendService(engine_factory=Engine)
    service.create_session("phase366-e2e")
    service.set_response_model(PretrainedResponseRealizer(GenerationModel(), Tokenizer()))

    result = service.step("phase366-e2e", "Explain Python")

    assert result.response == "Mirror response"
    assert result.realization_contract is not None
