from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def build_realizer_prompt(contract: Any) -> str:
    payload = {
        "mode": contract.mode,
        "semantic_context": dict(contract.context),
        "planned_actions": list(contract.actions),
        "observation": contract.observation,
    }
    return (
        "You are the response realization layer for Mirror 7. "
        "Write the final answer for the user using only the supplied Mirror 7 context. "
        "Do not invent actions, observations, evidence, or completed work. "
        "Answer naturally and concisely.\n\n"
        "MIRROR 7 REALIZATION DATA:\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
        + "\n\nFINAL ANSWER:\n"
    )


@dataclass
class PretrainedResponseRealizer:
    model: Any
    tokenizer: Any
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9

    def generate(self, request: Any) -> str:
        import torch

        prompt = build_realizer_prompt(request.contract)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("pretrained response realizer generated empty text")
        return text


def load_pretrained_realizer(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device_map: str | None = "auto",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> PretrainedResponseRealizer:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for the pretrained response realizer"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)
    return PretrainedResponseRealizer(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
