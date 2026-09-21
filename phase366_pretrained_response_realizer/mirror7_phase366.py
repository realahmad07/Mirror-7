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
    instruction = (
        "You are the response realization layer for Mirror 7. "
        "Use only the supplied Mirror 7 realization data. "
        "Turn that state into the final user-facing answer. "
        "Preserve verified facts and planned actions exactly. "
        "Do not claim that an action was completed unless the supplied observation "
        "or context explicitly supports that claim. "
        "Do not invent evidence, tool results, measurements, completed work, "
        "assumptions, or hypothetical scenarios. "
        "If the supplied state explicitly establishes an outcome, state it directly. "
        "Do not ask for more context when the supplied state is sufficient. "
        "Be concise, natural, factual, and directly answer the user's request."
    )
    return (
        instruction
        + "\n\nMIRROR 7 REALIZATION DATA:\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
        + "\n\nFINAL ANSWER:\n"
    )


def _tokenize_prompt(tokenizer: Any, prompt: str) -> Any:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Mirror 7's response realization layer. "
                    "Return only the final answer to the user. "
                    "Use only the supplied state. Do not add assumptions."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            pass
    return tokenizer(prompt, return_tensors="pt")


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
        inputs = _tokenize_prompt(self.tokenizer, prompt)
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=(
                    getattr(self.tokenizer, "pad_token_id", None)
                    or getattr(self.tokenizer, "eos_token_id", None)
                ),
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
    model.eval()
    return PretrainedResponseRealizer(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
