"""
Teacher Provider Interface for knowledge collection and distillation.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping
from .categories import KnowledgeCategory


@dataclass(frozen=True)
class TeacherResponse:
    prompt: str
    response_text: str
    category: KnowledgeCategory
    model_id: str
    provider: str
    latency_seconds: float = 0.0
    finish_reason: str = "stop"
    tokens_generated: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not self.response_text.strip():
            raise ValueError("response_text must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")


class TeacherProvider(ABC):
    """Abstract interface for teacher model providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        category: KnowledgeCategory,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> TeacherResponse:
        """Generate a response from the teacher model."""
        pass


class MockTeacher(TeacherProvider):
    """Deterministic mock teacher for offline execution, unit tests, and CI."""

    def __init__(self, model_id: str = "mock-teacher-v1"):
        self.model_id = model_id
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        category: KnowledgeCategory,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> TeacherResponse:
        self.call_count += 1
        t0 = time.perf_counter()
        
        # Category-tailored mock completions
        if category == KnowledgeCategory.MATHEMATICS:
            resp = f"Step 1: Understand the given values.\nStep 2: Apply the appropriate formula.\nStep 3: Solution for '{prompt}': The calculated result is accurate and verified."
        elif category == KnowledgeCategory.PROGRAMMING_AND_DEBUGGING:
            resp = f"```python\ndef solution():\n    # Implementation for: {prompt}\n    return True\n```\nExplanation: The solution is structured, tested, and guarded against edge cases."
        elif category == KnowledgeCategory.SELF_CORRECTION:
            resp = f"Analysis of original attempt: An error was identified.\nCorrection: The corrected reasoning confirms the updated output."
        elif category == KnowledgeCategory.INSTRUCTION_FOLLOWING:
            resp = f"1. Item Alpha\n2. Item Beta\n3. Item Gamma"
        else:
            resp = f"Detailed explanation addressing: '{prompt}'. This response covers the foundational concepts, structured breakdown, and clear conclusion."

        latency = time.perf_counter() - t0
        return TeacherResponse(
            prompt=prompt,
            response_text=resp,
            category=category,
            model_id=self.model_id,
            provider="mock",
            latency_seconds=latency,
            tokens_generated=len(resp.split()),
            metadata={"call_index": self.call_count, "temperature": temperature},
        )


class LocalHuggingFaceTeacher(TeacherProvider):
    """Provider for locally hosted open-weight models (e.g., via transformers)."""

    def __init__(self, model_id: str, device: str = "auto", **model_kwargs: Any):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
        self._model_kwargs = model_kwargs

    def _load(self):
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required to use LocalHuggingFaceTeacher"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device, **self._model_kwargs
            )
            self._model.eval()

    def generate(
        self,
        prompt: str,
        category: KnowledgeCategory,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> TeacherResponse:
        self._load()
        import torch

        t0 = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            inputs = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt")["input_ids"]

        device = next(self._model.parameters()).device
        inputs = inputs.to(device)

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        # Transformers warns or rejects temperature when sampling is disabled.
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output = self._model.generate(inputs, **generation_kwargs)

        new_tokens = output[0][inputs.shape[1] :]
        response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        latency = time.perf_counter() - t0

        return TeacherResponse(
            prompt=prompt,
            response_text=response_text,
            category=category,
            model_id=self.model_id,
            provider="local_huggingface",
            latency_seconds=latency,
            tokens_generated=len(new_tokens),
            metadata={"device": str(device)},
        )
