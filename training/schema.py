from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json


VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    user_text: str
    target_text: str
    goal: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    actions: tuple[str, ...] = ()
    split: str = "train"
    quality: float = 1.0

    def validate(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must be non-empty")
        if not self.user_text.strip():
            raise ValueError("user_text must be non-empty")
        if not self.target_text.strip():
            raise ValueError("target_text must be non-empty")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        if not isinstance(self.state, dict):
            raise ValueError("state must be a JSON object")
        if not isinstance(self.evidence, dict):
            raise ValueError("evidence must be a JSON object")
        if not 0.0 <= float(self.quality) <= 1.0:
            raise ValueError("quality must be between 0 and 1")

    def canonical_context(self) -> bytes:
        payload = {
            "goal": self.goal,
            "state": self.state,
            "evidence": self.evidence,
            "actions": list(self.actions),
        }
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def fingerprint(self) -> str:
        self.validate()
        payload = {
            "example_id": self.example_id,
            "user_text": self.user_text,
            "target_text": self.target_text,
            "goal": self.goal,
            "state": self.state,
            "evidence": self.evidence,
            "actions": list(self.actions),
            "split": self.split,
            "quality": float(self.quality),
        }
        return hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


def example_from_dict(value: dict[str, Any]) -> TrainingExample:
    if not isinstance(value, dict):
        raise ValueError("training record must be a JSON object")
    example = TrainingExample(
        example_id=str(value.get("example_id", "")),
        user_text=str(value.get("user_text", "")),
        target_text=str(value.get("target_text", "")),
        goal=value.get("goal"),
        state=dict(value.get("state", {})),
        evidence=dict(value.get("evidence", {})),
        actions=tuple(str(x) for x in value.get("actions", [])),
        split=str(value.get("split", "train")),
        quality=float(value.get("quality", 1.0)),
    )
    example.validate()
    return example
