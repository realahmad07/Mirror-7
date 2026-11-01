"""
Dataset Normalization and Provenance Tracking.
Converts teacher responses into Mirror 7 canonical TrainingExample records.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from training.schema import TrainingExample
from .teacher_interface import TeacherResponse
from .validator import ValidationResult


class DatasetNormalizer:
    """Normalizes teacher responses into Mirror 7 training records with provenance metadata."""

    def __init__(self, default_split: str = "train"):
        self.default_split = default_split

    def to_training_example(
        self,
        task_id: str,
        response: TeacherResponse,
        validation: ValidationResult,
        split: str | None = None,
    ) -> TrainingExample:
        split_to_use = split or self.default_split
        
        # Provenance metadata embedded in structured state
        state = {
            "origin": "knowledge_transfer",
            "teacher_model": response.model_id,
            "teacher_provider": response.provider,
            "category": response.category.value,
            "latency_seconds": round(response.latency_seconds, 4),
            "tokens_generated": response.tokens_generated,
            "validation_metrics": validation.metrics,
        }

        evidence = {
            "prompt_hash": hashlib.sha256(response.prompt.encode("utf-8")).hexdigest()[:16],
            "response_hash": hashlib.sha256(response.response_text.encode("utf-8")).hexdigest()[:16],
            "validation_reasons": list(validation.reasons),
        }

        example = TrainingExample(
            example_id=f"kt_{task_id}",
            user_text=response.prompt.strip(),
            target_text=response.response_text.strip(),
            goal=f"transfer_{response.category.value}",
            state=state,
            evidence=evidence,
            actions=(),
            split=split_to_use,
            quality=float(validation.quality_score),
        )
        example.validate()
        return example

    def export_jsonl(
        self, examples: Sequence[TrainingExample], output_path: str | Path
    ) -> int:
        """Write examples to a JSONL file and return count."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for ex in examples:
                payload = {
                    "example_id": ex.example_id,
                    "user_text": ex.user_text,
                    "target_text": ex.target_text,
                    "goal": ex.goal,
                    "state": ex.state,
                    "evidence": ex.evidence,
                    "actions": list(ex.actions),
                    "split": ex.split,
                    "quality": ex.quality,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                count += 1
        return count
