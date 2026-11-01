"""
Knowledge Collector.
Coordinates task generation, teacher querying, quality filtering, and dataset export.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from training.schema import TrainingExample
from .categories import KnowledgeCategory
from .teacher_interface import TeacherProvider, TeacherResponse
from .task_generator import TaskPrompt, TaskGenerator
from .validator import ResponseValidator, ValidationResult
from .normalizer import DatasetNormalizer

logger = logging.getLogger(__name__)


def _split_for_prompt(
    prompt: str, split_ratio: tuple[float, float, float]
) -> str:
    """Assign an identical normalized prompt to the same deterministic split."""
    normalized_prompt = re.sub(r"\s+", " ", prompt).strip().casefold()
    prompt_hash = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
    h_val = int(prompt_hash[:8], 16) / 0xFFFFFFFF
    if h_val < split_ratio[0]:
        return "train"
    if h_val < split_ratio[0] + split_ratio[1]:
        return "validation"
    return "test"


@dataclass
class CollectionReport:
    total_tasks: int = 0
    successful_generations: int = 0
    valid_examples: int = 0
    rejected_examples: int = 0
    examples_by_category: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)


class KnowledgeCollector:
    """End-to-end pipeline for harvesting validated teacher knowledge."""

    def __init__(
        self,
        teacher: TeacherProvider,
        validator: ResponseValidator | None = None,
        normalizer: DatasetNormalizer | None = None,
    ):
        self.teacher = teacher
        self.validator = validator or ResponseValidator()
        self.normalizer = normalizer or DatasetNormalizer()

    def collect(
        self,
        tasks: Sequence[TaskPrompt],
        split_ratio: tuple[float, float, float] = (0.80, 0.10, 0.10),
    ) -> tuple[list[TrainingExample], CollectionReport]:
        if len(split_ratio) != 3 or any(x < 0 for x in split_ratio):
            raise ValueError("split_ratio must contain three non-negative values")
        if not 0.0 < sum(split_ratio) <= 1.0:
            raise ValueError("split_ratio values must sum to a value in (0, 1]")

        report = CollectionReport(total_tasks=len(tasks))
        valid_examples: list[TrainingExample] = []

        for idx, task in enumerate(tasks):
            try:
                resp = self.teacher.generate(task.prompt, task.category)
                report.successful_generations += 1
            except Exception as exc:
                logger.warning(f"Teacher generation failed for task {task.task_id}: {exc}")
                continue

            val = self.validator.validate(resp)
            if not val.is_valid:
                report.rejected_examples += 1
                for r in val.reasons:
                    report.rejection_reasons[r] = report.rejection_reasons.get(r, 0) + 1
                continue

            # Split by normalized prompt content, not task_id. Variants with
            # different IDs but identical prompts must never leak across splits.
            split = _split_for_prompt(task.prompt, split_ratio)

            example = self.normalizer.to_training_example(task.task_id, resp, val, split=split)
            valid_examples.append(example)
            report.valid_examples += 1
            c_name = task.category.value
            report.examples_by_category[c_name] = report.examples_by_category.get(c_name, 0) + 1

        return valid_examples, report
