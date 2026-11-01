"""
Task Generator for Knowledge Transfer.
Generates systematic prompts and tasks across all 13 knowledge categories.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterator, Sequence

from .categories import KnowledgeCategory, CATEGORY_REGISTRY, ALL_CATEGORIES


@dataclass(frozen=True)
class TaskPrompt:
    task_id: str
    category: KnowledgeCategory
    prompt: str
    difficulty: str = "medium"
    context: dict[str, str] | None = None

    def fingerprint(self) -> str:
        payload = f"{self.task_id}:{self.category.value}:{self.prompt.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TaskGenerator:
    """Generates structured prompts and problem tasks for teacher querying."""

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def generate_seed_tasks(self) -> list[TaskPrompt]:
        """Generate tasks directly from the curated seed prompts of all categories."""
        tasks: list[TaskPrompt] = []
        for cat in ALL_CATEGORIES:
            meta = CATEGORY_REGISTRY[cat]
            for i, p in enumerate(meta.sample_seed_prompts):
                tid = f"{cat.value}_{i+1:03d}"
                tasks.append(
                    TaskPrompt(
                        task_id=tid,
                        category=cat,
                        prompt=p,
                        difficulty="medium",
                    )
                )
        return tasks

    def generate_variations(
        self, base_task: TaskPrompt, count: int = 3
    ) -> list[TaskPrompt]:
        """Produce slight contextual or stylistic variations of a task."""
        prefixes = [
            "Please provide a clear and structured explanation: ",
            "Explain step by step: ",
            "Solve the following with complete clarity: ",
            "Provide a comprehensive, verified answer to: ",
        ]
        variations: list[TaskPrompt] = []
        for i in range(count):
            prefix = prefixes[i % len(prefixes)]
            var_prompt = f"{prefix}{base_task.prompt}"
            var_id = f"{base_task.task_id}_var_{i+1}"
            variations.append(
                TaskPrompt(
                    task_id=var_id,
                    category=base_task.category,
                    prompt=var_prompt,
                    difficulty=base_task.difficulty,
                )
            )
        return variations

    def generate_corpus(
        self, target_per_category: int = 5
    ) -> list[TaskPrompt]:
        """Produce a balanced corpus across all 13 categories."""
        corpus: list[TaskPrompt] = []
        for cat in ALL_CATEGORIES:
            meta = CATEGORY_REGISTRY[cat]
            seeds = list(meta.sample_seed_prompts)
            for idx in range(target_per_category):
                base_p = seeds[idx % len(seeds)]
                tid = f"{cat.value}_{idx+1:04d}"
                if idx < len(seeds):
                    p_text = base_p
                else:
                    # Variation
                    p_text = f"Analyze and resolve step-by-step: {base_p}"
                corpus.append(
                    TaskPrompt(
                        task_id=tid,
                        category=cat,
                        prompt=p_text,
                        difficulty="medium",
                    )
                )
        return corpus
