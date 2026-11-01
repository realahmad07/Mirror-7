"""
Knowledge Distillation Strategies for Mirror 7.

Addresses the architectural constraints between subword Transformer teachers
and Mirror 7's native byte-level recurrent student model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from training.schema import TrainingExample


class DistillationMethod(str, Enum):
    SUPERVISED_DEMONSTRATION = "supervised_demonstration"  # Method A
    PREFERENCE_COMPARISON = "preference_comparison"        # Method B
    VERIFIED_CORRECTION = "verified_correction"            # Method C
    PROBLEM_SOLVING_DECOMPOSITION = "problem_solving_decomposition"  # Method D


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    prompt: str
    chosen_response: str
    rejected_response: str
    category: str
    quality_delta: float

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "prompt": self.prompt,
            "chosen": self.chosen_response,
            "rejected": self.rejected_response,
            "category": self.category,
            "quality_delta": self.quality_delta,
        }


@dataclass(frozen=True)
class CorrectionPair:
    correction_id: str
    prompt: str
    erroneous_response: str
    critique: str
    corrected_response: str
    category: str

    def to_training_example(self) -> TrainingExample:
        # Converts critique and correction into a unified training demonstration
        formatted_target = (
            f"Preliminary Analysis: {self.erroneous_response}\n\n"
            f"Critique: {self.critique}\n\n"
            f"Corrected Response: {self.corrected_response}"
        )
        return TrainingExample(
            example_id=f"corr_{self.correction_id}",
            user_text=self.prompt,
            target_text=formatted_target,
            goal="self_correction_demonstration",
            state={"type": "verified_correction", "category": self.category},
            evidence={},
            actions=(),
            split="train",
            quality=1.0,
        )


class DistillationManager:
    """Manages distillation dataset generation across different transfer regimes."""

    @staticmethod
    def explain_vocabulary_alignment_constraint() -> dict[str, str]:
        """Provides theoretical and mathematical basis for vocabulary incompatibility."""
        return {
            "teacher_vocabulary": "Subword BPE / SentencePiece (32,000 to 152,000 tokens)",
            "student_vocabulary": "Raw UTF-8 Bytes + 4 special tokens (260 tokens)",
            "logit_distillation_status": "INCOMPATIBLE WITHOUT INTERPOLATION",
            "justification": (
                "Direct KL divergence over logits requires identical probability spaces "
                "P(v) and Q(v) where |V_teacher| == |V_student|. Because the teacher emits "
                "subwords (e.g. 'function' as a single token ID 1423) while the student emits "
                "8 separate bytes ('f', 'u', 'n', 'c', 't', 'i', 'o', 'n'), teacher logits cannot "
                "be projected onto student byte positions without a dynamic alignment matrix. "
                "Response-level distillation (sequence-level knowledge distillation, Kim & Rush 2016) "
                "bypasses this constraint entirely, preserving mathematical purity."
            ),
            "recommended_strategy": "Response-level supervised fine-tuning with verified multi-step reasoning.",
        }

    def build_demonstration_dataset(
        self, examples: Sequence[TrainingExample]
    ) -> list[TrainingExample]:
        """Build standard SFT demonstration dataset from validated examples."""
        return [ex for ex in examples if ex.quality >= 0.70]

    def build_preference_dataset(
        self,
        chosen_examples: Sequence[TrainingExample],
        rejected_examples: Sequence[TrainingExample],
    ) -> list[PreferencePair]:
        """Pair higher-quality teacher completions with lower-quality/perturbed completions."""
        pairs: list[PreferencePair] = []
        rejected_by_prompt = {r.user_text.strip().lower(): r for r in rejected_examples}

        for idx, ch in enumerate(chosen_examples):
            key = ch.user_text.strip().lower()
            if key in rejected_by_prompt:
                rej = rejected_by_prompt[key]
                pairs.append(
                    PreferencePair(
                        pair_id=f"pref_{idx+1:04d}",
                        prompt=ch.user_text,
                        chosen_response=ch.target_text,
                        rejected_response=rej.target_text,
                        category=ch.state.get("category", "general"),
                        quality_delta=ch.quality - rej.quality,
                    )
                )
        return pairs
