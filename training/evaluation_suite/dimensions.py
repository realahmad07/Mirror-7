"""
Evaluation dimensions and score metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvaluationDimension(str, Enum):
    CONVERSATIONAL_QUALITY = "conversational_quality"
    PROMPT_UNDERSTANDING = "prompt_understanding"
    FACTUAL_ACCURACY = "factual_accuracy"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    CODING_ABILITY = "coding_ability"
    PLANNING = "planning"
    TOOL_USE_CORRECTNESS = "tool_use_correctness"
    GENERALIZATION = "generalization"
    STRATEGY_DISCOVERY = "strategy_discovery"


ALL_EVAL_DIMENSIONS: tuple[EvaluationDimension, ...] = tuple(EvaluationDimension)


@dataclass(frozen=True)
class DimensionScore:
    dimension: EvaluationDimension
    score: float  # 0.0 to 1.0
    passed_cases: int
    total_cases: int
    mean_f1: float
    repetition_ratio: float
    representative_success: str | None = None
    representative_failure: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
