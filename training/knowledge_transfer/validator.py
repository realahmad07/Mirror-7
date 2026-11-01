"""
Response Validation and Quality Filtering for Teacher Completions.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any
from .categories import KnowledgeCategory
from .teacher_interface import TeacherResponse


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    quality_score: float
    reasons: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)


class ResponseValidator:
    """Apply heuristic structure/format filters, not factual verification.

    A response passing this validator may still be factually, mathematically,
    or semantically incorrect. Keep that distinction in downstream reports.
    """

    def __init__(
        self,
        min_bytes: int = 15,
        max_bytes: int = 4096,
        max_repetition_ratio: float = 0.30,
        jaccard_dedup_threshold: float = 0.90,
    ):
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.max_repetition_ratio = max_repetition_ratio
        self.jaccard_dedup_threshold = jaccard_dedup_threshold
        self._seen_response_signatures: set[str] = set()

    def validate(self, response: TeacherResponse) -> ValidationResult:
        reasons: list[str] = []
        metrics: dict[str, Any] = {
            "factual_verification": "not_performed",
            "semantic_correctness": "not_verified",
        }
        score = 1.0

        raw_text = response.response_text.strip()
        encoded = raw_text.encode("utf-8")
        byte_len = len(encoded)
        metrics["byte_length"] = byte_len

        # 1. Length checks
        if byte_len < self.min_bytes:
            reasons.append(f"Response too short ({byte_len} bytes < {self.min_bytes})")
            return ValidationResult(is_valid=False, quality_score=0.0, reasons=tuple(reasons), metrics=metrics)

        if byte_len > self.max_bytes:
            reasons.append(f"Response exceeded maximum length ({byte_len} > {self.max_bytes})")
            score -= 0.3

        # 2. Degenerate repetition check
        words = raw_text.split()
        if len(words) >= 6:
            # 3-gram repetition ratio
            tri_grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
            unique_tri = len(set(tri_grams))
            rep_ratio = 1.0 - (unique_tri / max(1, len(tri_grams)))
            metrics["repetition_ratio"] = rep_ratio
            if rep_ratio > self.max_repetition_ratio:
                reasons.append(f"Excessive repetition ratio: {rep_ratio:.2f}")
                return ValidationResult(is_valid=False, quality_score=0.0, reasons=tuple(reasons), metrics=metrics)

        # 3. Programming and Code Validation
        if response.category == KnowledgeCategory.PROGRAMMING_AND_DEBUGGING or "```python" in raw_text:
            code_blocks = re.findall(r"```python(.*?)```", raw_text, re.DOTALL)
            if code_blocks:
                for block in code_blocks:
                    block_stripped = block.strip()
                    if block_stripped:
                        try:
                            ast.parse(block_stripped)
                            metrics["python_syntax_valid"] = True
                        except SyntaxError as e:
                            reasons.append(f"Python code block failed syntax validation: {e.msg}")
                            score -= 0.4
            else:
                # Expected code block for programming tasks
                if response.category == KnowledgeCategory.PROGRAMMING_AND_DEBUGGING:
                    reasons.append("Programming response lacked formatted code block")
                    score -= 0.2

        # 4. Mathematics and Step Decomposition
        if response.category in (
            KnowledgeCategory.MATHEMATICS,
            KnowledgeCategory.MULTI_STEP_PROBLEM_SOLVING,
        ):
            has_steps = bool(
                re.search(r"(step\s*\d|first|then|finally|calculate|=|\+|\*|/)", raw_text, re.IGNORECASE)
            )
            metrics["has_mathematical_derivation"] = has_steps
            if not has_steps:
                reasons.append("Mathematical response lacked clear derivation steps")
                score -= 0.3

        # 5. Exact and Near-Deduplication
        signature = re.sub(r"\s+", " ", raw_text.lower())[:200]
        if signature in self._seen_response_signatures:
            reasons.append("Duplicate response signature detected")
            return ValidationResult(is_valid=False, quality_score=0.0, reasons=tuple(reasons), metrics=metrics)
        self._seen_response_signatures.add(signature)

        final_score = max(0.0, min(1.0, score))
        is_valid = final_score >= 0.5 and len(reasons) <= 1
        return ValidationResult(
            is_valid=is_valid,
            quality_score=final_score,
            reasons=tuple(reasons),
            metrics=metrics,
        )
