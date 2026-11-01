"""
Comprehensive Independent Evaluator across the 10 core dimensions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .dimensions import EvaluationDimension, ALL_EVAL_DIMENSIONS, DimensionScore
from .benchmark_dataset import EvalBenchmarkExample, load_evaluation_benchmark


def _char_4grams(text: str) -> Counter[str]:
    clean = text.strip().lower()
    if len(clean) < 4:
        return Counter([clean] if clean else [])
    return Counter(clean[i : i + 4] for i in range(len(clean) - 3))


def compute_char_f1(pred: str, target: str) -> float:
    p_counts = _char_4grams(pred)
    t_counts = _char_4grams(target)
    common = sum((p_counts & t_counts).values())
    p_total = sum(p_counts.values())
    t_total = sum(t_counts.values())
    if p_total == 0 or t_total == 0:
        return 0.0
    precision = common / p_total
    recall = common / t_total
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def compute_repetition_ratio(text: str) -> float:
    words = text.strip().split()
    if len(words) < 6:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    unique = len(set(trigrams))
    return max(0.0, 1.0 - (unique / len(trigrams)))


@dataclass
class ComprehensiveEvaluationReport:
    model_name: str
    partition: str
    overall_score: float
    dimension_scores: dict[str, DimensionScore] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    successes: list[dict[str, Any]] = field(default_factory=list)


class IndependentEvaluator:
    """Evaluates text generation models or functions against the 10-dimension benchmark."""

    def __init__(self, partition: str = "dev"):
        self.partition = partition
        self.benchmark = load_evaluation_benchmark(partition=partition)

    def evaluate(
        self,
        generate_fn: Callable[[str], str],
        model_name: str = "evaluated_model",
    ) -> ComprehensiveEvaluationReport:
        dim_examples: dict[EvaluationDimension, list[EvalBenchmarkExample]] = {
            d: [] for d in ALL_EVAL_DIMENSIONS
        }
        for ex in self.benchmark:
            dim_examples[ex.dimension].append(ex)

        dim_scores: dict[str, DimensionScore] = {}
        all_dim_ratios: list[float] = []
        failures: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []

        for dim, examples in dim_examples.items():
            if not examples:
                continue

            passed = 0
            f1s: list[float] = []
            rep_ratios: list[float] = []
            rep_success: str | None = None
            rep_failure: str | None = None

            for ex in examples:
                output = generate_fn(ex.prompt).strip()
                out_lower = output.lower()

                f1 = max(compute_char_f1(output, ref) for ref in ex.reference_targets)
                f1s.append(f1)

                rep = compute_repetition_ratio(output)
                rep_ratios.append(rep)

                # Check keyword criteria
                has_all_expected = all(kw.lower() in out_lower for kw in ex.keywords_expected)
                has_no_forbidden = not any(kw.lower() in out_lower for kw in ex.keywords_forbidden)
                is_passing = has_all_expected and has_no_forbidden and f1 >= 0.15 and rep < 0.25

                if is_passing:
                    passed += 1
                    if not rep_success:
                        rep_success = f"Prompt: '{ex.prompt}' -> Response: '{output}'"
                    successes.append({"dimension": dim.value, "prompt": ex.prompt, "output": output, "f1": f1})
                else:
                    if not rep_failure:
                        rep_failure = f"Prompt: '{ex.prompt}' -> Response: '{output}' (F1: {f1:.2f}, Rep: {rep:.2f})"
                    failures.append({"dimension": dim.value, "prompt": ex.prompt, "output": output, "f1": f1})

            score_val = passed / max(1, len(examples))
            all_dim_ratios.append(score_val)

            dim_scores[dim.value] = DimensionScore(
                dimension=dim,
                score=round(score_val, 3),
                passed_cases=passed,
                total_cases=len(examples),
                mean_f1=round(sum(f1s) / max(1, len(f1s)), 4),
                repetition_ratio=round(sum(rep_ratios) / max(1, len(rep_ratios)), 4),
                representative_success=rep_success,
                representative_failure=rep_failure,
            )

        overall = sum(all_dim_ratios) / max(1, len(all_dim_ratios))
        return ComprehensiveEvaluationReport(
            model_name=model_name,
            partition=self.partition,
            overall_score=round(overall, 3),
            dimension_scores=dim_scores,
            failures=failures,
            successes=successes,
        )
