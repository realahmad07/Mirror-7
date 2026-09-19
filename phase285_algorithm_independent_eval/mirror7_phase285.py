from dataclasses import dataclass
from typing import Iterable, List, Tuple

from phase280_sealed_candidate_eval import EvaluationPack
from phase283_algorithm_variant_space import AlgorithmVariant, AlgorithmVariantSpace

@dataclass(frozen=True)
class AlgorithmEvaluation:
    variant: AlgorithmVariant
    train: float
    held_out: float
    regression_ok: bool
    total_cases: int

class AlgorithmIndependentEvaluator:
    """Evaluates declarative algorithms against hidden targets supplied only to the evaluator."""
    def __init__(self, space: AlgorithmVariantSpace):
        self.space = space

    def _score(self, variant: AlgorithmVariant, tasks: Iterable[dict]) -> tuple[int, int]:
        good = total = 0
        for task in tasks:
            total += 1
            public = task.get("public", {})
            values = public.get("values")
            if values is None:
                continue
            try:
                answer = self.space.execute(variant, values)
            except Exception:
                continue
            if answer == task.get("target"):
                good += 1
        return good, total

    def evaluate(self, variant: AlgorithmVariant, pack: EvaluationPack) -> AlgorithmEvaluation:
        if not self.space.validate(variant):
            raise ValueError("invalid variant")
        train_good, train_total = self._score(variant, pack.train)
        held_good, held_total = self._score(variant, pack.held_out)
        reg_good, reg_total = self._score(variant, pack.regression)
        return AlgorithmEvaluation(
            variant,
            train_good / train_total if train_total else 0.0,
            held_good / held_total if held_total else 0.0,
            reg_total > 0 and reg_good == reg_total,
            train_total + held_total + reg_total,
        )

    def improves(self, baseline: AlgorithmEvaluation, candidate: AlgorithmEvaluation, min_gain: float = .01) -> bool:
        return (
            candidate.train - baseline.train >= min_gain
            and candidate.held_out > 0.0
            and candidate.held_out >= baseline.held_out
            and candidate.regression_ok
        )
