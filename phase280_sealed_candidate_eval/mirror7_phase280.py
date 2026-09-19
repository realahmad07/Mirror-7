from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence

@dataclass(frozen=True)
class EvaluationPack:
    train: Sequence[dict]
    held_out: Sequence[dict]
    regression: Sequence[dict]

@dataclass(frozen=True)
class CandidateScore:
    train: float
    held_out: float
    regression_ok: bool
    total_cases: int

class SealedCandidateEvaluator:
    """Scores a candidate using public task fields while keeping targets in the evaluator."""
    def _score(self, solver: Callable[[Any], Any], tasks: Iterable[dict]) -> tuple[int, int]:
        good = 0
        total = 0
        for task in tasks:
            total += 1
            try:
                answer = solver(task["public"])
            except Exception:
                continue
            if answer == task["target"]:
                good += 1
        return good, total

    def evaluate(self, solver: Callable[[Any], Any], pack: EvaluationPack) -> CandidateScore:
        train_good, train_total = self._score(solver, pack.train)
        held_good, held_total = self._score(solver, pack.held_out)
        reg_good, reg_total = self._score(solver, pack.regression)
        train = train_good / train_total if train_total else 0.0
        held = held_good / held_total if held_total else 0.0
        regression_ok = reg_total > 0 and reg_good == reg_total
        return CandidateScore(train, held, regression_ok, train_total + held_total + reg_total)

    def improves(self, baseline: CandidateScore, candidate: CandidateScore, *, min_gain: float = 0.01) -> bool:
        if candidate.train - baseline.train < min_gain:
            return False
        if candidate.held_out <= 0.0:
            return False
        if candidate.held_out < baseline.held_out:
            return False
        if not candidate.regression_ok:
            return False
        return True
