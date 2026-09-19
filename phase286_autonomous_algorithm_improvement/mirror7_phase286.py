from dataclasses import dataclass
from typing import List, Mapping

from phase280_sealed_candidate_eval import EvaluationPack
from phase281_promotion_rollback import PromotionRegistry
from phase283_algorithm_variant_space import AlgorithmVariant, AlgorithmVariantSpace
from phase284_algorithm_mutation import AlgorithmMutator
from phase285_algorithm_independent_eval import AlgorithmEvaluation, AlgorithmIndependentEvaluator

@dataclass(frozen=True)
class AlgorithmImprovementReport:
    round: int
    baseline_train: float
    baseline_held_out: float
    candidates_tested: int
    promoted: bool
    selected_variant: AlgorithmVariant | None
    selected_train: float
    selected_held_out: float

class AlgorithmImprovementEngine:
    """Autonomously searches a bounded declarative algorithm space and promotes verified variants."""
    def __init__(self, initial_variant: AlgorithmVariant, space: AlgorithmVariantSpace | None = None):
        self.space = space or AlgorithmVariantSpace()
        if not self.space.validate(initial_variant):
            raise ValueError("invalid initial variant")
        self.mutator = AlgorithmMutator(self.space)
        self.evaluator = AlgorithmIndependentEvaluator(self.space)
        self.registry = PromotionRegistry({"algorithm": initial_variant})

    @property
    def current_variant(self) -> AlgorithmVariant:
        return self.registry.current["algorithm"]

    def run_round(self, pack: EvaluationPack, *, round_number: int, max_candidates: int = 8) -> AlgorithmImprovementReport:
        baseline = self.evaluator.evaluate(self.current_variant, pack)
        mutations = self.mutator.mutate(self.current_variant, max_children=max_candidates)
        best: AlgorithmEvaluation | None = None
        for mutation in mutations:
            score = self.evaluator.evaluate(mutation.child, pack)
            if not self.evaluator.improves(baseline, score):
                continue
            if best is None or (score.train, score.held_out) > (best.train, best.held_out):
                best = score
        if best is None:
            return AlgorithmImprovementReport(round_number, baseline.train, baseline.held_out, len(mutations), False, None, baseline.train, baseline.held_out)
        gain = best.train - baseline.train
        rec = self.registry.promote({"algorithm": best.variant}, train_gain=gain, held_out_score=best.held_out, regression_ok=best.regression_ok)
        return AlgorithmImprovementReport(round_number, baseline.train, baseline.held_out, len(mutations), rec.accepted, best.variant if rec.accepted else None, best.train if rec.accepted else baseline.train, best.held_out if rec.accepted else baseline.held_out)

    def run_until_stable(self, pack: EvaluationPack, *, max_rounds: int = 4, max_candidates: int = 8) -> List[AlgorithmImprovementReport]:
        if max_rounds < 1 or max_candidates < 1:
            raise ValueError("bounds must be positive")
        reports: List[AlgorithmImprovementReport] = []
        for round_number in range(1, max_rounds + 1):
            report = self.run_round(pack, round_number=round_number, max_candidates=max_candidates)
            reports.append(report)
            if not report.promoted:
                break
        return reports

    def rollback(self, version: int | None = None):
        return self.registry.rollback(version)
