from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Sequence

from phase279_gap_hypothesis import CapabilityGapDetector, HypothesisGenerator
from phase280_sealed_candidate_eval import EvaluationPack, CandidateScore, SealedCandidateEvaluator
from phase281_promotion_rollback import PromotionRegistry

@dataclass(frozen=True)
class ImprovementReport:
    round: int
    gap_detected: bool
    candidates_tested: int
    promoted: bool
    selected_parameter: str | None
    selected_value: object | None
    baseline_train: float
    selected_train: float
    selected_held_out: float

class SelfImprovementEngine:
    """Bounded self-improvement loop over an explicit, finite variant space."""
    def __init__(self, initial_config: Mapping[str, Any], *, target_score: float = 1.0):
        if not (0.0 <= float(target_score) <= 1.0):
            raise ValueError("target_score must be in [0,1]")
        self.detector = CapabilityGapDetector()
        self.generator = HypothesisGenerator()
        self.evaluator = SealedCandidateEvaluator()
        self.registry = PromotionRegistry(initial_config)
        self.target_score = float(target_score)

    @staticmethod
    def _evidence(solver: Callable[[Any], Any], tasks: Iterable[dict]) -> List[bool]:
        evidence = []
        for task in tasks:
            try:
                evidence.append(solver(task["public"]) == task["target"])
            except Exception:
                evidence.append(False)
        return evidence

    def run_round(
        self,
        *,
        capability: str,
        pack: EvaluationPack,
        parameter_space: Mapping[str, Iterable[object]],
        solver_factory: Callable[[Mapping[str, Any]], Callable[[Any], Any]],
        round_number: int,
        max_candidates: int = 8,
    ) -> ImprovementReport:
        baseline_config = self.registry.current
        baseline_solver = solver_factory(baseline_config)
        baseline = self.evaluator.evaluate(baseline_solver, pack)
        evidence = self._evidence(baseline_solver, pack.train)
        gap = self.detector.detect(capability, baseline.train, self.target_score, evidence)
        if gap is None:
            return ImprovementReport(round_number, False, 0, False, None, None, baseline.train, baseline.train, baseline.held_out)

        hypotheses = self.generator.generate(gap, parameter_space, max_candidates=max_candidates)
        best = None
        best_score: CandidateScore | None = None
        best_config = None
        for h in hypotheses:
            candidate_config = dict(baseline_config)
            candidate_config[h.parameter] = h.value
            candidate = self.evaluator.evaluate(solver_factory(candidate_config), pack)
            if not self.evaluator.improves(baseline, candidate):
                continue
            if best_score is None or (candidate.train, candidate.held_out) > (best_score.train, best_score.held_out):
                best = h
                best_score = candidate
                best_config = candidate_config

        if best is None or best_score is None or best_config is None:
            return ImprovementReport(round_number, True, len(hypotheses), False, None, None, baseline.train, baseline.train, baseline.held_out)

        gain = best_score.train - baseline.train
        rec = self.registry.promote(best_config, train_gain=gain, held_out_score=best_score.held_out, regression_ok=best_score.regression_ok)
        return ImprovementReport(
            round_number,
            True,
            len(hypotheses),
            rec.accepted,
            best.parameter if rec.accepted else None,
            best.value if rec.accepted else None,
            baseline.train,
            best_score.train if rec.accepted else baseline.train,
            best_score.held_out if rec.accepted else baseline.held_out,
        )

    def run_until_stable(
        self,
        *,
        capability: str,
        pack: EvaluationPack,
        parameter_space: Mapping[str, Iterable[object]],
        solver_factory: Callable[[Mapping[str, Any]], Callable[[Any], Any]],
        max_rounds: int = 4,
        max_candidates: int = 8,
    ) -> List[ImprovementReport]:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        reports = []
        for round_number in range(1, max_rounds + 1):
            report = self.run_round(
                capability=capability,
                pack=pack,
                parameter_space=parameter_space,
                solver_factory=solver_factory,
                round_number=round_number,
                max_candidates=max_candidates,
            )
            reports.append(report)
            if not report.promoted:
                break
        return reports
