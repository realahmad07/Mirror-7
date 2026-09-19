from dataclasses import dataclass

from phase280_sealed_candidate_eval import EvaluationPack
from phase287_self_generated_tasks import SelfGeneratedTaskSuite
from phase303_cross_capability_guard import RegressionGuard
from phase307_capability_adapter_contract import AdapterResult
from phase308_planning_frontier_adapter import PlanningFrontierAdapter
from phase309_language_frontier_adapter import LanguageFrontierAdapter
from phase310_reasoning_frontier_adapter import ReasoningFrontierAdapter
from phase311_adapter_registry import AdapterRegistry
from phase312_multicapability_frontier import build_default_frontier
from phase315_production_sandbox import ProductionSandbox
from phase317_production_self_redesign import ProductionSelfRedesign


BASE_SOURCE = """def solve(values):
    return values[-1]
"""


@dataclass(frozen=True)
class ProductionUpgradeEvent:
    capability: str
    accepted: bool
    score: float
    variant: str
    sandboxed: bool


class ProductionSequenceFrontierAdapter:
    """Source-redesign adapter whose every evaluation is forced through Docker."""

    name = "sequence_extrapolation"

    def __init__(self, sandbox: ProductionSandbox):
        if not isinstance(sandbox, ProductionSandbox):
            raise TypeError("sandbox must be a ProductionSandbox")
        if not sandbox.docker_available():
            raise RuntimeError("production sequence adapter requires Docker")
        self.sandbox = sandbox
        self.system = ProductionSelfRedesign(BASE_SOURCE, sandbox)
        self._round = 0

    @staticmethod
    def _pack(seed: int) -> EvaluationPack:
        suite = SelfGeneratedTaskSuite(seed)
        return suite.as_evaluation_pack(
            suite.linear_sequence(5),
            suite.linear_sequence(5),
            suite.linear_sequence(3),
        )

    @property
    def source(self) -> str:
        return self.system.source

    def improve(
        self, seed: int, rounds: int, candidates: int
    ) -> AdapterResult:
        if rounds < 1 or candidates < 1:
            raise ValueError("rounds and candidates must be positive")
        self._round += 1
        report = self.system.run_round(
            self._pack(seed),
            "solver.py",
            ["sequence extrapolation deficit"],
            self._round,
            candidates,
        )
        score = (
            report.selected_held_out
            if report.promoted
            else report.baseline_held_out
        )
        return AdapterResult(
            float(score),
            bool(report.promoted),
            report.selected_source or "no-change",
        )

    def rollback(self):
        return self.system.rollback()


class ProductionMultiCapabilityUpgrade:
    """End-to-end four-capability frontier with mandatory production sandboxing.

    Planning, language, and symbolic-reasoning adapters retain their existing
    bounded mechanisms. The source-changing sequence adapter is routed through
    the production Docker evaluator, then the shared regression guard decides
    whether the capability frontier advances.
    """

    def __init__(
        self,
        sandbox: ProductionSandbox | None = None,
        guard: RegressionGuard | None = None,
    ):
        self.sandbox = sandbox or ProductionSandbox()
        if not self.sandbox.docker_available():
            raise RuntimeError(
                "ProductionMultiCapabilityUpgrade requires a working Docker daemon"
            )

        self.frontier = build_default_frontier()
        self.sequence_adapter = ProductionSequenceFrontierAdapter(self.sandbox)
        self.adapters = AdapterRegistry(
            [
                self.sequence_adapter,
                PlanningFrontierAdapter(),
                LanguageFrontierAdapter(),
                ReasoningFrontierAdapter(),
            ]
        )
        self.scores = {
            target.name: target.baseline for target in self.frontier.targets
        }
        self.guard = guard or RegressionGuard()
        self.events: list[ProductionUpgradeEvent] = []

    @property
    def capability_names(self) -> tuple[str, ...]:
        return self.adapters.names

    @property
    def source(self) -> str:
        return self.sequence_adapter.source

    def step(
        self,
        seed: int = 7,
        rounds: int = 4,
        max_candidates: int = 1,
    ) -> ProductionUpgradeEvent | None:
        if rounds < 1 or max_candidates < 1:
            raise ValueError("rounds and max_candidates must be positive")

        target = self.frontier.top_gap()
        if target is None:
            return None

        adapter = self.adapters.get(target.name)
        result = adapter.improve(seed, rounds, max_candidates)

        baseline = dict(self.scores)
        candidate = dict(baseline)
        candidate[target.name] = float(result.score)

        guard = self.guard.evaluate(
            baseline, candidate, target.name
        )
        accepted = bool(result.changed and guard.accepted)

        if accepted:
            self.scores[target.name] = float(result.score)
            self.frontier.update(target.name, result.score)
        elif result.changed and hasattr(adapter, "rollback"):
            adapter.rollback()

        event = ProductionUpgradeEvent(
            target.name,
            accepted,
            float(result.score),
            result.variant,
            adapter is self.sequence_adapter,
        )
        self.events.append(event)
        return event

    def run(
        self,
        steps: int = 4,
        rounds: int = 4,
        max_candidates: int = 1,
    ) -> list[ProductionUpgradeEvent]:
        if steps < 1:
            raise ValueError("steps must be positive")
        events: list[ProductionUpgradeEvent] = []
        for offset in range(steps):
            event = self.step(
                seed=7 + offset,
                rounds=rounds,
                max_candidates=max_candidates,
            )
            if event is None:
                break
            events.append(event)
        return events
