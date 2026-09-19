from phase297_autonomous_redesign import AutonomousRedesignEngine
from phase315_production_sandbox import ProductionSandbox
from phase316_sandboxed_evaluator import ProductionSealedSourceEvaluator

class ProductionSelfRedesign:
    """Bounded self-redesign controller with mandatory sandboxed evaluation."""

    def __init__(self, initial_source: str, sandbox: ProductionSandbox | None = None):
        self.sandbox = sandbox or ProductionSandbox()
        if not self.sandbox.docker_available():
            raise RuntimeError(
                "ProductionSelfRedesign refuses to run without Docker sandboxing"
            )
        self.evaluator = ProductionSealedSourceEvaluator(self.sandbox)
        self.engine = AutonomousRedesignEngine(
            initial_source,
            evaluator=self.evaluator,
        )

    @property
    def source(self) -> str:
        return self.engine.source

    def run_round(
        self,
        pack,
        capability: str,
        failures,
        round_number: int = 1,
        max_candidates: int = 8,
    ):
        return self.engine.run_round(
            pack, capability, failures, round_number, max_candidates
        )

    def rollback(self, version: int | None = None):
        return self.engine.rollback(version)
