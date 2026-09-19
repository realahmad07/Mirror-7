from phase294_source_evaluator import SealedSourceEvaluator
from phase315_production_sandbox import ProductionSandbox

class ProductionSealedSourceEvaluator(SealedSourceEvaluator):
    """Sealed evaluator that refuses to execute candidates outside the Docker sandbox."""

    def __init__(self, sandbox: ProductionSandbox | None = None):
        self.sandbox = sandbox or ProductionSandbox()
        if not self.sandbox.docker_available():
            raise RuntimeError("production sandbox requires a working Docker daemon")
        super().__init__(sandbox=self.sandbox)
