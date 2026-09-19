"""Phase 82: fail-closed invariant gate for core research mechanisms."""
from dataclasses import dataclass

@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: tuple[str,...]
    failures: tuple[str,...]

class RegressionGate:
    def __init__(self): self.checks=[]; self.failures=[]
    def check(self,name,condition):
        self.checks.append(name)
        if not condition: self.failures.append(name)
        return bool(condition)
    def finish(self): return GateResult(not self.failures,tuple(self.checks),tuple(self.failures))
