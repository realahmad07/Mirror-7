from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class RecoveryReport:
    success: bool
    attempts: int
    state: Any
    reason: str

class FailureRecovery:
    """Retries bounded failures and returns the last stable state if recovery fails."""
    def run(self,action:Callable[[Any],Any],state:Any,retries:int=2,rollback:Callable[[Any],Any]|None=None)->RecoveryReport:
        stable=state; attempts=0
        for _ in range(max(0,retries)+1):
            attempts+=1
            try:
                out=action(stable)
                if out is not None: return RecoveryReport(True,attempts,out,"recovered")
            except Exception:
                pass
        restored=rollback(stable) if rollback else stable
        return RecoveryReport(False,attempts,restored,"recovery failed; stable state retained")
