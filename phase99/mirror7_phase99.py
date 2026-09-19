from __future__ import annotations
from dataclasses import dataclass
import hashlib,json

@dataclass(frozen=True)
class CaseResult:
    case_id:str
    expected:object
    actual:object
    passed:bool
    valid:bool

@dataclass(frozen=True)
class EvaluationReport:
    suite_hash:str
    total:int
    passed:int
    invalid:int
    held_out_passed:int

class IndependentEvaluator:
    """Frozen black-box evaluation with held-out and invalid-response accounting."""
    def __init__(self,cases):
        if not cases: raise ValueError("empty evaluation suite")
        clean=[]
        for case in cases:
            if not all(k in case for k in ("id","input","expected")): raise ValueError("malformed case")
            clean.append({"id":str(case["id"]),"input":case["input"],"expected":case["expected"],"held_out":bool(case.get("held_out",False))})
        self.cases=tuple(clean)
        payload=json.dumps(self.cases,sort_keys=True,separators=(",",":"),default=repr).encode()
        self.suite_hash=hashlib.sha256(payload).hexdigest()

    def run(self,predict):
        results=[]
        for case in self.cases:
            try:
                actual=predict(case["input"]); valid=actual is not None
            except Exception:
                actual=None; valid=False
            results.append(CaseResult(case["id"],case["expected"],actual,valid and actual==case["expected"],valid))
        return tuple(results)

    def report(self,results):
        passed=sum(r.passed for r in results)
        invalid=sum(not r.valid for r in results)
        held=sum(r.passed for r,c in zip(results,self.cases) if c["held_out"])
        return EvaluationReport(self.suite_hash,len(self.cases),passed,invalid,held)
