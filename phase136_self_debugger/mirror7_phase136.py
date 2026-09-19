from dataclasses import dataclass
@dataclass(frozen=True)
class DebugProposal:
    fingerprint:str; change:str; candidate_score:float; accepted:bool
class SelfDebugger:
    def diagnose(self,fingerprint,failures):
        return DebugProposal(fingerprint,"investigate "+";".join(failures),0.0,False) if failures else None
    def gate(self,fingerprint,failures,baseline,candidate,held_out,regression_ok,cost,max_cost=10):
        accepted=bool(failures and candidate>baseline and held_out>=baseline and regression_ok and cost<=max_cost)
        return DebugProposal(fingerprint,"candidate revision",candidate,accepted)
