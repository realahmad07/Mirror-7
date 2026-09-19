from dataclasses import dataclass
from typing import Any, Sequence
from phase101_context_inquiry import ContextInquiry
from phase102_evidence_ledger import EvidenceLedger,Evidence
from phase103_validated_learning import ValidatedLearner
from phase104_improvement_proposer import ImprovementProposer
from phase105_self_mod_gate import SelfImprovementGate
from phase106_resource_governance import ResourceGovernor

@dataclass(frozen=True)
class LoopReport:
    needs_context: bool
    knowledge_accepted: bool
    proposal_created: bool
    improvement_adopted: bool
    resource_ok: bool

class ControlledSelfImprovingLoop:
    def __init__(self):
        self.inquiry=ContextInquiry(["goal"])
        self.ledger=EvidenceLedger()
        self.learn=ValidatedLearner()
        self.proposer=ImprovementProposer()
        self.gate=SelfImprovementGate(min_gain=.01,max_cost=10)
        self.resources=ResourceGovernor(memory_limit=32,step_limit=64)

    def step(self,context:dict, claim:str, value:Any, source:str, failures:Sequence[str], improvement:tuple[float,float,float,Sequence[bool]]|None=None)->LoopReport:
        q=self.inquiry.assess(context)
        self.ledger.add(Evidence(claim,value,source,.8,True))
        accepted=self.learn.observe(claim,value) is not None
        p=None
        adopted=False
        if failures:
            gain,risk,cost,regression=improvement or (0.0,1.0,999.0,(False,))
            p=self.proposer.propose("learner",failures,.5,gain,risk,cost)
            if p:
                rep=self.gate.evaluate(.5,.5+gain,.5+gain,regression,cost)
                adopted=rep.accepted
        rr=self.resources.remember(claim,value,priority=.8)
        return LoopReport(q is not None,accepted,p is not None,adopted,rr.admitted)
