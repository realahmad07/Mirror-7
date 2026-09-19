from dataclasses import dataclass
from typing import List, Sequence
from phase294_source_evaluator import SealedSourceEvaluator, SourceEvaluation
from phase295_redesign_proposer import RedesignProposer
from phase296_source_promotion import SourcePromotionRegistry

@dataclass(frozen=True)
class RedesignReport:
    round:int
    baseline_train:float
    baseline_held_out:float
    candidates_tested:int
    promoted:bool
    selected_source:str|None
    selected_train:float
    selected_held_out:float

class AutonomousRedesignEngine:
    """Runs bounded source redesign under sealed evaluation and promotion gates."""
    def __init__(self,initial_source:str):
        self.evaluator=SealedSourceEvaluator()
        self.proposer=RedesignProposer()
        self.registry=SourcePromotionRegistry(initial_source)

    @property
    def source(self)->str: return self.registry.source

    def run_round(self,pack,capability:str,failures:Sequence[str],round_number:int,max_candidates:int=8)->RedesignReport:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        baseline=self.evaluator.evaluate(self.source,pack)
        if baseline.train>=1.0:
            return RedesignReport(round_number,baseline.train,baseline.held_out,0,False,None,baseline.train,baseline.held_out)
        plans=self.proposer.propose(capability,failures)[:max_candidates]
        best=None
        for plan in plans:
            from phase293_patch_application import PatchApplier
            result=PatchApplier().apply(self.source,plan)
            if not result.applied: continue
            score=self.evaluator.evaluate(result.source,pack)
            if not self.evaluator.improves(baseline,score): continue
            if best is None or (score.train,score.held_out)>(best[0].train,best[0].held_out):
                best=(score,result.source)
        if best is None:
            return RedesignReport(round_number,baseline.train,baseline.held_out,len(plans),False,None,baseline.train,baseline.held_out)
        score,source=best
        rec=self.registry.promote(source,baseline,score)
        return RedesignReport(round_number,baseline.train,baseline.held_out,len(plans),rec.accepted,source if rec.accepted else None,score.train if rec.accepted else baseline.train,score.held_out if rec.accepted else baseline.held_out)

    def run_until_stable(self,pack,capability:str,failures:Sequence[str],max_rounds:int=4,max_candidates:int=8)->List[RedesignReport]:
        if max_rounds<1 or max_candidates<1: raise ValueError("bounds must be positive")
        reports=[]
        for n in range(1,max_rounds+1):
            r=self.run_round(pack,capability,failures,n,max_candidates)
            reports.append(r)
            if not r.promoted: break
        return reports

    def rollback(self,version:int|None=None):
        return self.registry.rollback(version)
