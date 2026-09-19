from dataclasses import dataclass
from typing import List

from phase280_sealed_candidate_eval import EvaluationPack
from phase281_promotion_rollback import PromotionRegistry
from phase283_algorithm_variant_space import AlgorithmVariant, AlgorithmVariantSpace
from phase284_algorithm_mutation import AlgorithmMutator
from phase285_algorithm_independent_eval import AlgorithmIndependentEvaluator
from phase288_resource_aware_selection import ResourceAwareSelector
from phase289_improvement_memory import ImprovementMemory

@dataclass(frozen=True)
class MetaImprovementReport:
    round:int
    candidates_seen:int
    candidates_skipped_from_memory:int
    promoted:bool
    variant:AlgorithmVariant|None

class MetaImprovementEngine:
    """Unified bounded search: mutate -> remember -> evaluate -> select -> promote."""
    def __init__(self,initial:AlgorithmVariant,space:AlgorithmVariantSpace|None=None):
        self.space=space or AlgorithmVariantSpace()
        if not self.space.validate(initial): raise ValueError("invalid initial variant")
        self.mutator=AlgorithmMutator(self.space)
        self.evaluator=AlgorithmIndependentEvaluator(self.space)
        self.selector=ResourceAwareSelector()
        self.memory=ImprovementMemory()
        self.registry=PromotionRegistry({"algorithm":initial})

    @property
    def current(self)->AlgorithmVariant:
        return self.registry.current["algorithm"]

    def run_round(self,pack:EvaluationPack,round_number:int,max_candidates:int=8)->MetaImprovementReport:
        baseline=self.evaluator.evaluate(self.current,pack)
        self.memory.add(self.current.operations,True,baseline.held_out,"active baseline")
        mutations=self.mutator.mutate(self.current,max_children=max_candidates)
        options=[]; skipped=0
        for mutation in mutations:
            if self.memory.seen(mutation.child.operations):
                skipped+=1
                continue
            score=self.evaluator.evaluate(mutation.child,pack)
            self.memory.add(mutation.child.operations,False,score.held_out,"evaluated")
            if not self.evaluator.improves(baseline,score): continue
            utility=self.selector.make(
                mutation.child.name,
                score.train-baseline.train,
                score.held_out,
                len(mutation.child.operations),
                len(mutation.child.operations),
            )
            options.append((utility,score))
        if not options:
            return MetaImprovementReport(round_number,len(mutations),skipped,False,None)
        options.sort(key=lambda x:x[0].utility,reverse=True)
        selected=options[0][1]
        rec=self.registry.promote(
            {"algorithm":selected.variant},
            train_gain=selected.train-baseline.train,
            held_out_score=selected.held_out,
            regression_ok=selected.regression_ok,
        )
        if rec.accepted:
            self.memory.add(selected.variant.operations,True,selected.held_out,"promoted")
        return MetaImprovementReport(round_number,len(mutations),skipped,rec.accepted,selected.variant if rec.accepted else None)

    def run_until_stable(self,pack:EvaluationPack,max_rounds:int=4,max_candidates:int=8)->List[MetaImprovementReport]:
        if max_rounds<1 or max_candidates<1: raise ValueError("bounds must be positive")
        reports=[]
        for n in range(1,max_rounds+1):
            r=self.run_round(pack,n,max_candidates)
            reports.append(r)
            if not r.promoted: break
        return reports
