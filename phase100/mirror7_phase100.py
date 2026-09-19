from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

from phase94 import OpenEndedResearchLoop
from phase95 import AssociativeMemory
from phase96 import ContinualConsolidator
from phase97 import UncertaintyAwarePlanner
from phase98 import MixedViewGrounder
from phase99 import IndependentEvaluator

@dataclass(frozen=True)
class BoundaryReport:
    suite_hash:str
    passed:int
    total:int
    invalid:int
    held_out_passed:int
    memory_items:int
    consolidated_rules:int
    concepts:int
    integrated:bool

class Mirror7FinalBoundary:
    """Bounded end-to-end integration plus frozen black-box evaluation boundary."""
    def __init__(self,cases:Sequence[dict]):
        self.base=OpenEndedResearchLoop()
        self.memory=AssociativeMemory()
        self.consolidator=ContinualConsolidator()
        self.planner=None
        self.grounder=MixedViewGrounder()
        self.evaluator=IndependentEvaluator(cases)

    def learn_experience(self,signature,state,action,result,confidence=1.0):
        return self.memory.add(signature,state,action,result,confidence)

    def consolidate(self,key,value):
        return self.consolidator.observe(tuple(key),value)

    def ground(self,view):
        return self.grounder.observe(view)

    def plan(self,transitions,actions,start,goal):
        self.planner=UncertaintyAwarePlanner(transitions,actions)
        return self.planner.plan(start,goal)

    def evaluate(self,predict:Callable[[object],object]):
        results=self.evaluator.run(predict); report=self.evaluator.report(results)
        integrated=all(x is not None for x in (self.base,self.memory,self.consolidator,self.grounder))
        return BoundaryReport(report.suite_hash,report.passed,report.total,report.invalid,report.held_out_passed,len(self.memory.items),len(self.consolidator.rules),len(self.grounder.concepts),integrated)

    def fail_closed(self):
        return {"memory":len(self.memory.items),"rules":len(self.consolidator.rules),"concepts":len(self.grounder.concepts)}
