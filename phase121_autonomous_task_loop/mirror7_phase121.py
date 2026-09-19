from dataclasses import dataclass
from phase115_action_executor import ActionExecutor
from phase116_tool_composition import ToolComposer
from phase118_memory_retrieval import ContextMemory
from phase119_plan_verifier import PlanVerifier
from phase120_failure_recovery import FailureRecovery

@dataclass(frozen=True)
class LoopReport:
    completed: bool
    steps: int
    recovered: int
    memory_items: int
    reason: str

class AutonomousTaskLoop:
    """Bounded task execution: verify -> execute -> recover -> remember -> continue."""
    def __init__(self,budget=16):
        self.budget=budget
        self.actions=ActionExecutor()
        self.composer=ToolComposer()
        self.verifier=PlanVerifier()
        self.recovery=FailureRecovery()
        self.memory=ContextMemory()
    def run(self,steps:list[tuple[str,dict]],context:set[str],goal_check):
        if len(steps)>self.budget:
            return LoopReport(False,0,0,len(self.memory.items),"plan exceeds budget")
        check=self.verifier.verify([lambda n: n + 1 for _ in steps],0,lambda n:n==len(steps),self.budget)
        if not check.valid:
            return LoopReport(False,0,0,len(self.memory.items),"plan verification failed")
        if any(name not in self.actions.actions for name,_ in steps):
            return LoopReport(False,0,0,len(self.memory.items),"plan verification failed")
        completed=0
        recovered=0
        state=None
        for name,args in steps:
            rep=self.actions.execute(name,args)
            if not rep.verified:
                def retry(_):
                    attempt=self.actions.execute(name,args)
                    return attempt.output if attempt.verified else None
                rr=self.recovery.run(retry,state,retries=1)
                if not rr.success:
                    return LoopReport(False,completed,recovered,len(self.memory.items),"step failed after recovery")
                recovered+=1
                state=rr.state
                rep=None
            if rep is not None:
                state=rep.output
            completed+=1
            self.memory.remember(name,state,context,.9 if recovered==0 else .7)
            if goal_check(state):
                return LoopReport(True,completed,recovered,len(self.memory.items),"goal reached")
        return LoopReport(False,completed,recovered,len(self.memory.items),"steps exhausted")
