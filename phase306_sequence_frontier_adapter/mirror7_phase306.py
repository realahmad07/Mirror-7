from phase280_sealed_candidate_eval import EvaluationPack
from phase287_self_generated_tasks import SelfGeneratedTaskSuite
from phase294_source_evaluator import SealedSourceEvaluator
from phase297_autonomous_redesign import AutonomousRedesignEngine
from phase299_capability_frontier import CapabilityFrontier, CapabilityTarget
from phase305_frontier_upgrade_loop import FrontierUpgradeLoop, UpgradeOutcome

class SequenceFrontierAdapter:
    """Connects the existing source-redesign engine to the capability frontier."""
    def __init__(self):
        self.engine=AutonomousRedesignEngine(AutonomousRedesignEngine.__init__.__defaults__[0] if False else "def solve(values):
    return values[-1]
")

    def _pack(self,seed:int)->EvaluationPack:
        s=SelfGeneratedTaskSuite(seed)
        return s.as_evaluation_pack(s.linear_sequence(5),s.linear_sequence(5),s.linear_sequence(3))

    def evaluate(self,seed:int)->float:
        return SealedSourceEvaluator().evaluate(self.engine.source,self._pack(seed)).held_out

    def improve(self,seed:int,rounds:int,candidates:int)->UpgradeOutcome:
        pack=self._pack(seed)
        report=self.engine.run_round(pack,"solver.py",["sequence extrapolation deficit"],1,candidates)
        return UpgradeOutcome(report.selected_held_out,report.promoted,report.selected_source or "no-change",{})

class SequenceFrontierDemo:
    """Ready-to-run example of the targeted frontier loop using a real Mirror 7 adapter."""
    def run(self,steps:int=4):
        adapter=SequenceFrontierAdapter()
        frontier=CapabilityFrontier([CapabilityTarget("sequence_extrapolation",0.0,1.0,1.0)])
        loop=FrontierUpgradeLoop(frontier,{"sequence_extrapolation":adapter})
        return loop.run(steps=steps)
