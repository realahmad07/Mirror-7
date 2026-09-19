from phase287_self_generated_tasks import SelfGeneratedTaskSuite
from phase297_autonomous_redesign import AutonomousRedesignEngine

class IntegratedSelfRedesign:
    """Feeds fresh self-generated evaluation tasks into the bounded source redesign engine."""
    BASE_SOURCE="def solve(values):\n    return values[-1]\n"

    def run(self,seed:int=7,max_rounds:int=4):
        suite=SelfGeneratedTaskSuite(seed)
        train=suite.linear_sequence(5)
        held=suite.linear_sequence(5)
        regression=suite.linear_sequence(3)
        pack=suite.as_evaluation_pack(train,held,regression)
        engine=AutonomousRedesignEngine(self.BASE_SOURCE)
        reports=engine.run_until_stable(pack,"solver.py",["sequence extrapolation deficit"],max_rounds=max_rounds)
        return engine,reports
