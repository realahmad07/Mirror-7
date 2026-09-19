from .mirror7_phase135 import ResearchScheduler,ResearchTask
def test_high_value_task_selected():
 a=ResearchTask("a",.9,.9,.9,1); b=ResearchTask("b",.2,.9,.9,1); assert ResearchScheduler().select([a,b],1).name=="a"
def test_budget_filters(): assert ResearchScheduler().select([ResearchTask("x",1,1,1,2)],1) is None
def test_empty_safe(): assert ResearchScheduler().select([],1) is None
