from .mirror7_phase114 import IntegratedAgent
from phase111_task_decomposition import TaskNode

def test_full_loop_integrates_grounding_learning_plan_tool_monitor_memory():
    a=IntegratedAgent(); a.tools.register("double",lambda x:x["n"]*2,lambda y:y==6)
    r=a.run({"goal":"ship","platform":"cpu"},"answer",3,"user",[TaskNode("discover"),TaskNode("build",("discover",))],"double",{"n":3})
    assert r.knowledge_accepted and r.plan==("discover","build") and r.tool_verified and not r.ask_context

def test_missing_context_is_exposed():
    r=IntegratedAgent().run({"goal":"ship"},"x",1,"user",[TaskNode("a")])
    assert r.needs_context

def test_unverified_tool_reduces_confidence_and_requests_caution():
    a=IntegratedAgent(); a.tools.register("bad",lambda x:"wrong",lambda y:False)
    r=a.run({"goal":"ship","platform":"cpu"},"x",1,"u",[TaskNode("a")],"bad",{})
    assert not r.tool_verified and r.ask_context

def test_plan_rejects_cycles():
    try: IntegratedAgent().run({"goal":"ship","platform":"cpu"},"x",1,"u",[TaskNode("a",("b",)),TaskNode("b",("a",))])
    except ValueError: pass
    else: assert False

def test_unknown_tool_is_not_treated_as_success():
    a=IntegratedAgent(); r=a.run({"goal":"ship","platform":"cpu"},"x",1,"u",[TaskNode("a")],"missing",{})
    assert not r.tool_verified

def test_memory_bound_is_preserved_after_many_runs():
    a=IntegratedAgent()
    for i in range(100): a.run({"goal":"g","platform":"cpu"},str(i),i,"u",[TaskNode("a")])
    assert len(a.memory.items)<=64
