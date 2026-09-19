from .mirror7_phase211_220 import *

def test_211_goal_tree():
    r=HierarchicalReasoner(); g=GoalNode("root",[GoalNode("a"),GoalNode("b")])
    assert [x.name for x in g.children]==["a","b"]

def test_212_skill_abstraction():
    r=HierarchicalReasoner(); r.register_skill("build",["a","b"])
    g=r.expand(GoalNode("build"))
    assert [x.name for x in g.children]==["a","b"]

def test_213_topological_order():
    r=HierarchicalReasoner(); g=GoalNode("r",[GoalNode("b",depends_on=("a",)),GoalNode("a")])
    assert r.topological(g)==("r","a","b")

def test_214_verification_gate():
    r=HierarchicalReasoner(); g=GoalNode("a")
    assert r.verify(g,lambda n: True) and g.verified

def test_215_unverified_child_blocks_parent():
    r=HierarchicalReasoner(); c=GoalNode("c"); g=GoalNode("g",[c])
    assert not r.verify(g,lambda n: True)

def test_216_budget():
    r=HierarchicalReasoner(1); r.expand(GoalNode("x"))
    try: r.expand(GoalNode("y"))
    except RuntimeError: pass
    else: assert False

def test_217_composition():
    r=HierarchicalReasoner(); r.register_skill("a",["x"]); r.register_skill("b",["y"])
    g=r.compose(["a","b"])
    assert len(g.children)==2

def test_218_transfer():
    r=HierarchicalReasoner(); r.register_skill("a",["x"])
    g=r.transfer("a",{"a":"q","x":"z"})
    assert g.name=="q" and g.children[0].name=="z"

def test_219_fail_closed_cycle():
    r=HierarchicalReasoner(); g=GoalNode("a",depends_on=("a",))
    try: r.topological(g)
    except ValueError: pass
    else: assert False

def test_220_integrated_execution():
    r=HierarchicalReasoner(20); g=GoalNode("root",[GoalNode("a"),GoalNode("b")])
    assert r.execute(g,lambda n: True)
    assert g.verified
