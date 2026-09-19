from .mirror7_phase191_200 import *

def h(action,before,after,success=True):
    return ActionOutcome(action,before,after,success)

def test_191_opaque_actions():
    a=AffordanceLearner(); a.observe(h(b"\\x01",[0],[1]))
    assert a.candidate_actions()==(b"\\x01",)

def test_192_effect_discovery():
    m=discover_affordance([h(b"a",[0],[1]),h(b"a",[0],[1])])
    assert m.successes==2 and m.effect_counts

def test_193_precondition_discovery():
    m=discover_affordance([h(b"a",[0],[1]),h(b"a",[2],[2],False)])
    assert len(m.precondition_counts)==2

def test_194_failure_is_recorded():
    m=discover_affordance([h(b"a",[0],[0],False)])
    assert m.failures==1 and m.success_rate==0

def test_195_safe_selection():
    a=AffordanceLearner()
    a.observe(h(b"a",[0],[0],False))
    a.observe(h(b"b",[0],[1],True))
    assert select_safe_action(a,[b"a",b"b"],[0])==b"b"

def test_196_unknown_probe():
    a=AffordanceLearner()
    assert select_safe_action(a,[b"b",b"a"],[0])==b"a"

def test_197_context_gate():
    a=AffordanceLearner()
    a.observe(h(b"a",[0],[1]))
    assert a.choose([b"a"],[9]) is None

def test_198_goal_match():
    a=AffordanceLearner()
    a.observe(h(b"a",[0],[1]))
    a.observe(h(b"b",[0],[2]))
    assert a.choose([b"a",b"b"],[0],goal_effect=((0,2),))==b"b"

def test_199_composition():
    a=AffordanceLearner()
    a.observe(h(b"a",[0],[1]))
    a.observe(h(b"b",[1],[2]))
    assert compose_actions(a,[b"a",b"b"],[0])[-1]==(2,)

def test_200_fail_closed_composition():
    a=AffordanceLearner()
    a.observe(h(b"a",[0],[1]))
    assert compose_actions(a,[b"a",b"b"],[0]) is None

def test_action_type_fail_closed():
    try: canonical_observation(object())
    except TypeError: pass
    else: assert False

def test_deterministic_action_identity():
    a=AffordanceLearner()
    a.observe(h(b"x",[0],[1]))
    a.observe(h(b"y",[0],[2]))
    assert set(a.candidate_actions())=={b"x",b"y"}
