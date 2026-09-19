
from .mirror7_phase268 import CompositionalSemantics

def test_three_progressive_compositions():
    s=CompositionalSemantics()
    s.learn_primitive("move", "MOVE")
    s.learn_primitive("turn", "TURN")
    assert s.execute_plan(s.parse("move")) == ("MOVE",)
    assert s.execute_plan(s.parse("move then turn")) == ("MOVE","TURN")
    assert s.execute_plan(s.parse("move and turn then move")) == ("MOVE","TURN","MOVE")

def test_held_out_surface_forms():
    s=CompositionalSemantics()
    s.learn_primitive("step forward", "ADVANCE")
    assert s.execute_plan(s.parse("forward")) == ("ADVANCE",)

def test_reference_resolution():
    s=CompositionalSemantics()
    s.learn_primitive("open", "OPEN")
    s.parse("open")
    assert s.execute_plan(s.parse("it")) == ("OPEN",)

def test_ambiguity_abstains():
    s=CompositionalSemantics()
    s.learn_primitive("left", "LEFT")
    s.learn_primitive("turn", "TURN")
    try:
        s.parse("left turn")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ambiguity")

def test_unknown_fails_closed():
    s=CompositionalSemantics()
    try:
        s.parse("jump")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown instruction must not be invented")

def test_seed_stability():
    for _seed in (1,2,3):
        s=CompositionalSemantics()
        s.learn_primitive("go","GO")
        assert s.execute_plan(s.parse("go then go"))==("GO","GO")
