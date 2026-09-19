from .mirror7_phase264 import EndToEndAgent

def test_full_integration_loop():
    agent=EndToEndAgent()
    agent.observe(b"A","inc",b"B")
    agent.observe(b"B","inc",b"C")
    assert agent.act(b"A",b"C")=="inc"
    assert agent.act(b"A",b"Z") is None


def test_integration_three_seeds_same_structure_different_symbols():
    for a,b,c in ((b"A",b"B",b"C"),(b"D",b"E",b"F"),(b"G",b"H",b"I")):
        agent=EndToEndAgent(); agent.observe(a,"inc",b); agent.observe(b,"inc",c)
        assert agent.act(a,c)=="inc"

def test_integration_held_out_goal_abstains():
    agent=EndToEndAgent(); agent.observe(b"A","inc",b"B"); agent.observe(b"B","inc",b"C")
    assert agent.act(b"A",b"Q") is None
