from .mirror7_phase113 import AgentMemory

def test_remember_and_recall():
    m=AgentMemory(); m.remember("goal","ship",.9); assert m.recall("goal").value=="ship"

def test_confidence_does_not_drop_from_repeated_lower_observation():
    m=AgentMemory(); m.remember("x",1,.9); m.remember("x",2,.4); assert m.recall("x").confidence==.9

def test_low_confidence_can_be_filtered():
    m=AgentMemory(); m.remember("x",1,.3); assert m.recall("x",.5) is None

def test_memory_is_bounded():
    m=AgentMemory(2); m.remember("a",1,.1); m.remember("b",2,.9); m.remember("c",3,.8); assert len(m.items)<=2 and "a" not in m.items
