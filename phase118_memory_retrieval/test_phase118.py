from .mirror7_phase118 import ContextMemory

def test_related_context_is_retrieved_first():
 m=ContextMemory(); m.remember("a",1,{"ship","cpu"},.9); m.remember("b",2,{"cook"},.9); assert m.retrieve({"ship","cpu"})[0].key=="a"

def test_confidence_contributes_to_score():
 m=ContextMemory(); m.remember("low",1,{"x"},.2); m.remember("high",2,{"x"},.9); assert m.retrieve({"x"})[0].key=="high"

def test_memory_is_bounded():
 m=ContextMemory(2); [m.remember(str(i),i,{str(i)},1) for i in range(4)]; assert len(m.items)==2

def test_limit_is_respected():
 m=ContextMemory(); [m.remember(str(i),i,{"x"},1) for i in range(4)]; assert len(m.retrieve({"x"},limit=2))==2
