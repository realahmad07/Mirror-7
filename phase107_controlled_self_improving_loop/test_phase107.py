from .mirror7_phase107 import ControlledSelfImprovingLoop

def test_insufficient_context_drives_clarification():
    r=ControlledSelfImprovingLoop().step({},"x",1,"sensor",[])
    assert r.needs_context

def test_validated_learning_requires_repetition():
    l=ControlledSelfImprovingLoop(); r=l.step({"goal":"x"},"x",1,"a",[]); assert not r.knowledge_accepted
    r=l.step({"goal":"x"},"x",1,"b",[]); assert r.knowledge_accepted

def test_safe_improvement_can_be_adopted():
    l=ControlledSelfImprovingLoop(); r=l.step({"goal":"x"},"x",1,"a",["failure"],(.2,.1,2.0,(True,True))); assert r.proposal_created and r.improvement_adopted

def test_bad_improvement_is_not_adopted():
    l=ControlledSelfImprovingLoop(); r=l.step({"goal":"x"},"x",1,"a",["failure"],(.2,.1,2.0,(True,False))); assert r.proposal_created and not r.improvement_adopted

def test_resource_gate_stays_bounded():
    l=ControlledSelfImprovingLoop()
    for i in range(100): l.step({"goal":"x"},f"x{i}",i,"sensor",[])
    assert len(l.resources.memory)<=32
