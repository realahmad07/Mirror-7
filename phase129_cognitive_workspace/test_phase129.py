from .mirror7_phase129 import CognitiveWorkspace
def test_publish_and_read():
 w=CognitiveWorkspace(); w.publish("state",1,.9,"sensor"); assert w.read("state")[0].value==1
def test_confidence_filter():
 w=CognitiveWorkspace(); w.publish("x",1,.2); assert not w.read("x",.5)
def test_memory_bound():
 w=CognitiveWorkspace(2); [w.publish("x",i,i/10) for i in range(5)]; assert len(w.snapshot())==2
