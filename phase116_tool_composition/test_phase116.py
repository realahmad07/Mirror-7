from .mirror7_phase116 import ToolComposer

def test_composed_stages_pass_data_forward():
 r=ToolComposer().run([lambda x:x+1,lambda x:x*2],0); assert r.success and r.outputs[-1]==2

def test_failure_stops_downstream():
 hit=[]
 def bad(x): raise RuntimeError("no")
 def never(x): hit.append(1); return x
 r=ToolComposer().run([lambda x:x+1,bad,never],0); assert not r.success and r.failed_at==1 and not hit

def test_none_is_not_promoted():
 r=ToolComposer().run([lambda x:None],1); assert not r.success

def test_empty_composition_is_valid_noop():
 r=ToolComposer().run([],3); assert r.success and r.outputs==()
