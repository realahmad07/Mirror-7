from .mirror7_phase115 import ActionExecutor

def test_valid_action_is_verified():
 e=ActionExecutor(); e.register("add",lambda a:a["x"]+1,lambda a:a["x"]>=0,lambda o:o==3); r=e.execute("add",{"x":2}); assert r.executed and r.verified

def test_precondition_blocks_action():
 e=ActionExecutor(); e.register("x",lambda a:1,lambda a:False,lambda o:True); r=e.execute("x",{}); assert not r.executed

def test_postcondition_blocks_bad_result():
 e=ActionExecutor(); e.register("x",lambda a:1,postcondition=lambda o:o==2); r=e.execute("x",{}); assert r.executed and not r.verified

def test_unknown_action_fails_closed():
 assert not ActionExecutor().execute("unknown",{}).executed
