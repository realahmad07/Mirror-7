from .mirror7_phase110 import GuardedTools

def test_allowlisted_tool_executes_and_verifies():
    g=GuardedTools(); g.register("add",lambda a:a["x"]+a["y"],lambda o:o==5); r=g.call("add",{"x":2,"y":3}); assert r.success and r.verified

def test_unknown_tool_fails_closed():
    assert not GuardedTools().call("shell",{}).success

def test_bad_output_is_not_verified():
    g=GuardedTools(); g.register("x",lambda a:"bad",lambda o:isinstance(o,int)); r=g.call("x",{}); assert not r.verified

def test_exception_is_bounded_and_reported():
    g=GuardedTools(); g.register("x",lambda a:1/0,lambda o:True); r=g.call("x",{},retries=2); assert not r.success and "ZeroDivisionError" in r.reason
