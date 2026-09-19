from .mirror7_phase120 import FailureRecovery

def test_retry_recovers_transient_failure():
 n=[0]
 def a(x): n[0]+=1; return x+1 if n[0]>=2 else None
 r=FailureRecovery().run(a,0,2); assert r.success and r.attempts==2

def test_failure_returns_stable_state():
 r=FailureRecovery().run(lambda x:None,5,1); assert not r.success and r.state==5

def test_rollback_can_restore_state():
 r=FailureRecovery().run(lambda x:1/0,5,1,rollback=lambda x:0); assert not r.success and r.state==0

def test_retry_budget_is_hard():
 n=[0]
 def a(x): n[0]+=1; return None
 r=FailureRecovery().run(a,0,3); assert not r.success and r.attempts==4
