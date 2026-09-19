from .mirror7_phase126 import CheckpointExecutor

def test_completed_steps_are_checkpointed():
 e=CheckpointExecutor(); ok,state,_=e.run([lambda x:x+1,lambda x:x+1],0,lambda x:x<3); assert ok and e.latest().state==2

def test_failure_leaves_latest_checkpoint():
 e=CheckpointExecutor(); ok,_,reason=e.run([lambda x:x+1]*3,0,lambda x:True,fail_at=1); assert not ok and e.latest().index==0 and "paused" in reason

def test_resume_continues_from_checkpoint():
 e=CheckpointExecutor(); e.run([lambda x:x+1]*3,0,lambda x:True,fail_at=1); ok,state,_=e.run([lambda x:x+1]*3,1,lambda x:True); assert ok and state==3

def test_verification_failure_stops_without_promoting():
 e=CheckpointExecutor(); ok,_,_=e.run([lambda x:x+1],0,lambda x:False); assert not ok and e.latest() is None