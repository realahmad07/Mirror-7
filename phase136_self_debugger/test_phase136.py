from .mirror7_phase136 import SelfDebugger
def test_empty_failure_has_no_proposal(): assert SelfDebugger().diagnose("x",[]) is None
def test_bad_candidate_rejected(): assert not SelfDebugger().gate("x",["f"],.8,.7,.9,True,1).accepted
def test_valid_candidate_accepted(): assert SelfDebugger().gate("x",["f"],.8,.9,.85,True,1).accepted
