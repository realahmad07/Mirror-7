from .mirror7_phase295 import RedesignProposer

def test_sequence_failure_generates_multiple_hypotheses():
    xs=RedesignProposer().propose("solver.py",["sequence extrapolation failed"])
    assert len(xs)==3

def test_proposals_are_patch_plans():
    xs=RedesignProposer().propose("solver.py",["delta failure"])
    assert all(x.operations for x in xs)

def test_unknown_failure_generates_none():
    assert RedesignProposer().propose("solver.py",["unrelated"])==[]

def test_empty_failure_generates_none():
    assert RedesignProposer().propose("solver.py",[])==[]

def test_three_seed_generation_is_deterministic():
    for _ in (2,5,8):
        assert RedesignProposer().propose("solver.py",["sequence failure"])==RedesignProposer().propose("solver.py",["sequence failure"])

def test_duplicate_hypotheses_removed():
    xs=RedesignProposer().propose("solver.py",["sequence","sequence"])
    keys=[tuple((o.kind,o.old,o.new) for o in p.operations) for p in xs]
    assert len(keys)==len(set(keys))

def test_constant_control_is_available():
    assert len(RedesignProposer().propose("solver.py",["constant error"]))==1

def test_target_required():
    assert RedesignProposer().propose("",["sequence"])==[]

def test_patch_old_expression_is_exact():
    xs=RedesignProposer().propose("solver.py",["delta"])
    assert all(p.operations[0].old=="values[-1]" for p in xs)
