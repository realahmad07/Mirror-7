from phase280_sealed_candidate_eval import EvaluationPack
from .mirror7_phase297 import AutonomousRedesignEngine

PACK=EvaluationPack(
    [{"public":{"values":[0,2]},"target":4},{"public":{"values":[3,6]},"target":9},{"public":{"values":[10,14]},"target":18}],
    [{"public":{"values":[-5,-2]},"target":1},{"public":{"values":[8,12]},"target":16},{"public":{"values":[20,25]},"target":30}],
    [{"public":{"values":[1,1]},"target":1},{"public":{"values":[4,4]},"target":4}],
)
BASE="def solve(values):\n    return values[-1]\n"

def test_engine_finds_and_promotes_source_patch():
    e=AutonomousRedesignEngine(BASE)
    rs=e.run_until_stable(PACK,"solver.py",["sequence extrapolation failure"])
    assert rs[0].promoted and "values[-2]" in e.source

def test_second_round_stops_when_gap_is_closed():
    e=AutonomousRedesignEngine(BASE)
    rs=e.run_until_stable(PACK,"solver.py",["sequence failure"],max_rounds=3)
    assert len(rs)==2 and not rs[1].promoted and rs[1].baseline_train==1

def test_three_seed_runs_select_same_patch():
    for _ in (2,5,8):
        e=AutonomousRedesignEngine(BASE)
        e.run_round(PACK,"solver.py",["delta failure"],1)
        assert "values[-2]" in e.source

def test_heldout_failure_blocks_source_promotion():
    bad=EvaluationPack(PACK.train,[{"public":{"values":[8,12]},"target":999}],PACK.regression)
    e=AutonomousRedesignEngine(BASE)
    r=e.run_round(bad,"solver.py",["sequence failure"],1)
    assert not r.promoted and e.source==BASE

def test_regression_failure_blocks_source_promotion():
    bad=EvaluationPack(PACK.train,PACK.held_out,[{"public":{"values":[1,1]},"target":999}])
    e=AutonomousRedesignEngine(BASE)
    r=e.run_round(bad,"solver.py",["sequence failure"],1)
    assert not r.promoted and e.source==BASE

def test_unknown_failure_does_not_mutate():
    e=AutonomousRedesignEngine(BASE)
    r=e.run_round(PACK,"solver.py",["unknown failure"],1)
    assert not r.promoted and e.source==BASE

def test_candidate_budget_respected():
    e=AutonomousRedesignEngine(BASE)
    r=e.run_round(PACK,"solver.py",["sequence failure"],1,max_candidates=1)
    assert r.candidates_tested==1

def test_round_budget_respected():
    e=AutonomousRedesignEngine(BASE)
    assert len(e.run_until_stable(PACK,"solver.py",["sequence failure"],max_rounds=1))==1

def test_rollback_restores_base_source():
    e=AutonomousRedesignEngine(BASE)
    e.run_round(PACK,"solver.py",["sequence failure"],1)
    assert "values[-2]" in e.source
    rec=e.rollback()
    assert rec.accepted and e.source==BASE

def test_invalid_bounds_rejected():
    e=AutonomousRedesignEngine(BASE)
    try: e.run_until_stable(PACK,"solver.py",["sequence"],max_rounds=0)
    except ValueError: pass
    else: assert False
