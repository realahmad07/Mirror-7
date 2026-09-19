from .mirror7_phase104 import ImprovementProposer

def test_recurring_failure_creates_proposal():
    p=ImprovementProposer().propose("planner",["overexplores","timeout"],.6,.4,.1,2)
    assert p and p.target=="planner"

def test_no_failure_no_self_change():
    assert ImprovementProposer().propose("x",[],.8,.2,.1,1) is None

def test_high_risk_change_rejected():
    assert ImprovementProposer().propose("x",["f"],.5,.2,.3,1) is None

def test_negative_cost_rejected():
    try: ImprovementProposer().propose("x",["f"],.5,.3,.1,-1)
    except ValueError: pass
    else: assert False
