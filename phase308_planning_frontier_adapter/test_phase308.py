from .mirror7_phase308 import PlanningFrontierAdapter

def test_planning_adapter_improves_depth():
    a=PlanningFrontierAdapter()
    out=a.improve(7,4,8)
    assert out.changed and out.score==1.0

def test_planning_score_bounded():
    assert 0<=PlanningFrontierAdapter().evaluate(2,7)<=1

def test_three_seeds_are_supported():
    for seed in (2,5,8):
        assert PlanningFrontierAdapter().improve(seed,4,8).score==1.0

def test_no_backward_change():
    a=PlanningFrontierAdapter(); a.max_depth=5
    out=a.improve(7,4,2)
    assert not out.changed and a.max_depth==5
