from .mirror7_phase310 import ReasoningFrontierAdapter

def test_reasoning_adapter_increases_depth():
    a=ReasoningFrontierAdapter(); out=a.improve(7,4,8)
    assert out.changed and out.score==1.0

def test_reasoning_score_bounded():
    assert 0<=ReasoningFrontierAdapter().evaluate(1)<=1

def test_three_seed_runs_same():
    for seed in (2,5,8):
        assert ReasoningFrontierAdapter().improve(seed,4,8).score==1.0

def test_second_upgrade_is_stable():
    a=ReasoningFrontierAdapter(); a.improve(7,4,8); out=a.improve(7,4,8)
    assert not out.changed

def test_candidate_budget_can_block_upgrade():
    a=ReasoningFrontierAdapter(); out=a.improve(7,4,0)
    assert not out.changed and a.depth==1
