from .mirror7_phase314 import run_frontier_integration

def test_full_targeted_loop():
    campaign,events=run_frontier_integration(4)
    assert events and all(e.accepted for e in events)

def test_loop_advances_frontier():
    campaign,events=run_frontier_integration(4)
    assert len(events)==4
    assert campaign.frontier.top_gap() is None or campaign.frontier.top_gap().name not in {e.capability for e in events}

def test_scores_remain_bounded():
    campaign,_=run_frontier_integration(4)
    assert all(0<=v<=1 for v in campaign.scores.values())

def test_repeatability():
    a=run_frontier_integration(4)[1]
    b=run_frontier_integration(4)[1]
    assert a==b

def test_single_step_supported():
    campaign,events=run_frontier_integration(1)
    assert len(events)==1

def test_invalid_step_bound():
    try: run_frontier_integration(0)
    except ValueError: pass
    else: assert False
