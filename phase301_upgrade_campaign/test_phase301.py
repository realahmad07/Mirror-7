from .mirror7_phase301 import CampaignRunner, UpgradeCampaign

def test_campaign_accepts_improving_round():
    c=UpgradeCampaign("x",.5,1,3,4)
    r=CampaignRunner().run(c,.4,lambda _: (.7,True))
    assert r.promoted and r.end_score==.7

def test_campaign_stops_on_non_improvement():
    c=UpgradeCampaign("x",.5,1,3,4)
    r=CampaignRunner().run(c,.7,lambda _: (.6,True))
    assert not r.promoted and r.stable

def test_three_seed_campaign_is_deterministic():
    for _ in (2,5,8):
        r=CampaignRunner().run(UpgradeCampaign("x",.5,1,2,4),.4,lambda _: (.6,True))
        assert r.end_score==.6

def test_round_bound_is_respected():
    r=CampaignRunner().run(UpgradeCampaign("x",.5,1,1,4),.4,lambda _: (.6,True))
    assert r.rounds==1

def test_invalid_bounds_rejected():
    try: CampaignRunner().run(UpgradeCampaign("x",.5,1,0,4),.4,lambda _: (.6,True))
    except ValueError: pass
    else: assert False

def test_no_change_is_stable():
    r=CampaignRunner().run(UpgradeCampaign("x",.5,1,3,4),.4,lambda _: (.4,False))
    assert r.stable and not r.promoted

def test_never_moves_backward():
    r=CampaignRunner().run(UpgradeCampaign("x",.5,1,3,4),.4,lambda _: (.3,True))
    assert r.end_score==.4

def test_gap_metadata_preserved():
    c=UpgradeCampaign("x",.5,2); assert c.gap==.5 and c.priority==2
