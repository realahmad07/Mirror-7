from .mirror7_phase313 import MultiCapabilityCampaign

def test_campaign_progresses():
    c=MultiCapabilityCampaign(); events=c.run(4)
    assert len(events)==4 and all(e.accepted for e in events)

def test_campaign_moves_across_capabilities():
    c=MultiCapabilityCampaign(); names=[e.capability for e in c.run(4)]
    assert len(set(names))>=2

def test_scores_stay_bounded():
    c=MultiCapabilityCampaign()
    assert all(0<=e.score<=1 for e in c.run(4))

def test_repeatability():
    assert [e.capability for e in MultiCapabilityCampaign().run(4)]==[e.capability for e in MultiCapabilityCampaign().run(4)]

def test_invalid_run_bound():
    try: MultiCapabilityCampaign().run(0)
    except ValueError: pass
    else: assert False
