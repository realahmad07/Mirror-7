import random
from phase66_richer_hidden_state import Phase66Agent

def sample(delta,seed,n=8,dims=4):
    r=random.Random(seed)
    out=[]
    for _ in range(n):
        out.append(tuple(delta[d]+r.gauss(0,0.08) for d in range(dims)))
    return out

def feed(agent,action,base,delta,lag,seed=0,n=8):
    r=random.Random(seed)
    for _ in range(n):
        before=tuple(base)
        after=tuple(base[d]+delta[d]+r.gauss(0,0.08) for d in range(len(base)))
        agent.observe_effect(action,before,after,lag)

def test_multidimensional_stochastic_recovery():
    a=Phase66Agent(min_samples=4)
    for seed in range(3):
        feed(a,"pulse",(0,0,0,0),(1,2,3,4),2,seed)
    for d,v in enumerate((1,2,3,4)):
        e=a.distribution("pulse",2,d)
        assert e and abs(e.mean-v)<0.20 and e.samples>=12

def test_overlapping_delayed_effects_are_retained():
    a=Phase66Agent(min_samples=3)
    # Two actions have overlapping evidence windows and different lags.
    for seed in range(6):
        r=random.Random(seed)
        a.observe_effect("a",(0,0,0,0),(1+r.gauss(0,.03),0,0,0),2)
        a.observe_effect("b",(0,0,0,0),(0,2+r.gauss(0,.03),0,0),4)
    assert a.distribution("a",2,0).mean > .9
    assert a.distribution("b",4,1).mean > 1.9

def test_stochastic_transition_variance_is_preserved():
    a=Phase66Agent(min_samples=4)
    for x in (0.0,1.0,2.0,3.0,4.0,5.0):
        a.observe_effect("pulse",(0,),(x,),1)
    e=a.distribution("pulse",1,0)
    assert e and e.std>1.0 and e.samples==6

def test_hidden_hypothesis_revision_without_labels():
    a=Phase66Agent(min_samples=3,mismatch_window=2)
    feed(a,"pulse",(0,0,0,0),(1,0,0,0),2,1)
    before=len(a.hypotheses)
    feed(a,"pulse",(0,0,0,0),(3,0,0,0),2,2)
    feed(a,"pulse",(0,0,0,0),(3,0,0,0),2,3)
    assert len(a.hypotheses)>before

def test_longer_autonomous_experiment_uses_gap_slots():
    a=Phase66Agent(min_samples=2,experiment_budget=2)
    feed(a,"pulse",(0,0,0,0),(1,0,0,0),2,1,n=3)
    p=a.design_experiment(("pulse","__noop__","other"),slots=8)
    assert p and len(p["sequence"])==8 and p["sequence"][0]=="other"

def test_hard_experiment_budget():
    a=Phase66Agent(experiment_budget=3)
    for _ in range(10): a.design_experiment(("a","__noop__","b"))
    assert a.experiment_count==3
    assert a.design_experiment(("a","__noop__","b")) is None

def test_cross_environment_transfer_snapshot():
    a=Phase66Agent(min_samples=3)
    feed(a,"pulse",(0,0,0,0),(1,2,0,0),2,10,n=5)
    snap=a.export_hypotheses()
    b=Phase66Agent(min_samples=3)
    b.import_hypotheses(snap)
    assert b.infer(a.model)==0

def test_fail_closed_and_partial_observation():
    a=Phase66Agent(min_samples=2)
    for _ in range(3):
        a.observe_effect("pulse",(0,None,0),(1,None,0),2)
    assert a.distribution("pulse",2,1) is None
    assert a.fail_closed()["hypotheses"]>=1

def test_long_horizon_prediction():
    a=Phase66Agent(min_samples=3,max_lag=8)
    for _ in range(6):
        a.observe_effect("step",(0,0,0,0),(1,2,3,4),8)
    pred=a.predict_mean((10,10,10,10),"step",8)
    assert pred==(11.0,12.0,13.0,14.0)
