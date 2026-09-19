"""Mirror 7 Phase 32 acceptance tests against the current predictor/memory API."""
from __future__ import annotations
import random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase31_bootstrap"))
from mirror7_pipeline import raw_to_state
from mirror7_temporal import TemporalStateTracker
from phase32_prediction.mirror7_transition import TransitionMemory
from phase32_prediction.mirror7_predictor import PredictionStatus, TransitionPredictor


def states(raws):
    return tuple(raw_to_state(x) for x in raws)


def test_basic_prediction():
    a,b,c=states([b"AAAA",b"BBBB",b"CCCC"])
    p=TransitionPredictor(); p.online_update(a,b)
    assert p.predict_next(a).predicted_state_id==b.state_id
    assert p.predict_next(c).status==PredictionStatus.UNKNOWN


def test_ambiguous_successors():
    a,b,c=states([b"AA",b"BB",b"CC"])
    m=TransitionMemory(); m.record_transition(a,b); m.record_transition(a,c)
    r=TransitionPredictor(m).predict_next(a)
    assert r.status==PredictionStatus.AMBIGUOUS and r.predicted_state_id is None
    assert {x.state_id for x in r.candidates}=={b.state_id,c.state_id}


def test_support_accumulates():
    a,b,c=states([b"AA",b"BB",b"CC"])
    m=TransitionMemory(); m.record_transition(a,b); m.record_transition(a,b); m.record_transition(a,c)
    s={x.to_state_id:x.observation_count for x in m.get_successors(a)}
    assert s[b.state_id]==2 and s[c.state_id]==1


def test_held_out_sequence():
    train=states([bytes([1])*2,bytes([2])*2,bytes([3])*2,bytes([4])*2])
    p=TransitionPredictor()
    for x,y in zip(train,train[1:]): p.online_update(x,y)
    test=states([bytes([1])*2,bytes([2])*2,bytes([3])*2,bytes([4])*2])
    for x,y in zip(test,test[1:]):
        assert p.predict_next(x).predicted_state_id==y.state_id


def test_unseen_unknown():
    a,b,u=states([b"AA",b"BB",b"CD"]); p=TransitionPredictor(); p.online_update(a,b)
    assert p.predict_next(u).status==PredictionStatus.UNKNOWN


def test_discrepancy_online_update():
    a,b,c=states([b"AA",b"BB",b"CC"]); p=TransitionPredictor(); p.online_update(a,b)
    pred=p.predict_next(a); d=p.evaluate_prediction(pred,c)
    assert d.is_correct is False and d.actual_state_id==c.state_id
    p.online_update(a,c)
    assert p.predict_next(a).status==PredictionStatus.AMBIGUOUS


def test_multistep_rollout():
    a,b,c,d=states([b"AA",b"BB",b"CC",b"DD"]); p=TransitionPredictor()
    for x,y in zip((a,b,c,d),(b,c,d)): p.online_update(x,y)
    preds=p.predict_sequence(a,3)
    assert tuple(r.predicted_state_id for r in preds)==(b.state_id,c.state_id,d.state_id)


def test_three_seeds_and_negative():
    for seed in (11,22,33):
        rng=random.Random(seed); alphabet=rng.sample(range(1,255),4); ss=states([bytes([x,x]) for x in alphabet])
        p=TransitionPredictor()
        for x,y in zip(ss,ss[1:]): p.online_update(x,y)
        assert p.predict_next(ss[0]).predicted_state_id==ss[1].state_id
        negative=raw_to_state(bytes([alphabet[0],alphabet[2]]))
        assert p.predict_next(negative).status==PredictionStatus.UNKNOWN


def test_serialization():
    a,b,c=states([b"AA",b"BB",b"CC"]); p=TransitionPredictor(); p.online_update(a,b); p.online_update(b,c)
    restored=TransitionMemory.from_json(p.memory.to_json())
    assert restored.to_json()==p.memory.to_json()
    assert TransitionPredictor(restored).predict_next(a).predicted_state_id==b.state_id


def test_phase31_regression():
    a,b,c=states([b"AA",b"BB",b"CC"]); t=TemporalStateTracker()
    assert t.step(a).to_state_id==a.state_id and t.step(b).to_state_id==b.state_id and t.step(c).to_state_id==c.state_id


def test_end_to_end():
    p=TransitionPredictor(); train=states([b"AA",b"BB",b"CC",b"DD"])
    for x,y in zip(train,train[1:]): p.online_update(x,y)
    for x,y in zip(train,train[1:]): assert p.predict_next(x).predicted_state_id==y.state_id

if __name__=="__main__":
    tests=[v for n,v in sorted(globals().items()) if n.startswith("test_")]
    for test in tests: test(); print("[PASS]",test.__name__)
    print("PHASE_32_PASS",len(tests),"tests")
