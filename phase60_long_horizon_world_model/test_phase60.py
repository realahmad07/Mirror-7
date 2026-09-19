import pytest
from .mirror7_phase60 import LongHorizonWorldModel
from phase58_hierarchical_abstraction.mirror7_phase58 import discover_hierarchy
from phase59_predictive_concepts.mirror7_phase59 import PredictiveConceptModel

def make_episode(start=0, steps=30):
    state=(start,0)
    out=[]
    for _ in range(steps):
        nxt=(state[0]+1,state[1]^1)
        out.append((state,'advance',nxt))
        state=nxt
    return out

def test_progressive_long_horizon_3x3():
    for level,steps in enumerate((8,16,32)):
        for seed in range(3):
            m=LongHorizonWorldModel(min_support=2).fit([make_episode(seed,steps),make_episode(seed+100,steps)])
            s=(seed,0)
            actions=['advance']*steps
            rollout=m.rollout(s,actions)
            assert len(rollout)==steps
            assert rollout[-1]==(seed+steps,steps%2)

def test_heldout_state_generalization():
    train=[make_episode(0,20),make_episode(2,20),make_episode(4,20)]
    m=LongHorizonWorldModel(min_support=2).fit(train)
    r=m.predict((101,0),'advance')
    assert r.known and r.next_state==(102,1) and r.reason=='factorized'

def test_heldout_long_horizon_recombination():
    m=LongHorizonWorldModel(min_support=2).fit([make_episode(0,12),make_episode(10,12)])
    rollout=m.rollout((50,1),['advance']*25)
    assert len(rollout)==25 and rollout[-1]==(75,0)

def test_discrepancy_and_online_correction():
    m=LongHorizonWorldModel(min_support=2).fit([make_episode(0,10),make_episode(10,10)])
    d=m.discrepancy((100,0),'advance',(999,9))
    assert d['known'] and d['mismatch']
    m.update((100,0),'repair',(101,1))
    assert not m.predict((100,0),'repair').known
    m.update((100,0),'repair',(101,1))
    assert m.predict((100,0),'repair').next_state==(101,1)

def test_conflict_abstention():
    eps=[
        [((0,0),'x',(1,1)),((1,1),'x',(2,0))],
        [((0,0),'x',(9,9)),((9,9),'x',(8,8))]
    ]
    m=LongHorizonWorldModel(min_support=2).fit(eps)
    r=m.predict((0,0),'x')
    assert not r.known

def test_unknown_action_fails_closed():
    m=LongHorizonWorldModel(min_support=2).fit([make_episode(0,5),make_episode(10,5)])
    assert not m.predict((0,0),'teleport').known
    assert m.rollout((0,0),['advance','teleport','advance'])==((1,1),)

def test_phase58_59_integration():
    raw=[tuple('ABABCDCDABAB') for _ in range(6)]
    h=discover_hierarchy(raw,max_depth=3,min_episode_support=4)
    assert len(h.levels)>=2
    model=PredictiveConceptModel(max_order=3,min_support=2).fit(h.transformed_episodes)
    p=model.predict(h.transformed_episodes[0][:-1])
    assert p.value is not None
    s=(h.transformed_episodes[0][0],0)
    n=(h.transformed_episodes[0][0],1)
    w=LongHorizonWorldModel(min_support=2).fit([
        [(s,'tick',n),(n,'tick',(h.transformed_episodes[0][0],0))],
        [(s,'tick',n),(n,'tick',(h.transformed_episodes[0][0],0))]
    ])
    assert w.predict(s,'tick').known

def test_malformed():
    with pytest.raises(ValueError): LongHorizonWorldModel(min_support=1)
    with pytest.raises(ValueError): LongHorizonWorldModel().fit([])
    with pytest.raises(RuntimeError): LongHorizonWorldModel().rollout((0,),['a'])
