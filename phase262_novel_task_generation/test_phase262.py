import pytest
from .mirror7_phase262 import ProceduralTaskGenerator, IndependentEvaluator

def test_causal_generation_deterministic():
    assert ProceduralTaskGenerator(10).generate_causal_environment()==ProceduralTaskGenerator(10).generate_causal_environment()

def test_sequence_generation_and_validation():
    t=ProceduralTaskGenerator(42).generate_sequence_task()
    assert len(t["sequence"])==9 and isinstance(t["target"],int)
    with pytest.raises(ValueError): ProceduralTaskGenerator(1).generate_sequence_task(1)

def test_planning_bounds():
    g=ProceduralTaskGenerator(7); t=g.generate_planning_world(3,4)
    assert len(t["obstacles"])==4 and t["start"] not in t["obstacles"] and t["goal"] not in t["obstacles"]
    with pytest.raises(ValueError): g.generate_planning_world(2,3)

def test_independent_evaluator_sequence():
    ev=IndependentEvaluator(ProceduralTaskGenerator(42))
    assert ev.evaluate(lambda seq: seq[-1]+(seq[1]-seq[0]),"sequence",5)==1.0
    assert ev.evaluate(lambda seq: None,"sequence",5)==0.0
    with pytest.raises(ValueError): ev.evaluate(lambda x: x,"causal",1)


def test_sequence_three_seeds_and_held_out():
    for seed in (1, 17, 99):
        ev=IndependentEvaluator(ProceduralTaskGenerator(seed))
        assert ev.evaluate(lambda seq: seq[-1]+(seq[1]-seq[0]), "sequence", 7)==1.0
        assert ev.evaluate(lambda seq: object(), "sequence", 7)==0.0

def test_planning_held_out_and_determinism():
    a=ProceduralTaskGenerator(123).generate_planning_world(6,10)
    b=ProceduralTaskGenerator(123).generate_planning_world(6,10)
    c=ProceduralTaskGenerator(124).generate_planning_world(6,10)
    assert a==b and a!=c
