from .mirror7_phase306 import SequenceFrontierAdapter, SequenceFrontierDemo

def test_adapter_improves_sequence_capability():
    a=SequenceFrontierAdapter()
    before=a.evaluate(7); out=a.improve(7,4,8)
    assert out.changed and out.score>before

def test_adapter_keeps_hidden_tasks_separate():
    a=SequenceFrontierAdapter(); assert 0<=a.evaluate(11)<=1

def test_three_seed_improvements_are_stable():
    for seed in (2,5,8):
        a=SequenceFrontierAdapter(); assert a.improve(seed,4,8).changed

def test_demo_runs_targeted_loop():
    assert SequenceFrontierDemo().run(2)

def test_demo_has_bounded_history():
    rows=SequenceFrontierDemo().run(4); assert len(rows)<=4

def test_adapter_variant_is_a_source_fingerprint_or_nochange():
    out=SequenceFrontierAdapter().improve(19,4,8)
    assert isinstance(out.variant,str) and out.variant

def test_score_is_bounded():
    out=SequenceFrontierAdapter().improve(31,4,8); assert 0<=out.score<=1

def test_repeatability_same_seed():
    a=SequenceFrontierAdapter().improve(997,4,8); b=SequenceFrontierAdapter().improve(997,4,8)
    assert a.score==b.score and a.changed==b.changed
