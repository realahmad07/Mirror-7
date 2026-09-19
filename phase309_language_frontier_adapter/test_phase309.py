from .mirror7_phase309 import LanguageFrontierAdapter

def test_language_adapter_adds_composition():
    a=LanguageFrontierAdapter(); out=a.improve(7,4,8)
    assert out.changed and out.score==1.0 and a.mode=="composition"

def test_primitive_mode_is_bounded():
    assert 0<=LanguageFrontierAdapter().evaluate("primitive",7)<=1

def test_composition_mode_is_bounded():
    assert LanguageFrontierAdapter().evaluate("composition",7)==1.0

def test_three_seed_runs_are_stable():
    for seed in (2,5,8):
        assert LanguageFrontierAdapter().improve(seed,4,8).changed

def test_no_second_upgrade_after_composition():
    a=LanguageFrontierAdapter(); a.improve(7,4,8); out=a.improve(7,4,8)
    assert not out.changed and out.variant=="composition"
