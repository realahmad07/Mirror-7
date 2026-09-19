from .mirror7_phase287 import SelfGeneratedTaskSuite

def test_linear_tasks_are_generated():
    tasks=SelfGeneratedTaskSuite(1).linear_sequence(5)
    assert len(tasks)==5 and all(t.name=="linear" for t in tasks)

def test_three_seed_generation_is_deterministic():
    for seed in (2,7,19):
        assert SelfGeneratedTaskSuite(seed).mixed_suite(3)==SelfGeneratedTaskSuite(seed).mixed_suite(3)

def test_targets_are_hidden_from_public():
    assert all("target" not in t.public for t in SelfGeneratedTaskSuite(3).linear_sequence(4))

def test_held_out_seed_differs():
    assert SelfGeneratedTaskSuite(1).linear_sequence(2)!=SelfGeneratedTaskSuite(999).linear_sequence(2)

def test_invalid_count_rejected():
    try: SelfGeneratedTaskSuite(1).linear_sequence(0)
    except ValueError: pass
    else: assert False

def test_mixed_suite_contains_multiple_families():
    assert {t.name for t in SelfGeneratedTaskSuite(2).mixed_suite(3)}=={"linear","difference"}

def test_pack_conversion_keeps_hidden_targets_separate():
    s=SelfGeneratedTaskSuite(4); a=s.linear_sequence(2); p=s.as_evaluation_pack(a,a,a)
    assert all("target" not in x["public"] for x in p.train)

def test_zero_is_not_forced_as_target():
    assert any(t.target!=0 for t in SelfGeneratedTaskSuite(5).linear_sequence(8))

def test_negative_control_still_has_hidden_target():
    t=SelfGeneratedTaskSuite(6).linear_sequence(1)[0]
    assert t.target is not None
