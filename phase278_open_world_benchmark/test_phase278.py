
from .mirror7_phase278 import NovelBenchmark, perfect_solver

def test_progressive_families():
    for seed in (1,2,3):
        good,total=NovelBenchmark(seed).evaluate(perfect_solver,trials=2)
        assert good==total==4

def test_held_out_seed():
    good,total=NovelBenchmark(997).evaluate(perfect_solver,trials=3)
    assert good==total==6

def test_adversarial_bad_solver_rejected():
    def bad(_): return 0
    good,total=NovelBenchmark(7).evaluate(bad,trials=4)
    assert good<total

def test_opaque_names_do_not_change_score():
    a=NovelBenchmark(5).evaluate(perfect_solver,trials=3)
    b=NovelBenchmark(6).evaluate(perfect_solver,trials=3)
    assert a==(6,6) and b==(6,6)

def test_deterministic_generation():
    a=NovelBenchmark(42).sequence(6)
    b=NovelBenchmark(42).sequence(6)
    assert a==b

def test_task_family_unknown_solver_fails_closed():
    def empty(_): return None
    good,total=NovelBenchmark(13).evaluate(empty,trials=2)
    assert good==0 and total==4
