from phase77_integrated_autonomous_loop import IntegratedAutonomousLoop


def test_closed_loop_learns_and_reuses():
    m = IntegratedAutonomousLoop(max_steps=6, experiment_budget=3)
    def step(s, a): return (s[0] + (1 if a == "inc" else -1),)
    r = m.run((0,), ["inc"], step, lambda s: s[0] >= 2)
    assert r.reason == "goal reached"
    assert r.steps == 2
    assert m.predict((0,), "inc") == (1.0,)


def test_discrepancy_revision():
    m = IntegratedAutonomousLoop()
    assert m.learn((0,), "a", (1,))
    assert m.learn((0,), "a", (2,))
    assert m.revisions >= 1


def test_budget_and_fail_closed():
    m = IntegratedAutonomousLoop(max_steps=4, experiment_budget=0)
    r = m.run((0,), ["a"], lambda s,a: (1,))
    assert r.stopped and "budget" in r.reason


def test_bad_dimension_rejected():
    m = IntegratedAutonomousLoop()
    r = m.run((0,), ["a"], lambda s,a: (1,2))
    assert r.stopped and r.reason == "dimension mismatch"


def test_invalid_input_rejected():
    try:
        IntegratedAutonomousLoop(max_steps=0)
        assert False
    except ValueError:
        pass
