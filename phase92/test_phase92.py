from .mirror7_phase92 import AutonomousExperimentSequencer

def test_selects_discriminating_action():
    h = [{"a": {"d": 1}, "b": {"d": 5}}, {"a": {"d": 4}, "b": {"d": 5}}]
    p = AutonomousExperimentSequencer(max_depth=1).choose(h, ["a", "b"])
    assert p.sequence == ("a",)

def test_sequences_beyond_one_step():
    h = [{"a": {"d": 1}, "b": {"d": 3}}, {"a": {"d": 2}, "b": {"d": 9}}]
    p = AutonomousExperimentSequencer(max_depth=2).choose(h, ["a", "b"])
    assert 1 <= len(p.sequence) <= 2

def test_budget_fail_closed():
    s = AutonomousExperimentSequencer(budget=1)
    h = [{"a": {"d": 1}}, {"a": {"d": 2}}]
    assert s.choose(h, ["a"]) is not None
    assert s.choose(h, ["a"]) is None

def test_no_legal_actions():
    assert AutonomousExperimentSequencer().choose([{"a": {"d": 1}}], []) is None
