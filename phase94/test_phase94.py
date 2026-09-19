from .mirror7_phase94 import OpenEndedResearchLoop

def test_end_to_end_integration():
    m = OpenEndedResearchLoop()
    m.observe_stream([0, 0, 5, 5, 0, 0], (0.0, None))
    m.observe_stream([0, 0, 5, 5, 0, 0], (0.1, None))
    m.learn_effect("probe", 1, 0, 2)
    hid = m.revise_hypothesis()
    assert hid == 0
    assert m.choose_experiment(["probe"]) == ("probe",)

def test_overlapping_effect_path():
    m = OpenEndedResearchLoop()
    m.learn_effect("a", 1, 0, 2)
    m.learn_effect("b", 2, 0, 3)
    assert m.effects.predict((0,), [("a", 1), ("b", 2)]) == (5.0,)

def test_structural_transfer_handoff():
    m = OpenEndedResearchLoop()
    assert m.transfer_signature((1, 2, 3), (3, 1, 2)).accepted
