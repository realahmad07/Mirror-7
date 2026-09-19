from phase280_sealed_candidate_eval import EvaluationPack
from phase283_algorithm_variant_space import AlgorithmVariant, AlgorithmVariantSpace
from .mirror7_phase285 import AlgorithmIndependentEvaluator

def pack():
    return EvaluationPack(
        train=[
            {"public": {"values": [0, 2]}, "target": 4},
            {"public": {"values": [3, 6]}, "target": 9},
            {"public": {"values": [10, 14]}, "target": 18},
        ],
        held_out=[
            {"public": {"values": [-5, -2]}, "target": 1},
            {"public": {"values": [8, 12]}, "target": 16},
            {"public": {"values": [20, 25]}, "target": 30},
        ],
        regression=[
            {"public": {"values": [1, 1]}, "target": 1},
            {"public": {"values": [4, 4]}, "target": 4},
        ],
    )

def linear():
    return AlgorithmVariant("linear", ("last", "add_delta"))

def constant():
    return AlgorithmVariant("constant", ("constant",))

def test_linear_variant_passes_all_sets():
    s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
    e = s.evaluate(linear(), pack())
    assert e.train == 1.0 and e.held_out == 1.0 and e.regression_ok

def test_three_seed_evaluations_repeat():
    for seed in (3, 7, 19):
        s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
        e = s.evaluate(linear(), pack())
        assert (e.train, e.held_out, e.regression_ok) == (1.0, 1.0, True)

def test_bad_variant_fails():
    s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
    e = s.evaluate(constant(), pack())
    assert e.train == 0.0 and e.held_out == 0.0 and not e.regression_ok

def test_hidden_target_not_exposed_to_algorithm():
    class TrackingSpace(AlgorithmVariantSpace):
        def execute(self, variant, values):
            assert isinstance(values, list)
            return super().execute(variant, values)
    s = AlgorithmIndependentEvaluator(TrackingSpace())
    s.evaluate(linear(), pack())

def test_invalid_variant_rejected():
    s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
    try:
        s.evaluate(AlgorithmVariant("x", ("eval",)), pack())
    except ValueError:
        pass
    else:
        assert False

def test_improvement_gate_accepts_real_gain():
    s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
    base = s.evaluate(constant(), pack())
    cand = s.evaluate(linear(), pack())
    assert s.improves(base, cand)

def test_improvement_gate_rejects_heldout_regression():
    s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
    base = s.evaluate(linear(), pack())
    cand = s.evaluate(constant(), pack())
    assert not s.improves(base, cand)

def test_empty_pack_fails_closed():
    s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
    e = s.evaluate(linear(), EvaluationPack([], [], []))
    assert e.train == 0.0 and e.held_out == 0.0 and not e.regression_ok

def test_candidate_limit_does_not_change_evaluation():
    s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
    a = s.evaluate(linear(), pack())
    b = s.evaluate(linear(), pack())
    assert a == b

def test_negative_control_target_mismatch_is_not_promoted():
    altered = pack().__class__(pack().train, [{"public": {"values": [8, 12]}, "target": 999}], pack().regression)
    s = AlgorithmIndependentEvaluator(AlgorithmVariantSpace())
    e = s.evaluate(linear(), altered)
    base = s.evaluate(constant(), altered)
    assert e.held_out == 0.0 and not s.improves(base, e)
