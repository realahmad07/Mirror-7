from phase280_sealed_candidate_eval import EvaluationPack
from phase283_algorithm_variant_space import AlgorithmVariant, AlgorithmVariantSpace
from .mirror7_phase286 import AlgorithmImprovementEngine

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

def test_engine_discovers_algorithmic_improvement():
    engine = AlgorithmImprovementEngine(AlgorithmVariant("base", ("last", "delta")), AlgorithmVariantSpace(("last", "delta", "add_delta")))
    reports = engine.run_until_stable(pack(), max_rounds=3, max_candidates=8)
    assert reports[0].promoted
    assert engine.current_variant.operations == ("last", "add_delta")

def test_improvement_is_not_scalar_parameter_tuning():
    engine = AlgorithmImprovementEngine(AlgorithmVariant("base", ("last", "delta")), AlgorithmVariantSpace(("last", "delta", "add_delta")))
    engine.run_until_stable(pack(), max_rounds=1)
    assert engine.current_variant.operations != ("last", "delta")

def test_three_seed_runs_select_same_variant():
    for seed in (2, 5, 8):
        engine = AlgorithmImprovementEngine(AlgorithmVariant(f"base-{seed}", ("last", "delta")), AlgorithmVariantSpace(("last", "delta", "add_delta")))
        engine.run_until_stable(pack(), max_rounds=2)
        assert engine.current_variant.operations == ("last", "add_delta")

def test_held_out_failure_blocks_algorithm_promotion():
    bad = EvaluationPack(pack().train, [{"public": {"values": [8, 12]}, "target": 999}], pack().regression)
    engine = AlgorithmImprovementEngine(AlgorithmVariant("base", ("last", "delta")), AlgorithmVariantSpace(("last", "delta", "add_delta")))
    report = engine.run_round(bad, round_number=1)
    assert not report.promoted
    assert engine.current_variant.operations == ("last", "delta")

def test_regression_failure_blocks_algorithm_promotion():
    bad = EvaluationPack(pack().train, pack().held_out, [{"public": {"values": [1, 1]}, "target": 999}])
    engine = AlgorithmImprovementEngine(AlgorithmVariant("base", ("last", "delta")), AlgorithmVariantSpace(("last", "delta", "add_delta")))
    report = engine.run_round(bad, round_number=1)
    assert not report.promoted


def test_candidate_budget_is_respected():
    engine = AlgorithmImprovementEngine(AlgorithmVariant("base", ("last", "delta")), AlgorithmVariantSpace())
    report = engine.run_round(pack(), round_number=1, max_candidates=2)
    assert report.candidates_tested == 2

def test_round_budget_is_respected():
    engine = AlgorithmImprovementEngine(AlgorithmVariant("base", ("last", "delta")), AlgorithmVariantSpace())
    reports = engine.run_until_stable(pack(), max_rounds=1)
    assert len(reports) == 1

def test_no_improvement_leaves_variant_unchanged():
    linear = AlgorithmVariant("linear", ("last", "add_delta"))
    engine = AlgorithmImprovementEngine(linear, AlgorithmVariantSpace(("last", "delta", "add_delta")))
    report = engine.run_round(pack(), round_number=1)
    assert not report.promoted and engine.current_variant == linear

def test_rollback_restores_previous_algorithm():
    engine = AlgorithmImprovementEngine(AlgorithmVariant("base", ("last", "delta")), AlgorithmVariantSpace(("last", "delta", "add_delta")))
    engine.run_round(pack(), round_number=1)
    assert engine.current_variant.operations == ("last", "add_delta")
    rec = engine.rollback()
    assert rec.accepted and engine.current_variant.operations == ("last", "delta")

def test_invalid_bounds_fail_closed():
    engine = AlgorithmImprovementEngine(AlgorithmVariant("base", ("last",)), AlgorithmVariantSpace())
    try:
        engine.run_until_stable(pack(), max_rounds=0)
    except ValueError:
        pass
    else:
        assert False
