from .mirror7_phase282 import SelfImprovementEngine
from phase280_sealed_candidate_eval import EvaluationPack

def pack():
    train = [
        {"public": {"v": [0, 2]}, "target": 4},
        {"public": {"v": [3, 6]}, "target": 9},
        {"public": {"v": [10, 14]}, "target": 18},
    ]
    held = [
        {"public": {"v": [-5, -2]}, "target": 1},
        {"public": {"v": [8, 12]}, "target": 16},
        {"public": {"v": [20, 25]}, "target": 30},
    ]
    regression = [
        {"public": {"v": [1, 1]}, "target": 1},
        {"public": {"v": [4, 4]}, "target": 4},
    ]
    return EvaluationPack(train, held, regression)

def solver_factory(config):
    window = int(config["window"])
    def solve(public):
        v = public["v"]
        if window == 1:
            return v[-1] + 1
        if window == 2:
            return v[-1] + (v[-1] - v[-2])
        return v[-1] + (v[-1] - v[-2]) + (v[-2] - v[-3]) if len(v) >= 3 else v[-1] + 1
    return solve

def test_self_improvement_finds_better_variant():
    e = SelfImprovementEngine({"window": 1})
    reports = e.run_until_stable(capability="sequence extrapolation", pack=pack(), parameter_space={"window": [1, 2, 3]}, solver_factory=solver_factory)
    assert reports[0].promoted
    assert e.registry.current == {"window": 2}

def test_progressive_improvement_closes_gap():
    e = SelfImprovementEngine({"window": 1})
    reports = e.run_until_stable(capability="sequence extrapolation", pack=pack(), parameter_space={"window": [1, 2]}, solver_factory=solver_factory, max_rounds=3)
    assert len(reports) == 2
    assert reports[0].promoted and not reports[1].gap_detected

def test_three_seed_runs_are_deterministic():
    for seed in (2, 5, 8):
        e = SelfImprovementEngine({"window": 1, "seed": seed})
        reports = e.run_until_stable(capability="sequence extrapolation", pack=pack(), parameter_space={"window": [1, 2]}, solver_factory=solver_factory)
        assert reports[0].selected_value == 2
        assert e.registry.current["window"] == 2

def test_held_out_evidence_is_required():
    def overfit_factory(config):
        if int(config["window"]) == 2:
            def solve(public):
                v = public["v"]
                if v[0] in (0, 3, 10):
                    return v[-1] + (v[-1] - v[-2])
                return 0
            return solve
        return solver_factory(config)
    e = SelfImprovementEngine({"window": 1})
    reports = e.run_until_stable(
        capability="sequence extrapolation",
        pack=pack(),
        parameter_space={"window": [2]},
        solver_factory=overfit_factory,
    )
    assert reports[0].gap_detected and not reports[0].promoted
    assert e.registry.current["window"] == 1

def test_regression_prevents_promotion():
    def regressor(config):
        if int(config["window"]) == 2:
            return lambda public: public["v"][-1] + (public["v"][-1] - public["v"][-2]) + 999
        return solver_factory(config)
    e = SelfImprovementEngine({"window": 1})
    reports = e.run_until_stable(capability="sequence extrapolation", pack=pack(), parameter_space={"window": [2]}, solver_factory=regressor)
    assert not reports[0].promoted


def test_no_gap_means_no_mutation():
    e = SelfImprovementEngine({"window": 2})
    reports = e.run_until_stable(capability="sequence extrapolation", pack=pack(), parameter_space={"window": [1, 2, 3]}, solver_factory=solver_factory)
    assert not reports[0].gap_detected
    assert e.registry.current == {"window": 2}

def test_failed_candidates_are_not_current_state():
    e = SelfImprovementEngine({"window": 1})
    reports = e.run_until_stable(capability="sequence extrapolation", pack=pack(), parameter_space={"window": [1, 3]}, solver_factory=solver_factory)
    assert not reports[0].promoted
    assert e.registry.current == {"window": 1}

def test_candidate_limit_is_respected():
    e = SelfImprovementEngine({"window": 1})
    report = e.run_round(capability="sequence extrapolation", pack=pack(), parameter_space={"window": [1, 2, 3, 4]}, solver_factory=solver_factory, round_number=1, max_candidates=2)
    assert report.candidates_tested == 2

def test_round_limit_is_bounded():
    e = SelfImprovementEngine({"window": 1})
    reports = e.run_until_stable(capability="sequence extrapolation", pack=pack(), parameter_space={"window": [1, 3]}, solver_factory=solver_factory, max_rounds=1)
    assert len(reports) == 1

def test_invalid_round_limit_is_rejected():
    e = SelfImprovementEngine({"window": 1})
    try:
        e.run_until_stable(capability="x", pack=pack(), parameter_space={}, solver_factory=solver_factory, max_rounds=0)
    except ValueError:
        pass
    else:
        assert False
