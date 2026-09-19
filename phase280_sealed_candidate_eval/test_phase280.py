from .mirror7_phase280 import EvaluationPack, SealedCandidateEvaluator

def pack():
    return EvaluationPack(
        train=[
            {"public": {"v": [0, 1]}, "target": 2},
            {"public": {"v": [2, 4]}, "target": 6},
            {"public": {"v": [5, 8]}, "target": 11},
        ],
        held_out=[
            {"public": {"v": [-4, 0]}, "target": 4},
            {"public": {"v": [10, 14]}, "target": 18},
            {"public": {"v": [7, 12]}, "target": 17},
        ],
        regression=[
            {"public": {"v": [1, 1]}, "target": 1},
            {"public": {"v": [3, 3]}, "target": 3},
        ],
    )

def good_solver(public):
    v = public["v"]
    return v[-1] + (v[-1] - v[-2])

def baseline_solver(public):
    v = public["v"]
    return v[-1] + (1 if v[-1] >= v[-2] else 0)

def bad_solver(public):
    return 0

def test_progressive_good_candidate_scores():
    s = SealedCandidateEvaluator().evaluate(good_solver, pack())
    assert s.train == 1.0 and s.held_out == 1.0 and s.regression_ok

def test_three_seed_reproducibility():
    evaluator = SealedCandidateEvaluator()
    for seed in (3, 7, 19):
        tasks = [{"public": {"v": [seed, seed + 5]}, "target": seed + 10}]
        s = evaluator.evaluate(good_solver, pack().__class__(tasks, tasks, [{"public": {"v": [1, 1]}, "target": 1}]))
        assert s.train == s.held_out == 1.0 and s.regression_ok

def test_bad_candidate_fails_closed():
    s = SealedCandidateEvaluator().evaluate(bad_solver, pack())
    assert s.train == 0.0 and s.held_out == 0.0 and not s.regression_ok

def test_hidden_target_not_available_to_solver():
    seen = []
    def solver(public):
        seen.append(public)
        assert "target" not in public
        return None
    SealedCandidateEvaluator().evaluate(solver, pack())
    assert seen and all("target" not in p for p in seen)

def test_improvement_requires_held_out_evidence():
    e = SealedCandidateEvaluator()
    base = e.evaluate(baseline_solver, pack())
    cand = e.evaluate(lambda p: p["v"][-1] + 99, pack())
    assert not e.improves(base, cand)

def test_regression_blocks_promotion():
    e = SealedCandidateEvaluator()
    base = e.evaluate(baseline_solver, pack())
    candidate = e.evaluate(good_solver, pack())
    assert e.improves(base, candidate)
    failing = e.evaluate(lambda p: 1, pack())
    assert not e.improves(candidate, failing)

def test_empty_benchmark_fails_closed():
    e = SealedCandidateEvaluator()
    empty = EvaluationPack([], [], [])
    s = e.evaluate(good_solver, empty)
    assert s.train == s.held_out == 0.0 and not s.regression_ok

def test_threshold_blocks_tiny_gain():
    e = SealedCandidateEvaluator()
    base = e.evaluate(baseline_solver, pack())
    candidate = base
    assert not e.improves(base, candidate, min_gain=0.001)

def test_adversarial_solver_cannot_access_hidden_truth():
    def solver(public):
        return public.get("target", "LEAK")
    s = SealedCandidateEvaluator().evaluate(solver, pack())
    assert s.held_out == 0.0
