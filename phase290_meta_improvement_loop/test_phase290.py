from phase280_sealed_candidate_eval import EvaluationPack
from phase283_algorithm_variant_space import AlgorithmVariant, AlgorithmVariantSpace
from .mirror7_phase290 import MetaImprovementEngine

def pack():
    return EvaluationPack(
        [{"public":{"values":[0,2]},"target":4},{"public":{"values":[3,6]},"target":9},{"public":{"values":[10,14]},"target":18}],
        [{"public":{"values":[-5,-2]},"target":1},{"public":{"values":[8,12]},"target":16},{"public":{"values":[20,25]},"target":30}],
        [{"public":{"values":[1,1]},"target":1},{"public":{"values":[4,4]},"target":4}],
    )

def test_meta_loop_promotes_algorithm_variant():
    e=MetaImprovementEngine(AlgorithmVariant("base",("last","delta")),AlgorithmVariantSpace(("last","delta","add_delta")))
    reports=e.run_until_stable(pack(),max_rounds=2)
    assert reports[0].promoted and e.current.operations==("last","add_delta")

def test_memory_is_used_between_rounds():
    e=MetaImprovementEngine(AlgorithmVariant("base",("last","delta")),AlgorithmVariantSpace(("last","delta","add_delta")))
    e.run_round(pack(),1)
    r=e.run_round(pack(),2)
    assert r.candidates_skipped_from_memory>0

def test_three_seed_like_runs_are_stable():
    for seed in (2,5,8):
        e=MetaImprovementEngine(AlgorithmVariant(f"base-{seed}",("last","delta")),AlgorithmVariantSpace(("last","delta","add_delta")))
        e.run_round(pack(),1)
        assert e.current.operations==("last","add_delta")

def test_heldout_failure_blocks_promotion():
    p=pack(); bad=EvaluationPack(p.train,[{"public":{"values":[8,12]},"target":999}],p.regression)
    e=MetaImprovementEngine(AlgorithmVariant("base",("last","delta")),AlgorithmVariantSpace(("last","delta","add_delta")))
    r=e.run_round(bad,1)
    assert not r.promoted and e.current.operations==("last","delta")

def test_regression_failure_blocks_promotion():
    p=pack(); bad=EvaluationPack(p.train,p.held_out,[{"public":{"values":[1,1]},"target":999}])
    e=MetaImprovementEngine(AlgorithmVariant("base",("last","delta")),AlgorithmVariantSpace(("last","delta","add_delta")))
    r=e.run_round(bad,1)
    assert not r.promoted

def test_budget_is_respected():
    e=MetaImprovementEngine(AlgorithmVariant("base",("last","delta")),AlgorithmVariantSpace())
    assert e.run_round(pack(),1,max_candidates=2).candidates_seen==2

def test_round_bound_is_respected():
    e=MetaImprovementEngine(AlgorithmVariant("base",("last","delta")),AlgorithmVariantSpace(("last","delta","add_delta")))
    assert len(e.run_until_stable(pack(),max_rounds=1))==1

def test_no_promotion_when_already_optimal():
    e=MetaImprovementEngine(AlgorithmVariant("linear",("last","add_delta")),AlgorithmVariantSpace(("last","delta","add_delta")))
    r=e.run_round(pack(),1)
    assert not r.promoted and e.current.operations==("last","add_delta")

def test_invalid_bounds_fail_closed():
    e=MetaImprovementEngine(AlgorithmVariant("base",("last",)),AlgorithmVariantSpace())
    try: e.run_until_stable(pack(),max_rounds=0)
    except ValueError: pass
    else: assert False
