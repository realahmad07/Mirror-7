from phase280_sealed_candidate_eval import EvaluationPack
from .mirror7_phase294 import SealedSourceEvaluator

PACK=EvaluationPack(
    [{"public":{"values":[0,2]},"target":4},{"public":{"values":[3,6]},"target":9},{"public":{"values":[10,14]},"target":18}],
    [{"public":{"values":[-5,-2]},"target":1},{"public":{"values":[8,12]},"target":16},{"public":{"values":[20,25]},"target":30}],
    [{"public":{"values":[1,1]},"target":1},{"public":{"values":[4,4]},"target":4}],
)
BASE="def solve(values):\n    return values[-1]\n"
GOOD="def solve(values):\n    return values[-1] + (values[-1]-values[-2])\n"

def test_good_source_passes():
    e=SealedSourceEvaluator().evaluate(GOOD,PACK)
    assert e.train==1 and e.held_out==1 and e.regression_ok

def test_base_fails_task():
    e=SealedSourceEvaluator().evaluate(BASE,PACK)
    assert e.train==0

def test_hidden_target_not_in_candidate_input():
    e=SealedSourceEvaluator().evaluate("def solve(values):\n    return 0\n",PACK)
    assert e.held_out==0

def test_import_source_fails_closed():
    e=SealedSourceEvaluator().evaluate("import os\ndef solve(values):\n return 0\n",PACK)
    assert e.total_cases==8 and e.train==0

def test_three_seed_evaluation_is_stable():
    for _ in (2,5,8):
        assert SealedSourceEvaluator().evaluate(GOOD,PACK)==SealedSourceEvaluator().evaluate(GOOD,PACK)

def test_empty_pack_fails_closed():
    e=SealedSourceEvaluator().evaluate(GOOD,EvaluationPack([],[],[]))
    assert e.train==0 and e.held_out==0 and not e.regression_ok

def test_improvement_gate():
    ev=SealedSourceEvaluator()
    assert ev.improves(ev.evaluate(BASE,PACK),ev.evaluate(GOOD,PACK))

def test_regression_blocks():
    bad=EvaluationPack(PACK.train,PACK.held_out,[{"public":{"values":[1,1]},"target":999}])
    ev=SealedSourceEvaluator()
    assert not ev.improves(ev.evaluate(BASE,bad),ev.evaluate(GOOD,bad))

def test_attribute_source_fails_closed():
    assert SealedSourceEvaluator().evaluate("def solve(values):\n return values.real\n",PACK).train==0
