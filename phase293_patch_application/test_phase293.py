from phase291_patch_model import PatchOp, PatchPlan
from .mirror7_phase293 import PatchApplier

BASE="def solve(values):\n    return values[-1]\n"

def test_exact_patch_applies():
    p=PatchPlan("solver.py",(PatchOp("replace_expression","return","values[-1]","values[-1] + (values[-1]-values[-2])"),),"x")
    r=PatchApplier().apply(BASE,p)
    assert r.applied and "values[-2]" in r.source

def test_missing_match_fails_closed():
    p=PatchPlan("solver.py",(PatchOp("replace_expression","return","missing","x"),),"x")
    r=PatchApplier().apply(BASE,p)
    assert not r.applied and r.source==BASE

def test_ambiguous_match_fails_closed():
    src="def solve(values):\n    x=values[-1]\n    return values[-1]\n"
    p=PatchPlan("solver.py",(PatchOp("replace_expression","x","values[-1]","1"),),"x")
    r=PatchApplier().apply(src,p)
    assert not r.applied

def test_invalid_plan_does_not_mutate():
    p=PatchPlan("solver.py",(),"x")
    r=PatchApplier().apply(BASE,p)
    assert not r.applied and r.source==BASE

def test_invalid_source_result_is_rejected():
    p=PatchPlan("solver.py",(PatchOp("replace_expression","x","values[-1]","foo.bar"),),"x")
    r=PatchApplier().apply(BASE,p)
    assert not r.applied

def test_three_seed_application_is_deterministic():
    p=PatchPlan("solver.py",(PatchOp("replace_expression","x","values[-1]","values[-1]+1"),),"x")
    for _ in (2,5,8):
        assert PatchApplier().apply(BASE,p)==PatchApplier().apply(BASE,p)

def test_old_source_is_preserved_on_failure():
    p=PatchPlan("solver.py",(PatchOp("replace_expression","x","values[-1]","values[-1]+1"),PatchOp("replace_expression","y","missing","0")),"x")
    r=PatchApplier().apply(BASE,p)
    assert not r.applied and r.source==BASE

def test_multiple_exact_replacements_work():
    src="def solve(values):\n    return values[-1] + values[-1]\n"
    p=PatchPlan("solver.py",(PatchOp("replace_expression","x","values[-1] + values[-1]","values[-1] + 1"),),"x")
    r=PatchApplier().apply(src,p)
    assert r.applied

def test_result_reason_is_explicit():
    p=PatchPlan("solver.py",(),"x")
    assert PatchApplier().apply(BASE,p).reason=="plan rejected"
