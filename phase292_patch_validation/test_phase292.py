from .mirror7_phase292 import PatchValidator
from phase291_patch_model import PatchOp, PatchPlan

def plan():
    return PatchPlan("solver.py",(PatchOp("replace_literal","window","1","2"),),"x")

def test_valid_plan_passes():
    assert PatchValidator().validate_plan(plan()).accepted

def test_large_replacement_rejected():
    long="x"*201
    p=PatchPlan("solver.py",(PatchOp("replace_expression","x","a",long),),"x")
    assert not PatchValidator().validate_plan(p).accepted

def test_valid_source_passes():
    r=PatchValidator().validate_source("def solve(x):\n    return x+1\n")
    assert r.accepted

def test_syntax_error_rejected():
    assert not PatchValidator().validate_source("def solve(:").accepted

def test_import_rejected():
    assert not PatchValidator().validate_source("import os\ndef solve():\n return 1\n").accepted

def test_eval_rejected():
    assert not PatchValidator().validate_source("def solve(x):\n return eval(x)\n").accepted

def test_three_seed_validation_stable():
    for _ in (2,5,8):
        assert PatchValidator().validate_source("def solve(x):\n return x+1\n").accepted

def test_empty_plan_rejected():
    p=PatchPlan("solver.py",(),"x")
    assert not PatchValidator().validate_plan(p).accepted

def test_import_from_rejected():
    assert not PatchValidator().validate_source("from os import path\n\ndef solve():\n return 1\n").accepted

def test_compile_call_rejected():
    assert not PatchValidator().validate_source("def solve(x):\n return compile(x,'','exec')\n").accepted

def test_while_loop_rejected():
    assert not PatchValidator().validate_source("def solve(x):\n    while x:\n        x=x-1\n    return x\n").accepted

def test_attribute_access_rejected():
    assert not PatchValidator().validate_source("def solve(x):\n    return x.real\n").accepted

def test_if_expression_control_is_allowed():
    assert PatchValidator().validate_source("def solve(x):\n    if x[0] > 0:\n        return x[0]\n    return 0\n").accepted
