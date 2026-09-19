from .mirror7_phase291 import PatchOp, PatchPlan, PatchPlanValidator

def test_valid_patch_plan():
    p=PatchPlan("solver.py",(PatchOp("replace_literal","window","1","2"),),"measured gain")
    assert PatchPlanValidator().validate(p)

def test_multiple_operations_allowed():
    p=PatchPlan("solver.py",(PatchOp("replace_literal","a","1","2"),PatchOp("replace_expression","b","x","x+1")),"two edits")
    assert PatchPlanValidator().validate(p)

def test_empty_plan_rejected():
    p=PatchPlan("solver.py",(),"x")
    assert not PatchPlanValidator().validate(p)

def test_unknown_kind_rejected():
    p=PatchPlan("solver.py",(PatchOp("execute_source","x","a","b"),),"x")
    assert not PatchPlanValidator().validate(p)

def test_same_old_new_rejected():
    p=PatchPlan("solver.py",(PatchOp("replace_literal","x","1","1"),),"x")
    assert not PatchPlanValidator().validate(p)

def test_three_seed_shape_stable():
    v=PatchPlanValidator()
    for _ in (2,5,8):
        p=PatchPlan("solver.py",(PatchOp("replace_literal","window","1","2"),),"x")
        assert v.validate(p)

def test_missing_target_rejected():
    p=PatchPlan("",(PatchOp("replace_literal","x","1","2"),),"x")
    assert not PatchPlanValidator().validate(p)

def test_missing_rationale_rejected():
    p=PatchPlan("solver.py",(PatchOp("replace_literal","x","1","2"),),"")
    assert not PatchPlanValidator().validate(p)

def test_nonempty_path_required():
    p=PatchPlan("solver.py",(PatchOp("replace_literal","","1","2"),),"x")
    assert not PatchPlanValidator().validate(p)
