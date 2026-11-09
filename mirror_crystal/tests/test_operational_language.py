import pytest
from mirror_crystal.operational_language import (
    UOL, UOLError, UOLExecutionError, Lit, Var, Call, If, ArrayExpr,
    RecordExpr, Field, Index, Let, MapExpr, FoldExpr, Program,
    OperationalRule, RuleStatus, RuleCatalog, VariableSpec, expr_dict,
    expr_from_dict, program_dict, program_from_dict,
)

def program(body, inputs=(VariableSpec("x","integer"),), **kw):
    return Program("p:test","test",tuple(inputs),body,**kw)

def test_typed_arithmetic_and_postcondition():
    p=program(Call("multiply",(Call("add",(Var("x"),Lit(2))),Lit(3))),postconditions=(Call("greater_than",(Var("$result"),Lit(0))),))
    assert UOL().execute(p,{"x":4})==18

def test_type_error_and_wrong_inputs_are_rejected():
    with pytest.raises(UOLError): UOL().validate(program(Call("add",(Var("x"),Lit("bad")))))
    with pytest.raises(UOLExecutionError): UOL().execute(program(Var("x")),{"y":1})

def test_divide_by_zero_and_output_bounds():
    with pytest.raises(UOLExecutionError): UOL().execute(program(Call("divide",(Var("x"),Lit(0)))),{"x":3})
    bounded=UOL(max_items=2)
    with pytest.raises(UOLExecutionError): bounded.execute(program(MapExpr(Lit([1,2,3]),"item",Var("item")),inputs=()),{})

def test_conditional_and_boolean_runtime_gate():
    p=program(If(Call("greater_than",(Var("x"),Lit(0))),Lit(1),Lit(-1)))
    assert UOL().execute(p,{"x":4})==1
    bad=program(If(Lit("truthy"),Lit(1),Lit(0)))
    with pytest.raises(UOLError): UOL().validate(bad)

def test_structured_values_map_fold_and_index():
    mapped=program(MapExpr(Var("xs"),"item",Call("multiply",(Var("item"),Lit(2)))),inputs=(VariableSpec("xs","array"),))
    assert UOL().execute(mapped,{"xs":[1,2,3]})==(2,4,6)
    folded=program(FoldExpr(Var("xs"),Lit(0),"acc","item",Call("add",(Var("acc"),Var("item")))),inputs=(VariableSpec("xs","array"),))
    assert UOL().execute(folded,{"xs":[1,2,3]})==6
    record=program(Field(RecordExpr({"v":Index(Var("xs"),Lit(1))}),"v"),inputs=(VariableSpec("xs","array"),))
    assert UOL().execute(record,{"xs":[4,9]})==9

def test_rule_usage_preserves_status_and_requires_domain():
    rule=OperationalRule("double","double",(VariableSpec("n","integer"),),Call("multiply",(Var("n"),Lit(2))),("mathematics",),RuleStatus.ASSUMED,"curated-rule")
    language=UOL((rule,))
    p=Program("p:double","double",(VariableSpec("x","integer"),),Call("double",(Var("x"),)),"mathematics",rule_refs=("double",))
    assert language.execute(p,{"x":7})==14
    wrong=Program("p:wrong","wrong",p.inputs,p.body,"biology",rule_refs=("double",))
    with pytest.raises(UOLError):language.validate(wrong)
    disputed=OperationalRule("bad","bad",rule.inputs,rule.body,rule.domains,RuleStatus.DISPUTED)
    with pytest.raises(UOLExecutionError):UOL((disputed,)).execute(Program("p:bad","bad",p.inputs,Call("bad",(Var("x"),)),"mathematics",rule_refs=("bad",)),{"x":1})

def test_rule_cycle_is_rejected():
    r1=OperationalRule("r1","r1",(VariableSpec("x","integer"),),Call("r2",(Var("x"),)))
    r2=OperationalRule("r2","r2",(VariableSpec("x","integer"),),Call("r1",(Var("x"),)))
    with pytest.raises(UOLError):RuleCatalog((r1,r2))

def test_ast_and_program_round_trip():
    expr=Call("add",(Var("x"),Lit(5)))
    assert expr_from_dict(expr_dict(expr))==expr
    p=program(expr)
    assert program_from_dict(program_dict(p))==p

def test_precondition_and_postcondition_enforced():
    p=program(Var("x"),preconditions=(Call("greater_than",(Var("x"),Lit(0))),),postconditions=(Call("less_than",(Var("$result"),Lit(10))),))
    assert UOL().execute(p,{"x":5})==5
    with pytest.raises(UOLExecutionError):UOL().execute(p,{"x":-1})
    with pytest.raises(UOLExecutionError):UOL().execute(p,{"x":12})

def test_host_handler_is_only_bounded_uol_adapter():
    p=program(Call("add",(Var("x"),Lit(1))))
    assert UOL().native_handler(p)({"x":3})=={"result":4}
