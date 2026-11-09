import pytest
from mirror_crystal.operational_language import UOL, Lit, Var, Call, OperationalRule, RuleStatus, VariableSpec
from mirror_crystal.capability_synthesizer import CapabilitySynthesizer, SynthesisGoal, Example, SynthesisStatus, SynthesisError
from mirror_crystal.blueprint_contracts import EvidenceLevel

def test_synthesizes_new_arithmetic_expression_and_uses_holdout():
    goal=SynthesisGoal("increment-and-scale","mathematics",(VariableSpec("x","integer"),),"integer",(Example({"x":1},6),Example({"x":2},9)),(Example({"x":4},15),),constants=(2,3),max_depth=2,candidate_budget=10000)
    result=CapabilitySynthesizer().synthesize(goal)
    assert result.succeeded
    assert result.train_passed==2 and result.validation_passed==1
    assert UOL().execute(result.program,{"x":10})==33

def test_synthesizer_composes_user_fed_rule_into_new_solution():
    rule=OperationalRule("square","square",(VariableSpec("n","integer"),),Call("multiply",(Var("n"),Var("n"))),("mathematics",),RuleStatus.ASSUMED)
    language=UOL((rule,))
    goal=SynthesisGoal("square-plus-self","mathematics",(VariableSpec("x","integer"),),"integer",(Example({"x":2},6),Example({"x":3},12)),(Example({"x":5},30),),constants=(),max_depth=2,candidate_budget=10000)
    result=CapabilitySynthesizer(language).synthesize(goal)
    assert result.succeeded and result.rule_ids==("square",)
    assert language.execute(result.program,{"x":7})==56

def test_requires_disjoint_heldout_inputs():
    with pytest.raises(SynthesisError):SynthesisGoal("bad","math",(VariableSpec("x","integer"),),"integer",(Example({"x":1},1),Example({"x":2},2)),(Example({"x":2},2),))

def test_unverified_rule_is_not_accepted_and_candidate_remains_unverified():
    rule=OperationalRule("square","square",(VariableSpec("n","integer"),),Call("multiply",(Var("n"),Var("n"))),("mathematics",),RuleStatus.UNKNOWN)
    goal=SynthesisGoal("double","mathematics",(VariableSpec("x","integer"),),"integer",(Example({"x":1},2),Example({"x":2},4)),(Example({"x":3},6),),constants=(2,))
    synthesizer=CapabilitySynthesizer(UOL((rule,)))
    report=synthesizer.synthesize(goal)
    assert report.succeeded
    capability=synthesizer.to_candidate_capability(report)
    assert capability.verification.level is EvidenceLevel.UNVERIFIED
    assert not capability.executable

def test_budget_exhaustion_is_explicit():
    goal=SynthesisGoal("hard","mathematics",(VariableSpec("x","integer"),VariableSpec("y","integer")),"integer",(Example({"x":1,"y":2},15),Example({"x":2,"y":3},28)),(Example({"x":4,"y":6},80),),constants=(0,1,2,3,4),max_depth=4,candidate_budget=1)
    report=CapabilitySynthesizer().synthesize(goal)
    assert report.status is SynthesisStatus.BUDGET_EXHAUSTED
