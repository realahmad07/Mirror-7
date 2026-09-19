from .mirror7_phase103 import ValidatedLearner

def test_requires_corroboration():
    l=ValidatedLearner(min_support=2); assert l.observe("a",1) is None; c=l.observe("a",1); assert c and c.confidence==1

def test_single_conflict_does_not_replace_stable_rule():
    l=ValidatedLearner(min_support=2)
    l.observe("a",1); l.observe("a",1); c=l.observe("a",9)
    assert c and c.value==1 and l.accepted()[0].value==1

def test_majority_conflict_can_fail_closed():
    l=ValidatedLearner(min_support=3,min_confidence=.75)
    l.observe("a",1); l.observe("a",2); l.observe("a",2)
    assert l.accepted()==[]

def test_rule_budget_is_bounded():
    l=ValidatedLearner(min_support=1,min_confidence=1,max_rules=2)
    for i in range(5): l.observe(str(i),i)
    assert len(l.accepted())<=2
