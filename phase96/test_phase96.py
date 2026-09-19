from .mirror7_phase96 import ContinualConsolidator

def test_repeated_evidence_promotes():
    c=ContinualConsolidator(min_support=3)
    assert c.observe(("a",),(1,)) is None
    assert c.observe(("a",),(1,)) is None
    r=c.observe(("a",),(1,))
    assert r and r.support==3

def test_noise_does_not_replace_strong_rule():
    c=ContinualConsolidator(min_support=2)
    c.observe(("a",),(1,)); c.observe(("a",),(1,)); old=c.recall(("a",))
    c.observe(("a",),(9,))
    assert c.recall(("a",))==old

def test_consolidated_rules_bounded():
    c=ContinualConsolidator(max_rules=2,min_support=1)
    c.observe(("a",),(1,)); c.observe(("b",),(2,)); c.observe(("c",),(3,))
    assert len(c.rules)<=2

def test_invalid_value():
    c=ContinualConsolidator()
    try: c.observe(("a",),(float("inf"),)); assert False
    except ValueError: pass
