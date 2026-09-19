from .mirror7_phase101 import ContextInquiry

def test_missing_context_requests_specific_details():
    q=ContextInquiry(["goal","constraints"]).assess({"goal":"x"})
    assert q and "constraints" in q.question and q.missing==("constraints",)

def test_conflict_requests_context_not_guessing():
    q=ContextInquiry().assess({"goal":"x"}, conflicts=["source_a_vs_source_b"])
    assert q and "conflicting" in q.reason and q.assumptions==()

def test_complete_context_abstains_from_inquiry():
    assert ContextInquiry(["goal"]).assess({"goal":"x"}) is None

def test_empty_values_are_missing():
    q=ContextInquiry(["topic"]).assess({"topic":""})
    assert q and q.missing==("topic",)
