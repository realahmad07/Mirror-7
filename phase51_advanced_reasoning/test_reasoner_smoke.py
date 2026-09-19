from phase51_advanced_reasoning.mirror7_reasoner import (
    AdvancedReasoner,
    EpistemicCategory,
    FailureAttributionEngine,
)


def test_epistemic_ledgers():
    reasoner = AdvancedReasoner()
    reasoner.record_fact("f", "data")
    reasoner.record_assumption("a", "assumption")
    inference = reasoner.infer("i", "conclusion", ["f"], ["a"])
    assert reasoner.evidence.get("f").category == EpistemicCategory.FACT
    assert reasoner.assumptions.assumptions["a"].category == EpistemicCategory.ASSUMPTION
    assert inference.category == EpistemicCategory.INFERENCE
    assert inference.confidence < 1.0


def test_contradiction_detection():
    reasoner = AdvancedReasoner()
    reasoner.record_fact("f", "data")
    reasoner.infer("i", "conclusion", ["f"], [])
    assert reasoner.self_check("i")
    reasoner.record_fact("nf", ("NOT", "data"))
    reasoner.check_contradictions()
    assert not reasoner.self_check("i")


def test_failure_attribution():
    assert FailureAttributionEngine.diagnose_failure(1, 0, {"noise": True}) == "observation_or_external"
    assert FailureAttributionEngine.diagnose_failure(1, 0, {"rule_violated": True}) == "world_model"
    assert FailureAttributionEngine.diagnose_failure(1, 0, {}) == "reasoning_or_planning"


def test_early_termination():
    reasoner = AdvancedReasoner()
    assert reasoner.early_termination_check(0.95)
    assert not reasoner.early_termination_check(0.5)
