from __future__ import annotations
import pytest
from mirror_crystal.epistemic_policy import EpistemicTierPolicy
from mirror_crystal.phase2_cognitive_core import EpistemicConfidenceCalibrator

@pytest.mark.parametrize(
    ("confidence", "mode", "sandbox_verified", "coverage", "expected"),
    [
        (1.0, "symbolic_cas_exact", True, 1.0, "verified_exact"),
        (1.0, "session_constraint_lock", True, 1.0, "verified_exact"),
        (1.0, "slot_formula_lock", False, 1.0, "verified_exact"),
        (1.0, "mcts_program_synthesis", True, 1.0, "verified_exact"),
        (0.99, "direct_crystal_lock", False, 1.0, "moderate_confidence"),
        (0.99, "slot_template_lock", False, 1.0, "moderate_confidence"),
        (0.99, "multi_hop_rule_chain", True, 1.0, "moderate_confidence"),
        (0.99, "multi_hop_relational_chain", True, 1.0, "moderate_confidence"),
        (0.95, "direct_crystal_lock", True, 1.0, "high_confidence"),
        (0.80, "general_match", True, 0.9, "high_confidence"),
        (0.50, "general_match", False, 0.9, "moderate_confidence"),
        (0.0, "unmatched_honest_abstain", False, 0.0, "honest_abstain"),
        (0.75, "bicameral_creative_synthesis", False, 0.8, "creative_synthesis"),
        (0.75, "clarification_needed", False, 0.8, "diagnostic_clarification"),
    ],
)
def test_epistemic_policy_cases(confidence, mode, sandbox_verified, coverage, expected):
    assert EpistemicTierPolicy.classify(confidence, mode, sandbox_verified, coverage) == expected
    assert EpistemicConfidenceCalibrator.classify_tier(
        confidence, mode, sandbox_verified, query_coverage=coverage
    ) == expected

def test_non_finite_scores_never_escalate_to_verified():
    assert EpistemicTierPolicy.classify(float("nan"), "general_match", True) != "verified_exact"
    assert EpistemicTierPolicy.classify(float("inf"), "direct_crystal_lock", False) == "honest_abstain"
