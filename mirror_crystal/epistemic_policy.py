"""Conservative, shared policy for labeling answer evidence strength."""
from __future__ import annotations
import math

class EpistemicTierPolicy:
    """Separate deterministic execution from retrieval similarity and factual support."""
    EXACT_MODES = frozenset({
        "session_constraint_lock", "slot_formula_lock", "symbolic_cas_exact", "mcts_program_synthesis",
    })
    RETRIEVAL_ONLY_MODES = frozenset({
        "direct_crystal_lock", "slot_template_lock", "multi_hop_rule_chain", "multi_hop_relational_chain",
    })

    @classmethod
    def classify(cls, confidence: float, mode: str, sandbox_verified: bool, query_coverage: float = 1.0) -> str:
        """High retrieval similarity is not a proof of factual correctness."""
        score, coverage = float(confidence), float(query_coverage)
        if not math.isfinite(score):
            score = 0.0
        if not math.isfinite(coverage):
            coverage = 0.0
        score, coverage = min(1.0, max(0.0, score)), min(1.0, max(0.0, coverage))
        if mode == "bicameral_creative_synthesis":
            return "creative_synthesis"
        if mode == "clarification_needed":
            return "diagnostic_clarification"
        if mode == "unmatched_honest_abstain" or score <= 0.0:
            return "honest_abstain"
        if mode in cls.EXACT_MODES:
            return "verified_exact"
        if mode in cls.RETRIEVAL_ONLY_MODES:
            return "moderate_confidence"
        if sandbox_verified or (score >= 0.58 and coverage >= 0.70):
            return "high_confidence"
        return "moderate_confidence"
