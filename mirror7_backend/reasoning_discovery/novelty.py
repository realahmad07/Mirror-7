"""
Novelty and Generalization Evaluator Module.
Distinguishes genuine algorithmic novelty from superficial syntax or variable renaming.
"""
from __future__ import annotations

import ast
from typing import Sequence
from .schema import NoveltyReport, ReasoningStrategy, TestCase
from .sandbox import SafeSandbox


class ASTStructuralExtractor(ast.NodeVisitor):
    """Extracts sequence of normalized AST node types ignoring variable names and literals."""

    def __init__(self):
        self.node_sequence: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.node_sequence.append(type(node).__name__)
        super().generic_visit(node)


def compute_ast_structural_similarity(code_a: str, code_b: str) -> float:
    """Computes Jaccard similarity between the normalized AST node n-grams of two procedures."""
    try:
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)
    except SyntaxError:
        return 0.0

    ext_a = ASTStructuralExtractor()
    ext_b = ASTStructuralExtractor()
    ext_a.visit(tree_a)
    ext_b.visit(tree_b)

    seq_a = ext_a.node_sequence
    seq_b = ext_b.node_sequence

    if not seq_a or not seq_b:
        return 1.0 if seq_a == seq_b else 0.0

    # Bigrams of AST node types
    bigrams_a = set(zip(seq_a[:-1], seq_a[1:]))
    bigrams_b = set(zip(seq_b[:-1], seq_b[1:]))

    intersection = len(bigrams_a.intersection(bigrams_b))
    union = len(bigrams_a.union(bigrams_b))
    return intersection / max(1, union)


class NoveltyEvaluator:
    """Evaluates whether a candidate procedure is genuinely novel and generalizes to unseen cases."""

    def __init__(self, sandbox: SafeSandbox | None = None):
        self.sandbox = sandbox or SafeSandbox()

    def evaluate_novelty(
        self,
        candidate: ReasoningStrategy,
        known_strategies: Sequence[ReasoningStrategy],
        unseen_test_cases: Sequence[TestCase],
    ) -> NoveltyReport:
        differences: list[str] = []
        max_similarity = 0.0

        for known in known_strategies:
            sim = compute_ast_structural_similarity(
                candidate.procedure_code, known.procedure_code
            )
            if sim > max_similarity:
                max_similarity = sim

            # Check algorithmic paradigm differences
            if candidate.complexity_class != known.complexity_class and known.complexity_class != "unknown":
                differences.append(
                    f"Different asymptotic complexity from {known.name}: {candidate.complexity_class} vs {known.complexity_class}"
                )

            # Check control flow differences
            cand_has_while = "While" in candidate.procedure_code
            known_has_while = "While" in known.procedure_code
            if cand_has_while != known_has_while:
                differences.append(f"Distinct control-flow structure from {known.name} (While-loop presence differs)")

        # Evaluate generalization on unseen held-out test cases
        gen_result = self.sandbox.execute(candidate, unseen_test_cases)
        gen_score = gen_result.passed_tests / max(1, gen_result.total_tests)

        # Genuine novelty requires: structural similarity < 0.85 AND passing generalization
        is_novel = max_similarity < 0.85 and gen_result.is_correct

        confidence = 0.9 if gen_result.is_correct else 0.4

        return NoveltyReport(
            strategy_id=candidate.strategy_id,
            is_novel=is_novel,
            structural_similarity=round(max_similarity, 3),
            algorithmic_differences=tuple(set(differences)),
            generalization_score=round(gen_score, 3),
            confidence=confidence,
        )
