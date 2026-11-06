"""
Problem Exploration Module.
Discovers representations, identifies constraints vs assumptions, and formulates test contracts.
"""
from __future__ import annotations

import re
from typing import Any
from .schema import ProblemRepresentation, TestCase


class ProblemExplorer:
    """Explores and formalizes raw problem descriptions into structured testable representations."""

    def formalize_problem(
        self,
        problem_id: str,
        objective: str,
        raw_description: str,
        domain: str = "algorithm",
        sample_cases: list[tuple[tuple[Any, ...], Any]] | None = None,
    ) -> ProblemRepresentation:
        # Separate explicit constraints from assumed parameters
        constraints: list[str] = []
        assumptions: list[str] = []

        # Heuristic classification of statements
        sentences = [s.strip() for s in raw_description.split(".") if s.strip()]
        for s in sentences:
            s_lower = s.lower()
            if any(k in s_lower for k in ["must", "cannot", "only", "strictly", "positive", "non-negative", "limit"]):
                constraints.append(s)
            elif any(k in s_lower for k in ["assume", "assuming", "default", "typical", "standard"]):
                assumptions.append(s)
            else:
                constraints.append(s)

        test_cases: list[TestCase] = []
        if sample_cases:
            for args, expected in sample_cases:
                test_cases.append(TestCase(input_args=args, expected_output=expected))

        return ProblemRepresentation(
            problem_id=problem_id,
            objective=objective,
            knowns={"description": raw_description},
            constraints=tuple(constraints),
            assumptions=tuple(assumptions),
            domain=domain,
            test_cases=tuple(test_cases),
        )

    def formulate_dual_representations(
        self, base_problem: ProblemRepresentation
    ) -> tuple[ProblemRepresentation, ProblemRepresentation]:
        """Generate dual formulations (e.g., iterative state-space vs recursive functional)."""
        iterative_rep = ProblemRepresentation(
            problem_id=f"{base_problem.problem_id}_iterative",
            objective=f"Iterative accumulation formulation: {base_problem.objective}",
            knowns=dict(base_problem.knowns),
            constraints=base_problem.constraints + ("State must be updated in place or with bounded memory",),
            assumptions=base_problem.assumptions + ("Sequential loop evaluation is admissible",),
            domain=base_problem.domain,
            test_cases=base_problem.test_cases,
        )

        functional_rep = ProblemRepresentation(
            problem_id=f"{base_problem.problem_id}_functional",
            objective=f"Functional / divide-and-conquer formulation: {base_problem.objective}",
            knowns=dict(base_problem.knowns),
            constraints=base_problem.constraints + ("Subproblem recursion depth must terminate",),
            assumptions=base_problem.assumptions + ("Problem exhibits optimal substructure",),
            domain=base_problem.domain,
            test_cases=base_problem.test_cases,
        )
        return iterative_rep, functional_rep
