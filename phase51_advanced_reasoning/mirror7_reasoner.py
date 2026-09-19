"""
Mirror 7 — Phase 51: Advanced Reasoning System

Implements a 14-step epistemic reasoning loop:
OBSERVE -> EXTRACT RELEVANT FACTS -> IDENTIFY CURRENT STATE -> DEFINE GOAL ->
LIST CONSTRAINTS -> GENERATE HYPOTHESES -> CHECK EVIDENCE -> APPLY VALID RULES ->
PREDICT CONSEQUENCES -> TEST/SIMULATE -> CHECK CONTRADICTIONS -> SELECT ACTION/CONCLUSION ->
VERIFY RESULT -> UPDATE KNOWLEDGE
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class EpistemicCategory(Enum):
    FACT = auto()
    INFERENCE = auto()
    HYPOTHESIS = auto()
    ASSUMPTION = auto()
    PREDICTION = auto()
    COUNTERFACTUAL = auto()
    UNKNOWN = auto()
    CONTRADICTION = auto()


@dataclass
class ReasoningStatement:
    id: str
    category: EpistemicCategory
    content: Any
    confidence: float
    supporting_evidence: Set[str] = field(default_factory=set)
    assumptions_required: Set[str] = field(default_factory=set)


class EvidenceLedger:
    def __init__(self):
        self.statements: Dict[str, ReasoningStatement] = {}

    def add(self, statement: ReasoningStatement) -> None:
        self.statements[statement.id] = statement

    def get(self, sid: str) -> Optional[ReasoningStatement]:
        return self.statements.get(sid)


class AssumptionLedger:
    def __init__(self):
        self.assumptions: Dict[str, ReasoningStatement] = {}

    def add(self, assumption: ReasoningStatement) -> None:
        if assumption.category != EpistemicCategory.ASSUMPTION:
            raise ValueError("Must be an ASSUMPTION")
        self.assumptions[assumption.id] = assumption


class HypothesisManager:
    def __init__(self):
        self.hypotheses: Dict[str, ReasoningStatement] = {}

    def add(self, hyp: ReasoningStatement) -> None:
        self.hypotheses[hyp.id] = hyp

    def get_plausible(self, evidence_ledger: EvidenceLedger) -> List[ReasoningStatement]:
        return [h for h in self.hypotheses.values()
                if h.category != EpistemicCategory.CONTRADICTION]


class FailureAttributionEngine:
    @staticmethod
    def diagnose_failure(expected: Any, actual: Any, context: Dict[str, Any]) -> str:
        if context.get("novel_entity") or context.get("noise"):
            return "observation_or_external"
        elif context.get("rule_violated"):
            return "world_model"
        elif context.get("unforeseen_interaction"):
            return "causal_model"
        return "reasoning_or_planning"


class AdvancedReasoner:
    def __init__(self):
        self.evidence = EvidenceLedger()
        self.assumptions = AssumptionLedger()
        self.hypotheses = HypothesisManager()
        self.step_counts = 0

    def record_fact(self, fact_id: str, content: Any) -> None:
        self.evidence.add(
            ReasoningStatement(fact_id, EpistemicCategory.FACT, content, 1.0)
        )

    def record_assumption(self, assump_id: str, content: Any) -> None:
        stmt = ReasoningStatement(
            assump_id, EpistemicCategory.ASSUMPTION, content, 0.1
        )
        self.assumptions.add(stmt)
        self.evidence.add(stmt)

    def infer(
        self,
        inference_id: str,
        content: Any,
        evidence_ids: List[str],
        assumption_ids: List[str],
    ) -> ReasoningStatement:
        self.step_counts += 1
        confidence = 1.0

        for evidence_id in evidence_ids:
            evidence = self.evidence.get(evidence_id)
            if not evidence:
                raise ValueError(f"Evidence {evidence_id} not found in ledger.")
            if evidence.category == EpistemicCategory.CONTRADICTION:
                confidence = 0.0
            else:
                confidence *= evidence.confidence

        for assumption_id in assumption_ids:
            assumption = self.assumptions.assumptions.get(assumption_id)
            if not assumption:
                raise ValueError(
                    f"Assumption {assumption_id} not found in ledger."
                )
            confidence *= 0.5

        statement = ReasoningStatement(
            inference_id,
            EpistemicCategory.INFERENCE,
            content,
            confidence,
            set(evidence_ids),
            set(assumption_ids),
        )
        self.evidence.add(statement)
        return statement

    def check_contradictions(self) -> List[str]:
        self.step_counts += 1
        contradictions: List[str] = []
        content_map: Dict[Any, str] = {}

        for statement_id, statement in self.evidence.statements.items():
            if (
                isinstance(statement.content, tuple)
                and statement.content[0] == "NOT"
            ):
                opposite = statement.content[1]
                if opposite in content_map:
                    statement.category = EpistemicCategory.CONTRADICTION
                    self.evidence.statements[
                        content_map[opposite]
                    ].category = EpistemicCategory.CONTRADICTION
                    contradictions.append(statement_id)
            else:
                content_map[statement.content] = statement_id

        return contradictions

    def self_check(self, statement_id: str) -> bool:
        self.step_counts += 1
        statement = self.evidence.get(statement_id)
        if not statement:
            return False

        if statement.category == EpistemicCategory.FACT:
            return True

        for evidence_id in statement.supporting_evidence:
            evidence = self.evidence.get(evidence_id)
            if not evidence or evidence.category == EpistemicCategory.CONTRADICTION:
                return False

        return True

    def early_termination_check(
        self,
        current_confidence: float,
        threshold: float = 0.9,
    ) -> bool:
        return current_confidence >= threshold
