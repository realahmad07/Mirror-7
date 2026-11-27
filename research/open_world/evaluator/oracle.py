"""EVALUATOR-ONLY strictly isolated oracle ceiling baseline.

ISOLATION CONTRACT:
- Lives exclusively inside ``research/open_world/evaluator/oracle.py``.
- AST-verified that ZERO learner or non-oracle baseline modules import this file.
- Used solely to verify scoring-pipeline ceilings (100% accuracy, 100% evidence F1,
  100% unanswerable abstention recall, 0.0 Brier/ECE) and NEVER treated as a learner.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from research.open_world.contracts import (
    OpenWorldObservation,
    OpenWorldPrediction,
    OpenWorldSolver,
    OpenWorldTrainingExample,
    SolverMemoryFootprint,
)
from research.open_world.evaluator.hidden_eval import HiddenGoldRecord


class EvaluatorOnlyOracle(OpenWorldSolver):
    """Strictly isolated evaluator-only ceiling returning exact sealed gold records."""

    solver_id: str = "evaluator_only_oracle"
    category: str = "evaluator_ceiling"
    evaluator_only: bool = True

    def __init__(
        self,
        hidden_gold: Mapping[str, HiddenGoldRecord],
        seed: int = 101,
    ) -> None:
        super().__init__(seed=seed)
        self._hidden_gold: Mapping[str, HiddenGoldRecord] = hidden_gold

    def fit(self, benchmark_id: str, train_examples: Sequence[OpenWorldTrainingExample]) -> None:
        self._require_unfrozen()
        # Evaluator-only oracle does not learn from training examples.

    def predict(self, observation: OpenWorldObservation) -> OpenWorldPrediction:
        gold = self._hidden_gold.get(observation.task_id)
        if gold is None:
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                abstained=True,
                abstain_reason="oracle_missing_gold_record",
                confidence=0.0,
                uncertainty=1.0,
            )

        if gold.is_unanswerable:
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                predicted_answer=None,
                predicted_choice_index=None,
                abstained=True,
                abstain_reason="gold_unanswerable",
                confidence=1.0,
                uncertainty=0.0,
                cited_evidence_ids=(),
                memory_reads=1,
                metadata={"evaluator_only_oracle": True},
            )

        best_ans = gold.gold_answers[0] if gold.gold_answers else None
        return OpenWorldPrediction(
            task_id=observation.task_id,
            benchmark_id=observation.benchmark_id,
            predicted_answer=best_ans,
            predicted_choice_index=gold.gold_choice_index,
            abstained=False,
            confidence=1.0,
            uncertainty=0.0,
            cited_evidence_ids=gold.gold_evidence_ids,
            cited_evidence_spans=(best_ans,) if best_ans else (),
            memory_reads=1,
            metadata={"evaluator_only_oracle": True},
        )

    def memory_footprint(self) -> SolverMemoryFootprint:
        return SolverMemoryFootprint(
            serialized_bytes=0,
            compressed_mdl_bits=0,
            active_entry_count=0,
            details={"evaluator_only": True, "excluded_from_learner_memory": True},
        )
