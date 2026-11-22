"""EVALUATOR-ONLY hand-coded hidden simulator ceiling oracle.

Provides the 100% accuracy / 0% collateral ceiling sanity check for the S0 benchmark
harness. Never import this module from learner code or non-oracle baselines.
"""
from __future__ import annotations

from typing import Any, Sequence

from research.cpu_learning.evaluation.baselines import BaseS0Solver
from research.cpu_learning.evaluation.metrics import BitBudgetAudit, TaskPrediction
from research.cpu_learning.simulator.hidden_actions import (
    HIDDEN_OPERATOR_SCHEMAS,
    apply_hidden_sequence,
)
from research.cpu_learning.simulator.state_graph import (
    EvaluationTaskPublic,
    TrainingEpisode,
    canonical_json_bytes,
    compressed_bit_length,
)

EVALUATOR_ONLY: bool = True


class HiddenOracleBaseline(BaseS0Solver):
    """Evaluator-only ceiling oracle using the ground-truth hidden transition simulator."""

    name = "hand_coded_hidden_oracle"
    category = "evaluator_ceiling"
    evaluator_only = True

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        self._check_not_frozen()

    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        final_s, inter_s, _deltas, _applied, _reasons = apply_hidden_sequence(
            task.initial_state, task.actions, mask=task.observation_mask
        )
        return TaskPrediction(
            predicted_final_state=final_s,
            abstained=False,
            retrieval_hits=0,
            memory_accesses=0,
            step_predictions=inter_s,
            metadata={"oracle": True},
        )

    def recovered_operator_schemas(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for aid, schema in HIDDEN_OPERATOR_SCHEMAS.items():
            out[aid] = {
                "required_facts": sorted(schema.required_facts),
                "required_types": sorted(schema.required_types),
                "required_attr_guards": sorted(schema.required_attr_guards),
                "add_facts": sorted(schema.add_facts),
                "delete_facts": sorted(schema.delete_facts),
                "attribute_edits": sorted(schema.attribute_edits),
            }
        return out

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        schema_bytes = canonical_json_bytes(self.recovered_operator_schemas())
        op_bits = len(schema_bytes) * 8
        mdl_bits = compressed_bit_length(schema_bytes)
        return BitBudgetAudit(
            raw_training_data_bits=raw_training_bits,
            compressed_training_data_bits=compressed_training_bits,
            episode_store_bits=0,
            active_bundle_bits=0,
            operator_library_bits=op_bits,
            compressed_model_mdl_bits=mdl_bits,
            compression_ratio_vs_raw=mdl_bits / max(1, raw_training_bits),
        )
