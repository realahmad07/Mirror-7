"""Scoring metrics, retrieval-vs-learning audit, MDL bit-budget accounting, and CIs.

Implements all pre-registered metrics required by S0 and future S1-S10 gates:
- Exact final-state accuracy (strictly excluding retrieval hits and abstentions)
- Intermediate step accuracy
- Collateral change rate on unrelated facts/attributes
- Abstention rate and error rate
- Test-time episode-store retrieval audit
- Operator recovery precision/recall (evaluator-only against hidden schemas)
- Memory/MDL bit accounting (raw training bits, episode store, active bundle,
  operator library, compressed model MDL)
- Multi-seed exact Student-t 95% confidence intervals
"""
from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from research.cpu_learning.simulator.hidden_actions import (
    HIDDEN_OPERATOR_SCHEMAS,
    HiddenOperatorSchema,
)
from research.cpu_learning.simulator.state_graph import (
    EvaluationCase,
    StateGraph,
    TrainingEpisode,
    canonical_json_bytes,
    compressed_bit_length,
)

# Exact two-sided 95% Student-t critical values by degrees of freedom (df = n - 1)
_T_CRIT_95: dict[int, float] = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    14: 2.144787,
    19: 2.093024,
    29: 2.045230,
}


def t_critical_95(n: int) -> float:
    """Two-sided 95% Student-t critical value for sample size n."""
    if n <= 1:
        return 0.0
    df = n - 1
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    return 1.959964


def summarize_across_seeds(
    values: Sequence[float],
    *,
    bounds: tuple[float | None, float | None] | None = None,
) -> dict[str, Any]:
    """Compute mean, sample std, SE, and exact 95% Student-t CI (optionally clipped to bounds)."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "se": 0.0,
            "ci95_half": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "min": 0.0,
            "max": 0.0,
            "per_seed": [],
        }
    mean = sum(vals) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        std = math.sqrt(var)
        se = std / math.sqrt(n)
        half = t_critical_95(n) * se
    else:
        std = 0.0
        se = 0.0
        half = 0.0
    ci_lo = mean - half
    ci_hi = mean + half
    if bounds is not None:
        lo_bound, hi_bound = bounds
        if lo_bound is not None:
            ci_lo = max(lo_bound, ci_lo)
        if hi_bound is not None:
            ci_hi = min(hi_bound, ci_hi)
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "se": round(se, 6),
        "ci95_half": round(half, 6),
        "ci95_low": round(ci_lo, 6),
        "ci95_high": round(ci_hi, 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "per_seed": [round(x, 6) for x in vals],
    }


@dataclass(frozen=True)
class TaskPrediction:
    """Standard prediction output returned by every S0 baseline and future learner."""

    predicted_final_state: StateGraph | None
    abstained: bool = False
    retrieval_hits: int = 0
    memory_accesses: int = 0
    step_predictions: tuple[StateGraph | None, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskScore:
    """Per-task audited evaluation record."""

    task_id: str
    split_id: str
    depth: int
    exact_final_match: bool
    abstained: bool
    is_retrieval_hit: bool
    excluded_from_learning_score: bool
    collateral_changes: int
    unrelated_fact_count: int
    collateral_rate: float
    intermediate_matches: int
    total_steps: int
    memory_accesses: int
    gold_any_step_applied: bool
    exact_train_key_overlap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split_id": self.split_id,
            "depth": self.depth,
            "exact_final_match": self.exact_final_match,
            "abstained": self.abstained,
            "is_retrieval_hit": self.is_retrieval_hit,
            "exact_train_key_overlap": self.exact_train_key_overlap,
            "excluded_from_learning_score": self.excluded_from_learning_score,
            "collateral_changes": self.collateral_changes,
            "unrelated_fact_count": self.unrelated_fact_count,
            "collateral_rate": round(self.collateral_rate, 6),
            "intermediate_matches": self.intermediate_matches,
            "total_steps": self.total_steps,
            "memory_accesses": self.memory_accesses,
            "gold_any_step_applied": self.gold_any_step_applied,
        }


_MISSING_ATTR = object()


def compute_collateral_changes(
    initial_state: StateGraph,
    gold_final_state: StateGraph,
    predicted_final_state: StateGraph | None,
    abstained: bool,
) -> tuple[int, int, float]:
    """Count unrelated facts, attributes, entity types, and hallucinated entities altered incorrectly.

    Distinguishes primary action errors (wrong value on a fact predicate or entity attribute
    that genuinely changed in gold) from collateral corruption on unrelated facts, unrelated
    attributes/types (even on entities whose other facts changed), deleted entities, and
    hallucinated entities.
    """
    preserved_gold_facts = initial_state.facts & gold_final_state.facts
    init_ents = initial_state.entity_map()
    gold_ents = gold_final_state.entity_map()

    gold_changed_fact_keys = {(f[0], f[1]) for f in (initial_state.facts ^ gold_final_state.facts)}
    gold_changed_attr_keys: set[tuple[str, str]] = set()
    gold_changed_types: set[str] = set()
    preserved_attr_count = 0

    for eid in set(init_ents) | set(gold_ents):
        i_ent = init_ents.get(eid)
        g_ent = gold_ents.get(eid)
        if i_ent is None or g_ent is None or i_ent.entity_type != g_ent.entity_type:
            gold_changed_types.add(eid)
        i_attrs = i_ent.attr_dict() if i_ent is not None else {}
        g_attrs = g_ent.attr_dict() if g_ent is not None else {}
        for attr_name in set(i_attrs) | set(g_attrs):
            if i_attrs.get(attr_name, _MISSING_ATTR) != g_attrs.get(attr_name, _MISSING_ATTR):
                gold_changed_attr_keys.add((eid, attr_name))
            elif attr_name in g_attrs:
                preserved_attr_count += 1

    unrelated_count = max(1, len(preserved_gold_facts) + preserved_attr_count)
    if abstained or predicted_final_state is None:
        return (0, unrelated_count, 0.0)

    pred_ents = predicted_final_state.entity_map()

    # 1. Preserved gold facts erroneously deleted
    deleted_unrelated = len(preserved_gold_facts - predicted_final_state.facts)

    # 2. Spurious facts injected on ANY (entity, predicate) not modified by the gold transition
    spurious_facts = predicted_final_state.facts - gold_final_state.facts
    spurious_unrelated = sum(
        1 for f in spurious_facts if (f[0], f[1]) not in gold_changed_fact_keys
    )

    # 3. Unchanged entity deletions, entity_type mutations, unrelated attribute mutations,
    #    and hallucinated extraneous entities
    entity_attr_collateral = 0
    for eid, g_ent in gold_ents.items():
        p_ent = pred_ents.get(eid)
        if p_ent is None:
            entity_attr_collateral += 1
            continue
        if eid not in gold_changed_types and p_ent.entity_type != g_ent.entity_type:
            entity_attr_collateral += 1
        g_attrs = g_ent.attr_dict()
        p_attrs = p_ent.attr_dict()
        for attr_name in set(g_attrs) | set(p_attrs):
            if (eid, attr_name) in gold_changed_attr_keys:
                continue
            if g_attrs.get(attr_name, _MISSING_ATTR) != p_attrs.get(attr_name, _MISSING_ATTR):
                entity_attr_collateral += 1

    hallucinated_entities = len(set(pred_ents) - set(gold_ents))
    total_collateral = (
        deleted_unrelated + spurious_unrelated + entity_attr_collateral + hallucinated_entities
    )
    rate = total_collateral / float(unrelated_count)
    return (total_collateral, unrelated_count, rate)


def score_task(
    case: EvaluationCase,
    prediction: TaskPrediction,
    training_exact_keys: frozenset[str],
    *,
    is_canary_split: bool = False,
) -> TaskScore:
    """Score one predicted task against hidden ground truth with full multi-step retrieval auditing."""
    pub = case.public
    hid = case.hidden

    # Independent multi-step retrieval audit: inspect EVERY step (0..depth-1) on both the
    # true pre-step state and the solver's predicted pre-step state against training_exact_keys.
    any_step_exact = False
    for step_idx in range(len(pub.actions)):
        gold_pre_state = pub.initial_state if step_idx == 0 else hid.gold_intermediate_states[step_idx - 1]
        if pub.step_state_action_key(gold_pre_state, step_idx) in training_exact_keys:
            any_step_exact = True
            break
        if step_idx > 0 and step_idx - 1 < len(prediction.step_predictions):
            pred_pre_state = prediction.step_predictions[step_idx - 1]
            if (
                pred_pre_state is not None
                and pub.step_state_action_key(pred_pre_state, step_idx) in training_exact_keys
            ):
                any_step_exact = True
                break

    is_retrieval_hit = bool(
        prediction.retrieval_hits > 0 or (any_step_exact and prediction.memory_accesses > 0)
    )
    # Any exact training (state, action) overlap on ANY step is excluded from unseen learning
    # accuracy even if a dishonest solver reports memory_accesses == 0.
    excluded = bool(is_retrieval_hit or any_step_exact or is_canary_split)

    if prediction.abstained or prediction.predicted_final_state is None:
        exact_match = False
    else:
        exact_match = prediction.predicted_final_state == hid.gold_final_state

    col_changes, unrel_count, col_rate = compute_collateral_changes(
        pub.initial_state,
        hid.gold_final_state,
        prediction.predicted_final_state,
        prediction.abstained,
    )

    inter_matches = 0
    if prediction.step_predictions and not prediction.abstained:
        for p_state, g_state in zip(prediction.step_predictions, hid.gold_intermediate_states):
            if p_state is not None and p_state == g_state:
                inter_matches += 1
    elif exact_match and len(hid.gold_intermediate_states) == 1:
        inter_matches = 1

    return TaskScore(
        task_id=pub.task_id,
        split_id=pub.split_id,
        depth=pub.depth,
        exact_final_match=exact_match,
        abstained=prediction.abstained,
        is_retrieval_hit=is_retrieval_hit,
        exact_train_key_overlap=any_step_exact,
        excluded_from_learning_score=excluded,
        collateral_changes=col_changes,
        unrelated_fact_count=unrel_count,
        collateral_rate=col_rate,
        intermediate_matches=inter_matches,
        total_steps=len(hid.gold_intermediate_states),
        memory_accesses=prediction.memory_accesses,
        gold_any_step_applied=any(hid.gold_step_applied),
    )


def aggregate_split_scores(scores: Sequence[TaskScore]) -> dict[str, Any]:
    """Aggregate per-task scores into split metrics, keeping learning vs retrieval distinct."""
    total = len(scores)
    if total == 0:
        return {
            "total_tasks": 0,
            "learning_scored_tasks": 0,
            "retrieval_hits": 0,
            "learning_exact_accuracy": 0.0,
            "raw_exact_accuracy_including_retrieval": 0.0,
            "abstention_rate": 0.0,
            "wrong_non_abstain_rate": 0.0,
            "mean_collateral_rate": 0.0,
            "total_collateral_changes": 0,
            "intermediate_step_accuracy": 0.0,
        }

    retrieval_hits = sum(1 for s in scores if s.is_retrieval_hit)
    learning_items = [s for s in scores if not s.excluded_from_learning_score]
    n_learn = len(learning_items)
    learn_exact = sum(1 for s in learning_items if s.exact_final_match)
    raw_exact = sum(1 for s in scores if s.exact_final_match)
    abstained = sum(1 for s in scores if s.abstained)
    wrong_non_abstain = sum(1 for s in scores if (not s.abstained) and (not s.exact_final_match))
    total_col = sum(s.collateral_changes for s in scores)
    mean_col = sum(s.collateral_rate for s in scores) / total
    tot_steps = sum(s.total_steps for s in scores)
    inter_hits = sum(s.intermediate_matches for s in scores)

    return {
        "total_tasks": total,
        "learning_scored_tasks": n_learn,
        "retrieval_hits": retrieval_hits,
        "learning_exact_accuracy": round(learn_exact / n_learn, 6) if n_learn > 0 else 0.0,
        "raw_exact_accuracy_including_retrieval": round(raw_exact / total, 6),
        "abstention_rate": round(abstained / total, 6),
        "wrong_non_abstain_rate": round(wrong_non_abstain / total, 6),
        "mean_collateral_rate": round(mean_col, 6),
        "total_collateral_changes": total_col,
        "intermediate_step_accuracy": round(inter_hits / tot_steps, 6) if tot_steps > 0 else 0.0,
    }


@dataclass(frozen=True)
class BitBudgetAudit:
    """Learning-vs-storage memory accounting required by PDF page 15."""

    raw_training_data_bits: int
    compressed_training_data_bits: int
    episode_store_bits: int
    active_bundle_bits: int
    operator_library_bits: int
    compressed_model_mdl_bits: int
    compression_ratio_vs_raw: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_training_data_bits": self.raw_training_data_bits,
            "compressed_training_data_bits": self.compressed_training_data_bits,
            "episode_store_bits": self.episode_store_bits,
            "active_bundle_bits": self.active_bundle_bits,
            "operator_library_bits": self.operator_library_bits,
            "compressed_model_mdl_bits": self.compressed_model_mdl_bits,
            "compression_ratio_vs_raw": round(self.compression_ratio_vs_raw, 6),
        }


def compute_training_data_bits(episodes: Sequence[TrainingEpisode]) -> tuple[int, int]:
    """Return (raw_bits, zlib_compressed_bits) of canonical training episodes."""
    payload = canonical_json_bytes([ep.to_dict() for ep in episodes])
    return (len(payload) * 8, compressed_bit_length(payload))


def score_operator_recovery(
    candidate_schemas: Mapping[str, Mapping[str, Any]],
    gold_schemas: Mapping[str, HiddenOperatorSchema] = HIDDEN_OPERATOR_SCHEMAS,
) -> dict[str, Any]:
    """Score induced abstract operator preconditions/effects against hidden ground truth.

    Used by evaluator only. Returns precision, recall, and F1 across all actions.
    """
    tp = 0
    fp = 0
    fn = 0
    per_action: dict[str, Any] = {}

    for action_id in sorted(set(gold_schemas) | set(candidate_schemas)):
        gold = gold_schemas.get(action_id)
        gold_clauses: set[tuple[str, ...]] = set()
        if gold is not None:
            for f in gold.required_facts:
                gold_clauses.add(("pre_fact",) + f)
            for t in gold.required_types:
                gold_clauses.add(("pre_type",) + t)
            for g in gold.required_attr_guards:
                gold_clauses.add(("pre_attr",) + g)
            for f in gold.add_facts:
                gold_clauses.add(("add_fact",) + f)
            for f in gold.delete_facts:
                gold_clauses.add(("del_fact",) + f)
            for e in gold.attribute_edits:
                gold_clauses.add(("attr_edit",) + e)

        cand = candidate_schemas.get(action_id, {})
        cand_clauses: set[tuple[str, ...]] = set()
        for f in cand.get("required_facts", ()):
            cand_clauses.add(("pre_fact",) + tuple(f))
        for t in cand.get("required_types", ()):
            cand_clauses.add(("pre_type",) + tuple(t))
        for g in cand.get("required_attr_guards", ()):
            cand_clauses.add(("pre_attr",) + tuple(g))
        for f in cand.get("add_facts", ()):
            cand_clauses.add(("add_fact",) + tuple(f))
        for f in cand.get("delete_facts", ()):
            cand_clauses.add(("del_fact",) + tuple(f))
        for e in cand.get("attribute_edits", ()):
            cand_clauses.add(("attr_edit",) + tuple(e))

        a_tp = len(gold_clauses & cand_clauses)
        a_fp = len(cand_clauses - gold_clauses)
        a_fn = len(gold_clauses - cand_clauses)
        tp += a_tp
        fp += a_fp
        fn += a_fn
        prec = a_tp / (a_tp + a_fp) if (a_tp + a_fp) > 0 else 0.0
        rec = a_tp / (a_tp + a_fn) if (a_tp + a_fn) > 0 else 0.0
        per_action[action_id] = {
            "tp": a_tp,
            "fp": a_fp,
            "fn": a_fn,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_action": per_action,
    }


def capture_environment_info() -> dict[str, Any]:
    """Capture reproducible CPU, OS, Python, memory, and Git commit metadata."""
    git_commit = "unknown"
    git_branch = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass

    ram_total_gb: float | None = None
    try:
        import psutil  # type: ignore[import-not-found]

        ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": ram_total_gb,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "git_commit": git_commit,
        "git_branch": git_branch,
    }
