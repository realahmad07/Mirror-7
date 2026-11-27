"""EVALUATOR-ONLY 8-family open-world evaluation metrics (pure Python 3.11 stdlib).

Records ALL eight metric families mandated by Stage S11.0:
1. Accuracy (primary task accuracy, token-F1, exact match, mirrored consistency, selective accuracy)
2. Evidence grounding (citation precision/recall/F1, exact hit rate, unsupported answer rate)
3. Transfer (source-disjoint held-out accuracy, val->hidden gap, per-domain/category spread, shift scores)
4. Uncertainty (mean confidence, mean uncertainty, Brier score, 10-bin ECE)
5. Abstention (abstention rate, unanswerable abstention recall, answerable false-abstention rate,
   abstention F1, hallucination-on-unanswerable rate)
6. Memory use (serialized bytes, compressed MDL bits, active entry count, mean reads/writes)
7. CPU time (fit/predict wall & process CPU seconds, ms/item, preds/sec/core)
8. Memory consumption (peak tracemalloc heap KB & MB during fit and predict)
"""
from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any, Mapping, Sequence

from research.open_world.contracts import (
    OpenWorldPrediction,
    SolverMemoryFootprint,
    normalize_text,
)
from research.open_world.evaluator.hidden_eval import HiddenGoldRecord


def compute_token_f1(predicted: str, gold: str) -> float:
    """Standard normalized bag-of-tokens F1 (SQuAD / QA convention)."""
    p_toks = normalize_text(predicted).split()
    g_toks = normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    return 2.0 * prec * rec / (prec + rec)


def _best_answer_match(predicted: str | None, gold_answers: tuple[str, ...]) -> tuple[float, float, bool]:
    """Return (exact_match, max_token_f1, containment_match)."""
    if not predicted or not gold_answers:
        return 0.0, 0.0, False
    p_norm = normalize_text(predicted)
    if not p_norm:
        return 0.0, 0.0, False
    best_em = 0.0
    best_f1 = 0.0
    contained = False
    for g in gold_answers:
        g_norm = normalize_text(g)
        if not g_norm:
            continue
        if p_norm == g_norm:
            best_em = 1.0
        f1 = compute_token_f1(predicted, g)
        if f1 > best_f1:
            best_f1 = f1
        if g_norm in p_norm or (len(p_norm) >= 3 and p_norm in g_norm and f1 >= 0.5):
            contained = True
    return best_em, best_f1, contained


def _evidence_prf(
    cited_ids: tuple[str, ...],
    gold_ids: tuple[str, ...],
) -> tuple[float, float, float, bool]:
    """Compute evidence citation (precision, recall, f1, any_hit) for an answerable item."""
    c_set = {str(x) for x in cited_ids if str(x)}
    g_set = {str(x) for x in gold_ids if str(x)}
    if not g_set:
        return (1.0, 1.0, 1.0, True) if not c_set else (0.0, 1.0, 0.0, False)
    if not c_set:
        return 0.0, 0.0, 0.0, False
    inter = len(c_set & g_set)
    prec = inter / len(c_set)
    rec = inter / len(g_set)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, inter > 0


def compute_brier_and_ece(
    confidences: Sequence[float],
    outcomes: Sequence[float],
    *,
    num_bins: int = 10,
) -> tuple[float, float]:
    """Compute Brier score and 10-bin Expected Calibration Error (ECE) over [0, 1]."""
    n = len(confidences)
    if n == 0 or n != len(outcomes):
        return 0.0, 0.0
    brier = sum((float(c) - float(y)) ** 2 for c, y in zip(confidences, outcomes)) / n
    ece = 0.0
    for b in range(num_bins):
        lo = b / num_bins
        hi = (b + 1) / num_bins
        idxs = [
            i
            for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == num_bins - 1 and lo <= c <= hi)
        ]
        if not idxs:
            continue
        avg_conf = sum(confidences[i] for i in idxs) / len(idxs)
        avg_acc = sum(outcomes[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(avg_acc - avg_conf)
    return round(brier, 6), round(ece, 6)


def bootstrap_ci95(values: Sequence[float], *, seed: int = 101, n_boot: int = 200) -> dict[str, float]:
    """Deterministic percentile bootstrap 95% CI (pure Python stdlib)."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean_v = sum(values) / n
    if n == 1:
        return {"mean": round(mean_v, 6), "ci95_low": round(mean_v, 6), "ci95_high": round(mean_v, 6)}
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "mean": round(mean_v, 6),
        "ci95_low": round(lo, 6),
        "ci95_high": round(hi, 6),
    }


def compute_benchmark_metrics(
    *,
    benchmark_id: str,
    split: str,
    predictions: Sequence[OpenWorldPrediction],
    hidden_gold: Mapping[str, HiddenGoldRecord],
    memory_footprint: SolverMemoryFootprint,
    timing: Mapping[str, float],
    peak_heap: Mapping[str, float],
    val_primary_accuracy: float | None = None,
    transfer_shift_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute complete 8-family metrics for one solver on one benchmark split."""
    n = len(predictions)
    if n == 0:
        raise ValueError(f"No predictions to score for {benchmark_id}/{split}")

    primary_scores: list[float] = []
    exact_matches: list[float] = []
    token_f1s: list[float] = []
    containment_hits: list[float] = []

    ev_precs: list[float] = []
    ev_recs: list[float] = []
    ev_f1s: list[float] = []
    ev_any_hits: list[float] = []
    unsupported_answered = 0

    confidences: list[float] = []
    uncertainties: list[float] = []
    calibration_outcomes: list[float] = []

    abstained_count = 0
    answered_count = 0
    answered_scores: list[float] = []

    unans_total = 0
    unans_abstained = 0
    ans_total = 0
    ans_abstained = 0

    by_category_scores: dict[str, list[float]] = {}
    by_cluster_scores: dict[str, list[float]] = {}
    pair_correct: dict[str, list[bool]] = {}

    total_reads = 0
    total_writes = 0

    for pred in predictions:
        gold = hidden_gold[pred.task_id]
        total_reads += int(pred.memory_reads)
        total_writes += int(pred.memory_writes)
        confidences.append(float(pred.confidence))
        uncertainties.append(float(pred.uncertainty))

        if pred.abstained:
            abstained_count += 1
        else:
            answered_count += 1

        if gold.is_unanswerable:
            unans_total += 1
            if pred.abstained:
                unans_abstained += 1
                item_score = 1.0
                em = 1.0
                f1 = 1.0
                cont = True
            else:
                item_score = 0.0
                em = 0.0
                f1 = 0.0
                cont = False
                unsupported_answered += 1
        else:
            ans_total += 1
            if pred.abstained:
                ans_abstained += 1
                item_score = 0.0
                em = 0.0
                f1 = 0.0
                cont = False
            elif gold.task_kind == "causal_choice":
                ok = pred.predicted_choice_index == gold.gold_choice_index
                item_score = 1.0 if ok else 0.0
                em = item_score
                f1 = item_score
                cont = ok
            elif gold.task_kind == "extractive_qa":
                em, f1, cont = _best_answer_match(pred.predicted_answer, gold.gold_answers)
                # Primary score on SQuAD v2 is Token-F1 (standard official SQuAD v2 metric).
                item_score = f1
            else:  # long_memory_qa
                em, f1, cont = _best_answer_match(pred.predicted_answer, gold.gold_answers)
                # Primary score on LongMemEval: normalized answer containment OR token-F1 >= 0.5
                item_score = 1.0 if (cont or f1 >= 0.5) else f1

            # Evidence grounding scored on all answerable items
            ep, er, ef1, ehit = _evidence_prf(pred.cited_evidence_ids, gold.gold_evidence_ids)
            ev_precs.append(ep)
            ev_recs.append(er)
            ev_f1s.append(ef1)
            ev_any_hits.append(1.0 if ehit else 0.0)
            if (not pred.abstained) and (not ehit):
                unsupported_answered += 1

        primary_scores.append(item_score)
        exact_matches.append(em)
        token_f1s.append(f1)
        containment_hits.append(1.0 if cont else 0.0)
        calibration_outcomes.append(1.0 if item_score >= 0.5 else 0.0)
        if not pred.abstained:
            answered_scores.append(item_score)

        by_category_scores.setdefault(gold.category, []).append(item_score)
        by_cluster_scores.setdefault(gold.source_cluster, []).append(item_score)
        if gold.pair_cluster_id:
            pair_correct.setdefault(gold.pair_cluster_id, []).append(item_score >= 0.5)

    primary_ci = bootstrap_ci95(primary_scores)
    primary_acc = sum(primary_scores) / n
    selective_acc = (sum(answered_scores) / len(answered_scores)) if answered_scores else None
    brier, ece = compute_brier_and_ece(confidences, calibration_outcomes, num_bins=10)

    # Abstention metrics
    abstain_rate = abstained_count / n
    unans_abstain_recall = (unans_abstained / unans_total) if unans_total > 0 else None
    hallucination_on_unans = (1.0 - unans_abstain_recall) if unans_abstain_recall is not None else None
    ans_false_abstain_rate = (ans_abstained / ans_total) if ans_total > 0 else 0.0
    if unans_total > 0 and abstained_count > 0:
        abstain_prec = unans_abstained / abstained_count
        abstain_rec = unans_abstained / unans_total
        abstain_f1 = (
            2.0 * abstain_prec * abstain_rec / (abstain_prec + abstain_rec)
            if (abstain_prec + abstain_rec) > 0
            else 0.0
        )
    elif unans_total > 0:
        abstain_f1 = 0.0
    else:
        abstain_f1 = None

    # Mirrored-pair consistency (for Balanced COPA validation pairs)
    multi_pairs = [vals for vals in pair_correct.values() if len(vals) >= 2]
    pair_consistency = (
        sum(1 for vals in multi_pairs if all(vals)) / len(multi_pairs)
        if multi_pairs
        else None
    )

    # Per-cluster (topic/domain) transfer dispersion
    cluster_means = [sum(v) / len(v) for v in by_cluster_scores.values() if v]
    cluster_mean = sum(cluster_means) / len(cluster_means) if cluster_means else 0.0
    cluster_std = (
        math.sqrt(sum((x - cluster_mean) ** 2 for x in cluster_means) / len(cluster_means))
        if len(cluster_means) > 1
        else 0.0
    )

    raw_predict_cpu_s = float(timing.get("predict_cpu_s", 0.0))
    predict_wall_s = max(1e-6, float(timing.get("predict_wall_s", 0.0)))
    # On Windows, time.process_time() has ~15.6ms resolution; fall back to wall time for sub-tick runs.
    effective_predict_s = raw_predict_cpu_s if raw_predict_cpu_s > 0.0 else predict_wall_s

    return {
        "benchmark_id": benchmark_id,
        "split": split,
        "num_items": n,
        "accuracy": {
            "primary_score": round(primary_acc, 6),
            "primary_ci95_low": primary_ci["ci95_low"],
            "primary_ci95_high": primary_ci["ci95_high"],
            "exact_match": round(sum(exact_matches) / n, 6),
            "token_f1": round(sum(token_f1s) / n, 6),
            "containment_rate": round(sum(containment_hits) / n, 6),
            "selective_score_when_answered": round(selective_acc, 6) if selective_acc is not None else None,
            "answered_coverage": round(answered_count / n, 6),
            "mirrored_pair_consistency": round(pair_consistency, 6) if pair_consistency is not None else None,
            "by_category": {
                k: {"n": len(v), "mean_score": round(sum(v) / len(v), 6)}
                for k, v in sorted(by_category_scores.items())
            },
        },
        "evidence_grounding": {
            "answerable_items_scored": len(ev_f1s),
            "evidence_precision": round(sum(ev_precs) / len(ev_precs), 6) if ev_precs else None,
            "evidence_recall": round(sum(ev_recs) / len(ev_recs), 6) if ev_recs else None,
            "evidence_f1": round(sum(ev_f1s) / len(ev_f1s), 6) if ev_f1s else None,
            "evidence_any_hit_rate": round(sum(ev_any_hits) / len(ev_any_hits), 6) if ev_any_hits else None,
            "unsupported_answer_rate": round(unsupported_answered / max(1, answered_count), 6)
            if answered_count > 0
            else 0.0,
        },
        "transfer": {
            "source_disjoint_heldout_score": round(primary_acc, 6),
            "val_to_hidden_gap": round(primary_acc - val_primary_accuracy, 6)
            if val_primary_accuracy is not None
            else None,
            "num_unseen_source_clusters": len(cluster_means),
            "per_cluster_mean": round(cluster_mean, 6),
            "per_cluster_min": round(min(cluster_means), 6) if cluster_means else None,
            "per_cluster_std": round(cluster_std, 6),
            "cross_shift_transfer": dict(transfer_shift_summary) if transfer_shift_summary else None,
        },
        "uncertainty": {
            "mean_confidence": round(sum(confidences) / n, 6),
            "mean_uncertainty": round(sum(uncertainties) / n, 6),
            "brier_score": brier,
            "ece_10bin": ece,
        },
        "abstention": {
            "overall_abstention_rate": round(abstain_rate, 6),
            "unanswerable_total": unans_total,
            "unanswerable_abstention_recall": round(unans_abstain_recall, 6)
            if unans_abstain_recall is not None
            else None,
            "hallucination_on_unanswerable_rate": round(hallucination_on_unans, 6)
            if hallucination_on_unans is not None
            else None,
            "answerable_total": ans_total,
            "answerable_false_abstention_rate": round(ans_false_abstain_rate, 6),
            "abstention_f1": round(abstain_f1, 6) if abstain_f1 is not None else None,
        },
        "memory_use": {
            "serialized_bytes": int(memory_footprint.serialized_bytes),
            "compressed_mdl_bits": int(memory_footprint.compressed_mdl_bits),
            "active_entry_count": int(memory_footprint.active_entry_count),
            "external_readonly_pack_entries": int(memory_footprint.external_readonly_pack_entries),
            "mean_memory_reads_per_query": round(total_reads / n, 4),
            "total_memory_writes_during_eval": int(total_writes),
        },
        "cpu_time": {
            "fit_wall_s": round(float(timing.get("fit_wall_s", 0.0)), 6),
            "fit_cpu_s": round(float(timing.get("fit_cpu_s", 0.0)), 6),
            "calibrate_wall_s": round(float(timing.get("calibrate_wall_s", 0.0)), 6),
            "predict_wall_s": round(predict_wall_s, 6),
            "predict_cpu_s": round(raw_predict_cpu_s, 6),
            "ms_per_item_cpu": round((effective_predict_s / n) * 1000.0, 4),
            "preds_per_sec_core": round(n / effective_predict_s, 2),
        },
        "memory_consumption": {
            "fit_peak_tracemalloc_kb": round(float(peak_heap.get("fit_peak_kb", 0.0)), 2),
            "predict_peak_tracemalloc_kb": round(float(peak_heap.get("predict_peak_kb", 0.0)), 2),
            "max_peak_tracemalloc_mb": round(
                max(float(peak_heap.get("fit_peak_kb", 0.0)), float(peak_heap.get("predict_peak_kb", 0.0))) / 1024.0,
                4,
            ),
        },
    }
