"""One-command reproducible S0 benchmark harness and baseline experiment runner.

Usage:
    python -m research.cpu_learning.evaluation.run_experiments

Produces:
- ``research/cpu_learning/data/preregistration.json``
- ``research/cpu_learning/data/train_manifest.json``
- ``research/cpu_learning/data/heldout_manifest.json``
- ``research/cpu_learning/reports/raw_runs/s0_seed_<seed>.json``
- ``research/cpu_learning/reports/raw_runs/s0_per_split.csv``
- ``research/cpu_learning/reports/raw_runs/shard_canary_5seed.json``
- ``research/cpu_learning/reports/summary.json``
- ``research/cpu_learning/reports/research_report.md``
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Sequence

from research.cpu_learning.evaluation.baselines import (
    BaseS0Solver,
    CurrentMirrorSolverBaseline,
    EpisodeLookupBaseline,
    NoOpIdentityBaseline,
    SHARDHDCRetrievalBaseline,
    SmallCPUNeuralReferenceBaseline,
    StructuralKNNBaseline,
)
from research.cpu_learning.evaluation.metrics import (
    TaskScore,
    aggregate_split_scores,
    capture_environment_info,
    compute_training_data_bits,
    score_operator_recovery,
    score_task,
    summarize_across_seeds,
)
from research.cpu_learning.evaluation.oracle import HiddenOracleBaseline
from research.cpu_learning.evaluation.shard_canary import run_shard_arithmetic_canary
from research.cpu_learning.simulator.generators import (
    CANARY_SPLITS,
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    GENERATOR_VERSION,
    TEMPORAL_GENERATOR_VERSION,
    UNSEEN_EVAL_SPLITS,
    build_split_manifests,
    generate_heldout_suite,
    generate_training_episodes,
)
from research.cpu_learning.simulator.hidden_actions import HIDDEN_ACTION_NAMES

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
REPORTS_DIR = PACKAGE_ROOT / "reports"
RAW_RUNS_DIR = REPORTS_DIR / "raw_runs"

REPRO_COMMAND = "PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_experiments"
CPU_BUDGET_SECONDS_PER_SEED_MAX: float = 30.0


def _get_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def build_solver_factories(include_mirror_solver: bool = True) -> list[Callable[[int], BaseS0Solver]]:
    factories: list[Callable[[int], BaseS0Solver]] = [
        lambda seed: HiddenOracleBaseline(seed=seed),
        lambda seed: NoOpIdentityBaseline(seed=seed),
        lambda seed: EpisodeLookupBaseline(seed=seed),
        lambda seed: StructuralKNNBaseline(seed=seed),
        lambda seed: SHARDHDCRetrievalBaseline(seed=seed),
    ]
    if include_mirror_solver:
        factories.append(lambda seed: CurrentMirrorSolverBaseline(seed=seed))
    factories.append(lambda seed: SmallCPUNeuralReferenceBaseline(seed=seed))
    return factories


def build_preregistration(
    seeds: Sequence[int],
    demos_per_action: int,
    train_manifest: dict[str, Any],
    heldout_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Pre-registered experiment specification per PDF pages 10, 13, and 15."""
    return {
        "schema_version": "s0-preregistration-v1",
        "stage": "S0",
        "reproduce_command": REPRO_COMMAND,
        "generator_version": GENERATOR_VERSION,
        "temporal_generator_version": TEMPORAL_GENERATOR_VERSION,
        "seeds": list(seeds),
        "demos_per_action": demos_per_action,
        "hidden_actions": list(HIDDEN_ACTION_NAMES),
        "unseen_eval_splits": list(UNSEEN_EVAL_SPLITS),
        "canary_splits": list(CANARY_SPLITS),
        "cpu_budget_seconds_per_seed_max": CPU_BUDGET_SECONDS_PER_SEED_MAX,
        "s0_acceptance_gates": {
            "five_distinct_fixed_seeds_present": True,
            "deterministic_seed_replay_all_seeds": True,
            "pool_disjointness_verified": True,
            "all_solvers_frozen_before_temporal": True,
            "hidden_oracle_exact_accuracy": 1.0,
            "hidden_oracle_collateral_rate": 0.0,
            "hidden_oracle_operator_recovery_f1": 1.0,
            "episode_lookup_unseen_zero_leakage": True,
            "episode_lookup_canary_retrieval_audit_rate": 1.0,
            "episode_lookup_canary_excluded_from_learning": True,
            "cpu_budget_within_limit": True,
            "shard_modular_addition_canary_confirmed": True,
        },
        "future_itl_rdc_targets_from_pdf": {
            "s2_s3_exact_accuracy_depth4_min": 0.95,
            "s2_s3_exact_accuracy_depth6_min": 0.90,
            "s2_operator_recovery_precision_min": 0.90,
            "s2_operator_recovery_recall_min": 0.90,
            "s2_collateral_change_rate_max": 0.02,
            "s2_beat_episode_lookup_margin_min": 0.20,
            "s2_max_demos_per_action": 8,
            "s8_no_forgetting_retention_min": 0.95,
        },
        "train_manifest_sha256_by_seed": {
            s: info["sha256"] for s, info in train_manifest["per_seed"].items()
        },
        "heldout_manifest_split_counts": {
            s: info["total_unseen_tasks"] for s, info in heldout_manifest["per_seed"].items()
        },
    }


def _profile_peak_heap_bytes(
    factory: Callable[[int], BaseS0Solver],
    seed: int,
    train_eps: Sequence[Any],
    suite: dict[str, list[Any]],
    ordered_splits: Sequence[str],
) -> int:
    """Measure peak Python heap in an isolated pass (fit + 2 tasks/split incl. depth-6)
    so tracemalloc never distorts CPU throughput timing."""
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    try:
        mem_solver = factory(seed)
        mem_solver.fit(train_eps)
        for split_name in ordered_splits:
            if split_name == "heldout_temporal_frozen":
                mem_solver.freeze()
            for case in suite[split_name][:2]:
                mem_solver.predict(case.public)
        _cur_mem, peak_mem_bytes = tracemalloc.get_traced_memory()
        return int(peak_mem_bytes)
    finally:
        if not was_tracing:
            tracemalloc.stop()


def evaluate_solver_on_seed(
    solver: BaseS0Solver,
    seed: int,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
    *,
    memory_solver_factory: Callable[[int], BaseS0Solver] | None = None,
    precomputed: tuple[list[Any], dict[str, list[Any]], frozenset[str], int, int] | None = None,
) -> dict[str, Any]:
    """Run one solver on one seed with uninstrumented solver-only CPU timing and isolated heap profiling."""
    if precomputed is not None:
        train_eps, suite, training_exact_keys, raw_train_bits, comp_train_bits = precomputed
    else:
        train_eps = generate_training_episodes(seed, demos_per_action=demos_per_action)
        suite = generate_heldout_suite(seed, train_eps)
        training_exact_keys = frozenset(ep.state_action_key() for ep in train_eps)
        raw_train_bits, comp_train_bits = compute_training_data_bits(train_eps)

    ordered_splits = (
        [s for s in UNSEEN_EVAL_SPLITS if s != "heldout_temporal_frozen"]
        + list(CANARY_SPLITS)
        + ["heldout_temporal_frozen"]
    )
    inline_mem = memory_solver_factory is None
    was_tracing = tracemalloc.is_tracing()
    peak_mem_bytes = 0
    if inline_mem:
        if was_tracing:
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()

    try:
        t_fit0 = time.perf_counter()
        solver.fit(train_eps)
        fit_s = time.perf_counter() - t_fit0

        per_split_summary: dict[str, Any] = {}
        all_unseen_scores: list[TaskScore] = []
        failed_examples: list[dict[str, Any]] = []
        parser_stats = {"parsed_steps": 0, "total_steps": 0}
        total_pred_tasks = 0
        total_pred_time_s = 0.0

        for split_name in ordered_splits:
            if split_name == "heldout_temporal_frozen":
                solver.freeze()
            cases = suite[split_name]
            is_canary = split_name in CANARY_SPLITS
            split_scores: list[TaskScore] = []

            # Time solver.predict() exclusively (keep evaluator score_task() outside timing!)
            t_s0 = time.perf_counter()
            preds = [solver.predict(case.public) for case in cases]
            split_elapsed = time.perf_counter() - t_s0

            for case, pred in zip(cases, preds):
                sc = score_task(case, pred, training_exact_keys, is_canary_split=is_canary)
                split_scores.append(sc)
                if "parser_nonzero_steps" in pred.metadata:
                    parser_stats["parsed_steps"] += int(pred.metadata["parser_nonzero_steps"])
                    parser_stats["total_steps"] += int(pred.metadata["total_steps"])
                if (not is_canary) and (not sc.exact_final_match) and len(failed_examples) < 5:
                    failed_examples.append(
                        {
                            "task_id": case.public.task_id,
                            "split_id": split_name,
                            "depth": case.public.depth,
                            "instructions": [a.instruction_text() for a in case.public.actions],
                            "abstained": sc.abstained,
                            "collateral_changes": sc.collateral_changes,
                            "gold_step_applied": list(case.hidden.gold_step_applied),
                            "gold_block_reasons": list(case.hidden.gold_block_reasons),
                        }
                    )
            total_pred_time_s += split_elapsed
            total_pred_tasks += len(cases)

            agg = aggregate_split_scores(split_scores)
            agg["elapsed_seconds"] = round(split_elapsed, 6)
            per_split_summary[split_name] = agg
            if not is_canary:
                all_unseen_scores.extend(split_scores)

        if inline_mem:
            _cur_mem, peak_mem_bytes = tracemalloc.get_traced_memory()
    finally:
        if inline_mem and not was_tracing:
            tracemalloc.stop()

    if memory_solver_factory is not None:
        peak_mem_bytes = _profile_peak_heap_bytes(
            memory_solver_factory, seed, train_eps, suite, ordered_splits
        )

    unseen_agg = aggregate_split_scores(all_unseen_scores)
    op_recovery = score_operator_recovery(solver.recovered_operator_schemas())
    bit_audit = solver.bit_budget(raw_train_bits, comp_train_bits)
    preds_per_sec = total_pred_tasks / total_pred_time_s if total_pred_time_s > 0 else 0.0

    out: dict[str, Any] = {
        "solver": solver.name,
        "category": solver.category,
        "comparison_only": solver.comparison_only,
        "evaluator_only": solver.evaluator_only,
        "seed": seed,
        "frozen_before_temporal": solver.is_frozen,
        "fit_seconds": round(fit_s, 6),
        "predict_seconds": round(total_pred_time_s, 6),
        "predictions_per_sec_single_core": round(preds_per_sec, 2),
        "tracemalloc_peak_kb": round(peak_mem_bytes / 1024.0, 2),
        "rss_mb": _get_rss_mb(),
        "unseen_overall": unseen_agg,
        "per_split": per_split_summary,
        "operator_recovery": op_recovery,
        "bit_budget": bit_audit.to_dict(),
        "failed_examples_sample": failed_examples,
    }
    if parser_stats["total_steps"] > 0:
        out["mirror_parser_step_coverage"] = round(
            parser_stats["parsed_steps"] / float(parser_stats["total_steps"]), 6
        )
    return out

def _verify_s0_gate(
    summary_by_solver: dict[str, Any],
    runs_by_solver: dict[str, list[dict[str, Any]]],
    shard_canary: dict[str, Any],
    train_manifest: dict[str, Any],
    heldout_manifest: dict[str, Any],
    seeds: Sequence[int],
    demos_per_action: int,
) -> dict[str, Any]:
    """Programmatically verify every S0 acceptance criterion across ALL seeds."""
    checks: dict[str, bool] = {}

    # 1. Five distinct fixed seeds recorded
    checks["five_distinct_fixed_seeds_present"] = (
        len(seeds) >= 5 and len(set(seeds)) == len(seeds)
    )

    # 2. Deterministic seed replay across ALL seeds (double-generation hash equality)
    t_m2, h_m2 = build_split_manifests(seeds, demos_per_action=demos_per_action)
    checks["deterministic_seed_replay_all_seeds"] = (
        t_m2 == train_manifest and h_m2 == heldout_manifest
    )

    # 3. Disjoint entity/color pools certified
    checks["pool_disjointness_verified"] = bool(
        train_manifest["pool_disjointness"]["disjoint_verified"]
    )

    # 4. Every solver explicitly frozen before temporal holdout on every seed
    checks["all_solvers_frozen_before_temporal"] = all(
        all(bool(r["frozen_before_temporal"]) for r in recs)
        for recs in runs_by_solver.values()
    )

    # 5. Hidden oracle ceiling = 1.00 exact accuracy, 0.00 collateral, 1.00 op recovery
    oracle = summary_by_solver["hand_coded_hidden_oracle"]
    checks["hidden_oracle_exact_accuracy_100pct"] = (
        oracle["unseen_learning_exact_accuracy"]["mean"] == 1.0
        and oracle["unseen_learning_exact_accuracy"]["min"] == 1.0
    )
    checks["hidden_oracle_collateral_zero"] = (
        oracle["unseen_mean_collateral_rate"]["max"] == 0.0
    )
    checks["hidden_oracle_operator_recovery_f1_100pct"] = (
        oracle["operator_recovery_f1"]["min"] == 1.0
    )

    # 6. Episode lookup gets 0.00 unseen accuracy (zero train->unseen key leakage)
    lookup = summary_by_solver["episode_lookup"]
    checks["episode_lookup_unseen_zero_leakage"] = (
        lookup["unseen_learning_exact_accuracy"]["max"] == 0.0
    )

    # 7. Episode lookup retrieval audit catches 100% of canary exact hits & excludes them on ALL seeds
    canary = lookup["per_split"]["canary_exact_memorization"]
    checks["episode_lookup_canary_retrieval_audited_all_seeds"] = (
        canary["retrieval_hit_rate"]["min"] == 1.0
        and canary["learning_scored_tasks_max"] == 0
    )

    # 8. Every solver stayed within the pre-registered per-seed CPU time budget
    checks["cpu_budget_within_limit"] = all(
        all(
            (r["fit_seconds"] + r["predict_seconds"]) <= CPU_BUDGET_SECONDS_PER_SEED_MAX
            for r in recs
        )
        for recs in runs_by_solver.values()
    )

    # 9. Independent 5-seed SHARD modular-addition failure canary reproduced
    mod_task = shard_canary["summary_by_task"]["modular_addition"]
    checks["shard_modular_addition_canary_confirmed"] = (
        mod_task["hdc_pure_similarity"]["mean"] < 0.25
        and mod_task["shard_bit_perceptron"]["mean"] < 0.25
    )

    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def verify_committed_manifests(
    *,
    data_dir: Path = DATA_DIR,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> tuple[bool, list[str]]:
    """Verify committed pre-registered manifests match freshly generated ones bit-for-bit."""
    mismatches: list[str] = []
    train_manifest, heldout_manifest = build_split_manifests(
        seeds=seeds, demos_per_action=demos_per_action
    )
    prereg = build_preregistration(seeds, demos_per_action, train_manifest, heldout_manifest)
    expected_files = {
        "preregistration.json": prereg,
        "train_manifest.json": train_manifest,
        "heldout_manifest.json": heldout_manifest,
    }
    for fname, expected_obj in expected_files.items():
        fpath = data_dir / fname
        if not fpath.is_file():
            mismatches.append(f"Missing committed manifest: {fpath}")
            continue
        actual_obj = json.loads(fpath.read_text(encoding="utf-8"))
        if actual_obj != expected_obj:
            mismatches.append(f"Committed manifest drifted from deterministic generator: {fpath}")
    return (len(mismatches) == 0, mismatches)


def _render_markdown_report(summary: dict[str, Any]) -> str:
    env = summary["environment"]
    gate = summary["s0_gate"]
    lines: list[str] = [
        "# Mirror 7 CPU-Native Learning Research — Stage S0 Baseline Report",
        "",
        f"- **Stage:** `{summary['stage']}` (Benchmark Harness, Hidden Relational Simulator & Baselines)",
        f"- **S0 Acceptance Gate:** `{'PASS' if gate['passed'] else 'FAIL'}`",
        f"- **Reproduce Command:** `{summary['reproduce_command']}`",
        f"- **Generator Version:** `{summary['generator_version']}` / Temporal `{summary['temporal_generator_version']}`",
        f"- **Fixed Seeds:** `{summary['seeds']}` ({summary['demos_per_action']} demos/action = {summary['demos_per_action'] * 4} train episodes/seed)",
        f"- **Environment:** Python `{env['python_version']}` | `{env['platform']}` | CPUs `{env['cpu_count']}` | RAM `{env['ram_total_gb']} GB` | Git `{env['git_commit'][:12]}` (`{env['git_branch']}`)",
        "",
        "## 1. S0 Acceptance Gate Verification",
        "",
        "| Check | Status |",
        "|---|:---:|",
    ]
    for k, ok in gate["checks"].items():
        lines.append(f"| `{k}` | {'PASS' if ok else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## 2. Five-Seed Baseline Matrix on Unseen Generator Splits (9 Splits, 168 Unseen Tasks/Seed = 840 Total)",
            "",
            "| Solver | Category | Unseen Exact Acc (Mean ± 95% CI) | Depth-4 Acc | Depth-6 Acc | Temporal Frozen | Abstain Rate | Collateral Rate | Preds/s (1-core, uninstrumented) | Peak Heap (KB) | Model MDL (bits) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for sname, sdata in summary["solvers"].items():
        acc = sdata["unseen_learning_exact_accuracy"]
        ci_half = acc["ci95_half"]
        d4 = sdata["per_split"]["heldout_comp_depth4"]["learning_exact_accuracy"]["mean"]
        d6 = sdata["per_split"]["heldout_comp_depth6"]["learning_exact_accuracy"]["mean"]
        tf = sdata["per_split"]["heldout_temporal_frozen"]["learning_exact_accuracy"]["mean"]
        abst = sdata["unseen_abstention_rate"]["mean"]
        col = sdata["unseen_mean_collateral_rate"]["mean"]
        pps = sdata["predictions_per_sec_single_core"]["mean"]
        heap = sdata["tracemalloc_peak_kb"]["mean"]
        mdl = sdata["bit_budget_mean"]["compressed_model_mdl_bits"]
        lines.append(
            f"| `{sname}` | `{sdata['category']}` | {acc['mean']*100:.2f}% ± {ci_half*100:.2f}% | "
            f"{d4*100:.2f}% | {d6*100:.2f}% | {tf*100:.2f}% | {abst*100:.2f}% | "
            f"{col*100:.2f}% | {pps:.1f} | {heap:.1f} | {int(round(mdl)):,} |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Seed Unseen Learning Exact Accuracy Table",
            "",
            "| Solver | " + " | ".join(f"Seed {s}" for s in summary["seeds"]) + " | Mean | 95% CI |",
            "|---|" + "---:|" * len(summary["seeds"]) + "---:|---|",
        ]
    )
    for sname, sdata in summary["solvers"].items():
        acc = sdata["unseen_learning_exact_accuracy"]
        cells = " | ".join(f"{v*100:.2f}%" for v in acc["per_seed"])
        lines.append(
            f"| `{sname}` | {cells} | {acc['mean']*100:.2f}% | [{acc['ci95_low']*100:.2f}%, {acc['ci95_high']*100:.2f}%] |"
        )

    lines.extend(
        [
            "",
            "## 4. Retrieval-vs-Learning Canary Audit (`canary_exact_memorization`, 8 Exact Train Copies/Seed)",
            "",
            "| Solver | Raw Exact Match | Retrieval Hit Rate | Max Learning-Scored Tasks/Seed |",
            "|---|---:|---:|---:|",
        ]
    )
    for sname, sdata in summary["solvers"].items():
        c = sdata["per_split"]["canary_exact_memorization"]
        lines.append(
            f"| `{sname}` | {c['raw_exact_accuracy']['mean']*100:.2f}% | "
            f"{c['retrieval_hit_rate']['mean']*100:.2f}% | {c['learning_scored_tasks_max']} |"
        )

    lines.extend(
        [
            "",
            "## 5. Bit-Budget & MDL Memory Accounting (PDF p. 15 Mean Across 5 Seeds)",
            "",
            "| Solver | Raw Train Data (bits) | Compressed Train (bits) | Episode Store (bits) | Active Bundle (bits) | Operator Library (bits) | Compressed Model MDL (bits) | MDL / Raw Train |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sname, sdata in summary["solvers"].items():
        bb = sdata["bit_budget_mean"]
        lines.append(
            f"| `{sname}` | {int(round(bb['raw_training_data_bits'])):,} | "
            f"{int(round(bb['compressed_training_data_bits'])):,} | "
            f"{int(round(bb['episode_store_bits'])):,} | "
            f"{int(round(bb['active_bundle_bits'])):,} | "
            f"{int(round(bb['operator_library_bits'])):,} | "
            f"{int(round(bb['compressed_model_mdl_bits'])):,} | "
            f"{bb['compression_ratio_vs_raw']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 6. Independent 5-Seed SHARD HDC Arithmetic Canary (PDF p. 3 Reproduction)",
            "",
            "| Task | Classes | Chance | Pure HDC Similarity (Mean ± 95% CI) | SHARD Bit-Perceptron (Mean ± 95% CI) | Clipped 95% CI (Bit-Perceptron) | Learned Readout Bits |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    canary = summary["shard_arithmetic_canary"]
    for tname, tdata in canary["summary_by_task"].items():
        sim = tdata["hdc_pure_similarity"]
        perc = tdata["shard_bit_perceptron"]
        lines.append(
            f"| `{tname}` | {tdata['num_classes']} | {tdata['chance_baseline']*100:.1f}% | "
            f"{sim['mean']*100:.2f}% ± {sim['ci95_half']*100:.2f}% | "
            f"{perc['mean']*100:.2f}% ± {perc['ci95_half']*100:.2f}% | "
            f"[{perc['ci95_low']*100:.2f}%, {perc['ci95_high']*100:.2f}%] | "
            f"{tdata['learned_readout_bits']:,} |"
        )

    lines.extend(
        [
            "",
            "### SHARD Canary Per-Seed Bit-Perceptron Accuracy",
            "",
            "| Task | " + " | ".join(f"Seed {s}" for s in summary["seeds"]) + " |",
            "|---|" + "---:|" * len(summary["seeds"]),
        ]
    )
    for tname, tdata in canary["summary_by_task"].items():
        perc = tdata["shard_bit_perceptron"]
        cells = " | ".join(f"{v*100:.2f}%" for v in perc["per_seed"])
        lines.append(f"| `{tname}` | {cells} |")

    lines.extend(
        [
            "",
            "## 7. Representative Baseline Failure Samples (Seed 101)",
            "",
        ]
    )
    for sname, sdata in summary["solvers"].items():
        samples = sdata.get("failed_examples_seed0", [])
        if not samples:
            continue
        lines.append(f"### `{sname}`")
        for ex in samples[:3]:
            lines.append(
                f"- `{ex['task_id']}` (`{ex['split_id']}`, depth={ex['depth']}): "
                f"abstained=`{ex['abstained']}`, collateral_changes=`{ex['collateral_changes']}`, "
                f"instructions=`{ex['instructions']}`"
            )
        lines.append("")

    lines.extend(
        [
            "## 8. Next Branch Decision",
            "",
            "- **S0 Gate:** " + ("PASS — Ready to advance to **S1 (Typed State-Frame + Append-Only Episode Ledger)** after review." if gate["passed"] else "FAIL — Investigate failed checks before advancing."),
            "- **Preserved External Context (PDF p. 3):** Prior external five-organ evaluation (`scripts/run_five_organ_external_eval.py`) found useful constrained parsing and symbolic probes, weak long-memory retrieval, 0/20 MBPP passes, and below-chance StoryCloze ranking. S0 preserves those findings as baseline context without replacing them with internal test counts.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_s0_experiments(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
    data_dir: Path = DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    include_mirror_solver: bool = True,
) -> dict[str, Any]:
    """Execute the complete reproducible S0 benchmark suite and write all logs & reports."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_dir = reports_dir / "raw_runs"
    raw_runs_dir.mkdir(parents=True, exist_ok=True)

    train_manifest, heldout_manifest = build_split_manifests(
        seeds=seeds, demos_per_action=demos_per_action
    )
    prereg = build_preregistration(seeds, demos_per_action, train_manifest, heldout_manifest)

    (data_dir / "preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (data_dir / "train_manifest.json").write_text(
        json.dumps(train_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (data_dir / "heldout_manifest.json").write_text(
        json.dumps(heldout_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    factories = build_solver_factories(include_mirror_solver=include_mirror_solver)
    runs_by_solver: dict[str, list[dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []

    t_all0 = time.perf_counter()
    for seed in seeds:
        train_eps = generate_training_episodes(seed, demos_per_action=demos_per_action)
        suite = generate_heldout_suite(seed, train_eps)
        training_exact_keys = frozenset(ep.state_action_key() for ep in train_eps)
        raw_train_bits, comp_train_bits = compute_training_data_bits(train_eps)
        precomputed = (train_eps, suite, training_exact_keys, raw_train_bits, comp_train_bits)

        seed_runs: dict[str, Any] = {}
        for factory in factories:
            solver = factory(seed)
            rec = evaluate_solver_on_seed(
                solver,
                seed=seed,
                demos_per_action=demos_per_action,
                memory_solver_factory=factory,
                precomputed=precomputed,
            )
            runs_by_solver.setdefault(solver.name, []).append(rec)
            seed_runs[solver.name] = rec
            for split_name, sagg in rec["per_split"].items():
                csv_rows.append(
                    {
                        "seed": seed,
                        "solver": solver.name,
                        "category": solver.category,
                        "split_id": split_name,
                        "total_tasks": sagg["total_tasks"],
                        "learning_scored_tasks": sagg["learning_scored_tasks"],
                        "retrieval_hits": sagg["retrieval_hits"],
                        "learning_exact_accuracy": sagg["learning_exact_accuracy"],
                        "raw_exact_accuracy": sagg["raw_exact_accuracy_including_retrieval"],
                        "abstention_rate": sagg["abstention_rate"],
                        "wrong_non_abstain_rate": sagg["wrong_non_abstain_rate"],
                        "mean_collateral_rate": sagg["mean_collateral_rate"],
                        "intermediate_step_accuracy": sagg["intermediate_step_accuracy"],
                        "elapsed_seconds": sagg["elapsed_seconds"],
                    }
                )

        (raw_runs_dir / f"s0_seed_{seed}.json").write_text(
            json.dumps({"seed": seed, "solvers": seed_runs}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Write per-split CSV
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with (raw_runs_dir / "s0_per_split.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # Run independent 5-seed SHARD arithmetic canary
    shard_canary = run_shard_arithmetic_canary(seeds=seeds)
    (raw_runs_dir / "shard_canary_5seed.json").write_text(
        json.dumps(shard_canary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Aggregate 5-seed summaries
    all_splits = list(UNSEEN_EVAL_SPLITS) + list(CANARY_SPLITS)
    summary_by_solver: dict[str, Any] = {}
    for sname, recs in runs_by_solver.items():
        per_split_agg: dict[str, Any] = {}
        for split_name in all_splits:
            learn_accs = [r["per_split"][split_name]["learning_exact_accuracy"] for r in recs]
            raw_accs = [r["per_split"][split_name]["raw_exact_accuracy_including_retrieval"] for r in recs]
            col_rates = [r["per_split"][split_name]["mean_collateral_rate"] for r in recs]
            abst_rates = [r["per_split"][split_name]["abstention_rate"] for r in recs]
            ret_rates = [
                r["per_split"][split_name]["retrieval_hits"] / max(1, r["per_split"][split_name]["total_tasks"])
                for r in recs
            ]
            scored_counts = [r["per_split"][split_name]["learning_scored_tasks"] for r in recs]
            per_split_agg[split_name] = {
                "learning_scored_tasks_per_seed": scored_counts,
                "learning_scored_tasks_max": max(scored_counts) if scored_counts else 0,
                "learning_exact_accuracy": summarize_across_seeds(learn_accs, bounds=(0.0, 1.0)),
                "raw_exact_accuracy": summarize_across_seeds(raw_accs, bounds=(0.0, 1.0)),
                "mean_collateral_rate": summarize_across_seeds(col_rates, bounds=(0.0, None)),
                "abstention_rate": summarize_across_seeds(abst_rates, bounds=(0.0, 1.0)),
                "retrieval_hit_rate": summarize_across_seeds(ret_rates, bounds=(0.0, 1.0)),
            }

        bit_keys = list(recs[0]["bit_budget"].keys())
        bit_means = {
            k: round(sum(r["bit_budget"][k] for r in recs) / len(recs), 6)
            for k in bit_keys
        }

        s_entry: dict[str, Any] = {
            "category": recs[0]["category"],
            "comparison_only": recs[0]["comparison_only"],
            "evaluator_only": recs[0]["evaluator_only"],
            "unseen_learning_exact_accuracy": summarize_across_seeds(
                [r["unseen_overall"]["learning_exact_accuracy"] for r in recs], bounds=(0.0, 1.0)
            ),
            "unseen_abstention_rate": summarize_across_seeds(
                [r["unseen_overall"]["abstention_rate"] for r in recs], bounds=(0.0, 1.0)
            ),
            "unseen_wrong_non_abstain_rate": summarize_across_seeds(
                [r["unseen_overall"]["wrong_non_abstain_rate"] for r in recs], bounds=(0.0, 1.0)
            ),
            "unseen_mean_collateral_rate": summarize_across_seeds(
                [r["unseen_overall"]["mean_collateral_rate"] for r in recs], bounds=(0.0, None)
            ),
            "operator_recovery_precision": summarize_across_seeds(
                [r["operator_recovery"]["precision"] for r in recs], bounds=(0.0, 1.0)
            ),
            "operator_recovery_recall": summarize_across_seeds(
                [r["operator_recovery"]["recall"] for r in recs], bounds=(0.0, 1.0)
            ),
            "operator_recovery_f1": summarize_across_seeds(
                [r["operator_recovery"]["f1"] for r in recs], bounds=(0.0, 1.0)
            ),
            "fit_seconds": summarize_across_seeds(
                [r["fit_seconds"] for r in recs], bounds=(0.0, None)
            ),
            "predict_seconds": summarize_across_seeds(
                [r["predict_seconds"] for r in recs], bounds=(0.0, None)
            ),
            "predictions_per_sec_single_core": summarize_across_seeds(
                [r["predictions_per_sec_single_core"] for r in recs], bounds=(0.0, None)
            ),
            "tracemalloc_peak_kb": summarize_across_seeds(
                [r["tracemalloc_peak_kb"] for r in recs], bounds=(0.0, None)
            ),
            "bit_budget_mean": bit_means,
            "per_split": per_split_agg,
            "failed_examples_seed0": recs[0]["failed_examples_sample"],
        }
        if "mirror_parser_step_coverage" in recs[0]:
            s_entry["mirror_parser_step_coverage"] = summarize_across_seeds(
                [r["mirror_parser_step_coverage"] for r in recs], bounds=(0.0, 1.0)
            )
        summary_by_solver[sname] = s_entry

    s0_gate = _verify_s0_gate(
        summary_by_solver,
        runs_by_solver,
        shard_canary,
        train_manifest,
        heldout_manifest,
        seeds,
        demos_per_action,
    )
    total_elapsed = time.perf_counter() - t_all0

    summary = {
        "schema_version": "s0-summary-v1",
        "stage": "S0",
        "reproduce_command": REPRO_COMMAND,
        "generator_version": GENERATOR_VERSION,
        "temporal_generator_version": TEMPORAL_GENERATOR_VERSION,
        "seeds": list(seeds),
        "demos_per_action": demos_per_action,
        "total_elapsed_seconds": round(total_elapsed, 4),
        "environment": capture_environment_info(),
        "s0_gate": s0_gate,
        "solvers": summary_by_solver,
        "shard_arithmetic_canary": shard_canary,
    }

    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "research_report.md").write_text(
        _render_markdown_report(summary), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible Mirror 7 CPU-Native Learning S0 benchmark & baselines."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Fixed random seeds (default: 101 202 303 404 505).",
    )
    parser.add_argument(
        "--demos-per-action",
        type=int,
        default=DEFAULT_DEMOS_PER_ACTION,
        help="Demonstrations per hidden action in [3, 8] (default: 6).",
    )
    parser.add_argument(
        "--verify-manifests",
        action="store_true",
        help="Verify committed preregistration/train/heldout manifests match deterministic generation.",
    )
    args = parser.parse_args(argv)

    if args.verify_manifests:
        ok, mismatches = verify_committed_manifests(
            seeds=tuple(args.seeds), demos_per_action=args.demos_per_action
        )
        if ok:
            print("Committed S0 manifests verified bit-exact against deterministic generator.")
            return 0
        for msg in mismatches:
            print(f"MANIFEST VERIFICATION FAILED: {msg}")
        return 1

    summary = run_s0_experiments(
        seeds=tuple(args.seeds),
        demos_per_action=args.demos_per_action,
    )
    gate = summary["s0_gate"]
    print(f"=== Mirror 7 CPU-Native Learning S0 Complete ({summary['total_elapsed_seconds']:.2f}s) ===")
    print(f"S0 Gate: {'PASS' if gate['passed'] else 'FAIL'} | Checks: {gate['checks']}")
    for sname, sdata in summary["solvers"].items():
        acc = sdata["unseen_learning_exact_accuracy"]
        col = sdata["unseen_mean_collateral_rate"]["mean"]
        abst = sdata["unseen_abstention_rate"]["mean"]
        print(
            f"  {sname:28s} | unseen_exact={acc['mean']*100:6.2f}% "
            f"[{acc['ci95_low']*100:6.2f}%, {acc['ci95_high']*100:6.2f}%] | "
            f"abstain={abst*100:6.2f}% | collateral={col*100:5.2f}%"
        )
    print("--- SHARD Arithmetic Canary (5 seeds, unseen digit pairs) ---")
    for tname, tdata in summary["shard_arithmetic_canary"]["summary_by_task"].items():
        sim = tdata["hdc_pure_similarity"]["mean"]
        perc = tdata["shard_bit_perceptron"]["mean"]
        print(
            f"  {tname:18s} | pure_hdc={sim*100:6.2f}% | bit_perceptron={perc*100:6.2f}% "
            f"(chance={tdata['chance_baseline']*100:4.1f}%)"
        )
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
