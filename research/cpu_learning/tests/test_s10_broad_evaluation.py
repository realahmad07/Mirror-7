"""Stage S10 Independent Capstone Adversarial Tests: Broad Evaluation, Throughput, Calibration & Verdict.

STRICT ISOLATION:
- NEVER imports ``hidden_actions`` or ``oracle``.
- Pure Python 3.11 stdlib + pytest only (zero Python 3.12+ APIs).
- Verifies whole-tree AST/domain-literal/Python 3.11 compliance across all 9 learner packages,
  ECE calibration math, live single-seed throughput (>= 1,000 pred/sec/core), post-freeze
  temporal stress + 6-category OOD abstention calibration, transparent TDP-proxy energy
  reporting, brutally honest GPT-class readiness verdict (preserving prior external baseline
  context), bit-exact S0..S10 manifests, saved 5-seed 16/16 capstone gate, and an isolated
  2-seed subset run in ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path

from research.cpu_learning.controller import AutonomousMetaController
from research.cpu_learning.continual import SmallCPUNeuralContinualRef
from research.cpu_learning.evaluation.baselines import SmallCPUNeuralReferenceBaseline
from research.cpu_learning.evaluation.run_experiments import (
    DATA_DIR,
    PACKAGE_ROOT,
    REPORTS_DIR,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import verify_s1_committed_manifests
from research.cpu_learning.evaluation.run_s2_experiments import verify_s2_committed_manifests
from research.cpu_learning.evaluation.run_s3_experiments import verify_s3_committed_manifests
from research.cpu_learning.evaluation.run_s4_experiments import verify_s4_committed_manifests
from research.cpu_learning.evaluation.run_s5_experiments import verify_s5_committed_manifests
from research.cpu_learning.evaluation.run_s6_experiments import verify_s6_committed_manifests
from research.cpu_learning.evaluation.run_s7_experiments import verify_s7_committed_manifests
from research.cpu_learning.evaluation.run_s8_experiments import verify_s8_committed_manifests
from research.cpu_learning.evaluation.run_s9_experiments import verify_s9_committed_manifests
from research.cpu_learning.evaluation.run_s10_broad_eval import (
    ENERGY_MEASUREMENT_METHOD,
    REQUIRED_11_BASELINE_IDS,
    VERDICT_TIER,
    CalibrationEvalItem,
    _compute_ece_10bin,
    _run_live_s10_seed,
    audit_s10_whole_tree_isolation_and_py311,
    run_s10_broad_eval,
    verify_s10_committed_manifests,
)
from research.cpu_learning.grounding import (
    MirrorCTWConstructionByteLM,
    SmallCPUByteTransformerRef,
)


def test_s10_whole_tree_ast_isolation_zero_domain_literals_and_py311_compat() -> None:
    """Verify whole-tree AST import isolation, zero hardcoded domain literals, neural isolation, and Py3.11 compat."""
    audit = audit_s10_whole_tree_isolation_and_py311(PACKAGE_ROOT)
    assert audit["passed"] is True, f"Whole-tree audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["comparison_leak_violations"] == []
    assert audit["py312_violations"] == []
    assert audit["scanned_non_evaluator_modules"] >= 40
    assert audit["scanned_learner_modules"] >= 36

    assert SmallCPUNeuralReferenceBaseline.comparison_only is True
    assert SmallCPUByteTransformerRef.comparison_only is True
    assert SmallCPUByteTransformerRef.uses_backprop is True
    assert SmallCPUNeuralContinualRef.comparison_only is True
    assert SmallCPUNeuralContinualRef.uses_backprop is True
    assert MirrorCTWConstructionByteLM.comparison_only is False
    assert MirrorCTWConstructionByteLM.uses_backprop is False
    assert AutonomousMetaController.comparison_only is False
    assert AutonomousMetaController.uses_backprop is False


def test_s10_ece_calibration_math_catches_overconfident_errors() -> None:
    """Verify 10-bin ECE returns 0.0 on perfectly calibrated outputs and flags overconfident hallucinations."""
    perfect = [
        CalibrationEvalItem("a1", "in_scope", True, False, 1.0, True),
        CalibrationEvalItem("a2", "in_scope", True, False, 1.0, True),
        CalibrationEvalItem("o1", "ood", False, True, 0.0, False),
        CalibrationEvalItem("o2", "ood", False, True, 0.0, False),
    ]
    assert _compute_ece_10bin(perfect) == 0.0

    overconfident = [
        CalibrationEvalItem("a1", "in_scope", True, False, 1.0, True),
        CalibrationEvalItem("bad_ood", "ood", False, False, 1.0, False),
    ]
    assert abs(_compute_ece_10bin(overconfident) - 0.5) < 1e-9


def test_s10_single_seed_live_throughput_calibration_memory_and_energy() -> None:
    """Run live Stage S10 evaluation on seed 101 and verify freeze lock, calibration, throughput, RAM, and energy."""
    rec = _run_live_s10_seed(101)

    fi = rec["freeze_isolation"]
    assert fi["is_frozen"] is True
    assert fi["post_freeze_env_blocked"] is True
    assert fi["post_freeze_env_queries"] == 0
    assert fi["self_certified_promotions"] == 0

    cal = rec["post_freeze_stress_and_calibration"]
    assert cal["total_items"] == 220
    assert cal["in_scope_items"] == 120
    assert cal["unanswerable_ood_items"] == 100
    assert cal["post_freeze_temporal_exact_acc"] >= 0.95
    assert cal["unanswerable_abstention_recall"] == 1.0
    assert cal["in_scope_false_abstention_rate"] <= 0.02
    assert cal["selective_accuracy_when_answered"] >= 0.98
    assert cal["abstention_brier_score"] <= 0.05
    assert cal["expected_calibration_error_ece"] <= 0.05
    assert cal["post_freeze_collateral_rate"] < 0.02

    tp = rec["throughput_single_core"]
    assert tp["composite_all_4_workloads_preds_per_sec_core"] >= 1000.0
    assert tp["core_symbolic_and_functional_preds_per_sec_core"] >= 1000.0
    assert tp["relational_operator_apply"]["preds_per_sec_process_core"] >= 1000.0
    assert tp["compiled_functional_sandbox"]["preds_per_sec_process_core"] >= 10000.0

    en = rec["resource_and_energy_profile"]
    assert en["energy_measurement_method"] == ENERGY_MEASUREMENT_METHOD
    assert en["hardware_rapl_available"] is False
    assert 0.0 < en["peak_tracemalloc_mb"] < 256.0
    assert 0 < en["active_library_bits"] < en["raw_training_bits"]
    assert en["library_to_raw_ratio"] < 0.10
    assert en["estimated_train_joules_15w_laptop_core"] > 0.0
    assert en["estimated_train_joules_28w_server_core"] > en["estimated_train_joules_15w_laptop_core"]


def test_s10_saved_5seed_summary_13_gates_11_baselines_and_honest_verdict() -> None:
    """Verify saved 5-seed S10 summary passes 16/16 capstone checks, 13/13 PDF gates, and honest verdict rules."""
    summary = json.loads((REPORTS_DIR / "s10_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "S10"
    assert summary["seeds"] == [101, 202, 303, 404, 505]
    assert summary["s10_gate"]["passed"] is True
    assert len(summary["s10_gate"]["checks"]) == 16
    assert all(summary["s10_gate"]["checks"].values())

    gates = summary["master_13_gates"]
    assert len(gates) == 13
    assert all(g["passed"] is True for g in gates)

    matrix = summary["baseline_matrix"]
    assert set(matrix.keys()) == set(REQUIRED_11_BASELINE_IDS)
    assert matrix["small_cpu_neural_reference"]["comparison_only"] is True
    assert matrix["small_cpu_neural_reference"]["uses_backprop"] is True
    assert matrix["hand_coded_hidden_oracle"]["evaluator_only"] is True
    assert matrix["unified_mirror_shard_itl_rdc"]["comparison_only"] is False
    assert matrix["unified_mirror_shard_itl_rdc"]["uses_backprop"] is False

    verdict = summary["readiness_verdict"]
    assert verdict["verdict_tier"] == VERDICT_TIER
    assert verdict["narrow_verified_cpu_learning_achieved"] is True
    assert verdict["gpt_class_open_world_parity_achieved"] is False
    ext = verdict["preserved_prior_external_baseline_context"]
    assert "0 / 20" in ext["sanitized_mbpp_20_pass_rate"]
    assert "50.0%" in ext["rocstories_cloze_ranking"]
    assert len(verdict["experimentally_proven_on_cpu"]) >= 5
    assert len(verdict["explicitly_not_achieved_or_claimed"]) >= 4


def test_s10_all_s0_through_s10_manifests_and_isolated_2seed_subset_run(tmp_path: Path) -> None:
    """Verify bit-exact S0..S10 committed manifests and isolated 2-seed subset execution in tmp_path."""
    checks = [
        verify_committed_manifests(data_dir=DATA_DIR),
        verify_s1_committed_manifests(data_dir=DATA_DIR),
        verify_s2_committed_manifests(data_dir=DATA_DIR),
        verify_s3_committed_manifests(data_dir=DATA_DIR),
        verify_s4_committed_manifests(data_dir=DATA_DIR),
        verify_s5_committed_manifests(data_dir=DATA_DIR),
        verify_s6_committed_manifests(data_dir=DATA_DIR),
        verify_s7_committed_manifests(data_dir=DATA_DIR),
        verify_s8_committed_manifests(data_dir=DATA_DIR),
        verify_s9_committed_manifests(data_dir=DATA_DIR),
        verify_s10_committed_manifests(data_dir=DATA_DIR),
    ]
    all_errs = [e for ok, errs in checks if not ok for e in errs]
    assert all_errs == [], f"Committed S0..S10 manifest errors: {all_errs}"

    saved_summary_path = REPORTS_DIR / "s10_summary.json"
    saved_bytes_before = saved_summary_path.read_bytes()

    tmp_data = tmp_path / "data"
    tmp_reports = tmp_path / "reports"
    fresh = run_s10_broad_eval(
        seeds=[101, 202],
        data_dir=tmp_data,
        reports_dir=tmp_reports,
        verify_prior_manifests=True,
    )
    assert fresh["s10_gate"]["checks"]["prior_s0_through_s9_manifests_intact"] is True
    assert fresh["s10_gate"]["checks"]["whole_tree_ast_isolation_zero_literals_and_py311"] is True
    for k, ok in fresh["s10_gate"]["checks"].items():
        if k == "cpu_budget_within_15s_per_seed":
            continue
        assert ok is True, f"Fresh isolated 2-seed S10 gate failed check: {k}"

    assert (tmp_data / "s10_preregistration.json").exists()
    assert (tmp_reports / "s10_summary.json").exists()
    assert (tmp_reports / "s10_research_report.md").exists()
    assert (tmp_reports / "raw_runs" / "s10_baseline_matrix.csv").exists()
    assert (tmp_reports / "raw_runs" / "s10_gate_ledger.csv").exists()
    assert (tmp_reports / "raw_runs" / "s10_seed_101.json").exists()
    assert (tmp_reports / "raw_runs" / "s10_seed_202.json").exists()
    assert saved_summary_path.read_bytes() == saved_bytes_before
