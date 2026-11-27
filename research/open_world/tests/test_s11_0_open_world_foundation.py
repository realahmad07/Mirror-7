"""Independent Adversarial Tests for Stage S11.0 Open-World Generalization Foundation.

Verifies:
1. Whole-tree AST isolation (zero learner/baseline imports of evaluator/oracle/hidden_eval,
   zero GPU/LLM/external API dependencies, zero Python 3.12+ stdlib APIs).
2. Public-only observation schema invariant (zero gold answer/label/is_impossible/has_answer fields).
3. Source-disjoint connected-component splitting (zero leakage across train/val/hidden_eval).
4. Solver freeze locks (`SolverFrozenError` on post-freeze `.fit()` / `.calibrate()`).
5. Eight-family metric math (accuracy, evidence grounding, transfer, Brier/ECE calibration,
   abstention recall/false-abstention, memory use, CPU time, peak heap).
6. S0-S10 audit integrity, PR #43 non-merge & failed-gate record, and 14-dataset license audit.
7. SHA-256 tamper detection on verified dataset loader.
8. Full offline 5-baseline + isolated oracle ceiling end-to-end run in ``tmp_path``.
9. Committed real-dataset preregistration manifest, raw results, honest verdict, vault SHA-256
   preservation, and zero raw external benchmark text redistribution in git.
"""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import pytest

from research.open_world.audit import (
    audit_dataset_licenses,
    audit_pr43_status,
    audit_s0_s10_package,
)
from research.open_world.baselines import (
    ExistingMirrorSystemBaseline,
    ExistingRuleBlueprintBaseline,
    RetrievalOnlyBaseline,
    SimpleSymbolicBaseline,
)
from research.open_world.contracts import (
    ConversationTurn,
    OpenWorldObservation,
    OpenWorldPrediction,
    SolverFrozenError,
    SolverMemoryFootprint,
)
from research.open_world.datasets import (
    DatasetDownloadError,
    build_connected_components,
    download_or_load_verified,
    partition_source_clusters,
)
from research.open_world.evaluation.run_s11_0_foundation import (
    REPO_ROOT,
    S11_DATA_DIR,
    S11_PACKAGE_ROOT,
    S11_REPORTS_DIR,
    audit_s11_isolation_and_constraints,
    run_s11_0_foundation,
    verify_s11_0_preregistration,
)
from research.open_world.evaluator import (
    EvaluatorOnlyOracle,
    HiddenGoldRecord,
    build_offline_smoke_bundles,
    compute_benchmark_metrics,
    compute_brier_and_ece,
    compute_token_f1,
)


def test_s11_whole_tree_ast_isolation_zero_llm_gpu_and_py311_compat() -> None:
    """AST-verify strict evaluator/baseline isolation, zero external ML/LLM imports, and Py3.11 compat."""
    audit = audit_s11_isolation_and_constraints(S11_PACKAGE_ROOT)
    assert audit["passed"] is True, f"S11 AST audit failed: {audit}"
    assert audit["solver_import_violations"] == []
    assert audit["non_eval_oracle_violations"] == []
    assert audit["llm_gpu_violations"] == []
    assert audit["py312_violations"] == []
    assert audit["scanned_python_files"] >= 12
    assert audit["scanned_baseline_files"] >= 1

    assert EvaluatorOnlyOracle.evaluator_only is True
    for cls in (
        ExistingMirrorSystemBaseline,
        RetrievalOnlyBaseline,
        ExistingRuleBlueprintBaseline,
        SimpleSymbolicBaseline,
    ):
        assert cls.evaluator_only is False
        assert cls.uses_gpu is False
        assert cls.uses_pretrained_llm is False
        assert cls.uses_external_api is False
        assert cls.uses_neural_training is False


def test_s11_public_observation_schema_has_zero_hidden_gold_fields() -> None:
    """Verify learner-facing observation and turn dataclasses contain ZERO gold/hidden fields."""
    forbidden = {
        "answer",
        "answers",
        "gold_answer",
        "gold_answers",
        "gold_choice_index",
        "label",
        "is_impossible",
        "is_unanswerable",
        "has_answer",
        "answer_session_ids",
        "gold_evidence_ids",
        "mirrored",
    }
    obs_fields = {f.name for f in fields(OpenWorldObservation)}
    turn_fields = {f.name for f in fields(ConversationTurn)}
    assert obs_fields.isdisjoint(forbidden), f"Leaked fields in OpenWorldObservation: {obs_fields & forbidden}"
    assert turn_fields.isdisjoint(forbidden), f"Leaked fields in ConversationTurn: {turn_fields & forbidden}"


def test_s11_connected_component_source_disjoint_splitter_prevents_leakage() -> None:
    """Verify shared resources (sessions, mirrored pairs) never leak across train/val/hidden splits."""
    # Items q1 and q2 share session s_shared; q3 and q4 share s_shared2; q5..q12 independent.
    item_to_resources = {
        "q1": ["s_shared", "s_a"],
        "q2": ["s_shared", "s_b"],
        "q3": ["s_shared2", "s_c"],
        "q4": ["s_shared2", "s_d"],
        "q5": ["s_e"],
        "q6": ["s_f"],
        "q7": ["s_g"],
        "q8": ["s_h"],
        "q9": ["s_i"],
        "q10": ["s_j"],
    }
    comps = build_connected_components(item_to_resources)
    assert comps["q1"] == comps["q2"]
    assert comps["q3"] == comps["q4"]
    assert comps["q1"] != comps["q3"]

    res = partition_source_clusters(
        "synthetic_leak_test",
        comps,
        train_ratio=0.40,
        val_ratio=0.20,
        item_to_resources=item_to_resources,
    )
    assert res.disjointness_verified is True
    assert res.disjointness_report["item_overlaps"] == {"train_val": 0, "train_hidden": 0, "val_hidden": 0}
    assert res.disjointness_report["cluster_overlaps"] == {"train_val": 0, "train_hidden": 0, "val_hidden": 0}
    assert res.disjointness_report["resource_overlaps"] == {"train_val": 0, "train_hidden": 0, "val_hidden": 0}

    # q1 and q2 must be in the exact same split.
    for split_set in (set(res.train_item_ids), set(res.val_item_ids), set(res.hidden_item_ids)):
        assert ("q1" in split_set) == ("q2" in split_set)
        assert ("q3" in split_set) == ("q4" in split_set)

    # Fixed hidden cluster stays exclusively in hidden_eval.
    res_fixed = partition_source_clusters(
        "synthetic_fixed_hidden",
        comps,
        train_ratio=0.70,
        val_ratio=0.30,
        fixed_hidden_clusters={comps["q9"], comps["q10"]},
        item_to_resources=item_to_resources,
    )
    assert {"q9", "q10"} == set(res_fixed.hidden_item_ids)
    assert res_fixed.disjointness_verified is True

    with pytest.raises(ValueError):
        partition_source_clusters("bad", comps, train_ratio=0.8, val_ratio=0.5)


def test_s11_solver_freeze_lock_rejects_post_freeze_fit_and_calibrate() -> None:
    """Verify all 5 solvers enforce `.freeze()` and reject post-freeze training/calibration."""
    smoke = build_offline_smoke_bundles()["smoke_qa"]
    solvers = [
        ExistingMirrorSystemBaseline(seed=101),
        RetrievalOnlyBaseline(seed=101),
        ExistingRuleBlueprintBaseline(seed=101, repo_root=REPO_ROOT),
        SimpleSymbolicBaseline(seed=101),
        EvaluatorOnlyOracle(smoke.hidden_gold, seed=101),
    ]
    for solver in solvers:
        solver.fit(smoke.benchmark_id, smoke.train_examples)
        solver.calibrate(smoke.benchmark_id, smoke.val_examples)
        solver.freeze()
        assert solver.is_frozen is True
        with pytest.raises(SolverFrozenError):
            solver.fit(smoke.benchmark_id, smoke.train_examples)
        with pytest.raises(SolverFrozenError):
            solver.calibrate(smoke.benchmark_id, smoke.val_examples)
        pred = solver.predict(smoke.hidden_observations[0])
        assert pred.memory_writes == 0


def test_s11_metrics_calibration_evidence_and_abstention_edge_cases() -> None:
    """Verify token-F1, Brier/ECE calibration detection, evidence grounding, and abstention scoring."""
    assert compute_token_f1("The 1984 bridge!", "1984 bridge") == 1.0
    assert compute_token_f1("", "") == 1.0
    assert compute_token_f1("alpha", "") == 0.0
    assert abs(compute_token_f1("red blue", "red green") - 0.5) < 1e-9

    brier_perf, ece_perf = compute_brier_and_ece([1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    assert brier_perf == 0.0
    assert ece_perf == 0.0

    brier_bad, ece_bad = compute_brier_and_ece([1.0, 1.0], [1.0, 0.0])
    assert abs(brier_bad - 0.5) < 1e-6
    assert abs(ece_bad - 0.5) < 1e-6

    gold = {
        "t1": HiddenGoldRecord("t1", "b", "hidden_eval", "extractive_qa", "c1", "answerable", gold_answers=("42",), is_unanswerable=False, gold_evidence_ids=("t1:s0",)),
        "t2": HiddenGoldRecord("t2", "b", "hidden_eval", "extractive_qa", "c1", "answerable", gold_answers=("Paris",), is_unanswerable=False, gold_evidence_ids=("t2:s1",)),
        "t3": HiddenGoldRecord("t3", "b", "hidden_eval", "extractive_qa", "c2", "unanswerable", gold_answers=(), is_unanswerable=True, gold_evidence_ids=()),
        "t4": HiddenGoldRecord("t4", "b", "hidden_eval", "extractive_qa", "c2", "unanswerable", gold_answers=(), is_unanswerable=True, gold_evidence_ids=()),
    }
    preds = [
        # t1: correct answer + exact gold evidence
        OpenWorldPrediction("t1", "b", predicted_answer="42", abstained=False, confidence=0.9, uncertainty=0.1, cited_evidence_ids=("t1:s0",)),
        # t2: false abstention on answerable item
        OpenWorldPrediction("t2", "b", predicted_answer=None, abstained=True, confidence=0.6, uncertainty=0.4),
        # t3: correct abstention on unanswerable item
        OpenWorldPrediction("t3", "b", predicted_answer=None, abstained=True, confidence=0.8, uncertainty=0.2),
        # t4: hallucinated answer on unanswerable item
        OpenWorldPrediction("t4", "b", predicted_answer="Mars", abstained=False, confidence=0.9, uncertainty=0.1, cited_evidence_ids=("t4:s0",)),
    ]
    m = compute_benchmark_metrics(
        benchmark_id="b",
        split="hidden_eval",
        predictions=preds,
        hidden_gold=gold,
        memory_footprint=SolverMemoryFootprint(128, 512, 4),
        timing={"fit_wall_s": 0.01, "fit_cpu_s": 0.01, "predict_wall_s": 0.02, "predict_cpu_s": 0.02},
        peak_heap={"fit_peak_kb": 10.0, "predict_peak_kb": 12.0},
        val_primary_accuracy=0.75,
    )
    assert abs(m["accuracy"]["primary_score"] - 0.5) < 1e-6
    assert abs(m["abstention"]["unanswerable_abstention_recall"] - 0.5) < 1e-6
    assert abs(m["abstention"]["hallucination_on_unanswerable_rate"] - 0.5) < 1e-6
    assert abs(m["abstention"]["answerable_false_abstention_rate"] - 0.5) < 1e-6
    assert abs(m["evidence_grounding"]["evidence_f1"] - 0.5) < 1e-6
    assert abs(m["evidence_grounding"]["unsupported_answer_rate"] - 0.5) < 1e-6
    assert abs(m["transfer"]["val_to_hidden_gap"] - (-0.25)) < 1e-6


def test_s11_s0_s10_audit_pr43_governance_and_14_dataset_license_audit() -> None:
    """Verify S0-S10 limitation audit, PR #43 non-merge/failed-gate audit, and 14-dataset license audit."""
    s0_s10 = audit_s0_s10_package(REPO_ROOT)
    assert s0_s10["all_s0_s10_gates_passed"] is True
    assert s0_s10["s10_master_13_gates_passed"] == 13
    assert s0_s10["s10_master_13_gates_total"] == 13
    assert len(s0_s10["open_world_limitations"]) >= 8

    pr43 = audit_pr43_status(REPO_ROOT)
    assert pr43["pr_number"] == 43
    assert pr43["merged_into_main"] is False
    assert pr43["treated_as_passing_capability"] is False
    assert pr43["ci_check_runs_at_head"]["native-brain"] == "failure"
    assert pr43["governance_invariant_passed"] is True
    assert len(pr43["failed_semantic_reasoning_gates"]) >= 7
    assert len(pr43["unresolved_gates_and_open_blockers"]) >= 5

    lic = audit_dataset_licenses()
    assert lic["total_datasets_reviewed"] >= 14
    assert set(lic["active_s11_0_benchmarks"]) == {"balanced_copa", "squad_v2", "longmemeval"}
    assert all(lic["license_compliance_checks"].values())
    rejected = set(lic["datasets_by_status"].get("REJECTED_LICENSE", [])) | set(
        lic["datasets_by_status"].get("REJECTED_UNAVAILABLE", [])
    )
    assert {"clutrr", "rocstories_cloze", "locomo", "hellaswag"}.issubset(rejected)


def test_s11_loader_rejects_corrupted_sha256_cache(tmp_path: Path) -> None:
    """Verify SHA-256 verification fails closed on tampered cached dataset files."""
    bad_file = tmp_path / "tampered.json"
    bad_file.write_bytes(b'{"tampered": true}')
    with pytest.raises(DatasetDownloadError):
        download_or_load_verified(
            url="https://example.invalid/dataset.json",
            dest=bad_file,
            expected_sha256="0" * 64,
            allow_download=False,
        )


def test_s11_offline_smoke_end_to_end_all_5_baselines_and_oracle_ceiling(tmp_path: Path) -> None:
    """Run complete 5-baseline S11.0 pipeline offline in tmp_path and verify oracle ceiling & gates."""
    summary = run_s11_0_foundation(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        repo_root=REPO_ROOT,
        offline_smoke=True,
        allow_download=False,
    )
    assert summary["s11_0_gate"]["passed"] is True, f"Offline smoke gate failed: {summary['s11_0_gate']}"
    assert summary["vault_integrity"]["untouched"] is True
    assert summary["honest_verdict"]["open_world_generalization_claimed"] is False
    assert summary["honest_verdict"]["new_learning_architecture_implemented"] is False

    for bid in ("smoke_causal", "smoke_qa", "smoke_memory"):
        solvers = summary["benchmarks"][bid]["solvers"]
        assert set(solvers.keys()) == {
            "existing_mirror_system",
            "retrieval_only_bm25",
            "existing_rule_blueprint",
            "simple_symbolic_extractor",
            "evaluator_only_oracle",
        }
        oracle = solvers["evaluator_only_oracle"]["hidden_eval"]
        assert oracle["accuracy"]["primary_score"] == 1.0
        assert oracle["uncertainty"]["brier_score"] == 0.0
        assert oracle["uncertainty"]["ece_10bin"] == 0.0
        if oracle["abstention"]["unanswerable_total"] > 0:
            assert oracle["abstention"]["unanswerable_abstention_recall"] == 1.0

    assert (tmp_path / "data" / "s11_0_preregistration.json").is_file()
    assert (tmp_path / "reports" / "s11_0_summary.json").is_file()
    assert (tmp_path / "reports" / "s11_0_baseline_matrix.csv").is_file()
    assert (tmp_path / "reports" / "s11_0_research_report.md").is_file()


def test_s11_committed_preregistration_summary_and_zero_raw_text_redistribution() -> None:
    """Verify committed real-dataset S11.0 preregistration, summary, gates, and zero raw dataset files in git."""
    ok, errs = verify_s11_0_preregistration(data_dir=S11_DATA_DIR, allow_download=False)
    assert ok is True, f"Committed preregistration verification errors: {errs}"

    summary_path = S11_REPORTS_DIR / "s11_0_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["stage"] == "S11.0"
    assert summary["mode"] == "real_external_benchmarks"
    assert summary["s11_0_gate"]["passed"] is True
    assert all(summary["s11_0_gate"]["checks"].values())
    assert summary["vault_integrity"]["untouched"] is True
    assert summary["honest_verdict"]["open_world_generalization_claimed"] is False
    assert summary["honest_verdict"]["new_learning_architecture_implemented"] is False

    assert set(summary["benchmarks"].keys()) == {"balanced_copa", "squad_v2", "longmemeval"}
    for bid, bdata in summary["benchmarks"].items():
        assert bdata["split_manifest"]["disjointness_verified"] is True
        assert set(bdata["solvers"].keys()) == {
            "existing_mirror_system",
            "retrieval_only_bm25",
            "existing_rule_blueprint",
            "simple_symbolic_extractor",
            "evaluator_only_oracle",
        }
        oracle = bdata["solvers"]["evaluator_only_oracle"]["hidden_eval"]
        assert oracle["accuracy"]["primary_score"] == 1.0
        assert oracle["uncertainty"]["ece_10bin"] == 0.0
        assert (S11_REPORTS_DIR / "raw_runs" / f"s11_0_{bid}.json").is_file()

    assert (S11_REPORTS_DIR / "s11_0_baseline_matrix.csv").is_file()
    assert (S11_REPORTS_DIR / "s11_0_dataset_license_audit.json").is_file()
    assert (S11_REPORTS_DIR / "s11_0_s0_s10_and_pr43_audit.json").is_file()
    assert (S11_REPORTS_DIR / "s11_0_research_report.md").is_file()

    # Zero raw external benchmark text files may ever exist inside research/open_world/.
    forbidden_raw_names = {"train.csv", "test.csv", "dev-v2.0.json", "longmemeval_oracle.json", "longmemeval_s_cleaned.json"}
    found_raw = [p.name for p in S11_PACKAGE_ROOT.rglob("*") if p.name in forbidden_raw_names]
    assert found_raw == [], f"Raw external benchmark files must never be committed in repo: {found_raw}"
