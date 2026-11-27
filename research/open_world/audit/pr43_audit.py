"""Audit record of GitHub PR #43 (research-track-a-native-semantic-gate).

GOVERNANCE INVARIANT:
- PR #43 MUST NOT be merged into ``main`` or ``research/open-world-generalization-s11``.
- PR #43 MUST NOT be treated as a passing semantic-reasoning capability.
- All failed and unresolved semantic-reasoning gates from PR #43 are preserved verbatim below
  from direct inspection of commit ``2d70a161ba74c7dacdd884134ec7ce3267db29b3`` and its
  GitHub check-runs.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

PR43_HEAD_SHA: str = "2d70a161ba74c7dacdd884134ec7ce3267db29b3"
PR43_BRANCH_NAME: str = "research-track-a-native-semantic-gate"

PR43_AUDIT_RECORD: dict[str, Any] = {
    "pr_number": 43,
    "branch": PR43_BRANCH_NAME,
    "head_sha": PR43_HEAD_SHA,
    "state_at_audit": "open_not_merged",
    "merged_into_main": False,
    "treated_as_passing_capability": False,
    "ci_check_runs_at_head": {
        "source-guard": "success",
        "fast-ci": "success",
        "native-brain": "failure",
    },
    "primary_doc_inspected": "docs/38-track-a-native-intelligence-semantic-reasoning.md",
    "failed_semantic_reasoning_gates": [
        {
            "gate_id": "PR43_G1_exp6_60ep_positive_heldout_margin",
            "status": "FAIL (0/3 seeds for both MLP and Linear Operator)",
            "details": (
                "At 60 epochs with intermediate supervision: MLP top-1=2.61%, mean_margin=-0.3131; "
                "Linear operator top-1=2.06%, mean_margin=-0.3037. All 6 seed runs failed the "
                "positive held-out candidate-margin gate."
            ),
        },
        {
            "gate_id": "PR43_G2_exp6_240ep_positive_heldout_margin",
            "status": "FAIL (0/3 seeds for both MLP and Linear Operator)",
            "details": (
                "At 240 epochs (pre-correction historical evaluator): MLP top-1=34.96%, "
                "mean_margin=-0.0852; Linear operator top-1=40.72%, mean_margin=-0.0649. "
                "Both beat chance (~0.75-0.81%), passed order control and no-collapse, but ALL "
                "6 runs failed positive held-out candidate margin."
            ),
        },
        {
            "gate_id": "PR43_G3_contrastive_weight_0_25_ablation",
            "status": "FAIL (0/3 seeds, rejected)",
            "details": (
                "Linear operator contrastive weight 0.25 worsened metrics: top-1=34.10%, "
                "mean_margin=-0.1024, order_margin=0.2583. Failed all 3 seeds and rejected."
            ),
        },
        {
            "gate_id": "PR43_G4_rollout_supervision_weight_1_0_ablation",
            "status": "FAIL (0/3 seeds, rejected)",
            "details": (
                "Self-fed rollout supervision weight 1.0 fell below the 0.10 operator baseline "
                "(top-1=31.98%, mean_margin=-0.1112). Failed all 3 seeds and rejected."
            ),
        },
        {
            "gate_id": "PR43_G5_equivalent_target_corrected_evaluator_reruns",
            "status": "FAIL (0/3 seeds across all 4 tested configurations)",
            "details": (
                "After fixing equivalent-target false negatives, all 4 matched 3-seed configs "
                "still failed positive held-out margin on seeds 17/29/43: "
                "MLP c=0.10 (-0.1182/-0.0770/-0.0602; 31.34%/35.34%/38.21%); "
                "Linear c=0.10 (-0.0863/-0.0301/-0.0783; 35.82%/48.12%/38.21%); "
                "Linear c=0.25 (-0.1215/-0.0690/-0.1167; 28.36%/40.60%/33.33%); "
                "Linear rollout=1.0 (-0.1410/-0.0733/-0.1194; 26.12%/41.35%/28.46%)."
            ),
        },
        {
            "gate_id": "PR43_G6_composition_depth_and_suffix_degradation",
            "status": "FAIL (localized compositional collapse on longer chains)",
            "details": (
                "Stratified subgroup diagnostics (commit 0ee5859, Linear c=0.10): "
                "Unseen concept margin=-0.0694 (40.3%), Seen concept margin=-0.0625 (39.7%), "
                "Held-out prefix 0->1 margin=-0.0222 (44.4%), Held-out suffix 4->5 margin=-0.1314 (28.3%), "
                "Chain len 3=+0.0097 (62.3%), len 4=-0.0465 (41.0%), len 5=-0.1175 (29.0%; "
                "every seed negative between -0.1326 and -0.1015)."
            ),
        },
        {
            "gate_id": "PR43_G7_native_brain_ci_gate",
            "status": "FAIL (GitHub Actions check native-brain concluded failure)",
            "details": (
                "CI intentionally fails the native-brain research check when no predeclared "
                "operator configuration passes all 3 seeds on the unchanged semantic gate."
            ),
        },
    ],
    "unresolved_gates_and_open_blockers": [
        {
            "gate_id": "PR43_U1_rollout_dose_0_10_and_0_25_pending",
            "status": "UNRESOLVED / PENDING",
            "details": (
                "Low-dose rollout supervision ablations (weights 0.10 and 0.25) were added to CI "
                "to test accumulated rollout drift on 4->5 suffixes; no passing 3-seed result exists."
            ),
        },
        {
            "gate_id": "PR43_U2_frozen_exp5_exp7_matched_comparison",
            "status": "UNRESOLVED",
            "details": (
                "Required matched comparison against frozen Experiment 5/7 baselines under identical "
                "data, parameter budget, and compute budget has not been completed."
            ),
        },
        {
            "gate_id": "PR43_U3_natural_semantic_benchmark_and_license_boundaries",
            "status": "UNRESOLVED (partially addressed via RuleTaker controlled manifest only)",
            "details": (
                "CLUTRR was correctly excluded due to CC-BY-NC 4.0 vs unknown HF license. "
                "RuleTaker (Apache-2.0) was pinned in training/ruletaker_v1.manifest.json with an "
                "explicit boundary that it is templated/controlled language and does NOT establish "
                "organic natural-text reasoning."
            ),
        },
        {
            "gate_id": "PR43_U4_v5_free_running_prompt_disjointness_and_semantic_labels",
            "status": "UNRESOLVED (promotion_eligible=false)",
            "details": (
                "The 10-case legacy held-out V5 smoke prompt export explicitly sets "
                "promotion_eligible=false because source-disjointness from Drive training data is "
                "unaudited, and lexical/format metrics do not measure factual or semantic correctness."
            ),
        },
        {
            "gate_id": "PR43_U5_independent_external_reproduction",
            "status": "UNRESOLVED",
            "details": (
                "Independent verification of exact-answer tasks and blinded/external reproduction "
                "remain incomplete; no checkpoint is promoted."
            ),
        },
    ],
    "lessons_carried_into_s11": [
        "Never weaken a failing gate threshold; localize failure by stratum (depth, composition, unseen concept).",
        "Always check for target-equivalence / false-negative evaluator bugs and version evaluator schemas.",
        "Require explicit source-disjointness proofs and license verification before touching any external dataset.",
        "Keep neural-latent Track A research strictly separate from Mirror's CPU-native S11 open-world track.",
    ],
}


def audit_pr43_status(repo_root: Path) -> dict[str, Any]:
    """Verify live git ancestry (when git metadata is available) confirming PR #43 is NOT merged."""
    pr43_merged_into_head = False
    git_check_performed = False
    git_check_note = "git_check_skipped_shallow_or_no_git"

    if (repo_root / ".git").exists():
        try:
            # Check whether PR43's unique docs/38-track-a file or head commit is an ancestor of HEAD.
            track_a_doc = repo_root / "docs" / "38-track-a-native-intelligence-semantic-reasoning.md"
            ruletaker_manifest = repo_root / "training" / "ruletaker_v1.manifest.json"
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", PR43_HEAD_SHA, "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            git_check_performed = True
            if proc.returncode == 0 or track_a_doc.exists() or ruletaker_manifest.exists():
                pr43_merged_into_head = True
                git_check_note = "WARNING: PR #43 commit or exclusive files detected in HEAD"
            else:
                pr43_merged_into_head = False
                git_check_note = "Confirmed: PR #43 head commit and exclusive Track A files are NOT merged into HEAD"
        except Exception as exc:
            git_check_note = f"git_check_error: {type(exc).__name__}"

    record = dict(PR43_AUDIT_RECORD)
    record["live_git_verification"] = {
        "checked": git_check_performed,
        "pr43_merged_into_head": pr43_merged_into_head,
        "note": git_check_note,
    }
    record["governance_invariant_passed"] = (not pr43_merged_into_head) and (
        record["treated_as_passing_capability"] is False
    )
    return record
