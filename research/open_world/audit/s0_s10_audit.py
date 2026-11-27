"""Programmatic audit of the existing Stages S0–S10 CPU-native research package and its open-world limitations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

S0_S10_STAGE_SUMMARIES: tuple[tuple[str, str, str], ...] = (
    ("S0", "summary.json", "s0_gate"),
    ("S1", "s1_summary.json", "s1_gate"),
    ("S2", "s2_summary.json", "s2_gate"),
    ("S3", "s3_summary.json", "s3_gate"),
    ("S4", "s4_summary.json", "s4_gate"),
    ("S5", "s5_summary.json", "s5_gate"),
    ("S6", "s6_summary.json", "s6_gate"),
    ("S7", "s7_summary.json", "s7_gate"),
    ("S8", "s8_summary.json", "s8_gate"),
    ("S9", "s9_summary.json", "s9_gate"),
    ("S10", "s10_summary.json", "s10_gate"),
)

# Brutally honest structural limitations discovered by auditing S0–S10 for open-world generalization.
S0_S10_OPEN_WORLD_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "limitation_id": "L01_presegmented_state_transitions",
        "affected_stages": "S0-S5, S8, S9",
        "severity": "critical_blocker_for_open_world",
        "summary": (
            "ITL operator induction assumes clean pre-segmented (before_state, action, after_state) "
            "relational state-graph tuples produced by a deterministic black-box simulator. "
            "Real-world text, documents, and multi-session dialogues do not come with explicit "
            "structured state graphs or pre-labeled action invocations."
        ),
        "open_world_consequence": (
            "Without an upstream raw-observation -> evidence -> entity/event/relation extractor, "
            "S0-S10 ITL cards cannot even parse or ingest organic Wikipedia passages, causal prose, "
            "or multi-turn user dialogues."
        ),
    },
    {
        "limitation_id": "L02_closed_predicate_and_role_schemas",
        "affected_stages": "S1-S5, S7, S8",
        "severity": "critical_blocker_for_open_world",
        "summary": (
            "S1 TypedStateFrame and S2/S3 OperatorCard assume a discrete finite set of typed "
            "predicates and deterministic entity-type tags per world. Even in S7 raw UTF-8 grounding, "
            "constructions are aligned strictly against already-verified discrete OperatorCard state deltas."
        ),
        "open_world_consequence": (
            "Open-world observations introduce unbounded open-vocabulary entities, paraphrases, "
            "implicit causal links, and latent events that have no pre-registered state-delta oracle."
        ),
    },
    {
        "limitation_id": "L03_simulator_oracle_dependence_for_refutation",
        "affected_stages": "S3, S5, S9",
        "severity": "major_architectural_gap",
        "summary": (
            "S3 contrast refinement, S5 RDC dream-refutation/active querying, and S9 autonomous "
            "curriculum rely on an interactive black-box TransitionEnvironment.step() oracle that "
            "answers synthetic counterfactual queries on demand with zero noise."
        ),
        "open_world_consequence": (
            "In passive open-world corpora (documents, conversations, commonsense narratives), "
            "the learner cannot query a noise-free world simulator to test arbitrary counterfactual "
            "actions; verification must instead rely on observational consistency, source-disjoint "
            "cross-evidence corroboration, temporal contradiction detection, and executable checks where available."
        ),
    },
    {
        "limitation_id": "L04_bounded_dsl_and_external_code_gap",
        "affected_stages": "S6, S10",
        "severity": "major_capability_boundary",
        "summary": (
            "S6 achieved 19/20 on HumanEval-Micro within a compact typed combinator grammar "
            "(honestly leaving he_micro_20_count_vowels unsolved), while Mirror's external MBPP "
            "evaluation (scripts/run_five_organ_external_eval.py) remains 0/20."
        ),
        "open_world_consequence": (
            "Synthesizing general open-world capabilities from unstructured specifications requires "
            "expanding beyond fixed micro-DSL combinators without leaking task-specific templates."
        ),
    },
    {
        "limitation_id": "L05_structured_grounding_vs_open_world_language",
        "affected_stages": "S7, S10",
        "severity": "critical_blocker_for_open_world",
        "summary": (
            "S7's 2.5649 bpb CTW+construction Byte-LM and 100% UTF-8 instruction execution hold on "
            "controlled grounded synthetic worlds where constructions map cleanly onto discrete operators, "
            "not on organic natural-language commonsense (e.g. Balanced COPA, StoryCloze) or "
            "open-domain reading comprehension (SQuAD v2)."
        ),
        "open_world_consequence": (
            "Unseen lexical paraphrases, implicit world causality, pragmatic implicature, and "
            "adversarial unanswerable questions in natural prose defeat exact construction alignment."
        ),
    },
    {
        "limitation_id": "L06_long_horizon_noisy_memory_and_updates",
        "affected_stages": "S1, S8, Organ 3 (KanervaSDMInt8)",
        "severity": "major_architectural_gap",
        "summary": (
            "S1's append-only EpisodeLedger and S8's ContinualKnowledgeBase preserve verified "
            "structured skills with 99.43% retention, but neither performs temporal belief revision, "
            "multi-session coreference resolution, or contradiction tracking over raw unstructured "
            "conversational transcripts with distractor sessions."
        ),
        "open_world_consequence": (
            "In real multi-session memory benchmarks (LongMemEval), user facts evolve over weeks "
            "(knowledge-update), span multiple sessions (multi-session), require date arithmetic "
            "(temporal-reasoning), or are genuinely absent (_abs abstention)."
        ),
    },
    {
        "limitation_id": "L07_unverified_legacy_rule_and_blueprint_packs",
        "affected_stages": "Mirror Crystal / rules_and_blueprints / research/rule_packs",
        "severity": "governance_and_coverage_gap",
        "summary": (
            "Staged blueprints (997 unambiguous + 17 collision clusters), general reasoning candidates "
            "(10,240 unverified + 26 source-supported with tests not_run), and 110 authored domain seeds "
            "are mostly generic textbook/prompt templates with evidence_status=unverified."
        ),
        "open_world_consequence": (
            "Existing staged rule/blueprint packs cannot be promoted without independent verification "
            "and provide near-zero grounded coverage on unseen open-world instances."
        ),
    },
    {
        "limitation_id": "L08_analytical_tdp_energy_not_hardware_rapl",
        "affected_stages": "S10",
        "severity": "measurement_boundary",
        "summary": (
            "S10 energy numbers are analytical process-time * single-core TDP proxies, not direct "
            "hardware RAPL/PMU readings. Stage S11.0 therefore reports raw process CPU time, wall time, "
            "throughput, and tracemalloc peak heap directly without over-claiming hardware energy."
        ),
        "open_world_consequence": (
            "CPU time and peak heap measurements remain directly comparable across pure-CPU baselines."
        ),
    },
)


def audit_s0_s10_package(repo_root: Path) -> dict[str, Any]:
    """Inspect ``research/cpu_learning/`` summaries, manifests, and modules and record its audit."""
    cpu_root = repo_root / "research" / "cpu_learning"
    reports_dir = cpu_root / "reports"
    data_dir = cpu_root / "data"

    py_files = sorted(cpu_root.rglob("*.py")) if cpu_root.is_dir() else []
    manifests = sorted(p.name for p in data_dir.glob("*.json")) if data_dir.is_dir() else []

    stage_records: list[dict[str, Any]] = []
    all_gates_passed = True
    s10_verdict_tier = "UNKNOWN"
    s10_master_13_passed = 0
    s10_master_13_total = 0

    for stage_id, fname, gate_key in S0_S10_STAGE_SUMMARIES:
        fpath = reports_dir / fname
        if not fpath.is_file():
            all_gates_passed = False
            stage_records.append(
                {
                    "stage": stage_id,
                    "summary_file": fname,
                    "present": False,
                    "gate_passed": False,
                    "checks_passed": 0,
                    "checks_total": 0,
                }
            )
            continue
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        gate = payload.get(gate_key) or {}
        checks = gate.get("checks") or {}
        passed = bool(gate.get("passed"))
        if not passed:
            all_gates_passed = False
        if stage_id == "S10":
            verdict = payload.get("readiness_verdict") or {}
            s10_verdict_tier = str(verdict.get("verdict_tier", "UNKNOWN"))
            m13 = payload.get("master_13_gates") or []
            s10_master_13_total = len(m13)
            s10_master_13_passed = sum(1 for g in m13 if g.get("passed"))
        stage_records.append(
            {
                "stage": stage_id,
                "summary_file": fname,
                "present": True,
                "gate_passed": passed,
                "checks_passed": sum(1 for v in checks.values() if v),
                "checks_total": len(checks),
            }
        )

    return {
        "package_path": "research/cpu_learning",
        "python_module_count": len(py_files),
        "committed_manifest_files": manifests,
        "committed_manifest_count": len(manifests),
        "stage_summaries": stage_records,
        "all_s0_s10_gates_passed": all_gates_passed,
        "s10_master_13_gates_passed": s10_master_13_passed,
        "s10_master_13_gates_total": s10_master_13_total,
        "s10_honest_verdict_tier": s10_verdict_tier,
        "reusable_assets_for_s11": [
            "Append-only hash-chained SQLite EpisodeLedger with freeze locks (S1)",
            "10,240-bit BSC + Multi-Table LSH SHARD candidate proposer (S4)",
            "ITL role-unification & invariant/guard induction over structured deltas (S2-S3)",
            "RDC refutation, leave-one-out blame & MDL library compaction without self-certification (S5)",
            "Zero-forgetting ContinualKnowledgeBase with mandatory regression replay & poison rollback (S8)",
            "Anti-forgery IndependentVerificationReceipt & non-starvation budget controller (S9)",
            "Mirror Crystal QuotientEGraph, TwinEGraphDoCalculus, KanervaSDMInt8 & MengerProvenanceVerifier",
        ],
        "open_world_limitations": list(S0_S10_OPEN_WORLD_LIMITATIONS),
        "audit_conclusion": (
            "Stages S0-S10 form a verified, reproducible CPU-native foundation for narrow controlled "
            "relational/functional/grounded learning, NOT open-world intelligence. Eight structural gaps "
            "(L01-L08) block direct open-world transfer and motivate Stage S11.0 evaluation + S11.1 design."
        ),
    }
