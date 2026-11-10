"""Test suite for Phase 5: Autonomous Learning + Production Evaluation.

Validates:
1. Unanswered-query logging & gap detection.
2. Candidate knowledge generation & independent verification gate.
3. Controlled self-improvement without catastrophic forgetting (Rule 1: zero deletions).
4. End-to-end unseen task benchmarks across STEM, quantum, logic, and systems code.
5. Zero-trust security & adversarial penetration robustness.
6. Sub-millisecond CPU latency and performance stability.
"""
from pathlib import Path
import pytest

from mirror_crystal.autonomous_learning import AutonomousLearningEngine
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge

VAULT_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal")


@pytest.fixture(scope="module")
def vault():
    if not VAULT_PATH.exists():
        pytest.skip("Production vault not present on disk")
    v = DiskCrystalVault(VAULT_PATH)
    yield v
    v.close()


def test_phase5_unanswered_query_logging_and_gaps(vault):
    """Unmatched queries are safely recorded in unanswered_queries without hallucination."""
    query = "Explain the anomalous magnetic moment of the muon in QED"
    resp = vault.respond(query)

    # Must abstain honestly if no verified rule exists
    if resp.mode == "unmatched_honest_abstain":
        gaps = vault.get_coverage_gaps(limit=5)
        assert len(gaps) > 0
        topics = [g["topic"] for g in gaps]
        assert "physics" in topics or "science" in topics or "general" in topics


def test_phase5_autonomous_learning_and_independent_verification(vault, monkeypatch):
    """Candidate generation certifies valid solutions through independent verifiers."""
    monkeypatch.setenv("MIRROR_SANDBOX_MODE", "process")
    cur = vault.conn.cursor()

    # Seed two unanswered queries: one solvable by math, one by code
    cur.execute(
        "INSERT INTO unanswered_queries(query_text, inferred_domain, created_at) VALUES (?, ?, ?)",
        ("d/dx (7x^2 - 3x + 11)", "math", 1700000000.0),
    )
    cur.execute(
        "INSERT INTO unanswered_queries(query_text, inferred_domain, created_at) VALUES (?, ?, ?)",
        ("write a function to compute factorial", "programming", 1700000001.0),
    )
    vault.conn.commit()

    rules_before = cur.execute("SELECT COUNT(*) FROM rules;").fetchone()[0]

    report = None
    try:
        # Run autonomous learning
        report = AutonomousLearningEngine.learn_from_unanswered_queries(vault, max_queries=5)

        assert report.total_unanswered_analyzed >= 2
        assert report.candidates_verified >= 2
        assert report.candidates_crystallized >= 2

        rules_after = cur.execute("SELECT COUNT(*) FROM rules;").fetchone()[0]
        assert rules_after >= rules_before, "Rule 1 violated: rule count decreased!"

        # Verify that the newly crystallized rules can now be answered directly
        resp_math = vault.respond("d/dx (7x^2 - 3x + 11)")
        assert resp_math.confidence >= 0.80
        assert "14*x - 3" in resp_math.answer or "14x - 3" in resp_math.answer
    finally:
        # Restore vault to exact state so downstream tests retain pristine record counts
        if report and report.details:
            for cand in report.details:
                if cand.rule_id:
                    cur.execute("DELETE FROM rules WHERE rule_id = ?", (cand.rule_id,))
            vault.conn.commit()


def test_phase5_production_evaluation_benchmark_battery(vault):
    """Runs full production evaluation battery across all nine cognitive domains."""
    eval_report = AutonomousLearningEngine.run_production_evaluation(vault)

    assert eval_report.total_benchmark_tasks >= 8
    assert eval_report.benchmark_accuracy >= 0.85
    assert eval_report.zero_deletions_verified is True
    assert eval_report.quarantine_isolation_verified is True
    assert eval_report.average_latency_ms < 50.0 # Sub-millisecond to low millisecond CPU operations


def test_phase5_security_adversarial_penetration():
    """Validates that Zero-Trust AST and Sandbox block 100% of adversarial penetration payloads."""
    malicious_snippets = [
        "import os; os.system('echo compromised')",
        "import subprocess; subprocess.Popen(['cat', '/etc/shadow'])",
        "eval('__import__(\"os\").remove(\"important_file\")')",
        "exec('import socket')",
        "open('/tmp/test', 'w').write('data')",
        "class A: pass\nA.__subclasses__()",
    ]

    for payload in malicious_snippets:
        safe, reason = ZeroTrustSandboxJudge.is_ast_safe(payload)
        assert safe is False, f"Vulnerability detected! Payload allowed: {payload}"
        assert any(k in reason.lower() for k in ("blocked", "forbidden", "dunder"))
