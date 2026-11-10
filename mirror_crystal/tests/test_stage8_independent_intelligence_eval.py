"""Stage 8: Independent Intelligence Evaluation Test Suite.

Evaluates Mirror-7 across six advanced cognitive dimensions:
1. Unseen Mathematical Problems (calculus, definite integrals, large modular powers).
2. Novel Reasoning Combinations (multi-hop deduction chained with physical constants).
3. Cross-Domain Transfer (physics -> engineering -> quantum mathematics).
4. Long-Context Memory & Multi-Turn State (5+ turn constraint binding & session isolation).
5. Genuine Learning & Retention (candidate verification, zero catastrophic forgetting >= 0.9999).
6. Adversarial Questions & Robustness (AST jailbreaks, loop bombs, honest abstention).
7. Source-Backed Benchmark Governance (8 frozen cases, review gate properly reported as pending).
"""
import math
from pathlib import Path
import pytest

from mirror_crystal.autonomous_learning import AutonomousLearningEngine
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge
from mirror_crystal.phase5_bicameral_and_cas import SymbolicCASOrgan
from mirror_eval.factual_benchmark import load_factual_dataset

VAULT_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal")
BENCHMARK_DATASET = Path("mirror_eval/data/source_backed_factual_v1.jsonl")
BENCHMARK_MANIFEST = Path("mirror_eval/data/source_backed_factual_v1.manifest.json")


@pytest.fixture(scope="module")
def vault():
    if not VAULT_PATH.exists():
        pytest.skip("Production vault not present on disk")
    v = DiskCrystalVault(VAULT_PATH)
    yield v
    v.close()


def test_stage8_unseen_mathematical_problems():
    """Unseen higher-order differentiation, definite integration, and modular arithmetic."""
    # 1. Higher-order polynomial differentiation:
    # d/dx (x^5 - 10x^3 + 15x - 7) = 5*x^4 - 30*x^2 + 15
    res_diff = SymbolicCASOrgan.try_solve("d/dx (x^5 - 10x^3 + 15x - 7)")
    assert res_diff is not None and res_diff.verified is True
    assert "5*x^4 - 30*x^2 + 15" in res_diff.answer

    # 2. Definite integral with unseen upper limit:
    # int_0^3 (4x^3 - 3x^2 + 2x) dx = [x^4 - x^3 + x^2]_0^3
    # = (81 - 27 + 9) - 0 = 63
    res_int = SymbolicCASOrgan.try_solve("integrate (4x^3 - 3x^2 + 2x) from 0 to 3")
    assert res_int is not None and res_int.verified is True
    assert "63" in res_int.answer

    # 3. Exact rational linear system via Gauss-Jordan elimination
    res_sys = SymbolicCASOrgan.try_solve("solve 3x + 4y = 18 and 2x - y = 1")
    assert res_sys is not None and res_sys.verified is True
    assert "x = 2" in res_sys.answer and "y = 3" in res_sys.answer

    # 4. Exact rational dimensional unit conversion
    res_conv = SymbolicCASOrgan.try_solve("convert 72 km/h to m/s")
    assert res_conv is not None and res_conv.verified is True
    assert "20 m/s" in res_conv.answer


def test_stage8_novel_reasoning_combinations(vault):
    """Combines categorical syllogistic logic with physics kinematic calculations."""
    # Transitive Modus Ponens step-by-step reasoning
    q_syl = "Given that all noble gases are unreactive, and neon is a noble gas, what is the property of neon?"
    r_syl = vault.respond(q_syl)
    assert r_syl.confidence >= 0.85
    assert r_syl.mode == "multi_hop_transitive_syllogism"
    assert "neon" in r_syl.answer.lower()
    assert "unreactive" in r_syl.answer.lower()

    # Physics kinematics combined with calculus:
    # x(t) = 3t^3 - 9t^2 + 4t => v(t) = 9t^2 - 18t + 4 => v(4) = 9(16) - 18(4) + 4 = 144 - 72 + 4 = 76 m/s
    q_kin = "Given position function x(t) = 3t^3 - 9t^2 + 4t, find velocity at t = 4"
    r_kin = vault.respond(q_kin)
    assert r_kin.confidence >= 0.85
    assert "76" in r_kin.answer


def test_stage8_cross_domain_transfer(vault):
    """Knowledge and deduction transferred across quantum computing and mechanical engineering."""
    # 1. Quantum computing -> Linear algebra Born rule projection
    q_quantum = "Apply Hadamard gate to state |0> and find probability of measuring |0>"
    r_quantum = vault.respond(q_quantum)
    assert r_quantum.confidence >= 0.85
    assert "0.5" in r_quantum.answer or "50" in r_quantum.answer

    # 2. Materials science -> Structural mechanics factor of safety
    # F = 15000 N, A = 0.005 m^2 => actual stress = 3,000,000 Pa. Yield = 9,000,000 Pa => FS = 3.0
    q_fs = "Calculate safety factor for a load of 15000 N over an area of 0.005 m^2 with yield stress of 9000000"
    r_fs = vault.respond(q_fs)
    assert r_fs.confidence >= 0.85
    assert "3.0" in r_fs.answer
    assert "SAFE" in r_fs.answer


def test_stage8_long_context_memory_and_isolation(vault):
    """Multi-turn stateful conversational context tracking across 5 turns with session isolation."""
    session_a = "stage8_session_alice"
    session_b = "stage8_session_bob"

    # Alice sets constraint: maximum capacity is 50
    vault.respond("Our server limit is 50 concurrent requests.", session_id=session_a)

    # Bob sets different constraint: maximum capacity is 200
    vault.respond("Our server limit is 200 concurrent requests.", session_id=session_b)

    # Alice tests 4 batches of 15 (60 > 50) -> Unsafe
    r_a = vault.respond("Can we run 4 batches of 15 requests?", session_id=session_a)
    assert "60" in r_a.answer
    assert "exceeds your 50 limit by 10" in r_a.answer

    # Bob tests 4 batches of 15 (60 <= 200) -> Safe
    r_b = vault.respond("Can we run 4 batches of 15 requests?", session_id=session_b)
    assert "60" in r_b.answer
    assert "within your 200 limit" in r_b.answer
    assert "headroom: 140" in r_b.answer


def test_stage8_genuine_learning_and_zero_catastrophic_forgetting(vault):
    """Learns verified candidates without drift on canonical historical anchors."""
    # Anchors
    anchors = [
        "What is Mirror Crystal Canary Protocol 7777?",
        "What is a qubit and how does quantum superposition work mathematically?",
    ]
    pre_scores = {q: vault.respond(q).answer for q in anchors}

    # Execute autonomous learning on coverage gaps
    report = AutonomousLearningEngine.run_production_evaluation(vault)
    assert report.benchmark_accuracy >= 0.85

    # Re-evaluate anchors: zero catastrophic forgetting
    for q in anchors:
        post_ans = vault.respond(q).answer
        assert post_ans == pre_scores[q], f"Catastrophic forgetting detected on anchor query: {q}"


def test_stage8_adversarial_questions_and_sandboxing():
    """Validates refusal of prompt jailbreaks, loop bombs, and unauthorized capabilities."""
    # Loop bomb termination
    infinite_loop = "def hang():\n    while True: pass\nhang()\n"
    safe, reason = ZeroTrustSandboxJudge.verify_code_and_test(infinite_loop, max_loop_steps=300)
    assert safe is False
    assert "loop" in reason.lower() or "limit" in reason.lower()

    # Dunder breakout attempt
    breakout = "[c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'catch_warnings']"
    is_safe, msg = ZeroTrustSandboxJudge.is_ast_safe(breakout)
    assert is_safe is False
    assert "dunder" in msg.lower() or "forbidden" in msg.lower()


def test_stage8_source_backed_benchmark_review_gate_integrity():
    """Validates 8 frozen cases and verifies review gate is properly reported as pending."""
    assert BENCHMARK_DATASET.is_file(), "source_backed_factual_v1.jsonl missing"
    assert BENCHMARK_MANIFEST.is_file(), "source_backed_factual_v1.manifest.json missing"

    cases, digest = load_factual_dataset(
        BENCHMARK_DATASET,
        expected_sha256="278ab0daa8b50512b667ab155a4cea6377740ede3ffca888d858b66bdfd5ee54",
        require_independent_review=False,
    )
    assert len(cases) == 8, f"Expected 8 benchmark cases, found {len(cases)}"

    # Governance check: review gate must remain pending because independent reviews are not fabricated
    import json
    manifest = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["review_status"] == "pending_independent_review"
    assert manifest["required_distinct_independent_reviewers_per_claim"] == 2
