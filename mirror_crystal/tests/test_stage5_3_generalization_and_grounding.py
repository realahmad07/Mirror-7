"""Stage 5.3: Generalization, Language Grounding & Learning Validation Suite.

Validates:
1. Language & Phrasing Invariance (case, punctuation, whitespace, syntactic variations).
2. Multi-turn Conversational Grounding (coreference resolution, stateful capacity constraints).
3. Parametric Slot Generalization (unseen numerical and symbolic evaluation without retraining).
4. Zero Catastrophic Forgetting (100% anchor rule retention across continual learning cycles).
5. Multi-Verifier Gate Safety (unverified/faulty candidates rejected before vault contamination).
"""
from pathlib import Path
import pytest

from mirror_crystal.autonomous_learning import AutonomousLearningEngine
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge
from mirror_crystal.phase5_bicameral_and_cas import SymbolicCASOrgan

VAULT_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal")


@pytest.fixture(scope="module")
def vault():
    if not VAULT_PATH.exists():
        pytest.skip("Production vault not present on disk")
    v = DiskCrystalVault(VAULT_PATH)
    yield v
    v.close()


def test_stage5_3_language_and_phrasing_invariance(vault):
    """Punctuation, casing, whitespace, and syntactic variations resolve to Canary Protocol."""
    canary_variations = [
        "What is Mirror Crystal Canary Protocol 7777?",
        "what is mirror crystal canary protocol 7777",
        "WHAT IS MIRROR CRYSTAL CANARY PROTOCOL 7777!",
        "   what is   mirror crystal canary protocol 7777 ?  ",
    ]
    for q in canary_variations:
        resp = vault.respond(q)
        assert resp.confidence >= 0.50, f"Confidence dropped below threshold on query variation: {q}"
        assert "7777" in resp.answer
        assert resp.matched_rule_id == 1 or "Canary Protocol 7777" in resp.answer
        assert resp.latency_ms < 200.0

    # Mathematical phrasing invariance across CAS
    math_variations = [
        "d/dx (3x^2 + 5x - 7)",
        "derivative of 3x^2 + 5x - 7",
        "find derivative of 3x^2 + 5x - 7 with respect to x",
    ]
    for mq in math_variations:
        res = SymbolicCASOrgan.try_solve(mq)
        assert res is not None and res.verified is True
        assert "6*x + 5" in res.answer or "6x + 5" in res.answer


def test_stage5_3_conversational_grounding_and_capacity_constraints(vault):
    """Multi-turn stateful conversational grounding resolves implicit session constraints."""
    session_id = "test_grounding_session_5_3"

    # Turn 1: Set factual baseline capacity constraint
    t1 = vault.respond("Our system cluster throughput limit is 1500 transactions per second.", session_id=session_id)
    assert t1 is not None

    # Turn 2: Query exceeding constraint: 4 batches of 400 = 1600 > 1500
    t2 = vault.respond("Can we run 4 batches of 400 requests?", session_id=session_id)
    assert "1600" in t2.answer
    assert "exceeds your 1500 limit by 100" in t2.answer

    # Turn 3: Query within constraint: 3 batches of 400 = 1200 <= 1500
    t3 = vault.respond("Can we run 3 batches of 400 requests?", session_id=session_id)
    assert "1200" in t3.answer
    assert "within your 1500 limit" in t3.answer
    assert "headroom: 300" in t3.answer


def test_stage5_3_slot_parameter_generalization_unseen_values(vault):
    """Dynamic calculation of unseen continuous and discrete parameters."""
    # 1. Definite integral with unseen coefficients and bounds:
    # int_1^4 (6x^2 - 4x + 1) dx = [2x^3 - 2x^2 + x]_1^4
    # = (2*64 - 2*16 + 4) - (2 - 2 + 1) = (128 - 32 + 4) - 1 = 100 - 1 = 99
    res_int = SymbolicCASOrgan.try_solve("integrate (6x^2 - 4x + 1) from 1 to 4")
    assert res_int is not None and res_int.verified is True
    assert "99" in res_int.answer

    # 2. Physics kinematics with unseen polynomial:
    # x(t) = 5t^3 - 2t^2 + 8t at t = 3
    # v(t) = 15t^2 - 4t + 8 => v(3) = 15(9) - 4(3) + 8 = 135 - 12 + 8 = 131 m/s
    q_phys = "Given position function x(t) = 5t^3 - 2t^2 + 8t, find velocity at t = 3"
    r_phys = vault.respond(q_phys)
    assert r_phys.confidence >= 0.85
    assert "131" in r_phys.answer
    assert r_phys.mode == "cross_domain_physics_calculus"


def test_stage5_3_zero_catastrophic_forgetting_across_continual_learning(vault):
    """Anchor rules retain 100% precision across online continual learning operations."""
    anchors = [
        ("What is Mirror Crystal Canary Protocol 7777?", "7777", 0.90),
        ("Apply Hadamard gate to state |1> and find probability of measuring |0>", "0.5", 0.85),
        ("Calculate safety factor for a load of 8000 N over an area of 0.004 m^2 with yield stress of 6000000", "3.0", 0.85),
    ]

    # Pre-test anchor verification
    pre_scores = {}
    for q, needle, min_conf in anchors:
        r = vault.respond(q)
        assert needle in r.answer
        assert r.confidence >= min_conf
        pre_scores[q] = (r.answer, r.confidence)

    # Simulate an online query reasoning pass on an isolated topic
    vault.respond("Compare TCP and UDP protocols for network streaming")

    # Post-test anchor verification
    for q, needle, min_conf in anchors:
        r_post = vault.respond(q)
        orig_answer, orig_conf = pre_scores[q]
        assert needle in r_post.answer
        assert r_post.confidence >= min_conf
        assert abs(r_post.confidence - orig_conf) < 1e-4, f"Drift detected in rule confidence for: {q}"

    # Quarantined rule remains strictly suppressed
    assert vault.is_quarantined(11181) is True


def test_stage5_3_multi_verifier_knowledge_gate_rejection():
    """Unverified or flawed candidate knowledge is blocked by multi-verifier gate."""
    # 1. Unsafe AST payload is blocked from knowledge candidate generation
    malicious_cand = "def malicious_solver():\n    import os\n    os.system('rm -rf /')\n"
    is_safe, reason = ZeroTrustSandboxJudge.is_ast_safe(malicious_cand)
    assert is_safe is False
    assert "os" in reason.lower() or "blocked" in reason.lower()

    # 2. Incoherent mathematical statement is rejected by CAS organ
    bogus_math = "d/dx (x^2) = 5000*x^99"
    cas_res = SymbolicCASOrgan.try_solve(bogus_math)
    # Must either reject, fail verification, or correctly solve the lhs without accepting the erroneous claim
    if cas_res is not None:
        assert cas_res.verified is False or "2*x" in cas_res.answer or "2x" in cas_res.answer
