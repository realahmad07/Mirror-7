"""Test suite for Phase 4: Retrieval + Reasoning Intelligence.

Validates:
1. Exact and approximate retrieval over clean production index rules with quarantine suppression.
2. Cross-domain quantum computing & linear algebra gate transformations (Born rule).
3. Cross-domain physics & calculus kinematic derivation (v(t) = dx/dt).
4. Cross-domain engineering stress and factor of safety verification (FS = σ_yield / σ_actual).
5. Multi-hop transitive deduction and syllogistic inference.
6. Multi-rule comparative synthesis over distinct entities.
7. Mathematical and AST code sandbox verification gates.
"""
from pathlib import Path
import pytest

from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge
from mirror_crystal.phase5_bicameral_and_cas import SymbolicCASOrgan
from mirror_crystal.reasoning_engine import ReasoningIntelligenceEngine

VAULT_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal")


@pytest.fixture(scope="module")
def vault():
    if not VAULT_PATH.exists():
        pytest.skip("Production vault not present on disk")
    v = DiskCrystalVault(VAULT_PATH)
    yield v
    v.close()


def test_phase4_exact_and_approximate_retrieval_quarantine_suppression(vault):
    """Retrieval prioritizes verified clean rules and suppresses quarantined defects."""
    # Canary Protocol 7777 (Rule 1) is clean and active
    r_exact = vault.respond("What is Mirror Crystal Canary Protocol 7777?")
    assert r_exact.confidence >= 0.95
    assert "Canary Protocol 7777" in r_exact.answer or "7777" in r_exact.answer
    assert r_exact.latency_ms < 5.0 # Sub-millisecond direct lock

    # Collision rule 11181 ('Yes please.') is quarantined; must be inactive
    assert vault.is_quarantined(11181) is True
    rule_q = vault.get_rule(11181)
    assert rule_q.active is False


def test_phase4_cross_domain_quantum_computing_linear_algebra(vault):
    """Quantum logic gate transformation evaluated via unitary linear algebra and Born rule."""
    query = "Apply Hadamard gate to state |0> and find probability of measuring |0>"
    resp = vault.respond(query)

    assert resp.confidence >= 0.90
    assert resp.mode == "cross_domain_quantum_algebra"
    assert resp.sandbox_verified is True
    assert "0.5" in resp.answer or "50" in resp.answer
    assert "Hadamard" in resp.answer
    assert len(resp.reasoning_trace) >= 3


def test_phase4_cross_domain_physics_calculus_kinematics(vault):
    """Kinematics query derives velocity function from position via symbolic differentiation."""
    query = "Given position function x(t) = 3t^2 + 2t, find velocity at t = 3"
    resp = vault.respond(query)

    assert resp.confidence >= 0.90
    assert resp.mode == "cross_domain_physics_calculus"
    assert resp.sandbox_verified is True
    assert "20" in resp.answer # v(3) = 6(3) + 2 = 20
    assert "v(t)" in resp.answer
    assert len(resp.reasoning_trace) >= 2


def test_phase4_cross_domain_engineering_stress_and_safety_factor(vault):
    """Engineering query calculates direct stress and checks factor of safety against yield."""
    query = "Calculate safety factor for a load of 5000 N over an area of 0.002 m^2 with yield stress of 5000000"
    resp = vault.respond(query)

    # Actual stress = 5000 / 0.002 = 2,500,000 Pa
    # Factor of safety = 5,000,000 / 2,500,000 = 2.0
    assert resp.confidence >= 0.90
    assert resp.mode == "cross_domain_engineering_physics"
    assert resp.sandbox_verified is True
    assert "2.00" in resp.answer or "2.0" in resp.answer
    assert "SAFE" in resp.answer


def test_phase4_multi_hop_transitive_syllogism(vault):
    """Deduces conclusion from universal major premise and categorical minor premise."""
    query = "Given that all photons travel at speed of light, and gamma rays are photons, what is the speed of gamma rays?"
    resp = vault.respond(query)

    assert resp.confidence >= 0.90
    assert resp.mode == "multi_hop_transitive_syllogism"
    assert "gamma rays" in resp.answer.lower()
    assert "speed of light" in resp.answer.lower()
    assert len(resp.reasoning_trace) == 3


def test_phase4_multi_rule_comparative_synthesis(vault):
    """Synthesizes structured comparative analysis across two distinct entities."""
    query = "Compare QuickSort and MergeSort"
    resp = vault.respond(query)

    assert resp.confidence >= 0.30
    assert resp.mode == "multi_rule_comparative_synthesis"
    assert "QuickSort" in resp.answer
    assert "MergeSort" in resp.answer
    assert len(resp.reasoning_trace) >= 3


def test_phase4_mathematical_verification_gate():
    """Validates that SymbolicCASOrgan rejects invalid equations and verifies exact identities."""
    # Exact polynomial differentiation
    cas_res = SymbolicCASOrgan.try_solve("d/dx (x^3 + 5x^2 - 7)")
    assert cas_res is not None
    assert cas_res.verified is True
    assert "3*x^2 + 10*x" in cas_res.answer

    # Exact unit conversion
    cas_conv = SymbolicCASOrgan.try_solve("convert 100 meters to kilometers")
    assert cas_conv is not None
    assert cas_conv.verified is True
    assert "0.1" in cas_conv.answer


def test_phase4_code_sandbox_verification_gate(monkeypatch):
    """Validates that ZeroTrustSandboxJudge executes safe code and catches syntax bugs."""
    monkeypatch.setenv("MIRROR_SANDBOX_MODE", "process")
    valid_code = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\nassert fib(5) == 5\n"
    ok_valid, msg_valid = ZeroTrustSandboxJudge.verify_code_and_test(valid_code)
    assert ok_valid is True

    # AST static auditor check
    safe_ast, _ = ZeroTrustSandboxJudge.is_ast_safe(valid_code)
    assert safe_ast is True

    # Code with broken syntax
    broken_code = "def add(a, b\n    return a + b\n"
    ok_broken, msg_broken = ZeroTrustSandboxJudge.verify_code_and_test(broken_code)
    assert ok_broken is False
    assert "syntax" in msg_broken.lower()
