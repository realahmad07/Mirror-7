"""Stage 10: Independent Verification Test Suite.

Validates the five essential verification pillars:
1. External Reviewers & Governance (packet generation, independence validation, anti-fabrication).
2. Source & License Validation (HTTPS, cryptographic SHA-256 excerpt digests, SPDX license compliance).
3. Blind Benchmark Evaluation (held-out arithmetic, competition math, and negative controls).
4. Mathematical & Factual Accuracy (exact symbolic CAS, FTC+Simpson integral check, linear systems).
5. Cross-Domain Reasoning (quantum Born projection, kinematic calculus, structural mechanics, syllogisms).
"""
import json
from pathlib import Path
import pytest

from mirror_crystal import CerebrumCrystalBridge
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_eval.stage10_verification import (
    CrossDomainReasoningEvaluator,
    ExternalReviewerAuditEngine,
    MathematicalAndFactualVerifier,
    SourceAndLicenseValidator,
    BlindBenchmarkEvaluator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAULT_PATH = REPO_ROOT / "restorevalutecrytaled" / "mirror_crystal_all_datasets.dcrystal"
BENCHMARK_DATASET = REPO_ROOT / "mirror_eval" / "data" / "source_backed_factual_v1.jsonl"
BENCHMARK_MANIFEST = REPO_ROOT / "mirror_eval" / "data" / "source_backed_factual_v1.manifest.json"


@pytest.fixture(scope="module")
def vault():
    if not VAULT_PATH.exists():
        pytest.skip("Production vault not present on disk")
    v = DiskCrystalVault(VAULT_PATH)
    yield v
    v.close()


@pytest.fixture(scope="module")
def bridge():
    return CerebrumCrystalBridge()


# ============================================================================
# 1. External Reviewers & Governance
# ============================================================================

def test_stage10_external_reviewers_and_governance():
    """Validates unreviewed packet generation, reviewer independence rules, and zero fabrication."""
    assert BENCHMARK_DATASET.is_file()
    assert BENCHMARK_MANIFEST.is_file()

    # 1. Manifest governance check
    manifest = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["review_status"] == "pending_independent_review"
    assert manifest["required_distinct_independent_reviewers_per_claim"] == 2

    # 2. Reviewer packet audit (must not contain pre-filled approvals)
    packet = ExternalReviewerAuditEngine.audit_review_packet(BENCHMARK_DATASET, BENCHMARK_MANIFEST)
    assert packet["case_count"] == 8
    assert "No reviews have been created" in packet["notice"]

    # 3. Reviewer signature validation tests
    valid_review = {
        "reviewer_id": "independent-expert-alice",
        "independent": True,
        "method": "independent_source_comparison",
        "decision": "approve",
        "reviewed_at": "2026-11-10T12:00:00+00:00",
    }
    is_valid, msg = ExternalReviewerAuditEngine.validate_reviewer_signature(valid_review)
    assert is_valid is True, f"Valid review failed: {msg}"

    # Non-independent reviewer must fail
    non_indep = dict(valid_review, independent=False)
    assert ExternalReviewerAuditEngine.validate_reviewer_signature(non_indep)[0] is False

    # Naive non-timezone timestamp must fail
    bad_time = dict(valid_review, reviewed_at="2026-11-10 12:00:00")
    assert ExternalReviewerAuditEngine.validate_reviewer_signature(bad_time)[0] is False

    # 4. Multi-reviewer case approval threshold
    rev_bob = dict(valid_review, reviewer_id="independent-expert-bob")
    ok, _ = ExternalReviewerAuditEngine.verify_case_reviews([valid_review, rev_bob], min_reviewers=2)
    assert ok is True

    # Single reviewer fails threshold
    single_ok, _ = ExternalReviewerAuditEngine.verify_case_reviews([valid_review], min_reviewers=2)
    assert single_ok is False

    # Duplicate reviewer ID fails threshold
    dup_ok, _ = ExternalReviewerAuditEngine.verify_case_reviews([valid_review, valid_review], min_reviewers=2)
    assert dup_ok is False


# ============================================================================
# 2. Source & License Validation
# ============================================================================

def test_stage10_source_and_license_validation():
    """Validates HTTPS citations, exact SHA-256 excerpt digests, and SPDX compliance."""
    # 1. Primary source verification across all benchmark cases
    lines = BENCHMARK_DATASET.read_text(encoding="utf-8").strip().splitlines()
    claims_verified = 0
    for line in lines:
        case = json.loads(line)
        for claim in case.get("claims", []):
            valid, msg = SourceAndLicenseValidator.validate_primary_source(claim)
            assert valid is True, f"Claim {claim['claim_id']} failed source validation: {msg}"
            claims_verified += 1
    assert claims_verified >= 6

    # 2. Tampered excerpt detection
    corrupted_claim = {
        "claim_id": "test-corrupted",
        "source_publisher": "Test Org",
        "source_url": "https://example.org/doc",
        "source_locator": "Section 1",
        "source_quote": "Modified text that does not match original hash",
        "source_excerpt_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "license_spdx": "Apache-2.0",
    }
    is_valid, msg = SourceAndLicenseValidator.validate_primary_source(corrupted_claim)
    assert is_valid is False
    assert "digest mismatch" in msg

    # 3. Repository license audit
    repo_audit = SourceAndLicenseValidator.audit_repo_license(REPO_ROOT)
    assert repo_audit["ok"] is True
    assert repo_audit["license_type"] == "Apache-2.0"
    assert repo_audit["license_file_present"] is True


# ============================================================================
# 3. Blind Benchmark Evaluation
# ============================================================================

def test_stage10_blind_benchmark_evaluation(bridge):
    """Evaluates agent responses across held-out arithmetic, algebra, and negative controls."""
    held_out_battery = [
        # GSM8K external problems
        {
            "name": "gsm8k_baker",
            "prompt": "A baker made 60 muffins. He sold 20 in the morning and 10 in the afternoon. How many muffins are left?",
            "answerable": True,
            "expected_sub": "30",
        },
        {
            "name": "gsm8k_marbles",
            "prompt": "Leo has 24 marbles. Maya has half as many. How many marbles do they have altogether?",
            "answerable": True,
            "expected_sub": "36",
        },
        # Competition MATH external problems
        {
            "name": "math_large_pow",
            "prompt": "What is the remainder when 19^25 is divided by 1000000007?",
            "answerable": True,
            "expected_sub": "240323544",
        },
        {
            "name": "math_integral",
            "prompt": "Integrate 6*x^2 + 4*x from 0 to 3",
            "answerable": True,
            "expected_sub": "72",
        },
        {
            "name": "math_quadratic",
            "prompt": "Solve x^2 - 7*x + 12 = 0",
            "answerable": True,
            "expected_sub": "3",
        },
        {
            "name": "math_combinatorics",
            "prompt": "Compute 8 choose 3",
            "answerable": True,
            "expected_sub": "56",
        },
        # Negative control (unanswerable private maintainer token)
        {
            "name": "negative_control_secret",
            "prompt": "What is the private recovery token for the maintainer Mirror 7 account?",
            "answerable": False,
            "expected_sub": None,
        },
    ]

    class BridgeAgentAdapter:
        def __init__(self, b):
            self.b = b

        def respond(self, prompt):
            res = self.b.solve(prompt)
            from types import SimpleNamespace
            ok = res.get("ok", False)
            ans = str(res.get("answer", ""))
            return SimpleNamespace(
                answer=ans,
                confidence=1.0 if ok else 0.0,
                mode="symbolic_cas" if ok else "honest_abstain",
                verified=ok,
            )

    report = BlindBenchmarkEvaluator.evaluate_held_out_cases(BridgeAgentAdapter(bridge), held_out_battery)
    assert report["passed"] == report["total"], f"Failed cases: {report}"
    assert report["accuracy"] == 1.0


# ============================================================================
# 4. Mathematical & Factual Accuracy
# ============================================================================

def test_stage10_mathematical_and_factual_accuracy():
    """Validates exact algebraic solutions, definite integration, and primary factual data."""
    # 1. Higher-order polynomial derivative
    d_ok = MathematicalAndFactualVerifier.verify_polynomial_derivative(
        "3x^5 - 7x^3 + 11x - 4",
        "15*x^4 - 21*x^2 + 11",
    )
    assert d_ok is True

    # 2. Definite integral with exact rational and Simpson numerical check
    i_ok = MathematicalAndFactualVerifier.verify_definite_integral(
        "9x^2 - 4x + 2",
        a=1,
        b=3,
        expected_val=66,
    )
    assert i_ok is True

    # 3. Multivariable linear system over Q
    s_ok = MathematicalAndFactualVerifier.verify_linear_system(
        "4x + 5y = 26 and 3x - 2y = 8",
        {"x": "4", "y": "2"},
    )
    assert s_ok is True

    # 4. Exact dimensional unit conversion
    c_ok = MathematicalAndFactualVerifier.verify_unit_conversion(
        val="108",
        u_from="km/h",
        u_to="m/s",
        expected_val="30",
    )
    assert c_ok is True


# ============================================================================
# 5. Cross-Domain Reasoning
# ============================================================================

def test_stage10_cross_domain_reasoning(vault, bridge):
    """Validates multi-hop reasoning across quantum physics, kinematics, materials, and logic."""
    # 1. Quantum Born projection (Quantum -> Linear algebra)
    q_pass, q_ans = CrossDomainReasoningEvaluator.evaluate_quantum_born_projection(vault)
    assert q_pass is True, f"Quantum Born projection failed: {q_ans}"

    # 2. Kinematic calculus chain (Physics -> Calculus)
    k_pass, k_ans = CrossDomainReasoningEvaluator.evaluate_kinematic_calculus_chain(vault)
    assert k_pass is True, f"Kinematic chain failed: {k_ans}"

    # 3. Structural safety factor (Materials -> Engineering)
    s_pass, s_ans = CrossDomainReasoningEvaluator.evaluate_structural_safety_factor(vault)
    assert s_pass is True, f"Structural factor of safety failed: {s_ans}"

    # 4. Transitive categorical syllogism (Logic -> Chemistry)
    l_pass, l_ans = CrossDomainReasoningEvaluator.evaluate_transitive_categorical_syllogism(vault)
    assert l_pass is True, f"Categorical syllogism failed: {l_ans}"

    # 5. Quantum Grover scaling (Quantum algorithms -> Complexity)
    g_pass, g_ans = CrossDomainReasoningEvaluator.evaluate_quantum_grover_scaling(bridge)
    assert g_pass is True, f"Grover scaling failed: {g_ans}"
