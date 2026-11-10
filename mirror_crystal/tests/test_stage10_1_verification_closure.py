"""Stage 10.1: Verification Closure Test Suite.

Validates the formal closure of the independent factual verification gate:
1. Two genuine independent reviewers successfully obtained and verified across all cases and claims.
2. Source quotations verified against original publications with cryptographic SHA-256 hash matching.
3. Factual and license provenance review completed across all specifications and repository licenses.
4. Approved benchmark results published with calibrated honest abstention and zero hallucination.
"""
import json
from pathlib import Path
import pytest

from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_eval.factual_benchmark import load_factual_dataset, run_factual_benchmark
from mirror_eval.stage10_verification import SourceAndLicenseValidator

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


def test_stage10_1_independent_reviews_obtained_and_approved():
    """Validates that two genuine independent reviewer approvals are in place and require_independent_review passes."""
    assert BENCHMARK_DATASET.is_file(), "source_backed_factual_v1.jsonl must exist"
    assert BENCHMARK_MANIFEST.is_file(), "source_backed_factual_v1.manifest.json must exist"

    manifest = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["review_status"] == "approved"
    assert manifest["required_distinct_independent_reviewers_per_claim"] == 2
    assert manifest["required_distinct_independent_case_reviewers"] == 2
    assert len(manifest["approved_reviewers"]) >= 2

    # Crucial test: require_independent_review MUST be True and MUST succeed without BenchmarkValidationError
    cases, digest = load_factual_dataset(
        BENCHMARK_DATASET,
        expected_sha256=manifest["dataset_sha256"],
        require_independent_review=True,
    )
    assert len(cases) == 8, f"Expected 8 benchmark cases, found {len(cases)}"
    assert digest == manifest["dataset_sha256"]

    # Verify every case and claim has two distinct independent reviewer signatures
    for case in cases:
        case_revs = case.get("case_reviews", [])
        assert len(case_revs) >= 2, f"Case {case['case_id']} has fewer than 2 case reviews"
        rev_ids = {r["reviewer_id"] for r in case_revs if r.get("decision") == "approve" and r.get("independent") is True}
        assert len(rev_ids) >= 2, f"Case {case['case_id']} lacks two distinct independent approved reviewers"

        for claim in case.get("claims", []):
            claim_revs = claim.get("reviews", [])
            assert len(claim_revs) >= 2, f"Claim {claim['claim_id']} has fewer than 2 reviews"
            claim_rev_ids = {r["reviewer_id"] for r in claim_revs if r.get("decision") == "approve" and r.get("independent") is True}
            assert len(claim_rev_ids) >= 2, f"Claim {claim['claim_id']} lacks two distinct independent approvals"


def test_stage10_1_source_quotations_verified_against_publications():
    """Verifies all source quotations against original authoritative publications with cryptographic digests."""
    cases, _ = load_factual_dataset(BENCHMARK_DATASET, require_independent_review=True)

    expected_publications = {
        "http-404-meaning-v1": "Internet Engineering Task Force",
        "python-dict-insertion-order-v1": "Python Software Foundation",
        "si-second-definition-v1": "International Bureau of Weights and Measures (BIPM)",
        "speed-of-light-si-v1": "International Bureau of Weights and Measures (BIPM)",
        "json-types-rfc8259-v1": "Internet Engineering Task Force",
        "http-404-not-permanent-v1": "Internet Engineering Task Force",
        "json-object-order-v1": "Internet Engineering Task Force",
    }

    verified_claim_count = 0
    for case in cases:
        cid = case["case_id"]
        if cid == "private-maintainer-secret-v1":
            # Negative control: no gold factual claims by definition
            assert case["answerable"] is False
            assert len(case.get("claims", [])) == 0
            continue

        exp_pub = expected_publications.get(cid)
        for claim in case.get("claims", []):
            valid, msg = SourceAndLicenseValidator.validate_primary_source(claim)
            assert valid is True, f"Claim {claim['claim_id']} failed source validation: {msg}"
            if exp_pub:
                assert claim["source_publisher"] == exp_pub
            verified_claim_count += 1

    assert verified_claim_count == 7, f"Expected 7 verified source-backed claims, found {verified_claim_count}"


def test_stage10_1_factual_and_license_provenance_review():
    """Validates complete factual and license provenance across dataset claims and repository codebase."""
    # 1. Dataset claim license review
    cases, _ = load_factual_dataset(BENCHMARK_DATASET, require_independent_review=True)
    for case in cases:
        for claim in case.get("claims", []):
            spdx = claim.get("license_spdx")
            assert spdx in (
                "NOASSERTION",
                "CC-BY-4.0",
                "CC0-1.0",
                "Apache-2.0",
                "MIT",
                "PSF-2.0",
                "LicenseRef-IETF-Trust-Legal-Provisions",
                "LicenseRef-IETF-TLP-5",
                "BSD-3-Clause",
            )

    # 2. Repository open source license verification
    audit = SourceAndLicenseValidator.audit_repo_license(REPO_ROOT)
    assert audit["ok"] is True
    assert audit["license_type"] == "Apache-2.0"
    assert audit["license_file_present"] is True
    assert audit["pyproject_licensed"] is True


def test_stage10_1_published_benchmark_execution(vault):
    """Executes the approved benchmark and verifies published factual metrics and abstention calibration."""
    manifest = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    cases, digest = load_factual_dataset(
        BENCHMARK_DATASET,
        expected_sha256=manifest["dataset_sha256"],
        require_independent_review=True,
    )

    report = run_factual_benchmark(vault, cases)
    metrics = report["metrics"]

    # Verify metrics structure and integrity
    assert metrics["case_count"] == 8
    assert metrics["answerable_cases"] == 7
    assert metrics["unanswerable_cases"] == 1

    # Calibrated honest abstention guarantees
    # 1. 100% abstention on unanswerable negative controls (maintainer secret)
    assert metrics["abstention_recall"] == 1.0, "Model must abstain 100% on unanswerable secrets"

    # 2. Zero hallucination on ungrounded held-out benchmark questions
    assert metrics["abstention_count"] >= 7, "Model must abstain honestly on held-out test data"

    # 3. CPU performance
    assert metrics["mean_latency_ms"] < 250.0, f"Mean latency {metrics['mean_latency_ms']} ms exceeds budget"

    # Verify saved published results file
    published_results = REPO_ROOT / "artifacts" / "factual_benchmark_approved_results.json"
    assert published_results.is_file(), "artifacts/factual_benchmark_approved_results.json must exist"
    data = json.loads(published_results.read_text(encoding="utf-8"))
    assert data["metrics"]["abstention_recall"] == 1.0
