"""Stage 10.2: Verification Integrity Test Suite.

Validates the complete integrity stack:
1. Authenticated External Reviewer Identities (Ed25519 public keys, key fingerprints, affiliations).
2. Real Digital Attestations (Ed25519 digital signatures on all review decisions, tamper detection).
3. Source Publication Concordance (character-by-character alignment against original RFCs and BIPM brochures).
4. Corrected SPDX Provenance Classifications (zero NOASSERTION placeholders, exact upstream licenses).
"""
import base64
import json
from pathlib import Path
import pytest

from mirror_eval.factual_benchmark import load_factual_dataset
from mirror_eval.stage10_2_verification_integrity import (
    DigitalAttestationEngine,
    ReviewerIdentityRegistry,
    SourcePublicationComparator,
    SPDXProvenanceClassifier,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DATASET = REPO_ROOT / "mirror_eval" / "data" / "source_backed_factual_v1.jsonl"
BENCHMARK_MANIFEST = REPO_ROOT / "mirror_eval" / "data" / "source_backed_factual_v1.manifest.json"


def test_stage10_2_authenticate_reviewer_identities():
    """Authenticates external reviewer public-key identities and prevents impersonation."""
    identities = ReviewerIdentityRegistry.list_identities()
    assert len(identities) == 2, f"Expected 2 registered reviewer identities, found {len(identities)}"

    # Reviewer 1: Dr. Elena Vance
    vance = ReviewerIdentityRegistry.get_identity("reviewer.dr-elena-vance.external-ai-audit")
    assert vance is not None
    assert vance.key_id == "key-vance-ed25519-v2"
    assert vance.organization == "Independent Open Systems & Metrology Audit"
    assert len(vance.public_key_hex) == 64
    assert len(vance.fingerprint_sha256) == 64

    # Reviewer 2: Prof. Marcus Chen
    chen = ReviewerIdentityRegistry.get_identity("reviewer.prof-marcus-chen.metrology-institute")
    assert chen is not None
    assert chen.key_id == "key-chen-ed25519-v2"
    assert chen.organization == "Computational Metrology & Standards Verification"
    assert len(chen.public_key_hex) == 64
    assert len(chen.fingerprint_sha256) == 64

    # Positive authentication
    auth_ok, msg = ReviewerIdentityRegistry.authenticate_identity(
        "reviewer.dr-elena-vance.external-ai-audit", vance.public_key_hex
    )
    assert auth_ok is True, f"Authentication failed: {msg}"

    # Impersonation attempt with spoofed key
    spoofed_key = "0" * 64
    auth_spoof, _ = ReviewerIdentityRegistry.authenticate_identity(
        "reviewer.dr-elena-vance.external-ai-audit", spoofed_key
    )
    assert auth_spoof is False, "Security failed: spoofed public key was authenticated"

    # Unregistered identity authentication
    auth_unknown, _ = ReviewerIdentityRegistry.authenticate_identity("reviewer.unknown-attacker", vance.public_key_hex)
    assert auth_unknown is False, "Security failed: unknown identity was authenticated"


def test_stage10_2_digital_attestations_cryptographic_verification():
    """Cryptographically verifies all 30 Ed25519 digital signatures and validates tamper detection."""
    manifest = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    cases, digest = load_factual_dataset(
        BENCHMARK_DATASET,
        expected_sha256=manifest["dataset_sha256"],
        require_independent_review=True,
    )

    verified_signatures = 0
    for case in cases:
        cid = case["case_id"]

        # 1. Case-level digital signatures
        for att in case.get("case_reviews", []):
            assert att["algorithm"] == "Ed25519"
            assert "signature_b64" in att
            assert "payload_sha256" in att
            ok, msg = DigitalAttestationEngine.verify_attestation(att, cid, claim_id=None)
            assert ok is True, f"Case {cid} attestation verification failed: {msg}"
            verified_signatures += 1

        # 2. Claim-level digital signatures
        for claim in case.get("claims", []):
            claim_id = claim["claim_id"]
            for att in claim.get("reviews", []):
                assert att["algorithm"] == "Ed25519"
                assert "signature_b64" in att
                assert "payload_sha256" in att
                ok, msg = DigitalAttestationEngine.verify_attestation(att, cid, claim_id=claim_id)
                assert ok is True, f"Claim {claim_id} attestation verification failed: {msg}"
                verified_signatures += 1

    assert verified_signatures == 30, f"Expected 30 verified signatures (16 case + 14 claim), found {verified_signatures}"

    # Tamper detection test: modifying review decision invalidates signature
    first_case = cases[0]
    tampered_att = dict(first_case["case_reviews"][0], decision="reject")
    tampered_ok, _ = DigitalAttestationEngine.verify_attestation(tampered_att, first_case["case_id"], claim_id=None)
    assert tampered_ok is False, "Security failed: tampered decision signature verified"

    # Tamper detection test: modifying timestamp invalidates signature
    tampered_time = dict(first_case["case_reviews"][0], reviewed_at="2026-11-10T12:00:01+00:00")
    time_ok, _ = DigitalAttestationEngine.verify_attestation(tampered_time, first_case["case_id"], claim_id=None)
    assert time_ok is False, "Security failed: tampered timestamp signature verified"


def test_stage10_2_fetch_and_compare_source_publications():
    """Verifies character-by-character concordance of claims against canonical publication texts."""
    cases, _ = load_factual_dataset(BENCHMARK_DATASET, require_independent_review=True)

    concordance_checks = 0
    for case in cases:
        for claim in case.get("claims", []):
            ok, info = SourcePublicationComparator.verify_concordance(claim)
            assert ok is True, f"Claim {claim['claim_id']} concordance failed: {info}"
            assert info["digest_match"] is True
            assert info["character_offset"] >= 0
            assert len(info["context_before"]) >= 0
            concordance_checks += 1

    assert concordance_checks == 7, f"Expected 7 concordance checks, found {concordance_checks}"

    # Discrepancy test: tampered quote fails concordance
    fake_claim = {
        "claim_id": "http-404-core",
        "source_quote": "This sentence definitely does not appear in RFC 9110 Section 15.5.5.",
        "source_excerpt_sha256": "1234",
    }
    fake_ok, fake_info = SourcePublicationComparator.verify_concordance(fake_claim)
    assert fake_ok is False
    assert "quote not found in canonical source text" in fake_info["error"]


def test_stage10_2_spdx_provenance_classifications_corrected():
    """Verifies all claims have authentic, non-generic SPDX open-source license identifiers."""
    cases, _ = load_factual_dataset(BENCHMARK_DATASET, require_independent_review=True)

    for case in cases:
        cid = case["case_id"]
        # Audit provenance
        audit_ok, msg = SPDXProvenanceClassifier.audit_provenance(case)
        assert audit_ok is True, f"SPDX audit failed for {cid}: {msg}"

        for claim in case.get("claims", []):
            spdx = claim.get("license_spdx", "")
            # Ensure NO generic NOASSERTION exists in production dataset
            assert spdx != "NOASSERTION", f"Claim {claim['claim_id']} still has unasserted NOASSERTION license"
            assert spdx in ("CC-BY-4.0", "PSF-2.0", "CC0-1.0", "Apache-2.0", "LicenseRef-IETF-Trust-Legal-Provisions", "LicenseRef-IETF-TLP-5", "BSD-3-Clause")
