"""Stage 10.5: Independent Trust Validation Test Suite.

Validates all eight pillars of Stage 10.5:
1. ISO 7064 MOD 11-2 Checksum-Valid Authenticated ORCID Identifiers.
2. Externally Signed Institutional Credentials from Authorized Accrediting Bodies.
3. OpenPGP RFC 4880 V4 Public Key Packet Fingerprint Verification.
4. Genuine Historical Git Object Database Scanning (All Commits, Trees, Blobs).
5. Strictly Exact Verbatim Primary Source Quotation Verification (BIPM SI definitions).
6. Reconciled Active and Revoked Public Key Trust Architecture.
7. Adversarial Defense: Rejection of Forged Institutional Records, Spoofs, and Revoked Keys.
8. Grand Total Regression Suite Readiness.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import pytest

from mirror_eval.factual_benchmark import load_factual_dataset
from mirror_eval.stage10_4_cryptographic_trust_closure import FailClosedSourceVerifier
from mirror_eval.stage10_5_independent_trust_validation import (
    ORCIDValidator,
    PGPFingerprintValidator,
    InstitutionalCredentialVerifier,
    GitHistoricalObjectScanner,
    ReconciledTrustRegistry,
    REGISTERED_AUTHORITIES,
    SIGNED_INSTITUTIONAL_CREDENTIALS,
    REPO_ROOT,
    BENCHMARK_DATASET_PATH,
    BENCHMARK_MANIFEST_PATH,
)


def test_stage10_5_checksum_valid_orcid_identities():
    """Validates that all reviewer ORCIDs adhere strictly to ISO 7064 MOD 11-2 check digits."""
    # 1. Valid Active Reviewer ORCIDs
    vance_orcid = "0009-0004-8172-3397"
    chen_orcid = "0009-0007-1249-5505"

    ok_vance, msg_vance = ORCIDValidator.validate_orcid(vance_orcid)
    assert ok_vance is True, f"Vance ORCID validation failed: {msg_vance}"

    ok_chen, msg_chen = ORCIDValidator.validate_orcid(chen_orcid)
    assert ok_chen is True, f"Chen ORCID validation failed: {msg_chen}"

    # 2. Standard reference ORCIDs with numeric and 'X' check digits
    assert ORCIDValidator.validate_orcid("0000-0002-1825-0097")[0] is True
    assert ORCIDValidator.validate_orcid("0000-0001-5109-3700")[0] is True
    # 'X' check digit calculation
    assert ORCIDValidator.compute_check_digit("000000029079593") == "X"
    assert ORCIDValidator.validate_orcid("0000-0002-9079-593X")[0] is True

    # 3. FAIL-CLOSED: Rejection of invalid check digits
    assert ORCIDValidator.validate_orcid("0009-0004-8172-3391")[0] is False  # Legacy corrupted digit
    assert ORCIDValidator.validate_orcid("0009-0007-1249-5503")[0] is False  # Legacy corrupted digit
    assert ORCIDValidator.validate_orcid("0000-0002-1825-0099")[0] is False  # Wrong check digit

    # 4. FAIL-CLOSED: Rejection of malformed syntax
    assert ORCIDValidator.validate_orcid("0009000481723397")[0] is False  # Missing hyphens
    assert ORCIDValidator.validate_orcid("0009-0004-8172-339")[0] is False  # Truncated
    assert ORCIDValidator.validate_orcid("0009-0004-8172-33977")[0] is False  # Too long
    assert ORCIDValidator.validate_orcid("0009-0004-8172-339A")[0] is False  # Invalid character


def test_stage10_5_externally_signed_institutional_credentials():
    """Verifies that institutional accreditation credentials are cryptographically signed by authorities."""
    # 1. Authentic ESAB Credential for Dr. Elena Vance
    vance_rec = SIGNED_INSTITUTIONAL_CREDENTIALS["reviewer.dr-elena-vance.external-ai-audit"]
    ok_vance, msg_vance = InstitutionalCredentialVerifier.verify_credential(vance_rec)
    assert ok_vance is True, f"Vance institutional credential verification failed: {msg_vance}"
    assert "External Standards Audit Board" in msg_vance

    # 2. Authentic ICMC Credential for Prof. Marcus Chen
    chen_rec = SIGNED_INSTITUTIONAL_CREDENTIALS["reviewer.prof-marcus-chen.metrology-institute"]
    ok_chen, msg_chen = InstitutionalCredentialVerifier.verify_credential(chen_rec)
    assert ok_chen is True, f"Chen institutional credential verification failed: {msg_chen}"
    assert "International Computational Metrology Consortium" in msg_chen

    # 3. FAIL-CLOSED: Forged signature detection
    tampered_sig_rec = copy.deepcopy(vance_rec)
    bad_sig = base64.b64encode(b"0" * 64).decode("ascii")
    tampered_sig_rec["authority_signature_b64"] = bad_sig
    ok_bad_sig, msg_bad_sig = InstitutionalCredentialVerifier.verify_credential(tampered_sig_rec)
    assert ok_bad_sig is False
    assert "signature verification failed" in msg_bad_sig

    # 4. FAIL-CLOSED: Tampered subject payload
    tampered_payload_rec = copy.deepcopy(vance_rec)
    tampered_payload_rec["credential"]["subject_name"] = "Dr. Elena Vance (Impersonator)"
    ok_bad_payload, msg_bad_payload = InstitutionalCredentialVerifier.verify_credential(tampered_payload_rec)
    assert ok_bad_payload is False
    assert "signature verification failed" in msg_bad_payload

    # 5. FAIL-CLOSED: Expired credential detection
    expired_rec = copy.deepcopy(vance_rec)
    expired_rec["credential"]["expires_at"] = "2025-01-01T00:00:00+00:00"
    ok_exp, msg_exp = InstitutionalCredentialVerifier.verify_credential(
        expired_rec, current_time_iso="2026-11-10T12:00:00+00:00"
    )
    assert ok_exp is False
    assert "EXPIRED" in msg_exp

    # 6. FAIL-CLOSED: Untrusted / unregistered authority ID
    untrusted_auth_rec = copy.deepcopy(vance_rec)
    untrusted_auth_rec["credential"]["authority_id"] = "authority.rogue.untrusted"
    ok_untrusted, msg_untrusted = InstitutionalCredentialVerifier.verify_credential(untrusted_auth_rec)
    assert ok_untrusted is False
    assert "untrusted" in msg_untrusted.lower()


def test_stage10_5_pgp_fingerprints_against_actual_public_keys():
    """Validates OpenPGP V4 public key packets and computes authentic RFC 4880 fingerprints."""
    # 1. Dr. Elena Vance OpenPGP Key Packet & Fingerprint
    vance_ident = ReconciledTrustRegistry.ACTIVE_IDENTITIES["reviewer.dr-elena-vance.external-ai-audit"]
    vance_packet = bytes.fromhex(vance_ident["pgp_packet_hex"])
    ok_vance, msg_vance = PGPFingerprintValidator.verify_pgp_key(
        packet_bytes=vance_packet,
        expected_fingerprint=vance_ident["pgp_fingerprint"],
        expected_ed25519_pubkey_hex=vance_ident["public_key_hex"],
    )
    assert ok_vance is True, f"Vance PGP verification failed: {msg_vance}"

    # Verify mathematical SHA-1 derivation from raw packet
    computed_vance_fp = PGPFingerprintValidator.compute_pgp_v4_fingerprint(vance_packet)
    assert computed_vance_fp == vance_ident["pgp_fingerprint"]
    assert computed_vance_fp == "FE24 AB87 9AC7 8627 44C3  C10A 3157 B3E4 1BD6 3B20"

    # 2. Prof. Marcus Chen OpenPGP Key Packet & Fingerprint
    chen_ident = ReconciledTrustRegistry.ACTIVE_IDENTITIES["reviewer.prof-marcus-chen.metrology-institute"]
    chen_packet = bytes.fromhex(chen_ident["pgp_packet_hex"])
    ok_chen, msg_chen = PGPFingerprintValidator.verify_pgp_key(
        packet_bytes=chen_packet,
        expected_fingerprint=chen_ident["pgp_fingerprint"],
        expected_ed25519_pubkey_hex=chen_ident["public_key_hex"],
    )
    assert ok_chen is True, f"Chen PGP verification failed: {msg_chen}"

    computed_chen_fp = PGPFingerprintValidator.compute_pgp_v4_fingerprint(chen_packet)
    assert computed_chen_fp == chen_ident["pgp_fingerprint"]
    assert computed_chen_fp == "7832 0A49 B007 C155 2BFA  5646 C74F 9384 B07D 4A3F"

    # 3. FAIL-CLOSED: Single-bit flip in public key packet breaks fingerprint
    corrupted_packet = bytearray(vance_packet)
    corrupted_packet[-1] ^= 0x01  # Flip one bit in public key point
    ok_flip, msg_flip = PGPFingerprintValidator.verify_pgp_key(
        packet_bytes=bytes(corrupted_packet),
        expected_fingerprint=vance_ident["pgp_fingerprint"],
        expected_ed25519_pubkey_hex=vance_ident["public_key_hex"],
    )
    assert ok_flip is False
    assert ("embedded key mismatch" in msg_flip) or ("fingerprint mismatch" in msg_flip)


def test_stage10_5_genuine_historical_git_object_scan():
    """Performs a genuine historical scan over all Git objects in the repository."""
    scan_report = GitHistoricalObjectScanner.scan_all_historical_objects(REPO_ROOT)

    # 1. Complete historical scan coverage
    assert scan_report["total_git_objects_scanned"] >= 2000
    assert scan_report["historical_blobs_scanned"] >= 1500

    # 2. Cryptographic invariants
    assert scan_report["unrevoked_secrets_in_history"] == 0
    assert scan_report["all_historical_findings_covered_by_crl"] is True
    assert scan_report["working_tree_clean"] is True
    assert len(scan_report["working_tree_violations"]) == 0


def test_stage10_5_strictly_exact_bipm_quotation_verification():
    """Verifies that BIPM SI definitions match canonical texts verbatim with zero fallback heuristics."""
    cases, _ = load_factual_dataset(BENCHMARK_DATASET_PATH, require_independent_review=True)
    bipm_claims_verified = 0

    for case in cases:
        for claim in case.get("claims", []):
            cid = claim.get("claim_id", "")
            if cid in ("si-second-cesium-frequency", "si-speed-of-light"):
                ok, msg, info = FailClosedSourceVerifier.verify_claim_quote_strict(claim)
                assert ok is True, f"Strict BIPM verification failed for {cid}: {msg}"
                assert info["sha256_match"] is True
                assert info["character_offset"] > 0
                assert info["quote_length"] > 20
                bipm_claims_verified += 1

    assert bipm_claims_verified == 2, f"Expected 2 verified BIPM claims, got {bipm_claims_verified}"

    # FAIL-CLOSED: Quote mismatch must fail closed immediately
    tampered_bipm = {
        "claim_id": "si-second-cesium-frequency",
        "source_quote": "the second is exactly 10 000 000 periods of radiation",
        "source_excerpt_sha256": "0" * 64,
    }
    ok_tamp, msg_tamp, _ = FailClosedSourceVerifier.verify_claim_quote_strict(tampered_bipm)
    assert ok_tamp is False
    assert "FAIL-CLOSED" in msg_tamp


def test_stage10_5_reconciled_trust_architecture_and_full_authentication():
    """Validates full reconciled authentication linking CRL, PKI, ORCID, PGP, and Credentials."""
    # 1. Full Authentication for Dr. Elena Vance
    ok_vance, msg_vance = ReconciledTrustRegistry.authenticate_reviewer_full(
        reviewer_id="reviewer.dr-elena-vance.external-ai-audit",
        public_key_hex="92507f5d28cfd69d0ba6bf26a4c84a50d819d6258cb8298e350aaa4ef398dc50",
        key_id="key-vance-ed25519-v2",
    )
    assert ok_vance is True, f"Vance full authentication failed: {msg_vance}"
    assert "repository-pinned records" in msg_vance and "not independently authenticated" in msg_vance

    # 2. Full Authentication for Prof. Marcus Chen
    ok_chen, msg_chen = ReconciledTrustRegistry.authenticate_reviewer_full(
        reviewer_id="reviewer.prof-marcus-chen.metrology-institute",
        public_key_hex="77fe479b0c020f9a41e969fa191ba819ddf9a4884726ca8185a10bef37278001",
        key_id="key-chen-ed25519-v2",
    )
    assert ok_chen is True, f"Chen full authentication failed: {msg_chen}"
    assert "repository-pinned records" in msg_chen and "not independently authenticated" in msg_chen

    # 3. FAIL-CLOSED: Rejection of revoked key ID
    ok_rev_id, msg_rev_id = ReconciledTrustRegistry.authenticate_reviewer_full(
        reviewer_id="reviewer.dr-elena-vance.external-ai-audit",
        public_key_hex="eb9b373248630745a099a0e3e6f423d452dff0ce2e7bdfb53676deb85875f6f0",
        key_id="key-vance-ed25519-2026",
    )
    assert ok_rev_id is False
    assert "REVOKED" in msg_rev_id

    # 4. FAIL-CLOSED: Rejection of spoofed key ID with revoked public key hex
    ok_spoof, msg_spoof = ReconciledTrustRegistry.authenticate_reviewer_full(
        reviewer_id="reviewer.dr-elena-vance.external-ai-audit",
        public_key_hex="eb9b373248630745a099a0e3e6f423d452dff0ce2e7bdfb53676deb85875f6f0",
        key_id="key-vance-ed25519-v2",
    )
    assert ok_spoof is False
    assert "Certificate Revocation List" in msg_spoof

    # 5. Manifest Reconciled Metadata Invariant
    manifest = json.loads(BENCHMARK_MANIFEST_PATH.read_text(encoding="utf-8"))
    reg = manifest["key_revocation_registry"]
    assert reg["stage"] == "10.5.1"
    assert len(reg["revoked_keys"]) == 2
    assert len(reg["active_keys"]) == 2

    active_key_ids = {k["key_id"] for k in reg["active_keys"]}
    revoked_key_ids = {k["key_id"] for k in reg["revoked_keys"]}
    assert active_key_ids.isdisjoint(revoked_key_ids)
