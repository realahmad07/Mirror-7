from __future__ import annotations

from hashlib import sha256
import pytest

from mirror_crystal import (
    Blueprint, BlueprintKind, BlueprintProvenance, ClaimEvidence, ClaimEvidenceError,
    DiskCrystalVault, EvidenceLevel,
)
from mirror_crystal.blueprint_contracts import VerificationEvidence


def _claim_evidence(blueprint_id: str, reviewer_id: str) -> ClaimEvidence:
    quote = "The 404 (Not Found) status code indicates that the origin server did not find a current representation for the target resource"
    return ClaimEvidence(
        blueprint_id=blueprint_id,
        claim_id="http-404-core",
        claim_text="HTTP 404 means the origin server did not find a current representation for the target resource.",
        source_publisher="Internet Engineering Task Force",
        source_url="https://datatracker.ietf.org/doc/html/rfc9110#section-15.5.5",
        source_locator="RFC 9110, Section 15.5.5",
        source_quote=quote,
        source_excerpt_sha256="94d738e5a1212d798bf3fc3161c95f26f996997e6d993d9037c6c347f68c24e9",
        source_retrieved_at="2026-11-09T12:00:00+05:00",
        license_spdx="NOASSERTION",
        reviewer_id=reviewer_id,
        reviewer_independent=True,
        review_decision="approve",
        review_method="independent_source_comparison",
        reviewed_at="2026-11-09T13:00:00+05:00",
        review_notes="Compared the atomic claim with the cited RFC section.",
    )


def _blueprint() -> Blueprint:
    prompt = "What does HTTP 404 mean?"
    response = "HTTP 404 means the origin server did not find a current representation for the target resource."
    return Blueprint(
        blueprint_id="http-404-claim",
        kind=BlueprintKind.FACT,
        domain="web-protocols",
        intent="http_status",
        prompt_pattern=prompt,
        response_template=response,
        provenance=BlueprintProvenance(
            source="legacy_import",
            source_ref="legacy://rule/17",
            content_sha256=Blueprint._digest(prompt, response),
            captured_at="2026-11-09T12:00:00+05:00",
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )


def test_blueprint_requires_two_distinct_reviewers_per_claim(tmp_path):
    vault = DiskCrystalVault(tmp_path / "vault.dcrystal")
    try:
        bp = _blueprint()
        vault.store_blueprint(bp)
        vault.register_blueprint_claim_scope(bp.blueprint_id, ["http-404-core"])
        vault.store_claim_evidence(_claim_evidence(bp.blueprint_id, "reviewer-a"))
        with pytest.raises(ClaimEvidenceError, match="requires 2 distinct independent reviewers"):
            vault.curate_blueprint_from_evidence(bp.blueprint_id)
    finally:
        vault.close()


def test_independent_claim_review_promotes_and_verifies_fact_blueprint(tmp_path):
    vault = DiskCrystalVault(tmp_path / "vault.dcrystal")
    try:
        bp = _blueprint()
        vault.store_blueprint(bp)
        vault.register_blueprint_claim_scope(bp.blueprint_id, ["http-404-core"])
        vault.store_claim_evidence(_claim_evidence(bp.blueprint_id, "reviewer-a"))
        vault.store_claim_evidence(_claim_evidence(bp.blueprint_id, "reviewer-b"))

        promoted = vault.curate_blueprint_from_evidence(bp.blueprint_id)
        assert promoted.verification.level is EvidenceLevel.INDEPENDENTLY_REPRODUCED
        assert len(vault.list_claim_evidence(bp.blueprint_id)) == 2

        response = vault.respond_blueprint(bp.blueprint_id)
        assert response.mode == "blueprint_composition_reliable"
        assert response.epistemic_tier == "high_confidence"
        assert any("verified=http-404-core" in row for row in response.reasoning_trace)
    finally:
        vault.close()


def test_claim_evidence_rejects_bad_excerpt_digest():
    quote = "Source excerpt."
    with pytest.raises(ClaimEvidenceError, match="digest mismatch"):
        ClaimEvidence(
            blueprint_id="bp", claim_id="c1", claim_text="Claim",
            source_publisher="Publisher", source_url="https://example.org/source",
            source_locator="section 1", source_quote=quote, source_excerpt_sha256="0"*64,
            source_retrieved_at="2026-11-09T12:00:00+00:00", license_spdx="NOASSERTION",
            reviewer_id="reviewer-a", reviewer_independent=True, review_decision="approve",
            review_method="independent_source_comparison", reviewed_at="2026-11-09T12:00:00+00:00",
        )


def test_frozen_claim_scope_cannot_be_silently_changed(tmp_path):
    vault = DiskCrystalVault(tmp_path / "vault.dcrystal")
    try:
        bp = _blueprint()
        vault.store_blueprint(bp)
        vault.register_blueprint_claim_scope(bp.blueprint_id, ["http-404-core"])
        with pytest.raises(ClaimEvidenceError, match="frozen"):
            vault.register_blueprint_claim_scope(bp.blueprint_id, ["http-404-core", "extra"])
    finally:
        vault.close()
