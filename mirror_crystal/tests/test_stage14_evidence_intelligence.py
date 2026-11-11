"""Stage 14 Test Suite: Evidence-Based Intelligence Subsystem.

Tests:
1. Source provenance validation and cryptographic SHA-256 integrity.
2. Granular evidence passages and authoritativeness weighting.
3. Atomic claim extraction and explicit assumption tagging.
4. Multi-dimensional contradiction detection (negation, numerical disparity, antonyms).
5. Grounding verification against canonical source passages.
6. Calibrated epistemic confidence scoring and categorization.
7. Honest calibrated abstention contracts on contradictory or unsupported claims.
8. Evidence Provenance Graph DAG traversal, cryptographic audit trail, and integrity checking.
9. End-to-end synthesis segregating facts, assumptions, and inferences.
10. Backend HTTP server endpoints (/api/evidence/verify, /api/evidence/contradictions).
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Any
import pytest

from mirror_crystal.evidence_intelligence import (
    AtomicClaim,
    ClaimGroundingVerifier,
    ConfidenceCalibrator,
    ContradictionDetector,
    ContradictionType,
    EpistemicCategory,
    EvidenceBasedIntelligenceEngine,
    EvidenceError,
    EvidencePassage,
    EvidenceProvenanceGraph,
    SourceProvenance,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_source_provenance_validation_and_integrity():
    """Verify source provenance enforces valid URLs, SPDX licenses, and SHA-256 integrity."""
    valid_content = "The speed of light in vacuum is exactly 299,792,458 metres per second."
    valid_hash = _hash(valid_content)

    src = SourceProvenance(
        source_id="bipm_si_2019",
        title="SI Brochure: The International System of Units (9th edition)",
        url="https://www.bipm.org/en/publications/si-brochure",
        publisher="Bureau International des Poids et Mesures",
        license_spdx="CC-BY-3.0-IGO",
        retrieved_at="2026-01-15T12:00:00+00:00",
        sha256_hash=valid_hash,
        content=valid_content,
    )
    assert src.source_id == "bipm_si_2019"
    assert src.to_dict()["license_spdx"] == "CC-BY-3.0-IGO"

    # Mismatched SHA-256 content hash must raise EvidenceError
    with pytest.raises(EvidenceError, match="digest mismatch"):
        SourceProvenance(
            source_id="bad_src",
            title="Corrupted Document",
            url="https://example.org/doc",
            publisher="Example Org",
            license_spdx="MIT",
            retrieved_at="2026-01-15T12:00:00+00:00",
            sha256_hash=_hash("original"),
            content="tampered content",
        )

    # Invalid URL scheme must raise EvidenceError
    with pytest.raises(EvidenceError, match="valid HTTPS"):
        SourceProvenance(
            source_id="insecure_src",
            title="Insecure HTTP",
            url="http://insecure.example.org",
            publisher="Example Org",
            license_spdx="MIT",
            retrieved_at="2026-01-15T12:00:00+00:00",
            sha256_hash=_hash("sample"),
            content="sample",
        )


def test_evidence_passage_validation_and_authoritativeness():
    """Verify evidence passages enforce locator tags, weight bounds, and hash verification."""
    text = "The metre is defined by taking the fixed numerical value of the speed of light c."
    p_hash = _hash(text)

    passage = EvidencePassage(
        passage_id="bipm_p1",
        source_id="bipm_si_2019",
        locator="Chapter 2, Section 2.3.1, p. 131",
        text=text,
        sha256_hash=p_hash,
        authoritativeness_weight=0.98,
    )
    assert passage.passage_id == "bipm_p1"
    assert passage.authoritativeness_weight == 0.98

    # Corrupt hash must fail
    with pytest.raises(EvidenceError, match="SHA-256 digest mismatch"):
        EvidencePassage(
            passage_id="bad_p",
            source_id="bipm_si_2019",
            locator="Section 1",
            text=text,
            sha256_hash=_hash("wrong text"),
        )

    # Invalid weight must fail
    with pytest.raises(EvidenceError, match="authoritativeness_weight"):
        EvidencePassage(
            passage_id="p_weight",
            source_id="bipm_si_2019",
            locator="Section 1",
            text=text,
            sha256_hash=p_hash,
            authoritativeness_weight=1.5,
        )


def test_atomic_claim_decomposition_and_assumption_tagging():
    """Verify claim decomposition extracts sentences and distinguishes assumptions from asserted facts."""
    prose = (
        "The speed of light is 299792458 m/s. "
        "Assuming zero atmospheric resistance, the photon propagates continuously. "
        "Hypothetically, tachyons would violate causality."
    )
    claims = EvidenceBasedIntelligenceEngine.decompose_statements(prose)
    assert len(claims) == 3

    assert claims[0].statement.startswith("The speed of light")
    assert claims[0].is_assumption is False

    assert "zero atmospheric resistance" in claims[1].statement
    assert claims[1].is_assumption is True

    assert "tachyons would violate causality" in claims[2].statement
    assert claims[2].is_assumption is True


def test_contradiction_detection_direct_negation():
    """Verify detection of direct negation polarity conflict between statements."""
    c1 = AtomicClaim(
        claim_id="c1",
        statement="The database service is currently available and accepting queries.",
    )
    c2 = AtomicClaim(
        claim_id="c2",
        statement="The database service is not available and accepting queries.",
    )
    conflicts = ContradictionDetector.detect_conflicts([c1, c2])
    assert len(conflicts) >= 1
    direct = [c for c in conflicts if c.conflict_type == ContradictionType.DIRECT_NEGATION]
    assert len(direct) == 1
    assert direct[0].severity == 1.0
    assert "polarity conflict" in direct[0].explanation


def test_contradiction_detection_numerical_disparity():
    """Verify detection of conflicting numerical measurements with matching units."""
    c1 = AtomicClaim(
        claim_id="c1",
        statement="The orbital radius of the communications satellite is 42164 km.",
    )
    c2 = AtomicClaim(
        claim_id="c2",
        statement="The orbital radius of the communications satellite is 35786 km.",
    )
    conflicts = ContradictionDetector.detect_conflicts([c1, c2])
    assert len(conflicts) >= 1
    num_conflicts = [c for c in conflicts if c.conflict_type == ContradictionType.NUMERICAL_DISPARITY]
    assert len(num_conflicts) == 1
    assert "42164.0 km vs 35786.0 km" in num_conflicts[0].explanation


def test_contradiction_detection_antonym_polarity():
    """Verify detection of antonym polarity reversals regarding a shared topic."""
    c1 = AtomicClaim(
        claim_id="c1",
        statement="The quarterly operating revenue increased significantly in Europe.",
    )
    c2 = AtomicClaim(
        claim_id="c2",
        statement="The quarterly operating revenue decreased significantly in Europe.",
    )
    conflicts = ContradictionDetector.detect_conflicts([c1, c2])
    assert len(conflicts) >= 1
    ant_conflicts = [c for c in conflicts if c.conflict_type == ContradictionType.ANTONYM_POLARITY]
    assert len(ant_conflicts) == 1
    assert "increased" in ant_conflicts[0].explanation and "decreased" in ant_conflicts[0].explanation


def test_claim_grounding_and_provenance_tracking():
    """Verify claim grounding against canonical passages and full provenance traversal."""
    doc_text = "The French Republic is situated in Western Europe and Paris serves as its capital city."
    doc_hash = _hash(doc_text)

    source = SourceProvenance(
        source_id="geo_src_01",
        title="European Geographic Gazetteer",
        url="https://geography.example.org/france",
        publisher="Geographic Society",
        license_spdx="CC-BY-4.0",
        retrieved_at="2026-01-20T08:00:00+00:00",
        sha256_hash=doc_hash,
        content=doc_text,
    )
    passage = EvidencePassage(
        passage_id="p_geo_01",
        source_id="geo_src_01",
        locator="Section 1, p. 4",
        text=doc_text,
        sha256_hash=doc_hash,
        authoritativeness_weight=0.95,
    )

    claim = AtomicClaim(
        claim_id="claim_paris",
        statement="Paris is the capital city of France in Western Europe.",
    )

    score, pids, snippet = ClaimGroundingVerifier.verify_grounding(claim, [passage])
    assert score >= 0.80
    assert "p_geo_01" in pids
    assert snippet is not None

    # Track in Provenance Graph
    graph = EvidenceProvenanceGraph()
    graph.add_source(source)
    graph.add_passage(passage)
    claim.grounding_passage_ids = pids
    claim.calibrated_confidence = 0.95
    claim.epistemic_category = EpistemicCategory.VERIFIED_FACT
    graph.add_claim(claim)

    trail = graph.get_provenance_trail("claim_paris")
    assert trail["claim_id"] == "claim_paris"
    assert trail["epistemic_category"] == "verified_fact"
    assert len(trail["grounding_passages"]) == 1
    assert trail["grounding_passages"][0]["sha256_hash"] == doc_hash
    assert trail["source_provenance"][0]["publisher"] == "Geographic Society"


def test_calibrated_confidence_and_epistemic_categorization():
    """Verify confidence calibration distinguishes facts from assumptions and unverified claims."""
    # 1. Fact backed by evidence
    c_fact = AtomicClaim(claim_id="f1", statement="Water boils at 100 degrees Celsius at 1 atm.")
    conf1, cat1 = ConfidenceCalibrator.calibrate_claim(c_fact, grounding_score=0.92, has_contradiction=False, num_sources=2)
    assert cat1 == EpistemicCategory.VERIFIED_FACT
    assert conf1 >= 0.92

    # 2. Plausible inference
    c_inf = AtomicClaim(claim_id="inf1", statement="Atmospheric cooking times are longer at high altitude.")
    conf2, cat2 = ConfidenceCalibrator.calibrate_claim(c_inf, grounding_score=0.68, has_contradiction=False, num_sources=1)
    assert cat2 == EpistemicCategory.PLAUSIBLE_INFERENCE
    assert 0.50 <= conf2 <= 0.80

    # 3. Explicit assumption
    c_assump = AtomicClaim(claim_id="a1", statement="Assuming water behaves as an ideal incompressible liquid.", is_assumption=True)
    conf3, cat3 = ConfidenceCalibrator.calibrate_claim(c_assump, grounding_score=0.90, has_contradiction=False, num_sources=1)
    assert cat3 == EpistemicCategory.EXPLICIT_ASSUMPTION
    assert conf3 <= 0.40  # Assumptions must be capped

    # 4. Contradiction
    c_contra = AtomicClaim(claim_id="x1", statement="Water boils at 20 degrees Celsius at 1 atm.")
    conf4, cat4 = ConfidenceCalibrator.calibrate_claim(c_contra, grounding_score=0.90, has_contradiction=True, num_sources=1)
    assert cat4 == EpistemicCategory.CONTRADICTION
    assert conf4 == 0.0


def test_honest_abstention_on_unsupported_or_contradictory_queries():
    """Verify system honestly abstains when claims lack evidence or have severe contradictions."""
    # Zero grounding -> honest abstention
    c_unsupported = [
        AtomicClaim(claim_id="u1", statement="The ancient library contains three thousand secret manuscripts.", calibrated_confidence=0.15)
    ]
    should_abstain, reason = ConfidenceCalibrator.evaluate_abstention(
        claims=c_unsupported,
        conflicts=[],
        confidence_threshold=0.55,
    )
    assert should_abstain is True
    assert "Abstention required" in reason

    # Contradiction -> honest abstention
    conflicts = [
        ContradictionDetector._evaluate_pair(
            id_a="c1", text_a="Server is active", subj_a="", pred_a="",
            id_b="c2", text_b="Server is not active", subj_b="", pred_b="",
        )
    ]
    should_abstain2, reason2 = ConfidenceCalibrator.evaluate_abstention(
        claims=[AtomicClaim(claim_id="c1", statement="Server is active", calibrated_confidence=0.9)],
        conflicts=[conflicts[0]],
        confidence_threshold=0.55,
    )
    assert should_abstain2 is True
    assert "unresolved contradictions" in reason2


def test_evidence_provenance_graph_integrity_and_export():
    """Verify DAG structural validation catches dangling references and content tampering."""
    graph = EvidenceProvenanceGraph()

    src_content = "Canonical source text."
    src = SourceProvenance(
        source_id="src_1",
        title="Title",
        url="https://example.com/doc",
        publisher="Publisher",
        license_spdx="Apache-2.0",
        retrieved_at="2026-01-01T00:00:00+00:00",
        sha256_hash=_hash(src_content),
        content=src_content,
    )
    graph.add_source(src)

    p_content = "Passage excerpt."
    passage = EvidencePassage(
        passage_id="p_1",
        source_id="src_1",
        locator="Section 1",
        text=p_content,
        sha256_hash=_hash(p_content),
    )
    graph.add_passage(passage)

    claim = AtomicClaim(
        claim_id="c_1",
        statement="Passage excerpt is supported.",
        grounding_passage_ids=["p_1"],
    )
    graph.add_claim(claim)

    is_valid, issues = graph.verify_graph_integrity()
    assert is_valid is True
    assert len(issues) == 0

    # Corrupt claim passage reference
    claim.grounding_passage_ids.append("p_nonexistent")
    is_valid2, issues2 = graph.verify_graph_integrity()
    assert is_valid2 is False
    assert any("p_nonexistent" in err for err in issues2)


def test_end_to_end_synthesis_engine():
    """Verify full end-to-end synthesis segregating facts, assumptions, and inferences."""
    engine = EvidenceBasedIntelligenceEngine()

    content = "The James Webb Space Telescope (JWST) was launched on December 25 2021 from Kourou French Guiana."
    doc_hash = _hash(content)

    source = SourceProvenance(
        source_id="nasa_jwst_2021",
        title="NASA JWST Launch Factsheet",
        url="https://www.nasa.gov/mission_pages/webb/main/index.html",
        publisher="NASA",
        license_spdx="CC0-1.0",
        retrieved_at="2026-01-10T10:00:00+00:00",
        sha256_hash=doc_hash,
        content=content,
    )
    passage = EvidencePassage(
        passage_id="p_jwst_launch",
        source_id="nasa_jwst_2021",
        locator="Paragraph 1",
        text=content,
        sha256_hash=doc_hash,
    )

    claims = [
        AtomicClaim(
            claim_id="c1",
            statement="The James Webb Space Telescope was launched on December 25 2021.",
        ),
        AtomicClaim(
            claim_id="c2",
            statement="Assuming nominal orbital decay, the observatory operates for over ten years.",
            is_assumption=True,
        ),
    ]

    report = engine.process_and_synthesize(
        query="When was JWST launched and what is its expected mission duration?",
        claims=claims,
        sources=[source],
        passages=[passage],
    )

    assert report.abstained is False
    assert len(report.verified_facts) == 1
    assert "December 25 2021" in report.verified_facts[0]
    assert len(report.explicit_assumptions) == 1
    assert "nominal orbital decay" in report.explicit_assumptions[0]
    assert "### Verified Facts" in report.formatted_response
    assert "### Explicit Assumptions" in report.formatted_response
    assert report.overall_confidence >= 0.85
    assert len(report.provenance_summary) == 1


def test_stage14_http_server_endpoints():
    """Verify POST /api/evidence/verify and /api/evidence/contradictions on the HTTP server."""
    import http.server
    import threading
    import urllib.request
    from mirror7_backend.http_server import create_server
    from mirror7_backend.service import BackendService

    service = BackendService()
    server = create_server("127.0.0.1", 0, service)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{port}"

        # 1. Test /api/evidence/contradictions
        contra_payload = json.dumps({
            "claims": [
                "The database cluster is active and healthy.",
                "The database cluster is not active and healthy.",
            ]
        }).encode("utf-8")

        req1 = urllib.request.Request(
            f"{base_url}/api/evidence/contradictions",
            data=contra_payload,
            headers={"Content-Type": "application/json", "X-Request-ID": "contra_001"},
            method="POST",
        )
        with urllib.request.urlopen(req1, timeout=5) as resp1:
            assert resp1.status == 200
            assert resp1.headers.get("X-Request-ID") == "contra_001"
            data1 = json.loads(resp1.read().decode("utf-8"))
            assert data1["ok"] is True
            assert data1["total_conflicts"] >= 1
            assert data1["conflicts"][0]["conflict_type"] == "direct_negation"

        # 2. Test /api/evidence/verify
        doc_content = "Python 3.12 introduces improved error messages and isolated subinterpreters."
        doc_hash = _hash(doc_content)
        verify_payload = json.dumps({
            "query": "What are key features of Python 3.12?",
            "claims": [
                "Python 3.12 introduces improved error messages.",
                "Assuming steady adoption, runtime performance improves.",
            ],
            "sources": [{
                "source_id": "py312_notes",
                "title": "Python 3.12 Release Notes",
                "url": "https://docs.python.org/3/whatsnew/3.12.html",
                "publisher": "Python Software Foundation",
                "license_spdx": "PSF-2.0",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "sha256_hash": doc_hash,
                "content": doc_content,
            }],
            "passages": [{
                "passage_id": "py312_p1",
                "source_id": "py312_notes",
                "locator": "Summary",
                "text": doc_content,
                "sha256_hash": doc_hash,
                "authoritativeness_weight": 1.0,
            }],
        }).encode("utf-8")

        req2 = urllib.request.Request(
            f"{base_url}/api/evidence/verify",
            data=verify_payload,
            headers={"Content-Type": "application/json", "X-Request-ID": "verify_002"},
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=5) as resp2:
            assert resp2.status == 200
            assert resp2.headers.get("X-Request-ID") == "verify_002"
            data2 = json.loads(resp2.read().decode("utf-8"))
            assert data2["ok"] is True
            rep = data2["report"]
            assert rep["abstained"] is False
            assert len(rep["verified_facts"]) >= 1
            assert len(rep["explicit_assumptions"]) >= 1
            assert rep["overall_confidence"] >= 0.80

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_stage14_provenance_rejects_file_uri_and_unknown_spdx():
    base = dict(
        source_id="strict_src",
        title="Verified source",
        publisher="Institution",
        retrieved_at="2026-11-11T10:00:00Z",
        sha256_hash=_hash("source body"),
        content="source body",
    )
    with pytest.raises(EvidenceError, match="HTTPS or URN"):
        SourceProvenance(url="file:///etc/passwd", license_spdx="MIT", **base)
    with pytest.raises(EvidenceError, match="SPDX"):
        SourceProvenance(url="https://example.org/source", license_spdx="NOT-A-REAL-LICENSE", **base)


def test_stage14_provenance_requires_timezone_aware_iso_timestamp():
    base = dict(
        source_id="time_src",
        title="Timestamp source",
        url="https://example.org/source",
        publisher="Institution",
        license_spdx="MIT",
        sha256_hash=_hash("source body"),
        content="source body",
    )
    for bad_timestamp in ("2026-11-11", "not-a-date", "2026-11-11T10:00:00"):
        with pytest.raises(EvidenceError, match="ISO-8601"):
            SourceProvenance(retrieved_at=bad_timestamp, **base)


def test_stage14_source_digest_requires_the_actual_source_content():
    with pytest.raises(EvidenceError, match="source content is required"):
        SourceProvenance(
            source_id="digest_without_content",
            title="Digest only",
            url="urn:example:source:1",
            publisher="Institution",
            license_spdx="Apache-2.0",
            retrieved_at="2026-11-11T10:00:00+00:00",
            sha256_hash=_hash("not supplied"),
            content="",
        )


def test_stage14_provenance_graph_rejects_passage_not_in_source_and_duplicate_ids():
    source_text = "The original verified source contains this passage."
    source = SourceProvenance(
        source_id="containment_source",
        title="Source",
        url="https://example.org/verified",
        publisher="Institution",
        license_spdx="MIT",
        retrieved_at="2026-11-11T10:00:00+00:00",
        sha256_hash=_hash(source_text),
        content=source_text,
    )
    graph = EvidenceProvenanceGraph()
    graph.add_source(source)
    with pytest.raises(EvidenceError, match="not contained"):
        graph.add_passage(EvidencePassage(
            passage_id="foreign_passage",
            source_id=source.source_id,
            locator="Section 1",
            text="This text is not in the source.",
            sha256_hash=_hash("This text is not in the source."),
        ))
    with pytest.raises(EvidenceError, match="Duplicate source_id"):
        graph.add_source(source)

    valid_passage_text = "The original verified source contains this passage."
    passage = EvidencePassage(
        passage_id="valid_passage",
        source_id=source.source_id,
        locator="Section 1",
        text=valid_passage_text,
        sha256_hash=_hash(valid_passage_text),
    )
    graph.add_passage(passage)
    with pytest.raises(EvidenceError, match="Duplicate passage_id"):
        graph.add_passage(passage)


def test_stage14_source_provenance_accepts_valid_urn_and_z_timestamp():
    source_text = "Verified content from an archived authority."
    source = SourceProvenance(
        source_id="urn_source",
        title="URN source",
        url="urn:example:authority:document-7",
        publisher="Example Authority",
        license_spdx="CC-BY-4.0",
        retrieved_at="2026-11-11T10:00:00Z",
        sha256_hash=_hash(source_text),
        content=source_text,
    )
    assert source.sha256_hash == _hash(source_text)
