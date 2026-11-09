from mirror7_backend.evidence_synthesis import EvidenceSynthesisEngine


def test_synthesizes_unique_claims_and_tracks_source_support():
    result = EvidenceSynthesisEngine().synthesize(
        "Explain the system",
        [
            {
                "source": "source_a",
                "answer": "Python uses indentation. Python supports functions.",
                "verified": True,
            },
            {
                "source": "source_b",
                "answer": "Python uses indentation.",
                "verified": True,
            },
        ],
    )
    assert result["verified"] is True
    assert result["conflict"] is False
    assert len(result["claims"]) == 2
    first = next(item for item in result["claims"] if item["text"] == "Python uses indentation.")
    assert set(first["sources"]) == {"source_a", "source_b"}


def test_detects_direct_negation_conflict():
    result = EvidenceSynthesisEngine().synthesize(
        "Is the statement true?",
        [
            {"source": "source_a", "answer": "The service is available.", "verified": True},
            {"source": "source_b", "answer": "The service is not available.", "verified": True},
        ],
    )
    assert result["conflict"] is True
    assert result["verified"] is False
    assert result["answer"] == ""


def test_unverified_knowledge_and_memory_cannot_create_answer():
    result = EvidenceSynthesisEngine().synthesize(
        "What is the answer?",
        [{"source": "memory", "answer": "The answer is 42.", "verified": False}],
        memory_context="The answer is 42.",
    )
    assert result["verified"] is False
    assert result["answer"] == ""
    assert result["memory_used_as_evidence"] is False


def test_generator_must_preserve_exact_claim_coverage():
    engine = EvidenceSynthesisEngine()
    evidence = [{
        "source": "reviewed_blueprint",
        "answer": "A is supported. B is supported.",
        "verified": True,
    }]
    good = engine.synthesize(
        "Combine",
        evidence,
        generator=lambda payload: "B is supported. A is supported.",
    )
    assert good["generator_used"] is True

    bad = engine.synthesize(
        "Combine",
        evidence,
        generator=lambda payload: "A is supported. C is invented.",
    )
    assert bad["generator_rejected"] is True
    assert bad["answer"] == "A is supported. B is supported."
