import hashlib
import json

from mirror_eval.review_packet import build_review_packet


def test_review_packet_never_fabricates_independent_approvals(tmp_path):
    quote = "The primary source supports this exact statement."
    case = {
        "case_id": "case-1",
        "split": "test",
        "prompt": "What does the source state?",
        "answerable": True,
        "claims": [{
            "claim_id": "claim-1",
            "claim_text": "The primary source supports this exact statement.",
            "source_publisher": "Example Documentation",
            "source_url": "https://example.org/docs",
            "source_locator": "section 1",
            "source_quote": quote,
            "source_excerpt_sha256": hashlib.sha256(quote.encode()).hexdigest(),
            "source_retrieved_at": "2026-11-09T12:00:00+00:00",
            "license_spdx": "CC-BY-4.0",
            "required_fragments": ["primary source"],
            "reviews": [],
        }],
        "case_reviews": [],
    }
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(case) + "\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "dataset_sha256": digest,
        "review_status": "pending_independent_review",
    }), encoding="utf-8")

    packet = build_review_packet(dataset, manifest)
    assert packet["case_count"] == 1
    assert packet["dataset_sha256"] == digest
    assert packet["cases"][0]["claims"][0]["reviews"] == []
    assert packet["cases"][0]["case_reviews"] == []
    assert packet["cases"][0]["required_independent_reviewers"] == 2
    assert "No reviews have been created" in packet["notice"]
