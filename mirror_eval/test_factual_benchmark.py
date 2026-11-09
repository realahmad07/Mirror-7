from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from mirror_eval.factual_benchmark import (
    BenchmarkValidationError, load_factual_dataset, run_factual_benchmark,
)


def _reviewer(ident):
    return {
        "reviewer_id": ident,
        "independent": True,
        "method": "independent_source_comparison",
        "decision": "approve",
        "reviewed_at": "2026-11-09T12:00:00+00:00",
    }


def _case(case_id, answerable, answer):
    quote = "The fixture source says the value is forty two."
    claim = {
        "claim_id": "fixture-claim",
        "claim_text": "The value is forty two.",
        "required_fragments": ["value is forty two"],
        "source_publisher": "Fixture Authority",
        "source_url": "https://example.org/frozen-fixture",
        "source_locator": "Section 1",
        "license_spdx": "NOASSERTION",
        "source_quote": quote,
        "source_excerpt_sha256": sha256(quote.encode()).hexdigest(),
        "source_retrieved_at": "2026-11-09T12:00:00+00:00",
        "reviews": [_reviewer("reviewer-a"), _reviewer("reviewer-b")],
    }
    return {
        "case_id": case_id,
        "split": "test",
        "prompt": "What is the fixture value?",
        "answerable": answerable,
        "expected_answer": answer,
        "claims": [claim] if answerable else [],
        "case_reviews": [_reviewer("case-reviewer-a"), _reviewer("case-reviewer-b")],
    }


class FakeAgent:
    def __init__(self, responses):
        self.responses = iter(responses)

    def respond(self, prompt):
        return next(self.responses)


def test_source_backed_dataset_requires_two_distinct_independent_reviewers(tmp_path):
    item = _case("review-gate", True, "The value is forty two.")
    item["claims"][0]["reviews"] = [_reviewer("same-reviewer")]
    path = tmp_path / "pending.jsonl"
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkValidationError, match="two distinct independent approvals"):
        load_factual_dataset(path)


def test_dataset_digest_is_pinned_and_checked(tmp_path):
    item = _case("digest", True, "The value is forty two.")
    path = tmp_path / "frozen.jsonl"
    raw = json.dumps(item) + "\n"
    path.write_text(raw, encoding="utf-8")
    digest = sha256(raw.encode()).hexdigest()
    loaded, actual = load_factual_dataset(path, expected_sha256=digest)
    assert actual == digest and len(loaded) == 1
    with pytest.raises(BenchmarkValidationError, match="digest mismatch"):
        load_factual_dataset(path, expected_sha256="0"*64)


def test_metrics_cover_accuracy_unsupported_answers_abstention_and_calibration():
    correct = _case("correct", True, "The value is forty two.")
    wrong = _case("wrong", True, "The value is forty two.")
    impossible = _case("unanswerable", False, "")
    cases = [correct, wrong, impossible]
    agent = FakeAgent([
        SimpleNamespace(answer="The value is forty two.", mode="direct_crystal_lock", confidence=0.9),
        SimpleNamespace(answer="The value is forty three.", mode="direct_crystal_lock", confidence=0.8),
        SimpleNamespace(answer="I do not know.", mode="unmatched_honest_abstain", confidence=0.0),
    ])
    report = run_factual_benchmark(agent, cases)
    metrics = report["metrics"]
    assert metrics["factual_accuracy"] == pytest.approx(0.5)
    assert metrics["unsupported_answer_rate"] == pytest.approx(0.5)
    assert metrics["abstention_precision"] == pytest.approx(1.0)
    assert metrics["abstention_recall"] == pytest.approx(1.0)
    assert metrics["coverage"] == pytest.approx(2/3)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
    assert metrics["calibration_sample_count"] == 2
    assert report["cases"][0]["answer_text"] == "The value is forty two."
    assert len(report["cases"][0]["response_sha256"]) == 64


def test_unanswerable_case_answer_is_counted_unsupported():
    impossible = _case("unanswerable", False, "")
    agent = FakeAgent([SimpleNamespace(answer="secret=abc", mode="direct_crystal_lock", confidence=0.7)])
    metrics = run_factual_benchmark(agent, [impossible])["metrics"]
    assert metrics["unsupported_answer_rate"] == 1.0
    assert metrics["abstention_recall"] == 0.0
