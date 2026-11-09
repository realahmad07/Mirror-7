"""Frozen, source-backed factual benchmark and claim/abstention calibration metrics.

The default lexical judge is deliberately conservative. Use an independent semantic
judge adapter for published scores; do not treat string matching as entailment.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable
from urllib.parse import urlparse


class BenchmarkValidationError(ValueError):
    """Raised when a frozen benchmark record fails provenance/review policy."""


ABSTENTION_MODES = {
    "unmatched_honest_abstain",
    "honest_abstain",
    "blueprint_not_found",
    "blueprint_composition_blocked",
    "blueprint_contradiction_abstain",
    "blueprint_prompt_ambiguous",
    "clarification_needed",
}


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).casefold())).strip()


def _timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.tzinfo is not None
    except (TypeError, ValueError):
        return False


def _reviewers_are_independent(rows: Iterable[dict[str, Any]], *, min_reviewers: int = 2) -> bool:
    records = list(rows)
    if any(row.get("decision") == "reject" for row in records):
        return False
    approved = {
        str(row.get("reviewer_id", "")).strip()
        for row in records
        if row.get("decision") == "approve"
        and row.get("independent") is True
        and row.get("method") in {"independent_source_comparison", "independent_replication"}
        and str(row.get("reviewer_id", "")).strip()
        and _timestamp(str(row.get("reviewed_at", "")))
    }
    return len(approved) >= min_reviewers


def validate_case(case: dict[str, Any], *, require_independent_review: bool = True) -> None:
    for name in ("case_id", "split", "prompt"):
        if not str(case.get(name, "")).strip():
            raise BenchmarkValidationError(f"{name} must not be empty")
    if case["split"] != "test":
        raise BenchmarkValidationError(f"{case['case_id']}: factual benchmark must use frozen test split only")
    answerable = bool(case.get("answerable"))
    claims = case.get("claims") or []
    if answerable and not claims:
        raise BenchmarkValidationError(f"{case['case_id']}: answerable case needs at least one atomic claim")
    if not answerable and claims:
        raise BenchmarkValidationError(f"{case['case_id']}: unanswerable case must not contain gold factual claims")

    for claim in claims:
        for name in ("claim_id", "claim_text", "source_publisher", "source_url",
                     "source_locator", "source_quote", "source_excerpt_sha256", "license_spdx"):
            if not str(claim.get(name, "")).strip():
                raise BenchmarkValidationError(f"{case['case_id']}: claim field {name} is missing")
        parsed = urlparse(str(claim["source_url"]))
        if parsed.scheme != "https" or not parsed.netloc:
            raise BenchmarkValidationError(f"{case['case_id']}: source URL must use HTTPS")
        digest = hashlib.sha256(str(claim["source_quote"]).encode("utf-8")).hexdigest()
        if digest != claim["source_excerpt_sha256"]:
            raise BenchmarkValidationError(f"{case['case_id']}: source excerpt digest mismatch")
        if not _timestamp(str(claim.get("source_retrieved_at", ""))):
            raise BenchmarkValidationError(f"{case['case_id']}: source retrieval time must be timezone-aware")
        fragments = claim.get("required_fragments") or []
        if not fragments:
            raise BenchmarkValidationError(f"{case['case_id']}: each claim needs required answer fragments")
        if require_independent_review and not _reviewers_are_independent(claim.get("reviews") or []):
            raise BenchmarkValidationError(
                f"{case['case_id']}/{claim['claim_id']}: two distinct independent approvals are required"
            )

    if require_independent_review:
        reviews = case.get("case_reviews") or []
        if not _reviewers_are_independent(reviews):
            raise BenchmarkValidationError(f"{case['case_id']}: two independent case reviews are required")


def load_factual_dataset(
    dataset_path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_independent_review: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    path = Path(dataset_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise BenchmarkValidationError(
            f"frozen dataset digest mismatch: expected {expected_sha256}, got {digest}"
        )
    cases = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise BenchmarkValidationError("factual benchmark is empty")
    ids: set[str] = set()
    for case in cases:
        validate_case(case, require_independent_review=require_independent_review)
        if case["case_id"] in ids:
            raise BenchmarkValidationError(f"duplicate case_id: {case['case_id']}")
        ids.add(case["case_id"])
    return cases, digest


@dataclass(frozen=True)
class ClaimJudgement:
    supported: bool
    contradicted: bool = False
    note: str = ""


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str
    answerable: bool
    abstained: bool
    correct: bool
    confidence: float
    supported_claims: int
    unsupported_claims: int
    response_mode: str
    answer_text: str
    response_sha256: str
    latency_ms: float


def literal_claim_judge(answer: str, claim: dict[str, Any]) -> ClaimJudgement:
    """Exact-fragment baseline; this does not prove semantic support or catch every extra claim."""
    normalized = normalize_answer(answer)
    groups = claim.get("required_fragment_groups")
    if groups:
        matched = all(
            any(normalize_answer(item) and normalize_answer(item) in normalized for item in group)
            for group in groups
        )
    else:
        fragments = [normalize_answer(item) for item in claim.get("required_fragments", [])]
        matched = bool(fragments) and all(item and item in normalized for item in fragments)
    return ClaimJudgement(supported=matched, contradicted=False, note="required-fragment exact-match baseline")


def _is_abstention(response: Any) -> bool:
    mode = str(getattr(response, "mode", "") or "")
    answer = str(getattr(response, "answer", "") or "").strip()
    return mode in ABSTENTION_MODES or (not answer and float(getattr(response, "confidence", 0.0) or 0.0) <= 0.0)


def _calibration_metrics(rows: list[BenchmarkCaseResult], bins: int) -> dict[str, float | int]:
    # Response confidence is confidence in an answer; abstentions have no answer confidence.
    answered = [row for row in rows if not row.abstained]
    if not answered:
        return {
            "brier_score": 0.0,
            "expected_calibration_error": 0.0,
            "calibration_sample_count": 0,
        }
    brier = sum((min(1.0, max(0.0, row.confidence)) - float(row.correct)) ** 2 for row in answered) / len(answered)
    bucket: dict[int, list[BenchmarkCaseResult]] = defaultdict(list)
    for row in answered:
        index = min(bins - 1, int(min(1.0, max(0.0, row.confidence)) * bins))
        bucket[index].append(row)
    ece = 0.0
    for members in bucket.values():
        mean_conf = sum(row.confidence for row in members) / len(members)
        accuracy = sum(float(row.correct) for row in members) / len(members)
        ece += (len(members) / len(answered)) * abs(mean_conf - accuracy)
    return {
        "brier_score": round(brier, 6),
        "expected_calibration_error": round(ece, 6),
        "calibration_sample_count": len(answered),
    }


def run_factual_benchmark(
    agent: Any,
    cases: Iterable[dict[str, Any]],
    *,
    claim_judge: Callable[[str, dict[str, Any]], ClaimJudgement] = literal_claim_judge,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Run reviewed test cases and report accuracy, unsupported answers, abstention and calibration."""
    rows: list[BenchmarkCaseResult] = []
    for case in cases:
        validate_case(case, require_independent_review=True)
        response = agent.respond(case["prompt"])
        answer = str(getattr(response, "answer", "") or "")
        abstained = _is_abstention(response)
        confidence = min(1.0, max(0.0, float(getattr(response, "confidence", 0.0) or 0.0)))
        supported = 0
        unsupported = 0
        for claim in case.get("claims", []):
            judgement = claim_judge(answer, claim)
            if judgement.supported and not judgement.contradicted:
                supported += 1
            else:
                unsupported += 1
        if case["answerable"]:
            correct = (not abstained) and unsupported == 0 and supported == len(case["claims"])
        else:
            correct = abstained
        rows.append(BenchmarkCaseResult(
            case_id=str(case["case_id"]),
            answerable=bool(case["answerable"]),
            abstained=abstained,
            correct=correct,
            confidence=confidence,
            supported_claims=supported,
            unsupported_claims=unsupported,
            response_mode=str(getattr(response, "mode", "") or "unknown"),
            answer_text=answer,
            response_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            latency_ms=max(0.0, float(getattr(response, "latency_ms", 0.0) or 0.0)),
        ))

    answered = [row for row in rows if not row.abstained]
    answerable = [row for row in rows if row.answerable]
    unanswerable = [row for row in rows if not row.answerable]
    correct_answerable = sum(row.correct for row in answerable)
    correct_abstentions = sum(row.correct and row.abstained for row in rows)
    all_claims = sum(row.supported_claims + row.unsupported_claims for row in rows)
    supported_claims = sum(row.supported_claims for row in rows)
    unsupported_nonabstained = sum(
        (row.unsupported_claims > 0 or not row.answerable) for row in answered
    )

    metrics = {
        "case_count": len(rows),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "factual_accuracy": round(correct_answerable / len(answerable), 6) if answerable else 0.0,
        "answered_case_accuracy": round(sum(row.correct for row in answered) / len(answered), 6) if answered else 0.0,
        "coverage": round(len(answered) / len(rows), 6) if rows else 0.0,
        "unsupported_answer_rate": round(unsupported_nonabstained / len(answered), 6) if answered else 0.0,
        "claim_support_rate": round(supported_claims / all_claims, 6) if all_claims else 0.0,
        "abstention_precision": round(correct_abstentions / sum(row.abstained for row in rows), 6) if any(row.abstained for row in rows) else 0.0,
        "abstention_recall": round(correct_abstentions / len(unanswerable), 6) if unanswerable else 0.0,
        "abstention_count": sum(row.abstained for row in rows),
        "mean_latency_ms": round(sum(row.latency_ms for row in rows) / len(rows), 3) if rows else 0.0,
        **_calibration_metrics(rows, max(2, int(calibration_bins))),
    }
    return {
        "benchmark": "Mirror source-backed factual test",
        "metrics": metrics,
        "cases": [row.__dict__ for row in rows],
        "judge": getattr(claim_judge, "__name__", type(claim_judge).__name__),
        "limitations": [
            "The literal-fragment judge is a reproducible baseline, not semantic entailment.",
            "Unsupported extra claims require an independent claim-extraction/annotation judge.",
            "Publish factual scores only with the frozen dataset digest and review manifest.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--vault", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases, digest = load_factual_dataset(
        args.dataset,
        expected_sha256=manifest.get("dataset_sha256"),
        require_independent_review=not args.validate_only,
    )
    if args.validate_only:
        print(json.dumps({
            "validated": True,
            "cases": len(cases),
            "dataset_sha256": digest,
            "review_gate": "pending" if manifest.get("review_status") != "approved" else "approved",
        }, indent=2))
        return 0
    if args.vault is None or not args.vault.is_file():
        parser.error("--vault must point to an existing .dcrystal vault")
    from mirror_crystal import DiskCrystalVault
    vault = DiskCrystalVault(args.vault)
    try:
        result = run_factual_benchmark(vault, cases)
    finally:
        vault.close()
    result["dataset_sha256"] = digest
    result["manifest"] = str(args.manifest)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
