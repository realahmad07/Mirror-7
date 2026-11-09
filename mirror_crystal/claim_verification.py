"""Auditable source-backed claim evidence with independent human review gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import re
from urllib.parse import urlparse
from typing import Iterable

from .blueprint_contracts import Blueprint


class ClaimEvidenceError(ValueError):
    """Raised when claim evidence is incomplete or fails its review contract."""


def _aware_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ClaimEvidenceError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ClaimEvidenceError(f"{field} must include a timezone")
    return parsed


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


@dataclass(frozen=True)
class ClaimEvidence:
    blueprint_id: str
    claim_id: str
    claim_text: str
    source_publisher: str
    source_url: str
    source_locator: str
    source_quote: str
    source_excerpt_sha256: str
    source_retrieved_at: str
    license_spdx: str
    reviewer_id: str
    reviewer_independent: bool
    review_decision: str
    review_method: str
    reviewed_at: str
    review_notes: str = ""

    def __post_init__(self) -> None:
        required = (
            "blueprint_id", "claim_id", "claim_text", "source_publisher", "source_url",
            "source_locator", "source_quote", "source_excerpt_sha256", "source_retrieved_at",
            "license_spdx", "reviewer_id", "review_method", "reviewed_at",
        )
        for name in required:
            if not str(getattr(self, name)).strip():
                raise ClaimEvidenceError(f"{name} must not be empty")
        parsed = urlparse(self.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ClaimEvidenceError("source_url must be an HTTPS URL")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_excerpt_sha256):
            raise ClaimEvidenceError("source_excerpt_sha256 must be a lowercase SHA-256 digest")
        actual = hashlib.sha256(self.source_quote.encode("utf-8")).hexdigest()
        if actual != self.source_excerpt_sha256:
            raise ClaimEvidenceError("source quote digest mismatch")
        _aware_timestamp(self.source_retrieved_at, "source_retrieved_at")
        _aware_timestamp(self.reviewed_at, "reviewed_at")
        if self.review_decision not in {"approve", "reject"}:
            raise ClaimEvidenceError("review_decision must be 'approve' or 'reject'")
        if self.review_method not in {"independent_source_comparison", "independent_replication"}:
            raise ClaimEvidenceError("review_method must identify an independent review procedure")
        if self.review_decision == "approve" and not self.reviewer_independent:
            raise ClaimEvidenceError("an approving reviewer must be marked independent")
        if self.review_decision == "reject" and not self.review_notes.strip():
            raise ClaimEvidenceError("a rejection requires review_notes")

    @classmethod
    def from_dict(cls, data: dict) -> "ClaimEvidence":
        return cls(**dict(data))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClaimEvidenceReport:
    reliable: bool
    expected_claim_ids: tuple[str, ...]
    verified_claim_ids: tuple[str, ...]
    missing_claim_ids: tuple[str, ...]
    rejected_claim_ids: tuple[str, ...]
    reasons: tuple[str, ...]


class ClaimEvidenceVerifier:
    """Require exact claim-to-answer coverage and two distinct independent reviewers."""

    def verify(
        self,
        blueprint: Blueprint,
        evidence: Iterable[ClaimEvidence],
        expected_claim_ids: Iterable[str],
        *,
        min_independent_reviewers: int = 2,
    ) -> ClaimEvidenceReport:
        expected = tuple(sorted(set(str(x) for x in expected_claim_ids if str(x).strip())))
        rows = [item for item in evidence if item.blueprint_id == blueprint.blueprint_id]
        grouped: dict[str, list[ClaimEvidence]] = {}
        for row in rows:
            grouped.setdefault(row.claim_id, []).append(row)

        reasons: list[str] = []
        missing: list[str] = []
        rejected: list[str] = []
        verified: list[str] = []
        response_norm = _normalize(blueprint.response_template)

        if not expected:
            reasons.append("no explicit claim scope is registered for this blueprint")

        for claim_id in expected:
            claim_rows = grouped.get(claim_id, [])
            claim_texts = {_normalize(row.claim_text) for row in claim_rows}
            if len(claim_texts) != 1:
                missing.append(claim_id)
                reasons.append(f"{claim_id}: claim text is missing or inconsistent")
                continue
            claim_text = next(iter(claim_texts))
            if not claim_text or claim_text not in response_norm:
                missing.append(claim_id)
                reasons.append(f"{claim_id}: claim text is not covered by the blueprint response")
                continue
            if any(row.review_decision == "reject" for row in claim_rows):
                rejected.append(claim_id)
                reasons.append(f"{claim_id}: at least one reviewer rejected the claim")
                continue
            reviewers = {
                row.reviewer_id
                for row in claim_rows
                if row.review_decision == "approve" and row.reviewer_independent
            }
            if len(reviewers) < max(2, int(min_independent_reviewers)):
                missing.append(claim_id)
                reasons.append(
                    f"{claim_id}: requires {max(2, int(min_independent_reviewers))} distinct independent reviewers; found {len(reviewers)}"
                )
                continue
            verified.append(claim_id)

        reliable = bool(expected) and len(verified) == len(expected) and not rejected and not missing
        return ClaimEvidenceReport(
            reliable=reliable,
            expected_claim_ids=expected,
            verified_claim_ids=tuple(sorted(verified)),
            missing_claim_ids=tuple(sorted(set(missing))),
            rejected_claim_ids=tuple(sorted(set(rejected))),
            reasons=tuple(reasons),
        )
