"""Fail-closed verification of blueprint provenance, freshness and contradictions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
from typing import Iterable

from .blueprint_contracts import Blueprint, EvidenceLevel


@dataclass(frozen=True)
class BlueprintVerificationReport:
    reliable: bool
    freshness: str
    contradictions: tuple[str, ...]
    reasons: tuple[str, ...]
    source_ref: str
    evidence_level: str


class BlueprintAnswerVerifier:
    """Verify evidence metadata conservatively; this is not an external fact-checker."""

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.casefold()).strip()

    def verify(
        self,
        blueprint: Blueprint,
        peers: Iterable[Blueprint] = (),
        *,
        now: datetime | None = None,
        max_age_days: int = 180,
    ) -> BlueprintVerificationReport:
        reasons: list[str] = []
        provenance = blueprint.provenance
        expected_digest = hashlib.sha256(
            (blueprint.prompt_pattern + "\0" + blueprint.response_template).encode("utf-8")
        ).hexdigest()
        if not provenance.source or provenance.source == "unknown":
            reasons.append("source identity is missing")
        if not provenance.source_ref.strip():
            reasons.append("source reference is missing")
        if not provenance.content_sha256 or provenance.content_sha256 != expected_digest:
            reasons.append("source content digest is missing or mismatched")

        freshness = "unknown"
        captured = provenance.captured_at.strip()
        if captured:
            try:
                parsed = datetime.fromisoformat(captured.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("timestamp must include timezone")
                current = now or datetime.now(timezone.utc)
                age_days = (current - parsed.astimezone(timezone.utc)).total_seconds() / 86400.0
                if age_days < -1.0:
                    freshness = "future"
                    reasons.append("source capture time is in the future")
                elif age_days > max(0, int(max_age_days)):
                    freshness = "stale"
                    reasons.append(f"source is older than {max_age_days} days")
                else:
                    freshness = "fresh"
            except (TypeError, ValueError, OverflowError):
                reasons.append("source capture timestamp is invalid or timezone-naive")
        else:
            reasons.append("source capture timestamp is missing")

        contradictions: set[str] = set()
        response_norm = self._norm(blueprint.response_template)
        for peer in peers:
            if peer.blueprint_id == blueprint.blueprint_id:
                continue
            if peer.domain == blueprint.domain and peer.intent == blueprint.intent:
                other = self._norm(peer.response_template)
                if other and response_norm and other != response_norm:
                    contradictions.add(peer.blueprint_id)
        if contradictions:
            reasons.append("conflicting blueprint responses exist for the same domain and intent")

        trusted_levels = {
            EvidenceLevel.FORMALLY_VERIFIED,
            EvidenceLevel.INDEPENDENTLY_REPRODUCED,
        }
        if blueprint.verification.level not in trusted_levels:
            reasons.append("evidence level does not establish reliable factual verification")
        if not blueprint.verification.checks_passed:
            reasons.append("no named passing verification checks are attached")
        if blueprint.verification.checks_failed:
            reasons.append("one or more verification checks failed")
        if not blueprint.verification.verifier.strip():
            reasons.append("named verifier is missing")

        reliable = not reasons and freshness == "fresh" and not contradictions
        return BlueprintVerificationReport(
            reliable=reliable,
            freshness=freshness,
            contradictions=tuple(sorted(contradictions)),
            reasons=tuple(reasons),
            source_ref=provenance.source_ref,
            evidence_level=blueprint.verification.level.value,
        )