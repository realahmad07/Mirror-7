"""Explicit, review-gated curation helpers for legacy Crystal knowledge."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .claim_verification import ClaimEvidence, ClaimEvidenceError, ClaimEvidenceVerifier


def curate_blueprint_batch(vault, evidence_rows: Iterable[ClaimEvidence]) -> dict[str, object]:
    grouped: dict[str, list[ClaimEvidence]] = defaultdict(list)
    for evidence in evidence_rows:
        grouped[evidence.blueprint_id].append(evidence)

    promoted: list[str] = []
    pending: dict[str, list[str]] = {}
    for blueprint_id, rows in sorted(grouped.items()):
        required = sorted({row.claim_id for row in rows})
        vault.register_blueprint_claim_scope(blueprint_id, required)
        for row in rows:
            vault.store_claim_evidence(row)
        try:
            vault.curate_blueprint_from_evidence(blueprint_id)
            promoted.append(blueprint_id)
        except ClaimEvidenceError as exc:
            pending[blueprint_id] = [str(exc)]

    return {
        "reviewed_blueprints": len(grouped),
        "promoted_blueprints": promoted,
        "pending_blueprints": pending,
    }
