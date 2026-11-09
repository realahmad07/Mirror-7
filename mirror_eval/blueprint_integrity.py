"""Small black-box integrity suite kept separate from runtime implementation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mirror_crystal import Blueprint, BlueprintKind, BlueprintProvenance, EvidenceLevel, VerificationEvidence
from mirror_crystal.answer_verification import BlueprintAnswerVerifier
from mirror_crystal.blueprint_runtime import BlueprintCompositionEngine, BlueprintCompositionError


def _bp(ident, response, **kwargs):
    prompt = "fixture prompt"
    return Blueprint(
        blueprint_id=ident,
        kind=BlueprintKind.FACT,
        domain=kwargs.get("domain", "eval"),
        intent=kwargs.get("intent", "fact"),
        prompt_pattern=prompt,
        response_template=response,
        dependencies=tuple(kwargs.get("dependencies", ())),
        preconditions=tuple(kwargs.get("preconditions", ())),
        postconditions=tuple(kwargs.get("postconditions", ())),
        provenance=BlueprintProvenance(
            source="evaluation-fixture",
            source_ref="fixture://" + ident,
            content_sha256=Blueprint._digest(prompt, response),
            captured_at=kwargs.get("captured_at", datetime.now(timezone.utc).isoformat()),
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )


def run_blueprint_integrity_suite():
    """Run held-out synthetic negative controls using public runtime interfaces."""
    rows = []
    root = _bp("root", "answer", dependencies=("absent",))
    try:
        BlueprintCompositionEngine().compose("root", [root])
        rows.append(("missing_dependency", False))
    except BlueprintCompositionError:
        rows.append(("missing_dependency", True))

    blocked = _bp("pre", "answer", preconditions=("verified-input",))
    try:
        BlueprintCompositionEngine().compose("pre", [blocked])
        rows.append(("missing_precondition", False))
    except BlueprintCompositionError:
        rows.append(("missing_precondition", True))

    left = _bp("left", "truth-a")
    right = _bp("right", "truth-b")
    contradiction = BlueprintAnswerVerifier().verify(left, [left, right])
    rows.append(("contradiction_detected", bool(contradiction.contradictions)))

    old = _bp("old", "old fact", captured_at=(datetime.now(timezone.utc)-timedelta(days=4000)).isoformat())
    stale = BlueprintAnswerVerifier().verify(old, max_age_days=30)
    rows.append(("stale_source_rejected", not stale.reliable and stale.freshness == "stale"))

    legacy = _bp("legacy", "legacy fact")
    legacy_report = BlueprintAnswerVerifier().verify(legacy)
    rows.append(("unverified_legacy_rejected", not legacy_report.reliable))

    return [{"case": name, "passed": bool(ok)} for name, ok in rows]
