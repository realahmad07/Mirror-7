"""Canonical blueprint persistence and runtime entry points for DiskCrystalVault."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import time
from typing import Any, Callable, Iterable, Mapping

from .answer_verification import BlueprintAnswerVerifier
from .blueprint_contracts import (
    Blueprint, BlueprintKind, BlueprintProvenance, BlueprintValidationError,
    EvidenceLevel, VerificationEvidence,
)
from .claim_verification import ClaimEvidence, ClaimEvidenceError, ClaimEvidenceVerifier
from .blueprint_prompt import blueprint_query_prefix, blueprint_prompt_fields, match_blueprint_prompt
from .blueprint_runtime import BlueprintCompositionEngine, BlueprintCompositionError
from .response_types import CrystalResponse


class VaultBlueprintRuntimeMixin:
    """Persist canonical blueprints beside legacy rules without destructive migration."""

    def store_blueprint(self, blueprint: Blueprint) -> str:
        if not isinstance(blueprint, Blueprint):
            raise TypeError("blueprint must be a Blueprint")
        expected = Blueprint._digest(blueprint.prompt_pattern, blueprint.response_template)
        supplied = blueprint.provenance.content_sha256
        if supplied and supplied != expected:
            raise BlueprintValidationError("blueprint provenance digest does not match its content")
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO blueprints("
                    "blueprint_id, domain, intent, kind, canonical_json, prompt_key, prompt_prefix, updated_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        blueprint.blueprint_id,
                        blueprint.domain,
                        blueprint.intent,
                        blueprint.kind.value,
                        blueprint.canonical_json(),
                        *blueprint_prompt_fields(blueprint.prompt_pattern),
                        time.time(),
                    ),
                )
        return blueprint.blueprint_id

    def find_blueprint_matches(self, prompt: str, *, limit: int = 512) -> list[tuple[Blueprint, dict[str, Any]]]:
        """Find canonical blueprints matching a full literal/slot prompt pattern."""
        prefix = blueprint_query_prefix(prompt)
        rows = self.conn.execute(
            "SELECT canonical_json FROM blueprints WHERE prompt_prefix = ? OR prompt_prefix = '' "
            "ORDER BY blueprint_id LIMIT ?",
            (prefix, max(1, min(int(limit), 2048))),
        ).fetchall()
        matches: list[tuple[Blueprint, dict[str, Any]]] = []
        for (canonical_json,) in rows:
            try:
                blueprint = Blueprint.from_dict(json.loads(canonical_json))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            values = match_blueprint_prompt(blueprint.prompt_pattern, prompt)
            if values is not None:
                matches.append((blueprint, values))
        return matches

    def respond_from_blueprint_prompt(self, prompt: str) -> CrystalResponse | None:
        """Use a uniquely matching canonical blueprint before the legacy retrieval path."""
        matches = self.find_blueprint_matches(prompt)
        if not matches:
            return None
        if len(matches) != 1:
            ids = [blueprint.blueprint_id for blueprint, _ in matches[:12]]
            return CrystalResponse(
                answer="Multiple canonical blueprints match this request. Mirror abstained rather than selecting one silently.",
                confidence=0.0,
                mode="blueprint_prompt_ambiguous",
                matched_rule_id=None,
                latency_ms=0.0,
                sandbox_verified=False,
                reasoning_trace=["Ambiguous canonical prompt matches: " + ", ".join(ids)],
                epistemic_tier="honest_abstain",
            )
        blueprint, values = matches[0]
        return self.respond_blueprint(blueprint.blueprint_id, values=values)

    def store_claim_evidence(self, evidence: ClaimEvidence) -> None:
        """Persist one independent review for one atomic claim on a blueprint."""
        if not isinstance(evidence, ClaimEvidence):
            raise TypeError("evidence must be ClaimEvidence")
        if self.get_blueprint(evidence.blueprint_id) is None:
            raise ClaimEvidenceError(f"unknown blueprint: {evidence.blueprint_id}")
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO blueprint_claim_evidence("
                    "blueprint_id, claim_id, reviewer_id, canonical_json, updated_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (
                        evidence.blueprint_id,
                        evidence.claim_id,
                        evidence.reviewer_id,
                        json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":")),
                        time.time(),
                    ),
                )

    def list_claim_evidence(self, blueprint_id: str) -> list[ClaimEvidence]:
        rows = self.conn.execute(
            "SELECT canonical_json FROM blueprint_claim_evidence "
            "WHERE blueprint_id = ? ORDER BY claim_id, reviewer_id",
            (str(blueprint_id),),
        ).fetchall()
        return [ClaimEvidence.from_dict(json.loads(row[0])) for row in rows]

    def register_blueprint_claim_scope(self, blueprint_id: str, required_claim_ids: Iterable[str]) -> tuple[str, ...]:
        """Freeze the complete atomic claim scope before independent review."""
        if self.get_blueprint(blueprint_id) is None:
            raise ClaimEvidenceError(f"unknown blueprint: {blueprint_id}")
        claim_ids = tuple(sorted(set(str(x).strip() for x in required_claim_ids if str(x).strip())))
        if not claim_ids:
            raise ClaimEvidenceError("claim scope must contain at least one claim id")
        with self._lock:
            with self.conn:
                existing = self.conn.execute(
                    "SELECT required_claim_ids_json FROM blueprint_claim_scopes WHERE blueprint_id = ?",
                    (blueprint_id,),
                ).fetchone()
                if existing is not None:
                    old_scope = tuple(json.loads(existing[0]))
                    if old_scope != claim_ids:
                        raise ClaimEvidenceError("claim scope is frozen and cannot be silently changed")
                else:
                    self.conn.execute(
                        "INSERT INTO blueprint_claim_scopes(blueprint_id, required_claim_ids_json, updated_at) "
                        "VALUES (?, ?, ?)",
                        (blueprint_id, json.dumps(claim_ids), time.time()),
                    )
        return claim_ids

    def get_blueprint_claim_scope(self, blueprint_id: str) -> tuple[str, ...]:
        row = self.conn.execute(
            "SELECT required_claim_ids_json FROM blueprint_claim_scopes WHERE blueprint_id = ?",
            (str(blueprint_id),),
        ).fetchone()
        return tuple(json.loads(row[0])) if row else ()

    def verify_blueprint_claims(self, blueprint: Blueprint):
        return ClaimEvidenceVerifier().verify(
            blueprint,
            self.list_claim_evidence(blueprint.blueprint_id),
            self.get_blueprint_claim_scope(blueprint.blueprint_id),
        )

    def curate_blueprint_from_evidence(self, blueprint_id: str) -> Blueprint:
        """Promote only after two distinct independent reviewers approve every frozen claim."""
        blueprint = self.get_blueprint(blueprint_id)
        if blueprint is None:
            raise ClaimEvidenceError(f"unknown blueprint: {blueprint_id}")
        scope = self.get_blueprint_claim_scope(blueprint_id)
        evidence = self.list_claim_evidence(blueprint_id)
        report = ClaimEvidenceVerifier().verify(blueprint, evidence, scope)
        if not report.reliable:
            raise ClaimEvidenceError("blueprint curation blocked: " + "; ".join(report.reasons))

        approved = [
            item for item in evidence
            if item.claim_id in scope and item.review_decision == "approve" and item.reviewer_independent
        ]
        reviewers = sorted({item.reviewer_id for item in approved})
        source_urls = sorted({item.source_url for item in approved})
        source_publishers = sorted({item.source_publisher for item in approved})
        retrieved_at = max(item.source_retrieved_at for item in approved)
        license_ids = sorted({item.license_spdx for item in approved})
        now = datetime.now(timezone.utc).isoformat()
        promoted = replace(
            blueprint,
            provenance=BlueprintProvenance(
                source="independently_reviewed_primary_sources:" + ",".join(source_publishers),
                source_ref=";".join(source_urls),
                license_spdx=" AND ".join(license_ids),
                content_sha256=Blueprint._digest(blueprint.prompt_pattern, blueprint.response_template),
                captured_at=retrieved_at,
            ),
            verification=VerificationEvidence(
                level=EvidenceLevel.INDEPENDENTLY_REPRODUCED,
                checks_passed=tuple(f"claim:{claim_id}:two_independent_reviews" for claim_id in scope),
                verifier="independent-reviewers:" + ",".join(reviewers),
                verified_at=now,
            ),
        )
        self.store_blueprint(promoted)
        return promoted

    def _load_blueprint_dependency_closure(self, blueprint_id: str) -> list[Blueprint]:
        """Load only a root blueprint and its transitive dependencies.

        The previous implementation loaded the first 10,000 blueprints in the
        vault. That made otherwise valid records unreachable once the canonical
        store grew beyond that arbitrary limit. Composition needs only the
        dependency closure; peer records are fetched separately for verification.
        """
        pending = [str(blueprint_id)]
        loaded: dict[str, Blueprint] = {}
        while pending:
            current_id = pending.pop()
            if current_id in loaded:
                continue
            blueprint = self.get_blueprint(current_id)
            if blueprint is None:
                # Leave missing dependencies out so BlueprintDependencyGraph
                # returns the existing, explicit "missing dependency" failure.
                continue
            loaded[current_id] = blueprint
            pending.extend(
                dependency_id
                for dependency_id in reversed(sorted(blueprint.dependencies))
                if dependency_id not in loaded
            )
        return list(loaded.values())

    def get_blueprint(self, blueprint_id: str) -> Blueprint | None:
        row = self.conn.execute(
            "SELECT canonical_json FROM blueprints WHERE blueprint_id = ?",
            (str(blueprint_id),),
        ).fetchone()
        if row is None:
            return None
        return Blueprint.from_dict(json.loads(row[0]))

    def list_blueprints(
        self,
        *,
        domain: str | None = None,
        intent: str | None = None,
        limit: int = 10000,
    ) -> list[Blueprint]:
        clauses: list[str] = []
        params: list[Any] = []
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        if intent is not None:
            clauses.append("intent = ?")
            params.append(intent)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(int(limit), 10000)))
        rows = self.conn.execute(
            "SELECT canonical_json FROM blueprints" + where + " ORDER BY blueprint_id LIMIT ?",
            tuple(params),
        ).fetchall()
        return [Blueprint.from_dict(json.loads(row[0])) for row in rows]

    def backfill_legacy_blueprints(self, *, after_rule_id: int = -1, limit: int = 100) -> dict[str, int]:
        """Convert a bounded page of legacy rules to canonical, explicitly unverified blueprints."""
        page_size = max(1, min(int(limit), 1000))
        rows = self.conn.execute(
            "SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload "
            "FROM rules WHERE rule_id > ? ORDER BY rule_id LIMIT ?",
            (int(after_rule_id), page_size),
        ).fetchall()
        migrated = 0
        last_id = int(after_rule_id)
        for row in rows:
            rule = self._row_to_rule(row)
            blueprint = Blueprint.from_crystal_rule(rule)
            if self.get_blueprint(blueprint.blueprint_id) is None:
                self.store_blueprint(blueprint)
                migrated += 1
            last_id = max(last_id, rule.rule_id)
        return {"migrated": migrated, "scanned": len(rows), "next_after_rule_id": last_id}

    def respond_blueprint(
        self,
        blueprint_id: str,
        *,
        values: Mapping[str, Any] | None = None,
        verified_conditions: Iterable[str] = (),
        postcondition_verifier: Callable[[Blueprint, str, str, Mapping[str, Any]], bool] | None = None,
        executor: Callable[[Blueprint, Mapping[str, Any]], str] | None = None,
        max_age_days: int = 180,
    ) -> CrystalResponse:
        """Compose a persisted blueprint chain and never label unsupported evidence as reliable."""
        started = time.perf_counter_ns()
        root = self.get_blueprint(blueprint_id)
        if root is None:
            return CrystalResponse(
                answer="Blueprint not found; no composition was attempted.",
                confidence=0.0,
                mode="blueprint_not_found",
                matched_rule_id=None,
                latency_ms=0.0,
                sandbox_verified=False,
                reasoning_trace=[f"Unknown blueprint: {blueprint_id}"],
                epistemic_tier="honest_abstain",
            )
        try:
            all_blueprints = self._load_blueprint_dependency_closure(blueprint_id)
            composition = BlueprintCompositionEngine().compose(
                blueprint_id,
                all_blueprints,
                values=values,
                verified_conditions=verified_conditions,
                postcondition_verifier=postcondition_verifier,
                executor=executor,
            )
        except BlueprintCompositionError as exc:
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            return CrystalResponse(
                answer="Blueprint composition was blocked because a required verification gate was not met.",
                confidence=0.0,
                mode="blueprint_composition_blocked",
                matched_rule_id=None,
                latency_ms=round(elapsed, 3),
                sandbox_verified=False,
                reasoning_trace=[str(exc)],
                epistemic_tier="honest_abstain",
            )

        all_blueprints = self.list_blueprints()
        composed_blueprints = [
            item for item in all_blueprints if item.blueprint_id in set(composition.blueprint_ids)
        ]
        verification_reports = []
        claim_reports = []
        contradiction_ids: set[str] = set()
        for item in composed_blueprints:
            peers = self.list_blueprints(domain=item.domain, intent=item.intent)
            item_report = BlueprintAnswerVerifier().verify(item, peers, max_age_days=max_age_days)
            verification_reports.append((item, item_report))
            contradiction_ids.update(item_report.contradictions)
            if item.kind == BlueprintKind.FACT:
                claim_reports.append((item, self.verify_blueprint_claims(item)))

        evidence_reliable = all(report.reliable for _, report in verification_reports)
        claims_reliable = all(report.reliable for _, report in claim_reports)
        reliable = evidence_reliable and claims_reliable
        if contradiction_ids:
            answer = "Conflicting blueprint records were found. Mirror abstained instead of choosing one as factual."
            confidence = 0.0
            mode = "blueprint_contradiction_abstain"
            tier = "honest_abstain"
        else:
            answer = composition.answer
            confidence = 0.95 if reliable else 0.40
            mode = "blueprint_composition_reliable" if reliable else "blueprint_composition_unverified"
            tier = "high_confidence" if reliable else "unverified"

        trace = [f"Dependency-first composition: {', '.join(composition.blueprint_ids)}"]
        for item, item_report in verification_reports:
            trace.append(
                f"Blueprint {item.blueprint_id}: source={item_report.source_ref or 'missing'}; "
                f"evidence={item_report.evidence_level}; freshness={item_report.freshness}"
            )
            trace.extend(f"{item.blueprint_id} not reliable: {reason}" for reason in item_report.reasons)
        for item, claim_report in claim_reports:
            trace.append(
                f"Claim evidence {item.blueprint_id}: verified={','.join(claim_report.verified_claim_ids) or 'none'}"
            )
            trace.extend(f"{item.blueprint_id} claim review: {reason}" for reason in claim_report.reasons)
        if composition.verified_postconditions:
            trace.append("Postconditions verified: " + ", ".join(composition.verified_postconditions))
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        return CrystalResponse(
            answer=answer,
            confidence=confidence,
            mode=mode,
            matched_rule_id=None,
            latency_ms=round(elapsed, 3),
            sandbox_verified=False,
            reasoning_trace=trace,
            epistemic_tier=tier,
        )