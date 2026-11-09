"""Canonical blueprint persistence and runtime entry points for DiskCrystalVault."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Mapping

from .answer_verification import BlueprintAnswerVerifier
from .blueprint_contracts import Blueprint, BlueprintValidationError
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
                    "INSERT OR REPLACE INTO blueprints(blueprint_id, domain, intent, kind, canonical_json, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        blueprint.blueprint_id,
                        blueprint.domain,
                        blueprint.intent,
                        blueprint.kind.value,
                        blueprint.canonical_json(),
                        time.time(),
                    ),
                )
        return blueprint.blueprint_id

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
            all_blueprints = self.list_blueprints()
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

        peers = self.list_blueprints(domain=root.domain, intent=root.intent)
        report = BlueprintAnswerVerifier().verify(root, peers, max_age_days=max_age_days)
        if report.contradictions:
            answer = "Conflicting blueprint records were found. Mirror abstained instead of choosing one as factual."
            confidence = 0.0
            mode = "blueprint_contradiction_abstain"
            tier = "honest_abstain"
        else:
            answer = composition.answer
            confidence = 0.95 if report.reliable else 0.40
            mode = "blueprint_composition_reliable" if report.reliable else "blueprint_composition_unverified"
            tier = "high_confidence" if report.reliable else "unverified"

        trace = [
            f"Dependency-first composition: {', '.join(composition.blueprint_ids)}",
            f"Source: {report.source_ref or 'missing'}",
            f"Evidence: {report.evidence_level}; freshness: {report.freshness}",
        ]
        trace.extend(f"Not reliable: {reason}" for reason in report.reasons)
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