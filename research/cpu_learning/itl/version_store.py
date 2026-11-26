"""Append-only versioned OperatorCard store with shadow/promote/retire/rollback lifecycle.

STRICT ISOLATION (Stage S3):
- Contains ZERO hardcoded domain action, predicate, or attribute literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Enforces monotonic version histories per ``action_id``, non-destructive promotion
  (previous promoted version is retired, NEVER deleted), deterministic rollback to any
  historical version, freeze guards, and exact JSON serialization round-trips.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.simulator.state_graph import canonical_json

VERSION_STORE_SCHEMA_VERSION: str = "s3-operator-version-store-v1"
DEFAULT_VERSION_TIMESTAMP: str = "2026-11-26T00:00:00Z"

VersionLifecycleState = Literal["shadow", "promoted", "retired"]
_VALID_LIFECYCLE_STATES: frozenset[str] = frozenset({"shadow", "promoted", "retired"})


class VersionStoreError(ValueError):
    """Raised on invalid version registration, missing version, or lifecycle constraint error."""


class VersionStoreFrozenError(RuntimeError):
    """Raised when a mutating operation is attempted on a frozen OperatorVersionStore."""


@dataclass(frozen=True)
class OperatorVersionRecord:
    """Immutable snapshot of one versioned OperatorCard and its lifecycle metadata."""

    card: OperatorCard
    status: VersionLifecycleState
    evidence: dict[str, Any] = field(default_factory=dict)
    contrast_pass_rate: float = 0.0
    created_at: str = DEFAULT_VERSION_TIMESTAMP
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in _VALID_LIFECYCLE_STATES:
            raise VersionStoreError(
                f"Invalid lifecycle state {self.status!r}; expected one of {sorted(_VALID_LIFECYCLE_STATES)}"
            )
        if self.card.version < 1:
            raise VersionStoreError(
                f"OperatorCard version must be >= 1, got {self.card.version}"
            )
        rate = float(self.contrast_pass_rate)
        if not (0.0 <= rate <= 1.0):
            raise VersionStoreError(
                f"contrast_pass_rate must be in [0.0, 1.0], got {rate}"
            )
        object.__setattr__(self, "contrast_pass_rate", round(rate, 6))
        object.__setattr__(self, "evidence", copy.deepcopy(dict(self.evidence)))
        object.__setattr__(self, "created_at", str(self.created_at))
        object.__setattr__(self, "notes", str(self.notes))

    def with_lifecycle_state(
        self,
        new_state: VersionLifecycleState,
        *,
        extra_evidence: Mapping[str, Any] | None = None,
        contrast_pass_rate: float | None = None,
        notes: str | None = None,
    ) -> OperatorVersionRecord:
        merged_evidence = copy.deepcopy(self.evidence)
        if extra_evidence:
            merged_evidence.update(copy.deepcopy(dict(extra_evidence)))
        return OperatorVersionRecord(
            card=self.card,
            status=new_state,
            evidence=merged_evidence,
            contrast_pass_rate=(
                self.contrast_pass_rate
                if contrast_pass_rate is None
                else float(contrast_pass_rate)
            ),
            created_at=self.created_at,
            notes=self.notes if notes is None else str(notes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.to_dict(),
            "lifecycle_state": self.status,
            "evidence": copy.deepcopy(self.evidence),
            "contrast_pass_rate": self.contrast_pass_rate,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperatorVersionRecord:
        state_val = str(data["lifecycle_state"])
        if state_val not in _VALID_LIFECYCLE_STATES:
            raise VersionStoreError(f"Unknown lifecycle_state in serialized record: {state_val!r}")
        return cls(
            card=OperatorCard.from_dict(data["card"]),
            status=state_val,  # type: ignore[arg-type]
            evidence=dict(data.get("evidence", {})),
            contrast_pass_rate=float(data.get("contrast_pass_rate", 0.0)),
            created_at=str(data.get("created_at", DEFAULT_VERSION_TIMESTAMP)),
            notes=str(data.get("notes", "")),
        )


class OperatorVersionStore:
    """Append-only version history store per ``action_id`` with safe promotion and rollback.

    Guarantees:
    - Every registered version (``shadow``, ``promoted``, ``retired``) is preserved permanently;
      NO version is ever deleted.
    - At most one version per ``action_id`` holds ``promoted`` state at any point in time.
    - Promoting a candidate retires the previously promoted version without deleting it.
    - ``rollback(action_id, target_version)`` restores a historical version to ``promoted``
      and retires the displaced version without deleting any version.
    - Every lifecycle event is recorded in an append-only ``transition_log``.
    - Once ``freeze()`` is called, all mutating calls raise ``VersionStoreFrozenError``.
    """

    def __init__(self) -> None:
        self._histories: dict[str, list[OperatorVersionRecord]] = {}
        self._transition_log: list[dict[str, Any]] = []
        self._frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def _require_mutable(self) -> None:
        if self._frozen:
            raise VersionStoreFrozenError(
                "OperatorVersionStore is frozen; register_shadow/promote/rollback are forbidden"
            )

    def _log_event(
        self,
        *,
        event_kind: str,
        action_id: str,
        version: int,
        from_state: str | None,
        to_state: str,
        card_fingerprint: str,
        timestamp: str,
        reason: str = "",
    ) -> None:
        self._transition_log.append(
            {
                "event_seq": len(self._transition_log) + 1,
                "event_kind": event_kind,
                "action_id": action_id,
                "version": version,
                "from_state": from_state,
                "to_state": to_state,
                "card_fingerprint": card_fingerprint,
                "timestamp": timestamp,
                "reason": reason,
            }
        )

    def action_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._histories.keys()))

    def next_version(self, action_id: str) -> int:
        hist = self._histories.get(action_id)
        if not hist:
            return 1
        return max(rec.card.version for rec in hist) + 1

    def register_shadow(
        self,
        card: OperatorCard,
        *,
        evidence: Mapping[str, Any] | None = None,
        contrast_pass_rate: float = 0.0,
        created_at: str = DEFAULT_VERSION_TIMESTAMP,
        notes: str = "",
    ) -> OperatorVersionRecord:
        """Append a new candidate card version in ``shadow`` state (never overwrites or deletes)."""
        self._require_mutable()
        aid = card.action_id
        hist = self._histories.setdefault(aid, [])
        if hist:
            max_ver = max(r.card.version for r in hist)
            if card.version <= max_ver:
                raise VersionStoreError(
                    f"Version {card.version} for action {aid!r} must exceed latest version {max_ver}"
                )

        record = OperatorVersionRecord(
            card=card,
            status="shadow",
            evidence=dict(evidence or {}),
            contrast_pass_rate=contrast_pass_rate,
            created_at=created_at,
            notes=notes,
        )
        hist.append(record)
        self._log_event(
            event_kind="register_shadow",
            action_id=aid,
            version=card.version,
            from_state=None,
            to_state="shadow",
            card_fingerprint=card.fingerprint(),
            timestamp=created_at,
            reason=notes,
        )
        return record

    def promote(
        self,
        action_id: str,
        version: int,
        *,
        evidence: Mapping[str, Any] | None = None,
        contrast_pass_rate: float | None = None,
        notes: str | None = None,
        created_at: str = DEFAULT_VERSION_TIMESTAMP,
    ) -> OperatorVersionRecord:
        """Promote ``version`` for ``action_id``, retiring any previously promoted version intact."""
        self._require_mutable()
        hist = self._histories.get(action_id)
        if not hist:
            raise VersionStoreError(f"Unknown action_id {action_id!r} in OperatorVersionStore")

        target_idx: int | None = None
        for idx, rec in enumerate(hist):
            if rec.card.version == version:
                target_idx = idx
                break
        if target_idx is None:
            raise VersionStoreError(
                f"Version {version} not found for action_id {action_id!r}"
            )

        # Retire any currently promoted version without deleting it
        for idx, rec in enumerate(hist):
            if idx != target_idx and rec.status == "promoted":
                retired_rec = rec.with_lifecycle_state("retired")
                hist[idx] = retired_rec
                self._log_event(
                    event_kind="retire",
                    action_id=action_id,
                    version=rec.card.version,
                    from_state="promoted",
                    to_state="retired",
                    card_fingerprint=rec.card.fingerprint(),
                    timestamp=created_at,
                    reason=f"superseded_by_v{version}",
                )

        prev_state = hist[target_idx].status
        promoted_rec = hist[target_idx].with_lifecycle_state(
            "promoted",
            extra_evidence=evidence,
            contrast_pass_rate=contrast_pass_rate,
            notes=notes,
        )
        hist[target_idx] = promoted_rec
        self._log_event(
            event_kind="promote",
            action_id=action_id,
            version=version,
            from_state=prev_state,
            to_state="promoted",
            card_fingerprint=promoted_rec.card.fingerprint(),
            timestamp=created_at,
            reason=promoted_rec.notes,
        )
        return promoted_rec

    def rollback(
        self,
        action_id: str,
        target_version: int,
        *,
        reason: str = "",
        created_at: str = DEFAULT_VERSION_TIMESTAMP,
    ) -> OperatorVersionRecord:
        """Roll back ``action_id`` to ``target_version`` while preserving all versions in history."""
        self._require_mutable()
        hist = self._histories.get(action_id)
        if not hist:
            raise VersionStoreError(f"Unknown action_id {action_id!r} in OperatorVersionStore")

        target_idx: int | None = None
        for idx, rec in enumerate(hist):
            if rec.card.version == target_version:
                target_idx = idx
                break
        if target_idx is None:
            raise VersionStoreError(
                f"Cannot rollback: target_version {target_version} not found for action {action_id!r}"
            )
        if hist[target_idx].status == "shadow":
            raise VersionStoreError(
                f"Cannot rollback to unpromoted shadow version {target_version} for action {action_id!r}; "
                "only previously promoted ('retired' or 'promoted') versions may be rollback targets"
            )

        displaced_version: int | None = None
        for idx, rec in enumerate(hist):
            if idx != target_idx and rec.status == "promoted":
                displaced_version = rec.card.version
                hist[idx] = rec.with_lifecycle_state("retired")
                self._log_event(
                    event_kind="retire_on_rollback",
                    action_id=action_id,
                    version=rec.card.version,
                    from_state="promoted",
                    to_state="retired",
                    card_fingerprint=rec.card.fingerprint(),
                    timestamp=created_at,
                    reason=reason or f"rolled_back_to_v{target_version}",
                )

        prev_state = hist[target_idx].status
        restored = hist[target_idx].with_lifecycle_state(
            "promoted",
            extra_evidence={
                "last_rollback_from_version": displaced_version,
                "last_rollback_reason": reason,
            },
        )
        hist[target_idx] = restored
        self._log_event(
            event_kind="rollback",
            action_id=action_id,
            version=target_version,
            from_state=prev_state,
            to_state="promoted",
            card_fingerprint=restored.card.fingerprint(),
            timestamp=created_at,
            reason=reason or f"restored_v{target_version}",
        )
        return restored

    def get_active(self, action_id: str) -> OperatorVersionRecord | None:
        """Return the currently promoted OperatorVersionRecord for ``action_id`` (or None)."""
        for rec in self._histories.get(action_id, ()):
            if rec.status == "promoted":
                return rec
        return None

    def get_version(self, action_id: str, version: int) -> OperatorVersionRecord | None:
        """Return the specific historical record for ``(action_id, version)`` if present."""
        for rec in self._histories.get(action_id, ()):
            if rec.card.version == version:
                return rec
        return None

    def get_history(self, action_id: str) -> tuple[OperatorVersionRecord, ...]:
        """Return full append-only version history for ``action_id`` in ascending version order."""
        return tuple(self._histories.get(action_id, ()))

    def get_shadows(self, action_id: str) -> tuple[OperatorVersionRecord, ...]:
        """Return all unpromoted ``shadow`` records currently retained for ``action_id``."""
        return tuple(r for r in self._histories.get(action_id, ()) if r.status == "shadow")

    def active_cards(self) -> dict[str, OperatorCard]:
        """Return ``{action_id: promoted_card}`` sorted by ``action_id``."""
        out: dict[str, OperatorCard] = {}
        for aid in sorted(self._histories):
            active = self.get_active(aid)
            if active is not None:
                out[aid] = active.card
        return out

    def transition_log(self) -> tuple[dict[str, Any], ...]:
        """Return an immutable snapshot of the append-only lifecycle transition log."""
        return tuple(copy.deepcopy(ev) for ev in self._transition_log)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERSION_STORE_SCHEMA_VERSION,
            "frozen": self._frozen,
            "histories": {
                aid: [rec.to_dict() for rec in self._histories[aid]]
                for aid in sorted(self._histories)
            },
            "transition_log": [copy.deepcopy(ev) for ev in self._transition_log],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperatorVersionStore:
        store = cls()
        histories_raw = data.get("histories", {})
        for aid in sorted(histories_raw):
            recs = [OperatorVersionRecord.from_dict(r) for r in histories_raw[aid]]
            # Validate strictly monotonic versions and at most one promoted version
            seen_v: set[int] = set()
            promoted_count = 0
            last_v = 0
            for rec in recs:
                if rec.card.action_id != aid:
                    raise VersionStoreError(
                        f"History key {aid!r} disagrees with card action_id {rec.card.action_id!r}"
                    )
                if rec.card.version in seen_v or rec.card.version <= last_v:
                    raise VersionStoreError(
                        f"Non-monotonic or duplicate version {rec.card.version} in history for {aid!r}"
                    )
                seen_v.add(rec.card.version)
                last_v = rec.card.version
                if rec.status == "promoted":
                    promoted_count += 1
            if promoted_count > 1:
                raise VersionStoreError(
                    f"Action {aid!r} has {promoted_count} promoted versions (maximum 1 allowed)"
                )
            store._histories[aid] = recs

        store._transition_log = [copy.deepcopy(dict(ev)) for ev in data.get("transition_log", ())]
        store._frozen = bool(data.get("frozen", False))
        return store

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())
