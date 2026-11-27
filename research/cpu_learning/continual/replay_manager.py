"""Stage S8 Continual Knowledge Base, Mandatory Regression Replay & Forgetting Baselines.

STRICT ISOLATION & PYTHON 3.11 COMPATIBILITY (PDF pp. 8, 10, 12–15, 17):
- Contains ZERO hardcoded S0, D2 (Arcane), D3 (Logistics), or D4 (Biolab) domain literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Uses pure Python 3.11 standard library ONLY (zero Python 3.12+ APIs).
- ``ContinualKnowledgeBase`` + ``RegressionReplayManager`` manage heterogeneous versioned skills
  across chronological phases, enforce immutable SHA-256 verification (``rewritten_prior_skills == 0``),
  gate every candidate update on full cross-domain regression replay, automatically roll back and
  quarantine poisoned/regressing updates, and enforce ``.freeze()`` at evaluation time.
- Includes two genuine mechanistic continual forgetting baselines for head-to-head comparison:
  1. ``NaiveOverwriteNoReplayBaseline`` (``comparison_only = False``, ``uses_backprop = False``):
     unversioned, un-namespaced fixed-capacity rule table without replay gating.
  2. ``SmallCPUNeuralContinualRef`` (``comparison_only = True``, ``uses_backprop = True``):
     pure-Python shared-trunk 2-layer MLP trained sequentially with analytical backprop + SGD and
     no replay; strictly isolated from ``ContinualKnowledgeBase`` and ``RegressionReplayManager``.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.itl.version_store import OperatorVersionStore
from research.cpu_learning.simulator.state_graph import (
    canonical_json_bytes,
    compressed_bit_length,
    sha256_hex,
)

REPLAY_MANAGER_SCHEMA_VERSION: str = "s8-replay-manager-v1"

SkillLifecycleState = Literal["shadow", "promoted", "retired", "quarantined"]
_VALID_SKILL_STATES: frozenset[str] = frozenset({"shadow", "promoted", "retired", "quarantined"})


class ContinualLearnerFrozenError(RuntimeError):
    """Raised when any mutating operation is attempted on a frozen continual learner."""


# ---------------------------------------------------------------------------
# 1. Immutable Skill Version, Regression Case & Quarantine Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionedSkillEntry:
    """Immutable versioned skill record stored inside ``ContinualKnowledgeBase``."""

    domain_id: str
    skill_id: str
    kind: str
    version: int
    lifecycle_state: SkillLifecycleState
    canonical_sha256: str
    executable_dict: dict[str, Any]
    executable_object: Any
    phase_added: int
    parent_version: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.lifecycle_state not in _VALID_SKILL_STATES:
            raise ValueError(f"Invalid skill lifecycle_state: {self.lifecycle_state!r}")
        if int(self.version) < 1:
            raise ValueError(f"Skill version must be >= 1, got {self.version}")
        copied_dict = copy.deepcopy(dict(self.executable_dict))
        expected_sha = sha256_hex(canonical_json_bytes(copied_dict))
        if self.canonical_sha256 != expected_sha:
            raise ValueError(
                f"Skill SHA-256 mismatch for {self.domain_id}:{self.skill_id} v{self.version}"
            )
        object.__setattr__(self, "executable_dict", copied_dict)

    def with_state(self, new_state: SkillLifecycleState, *, notes: str | None = None) -> VersionedSkillEntry:
        return VersionedSkillEntry(
            domain_id=self.domain_id,
            skill_id=self.skill_id,
            kind=self.kind,
            version=self.version,
            lifecycle_state=new_state,
            canonical_sha256=self.canonical_sha256,
            executable_dict=self.executable_dict,
            executable_object=self.executable_object,
            phase_added=self.phase_added,
            parent_version=self.parent_version,
            notes=self.notes if notes is None else str(notes),
        )

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "skill_id": self.skill_id,
            "kind": self.kind,
            "version": self.version,
            "lifecycle_state": self.lifecycle_state,
            "canonical_sha256": self.canonical_sha256,
            "phase_added": self.phase_added,
            "parent_version": self.parent_version,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DomainRegressionCase:
    """Single immutable archived regression test case protecting a learned domain."""

    case_id: str
    domain_id: str
    evaluator_fn: Callable[[ContinualKnowledgeBase], tuple[bool, str]]
    case_payload_bytes: bytes = b""

    def compressed_bits(self) -> int:
        if self.case_payload_bytes:
            return compressed_bit_length(self.case_payload_bytes)
        return compressed_bit_length(
            canonical_json_bytes({"case_id": self.case_id, "domain_id": self.domain_id})
        )


@dataclass(frozen=True)
class ReplayVerificationReport:
    """Result of executing archived regression suites across all registered domains."""

    passed: bool
    total_cases_checked: int
    total_failures: int
    per_domain_pass_rate: dict[str, float]
    failed_domains: tuple[str, ...]
    offending_case_ids: tuple[str, ...]
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_cases_checked": self.total_cases_checked,
            "total_failures": self.total_failures,
            "per_domain_pass_rate": {
                k: round(v, 6) for k, v in sorted(self.per_domain_pass_rate.items())
            },
            "failed_domains": list(self.failed_domains),
            "offending_case_ids": list(self.offending_case_ids),
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class QuarantineRecord:
    """Immutable provenance record when a faulty candidate update fails regression replay."""

    quarantine_id: str
    domain_id: str
    skill_id: str
    kind: str
    candidate_version: int
    candidate_sha256: str
    restored_version: int | None
    restored_sha256: str | None
    failed_domains: tuple[str, ...]
    offending_case_ids: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    phase_index: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "quarantine_id": self.quarantine_id,
            "domain_id": self.domain_id,
            "skill_id": self.skill_id,
            "kind": self.kind,
            "candidate_version": self.candidate_version,
            "candidate_sha256": self.candidate_sha256,
            "restored_version": self.restored_version,
            "restored_sha256": self.restored_sha256,
            "failed_domains": list(self.failed_domains),
            "offending_case_ids": list(self.offending_case_ids),
            "failure_reasons": list(self.failure_reasons),
            "phase_index": self.phase_index,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CandidateUpdateOutcome:
    """Complete audit report for ``apply_candidate_update_with_replay``."""

    accepted: bool
    rolled_back: bool
    quarantined: bool
    domain_id: str
    skill_id: str
    candidate_version: int
    active_version_after: int | None
    active_sha256_after: str | None
    replay_on_candidate: ReplayVerificationReport
    replay_after_resolution: ReplayVerificationReport
    quarantine_record: QuarantineRecord | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rolled_back": self.rolled_back,
            "quarantined": self.quarantined,
            "domain_id": self.domain_id,
            "skill_id": self.skill_id,
            "candidate_version": self.candidate_version,
            "active_version_after": self.active_version_after,
            "active_sha256_after": self.active_sha256_after,
            "replay_on_candidate": self.replay_on_candidate.to_dict(),
            "replay_after_resolution": self.replay_after_resolution.to_dict(),
            "quarantine_record": (
                None if self.quarantine_record is None else self.quarantine_record.to_dict()
            ),
        }


@dataclass(frozen=True)
class LibraryGrowthSnapshot:
    """Sub-linear library growth snapshot recorded at chronological phase ``phase_index``."""

    phase_index: int
    phase_name: str
    domains_registered: tuple[str, ...]
    active_skill_count: int
    cumulative_raw_training_bits: int
    cumulative_compressed_training_bits: int
    active_library_bits: int
    episode_ledger_bits: int

    @property
    def library_to_raw_ratio(self) -> float:
        return self.active_library_bits / float(max(1, self.cumulative_raw_training_bits))

    @property
    def library_to_compressed_train_ratio(self) -> float:
        return self.active_library_bits / float(max(1, self.cumulative_compressed_training_bits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_index": self.phase_index,
            "phase_name": self.phase_name,
            "domains_registered": list(self.domains_registered),
            "active_skill_count": self.active_skill_count,
            "cumulative_raw_training_bits": self.cumulative_raw_training_bits,
            "cumulative_compressed_training_bits": self.cumulative_compressed_training_bits,
            "active_library_bits": self.active_library_bits,
            "episode_ledger_bits": self.episode_ledger_bits,
            "library_to_raw_ratio": round(self.library_to_raw_ratio, 6),
            "library_to_compressed_train_ratio": round(self.library_to_compressed_train_ratio, 6),
        }


# ---------------------------------------------------------------------------
# 2. Regression Replay Manager & Continual Knowledge Base
# ---------------------------------------------------------------------------


class RegressionReplayManager:
    """Append-only cross-domain regression replay archive enforcing zero silent regressions."""

    def __init__(self) -> None:
        self._cases_by_domain: dict[str, list[DomainRegressionCase]] = {}
        self._seen_case_ids: set[str] = set()
        self._frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def domain_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._cases_by_domain.keys()))

    def has_cases_for(self, domain_id: str) -> bool:
        return bool(self._cases_by_domain.get(str(domain_id)))

    def register_domain_regression_cases(
        self,
        domain_id: str,
        cases: Sequence[DomainRegressionCase],
    ) -> int:
        if self._frozen:
            raise ContinualLearnerFrozenError("RegressionReplayManager is frozen")
        bucket = self._cases_by_domain.setdefault(str(domain_id), [])
        added = 0
        for c in cases:
            if c.domain_id != domain_id:
                raise ValueError(
                    f"Case {c.case_id!r} domain_id {c.domain_id!r} != {domain_id!r}"
                )
            if c.case_id in self._seen_case_ids:
                raise ValueError(f"Duplicate regression case_id {c.case_id!r}")
            self._seen_case_ids.add(c.case_id)
            bucket.append(c)
            added += 1
        return added

    def run_all_prior_regressions(
        self,
        kb: ContinualKnowledgeBase,
        *,
        domains: Sequence[str] | None = None,
    ) -> ReplayVerificationReport:
        target_domains = (
            tuple(sorted(self._cases_by_domain.keys()))
            if domains is None
            else tuple(domains)
        )
        total_checked = 0
        total_failures = 0
        per_dom_rates: dict[str, float] = {}
        failed_doms: list[str] = []
        offending_ids: list[str] = []
        reasons: list[str] = []

        for dom in target_domains:
            dom_cases = self._cases_by_domain.get(dom, [])
            if not dom_cases:
                per_dom_rates[dom] = 1.0
                continue
            hits = 0
            for c in dom_cases:
                total_checked += 1
                try:
                    ok, detail = c.evaluator_fn(kb)
                except Exception as exc:
                    ok, detail = (False, f"exception:{type(exc).__name__}:{exc}")
                if ok:
                    hits += 1
                else:
                    total_failures += 1
                    offending_ids.append(c.case_id)
                    reasons.append(f"{dom}:{c.case_id}:{detail}")
            rate = hits / float(len(dom_cases))
            per_dom_rates[dom] = rate
            if hits < len(dom_cases):
                failed_doms.append(dom)

        return ReplayVerificationReport(
            passed=(total_failures == 0),
            total_cases_checked=total_checked,
            total_failures=total_failures,
            per_domain_pass_rate=per_dom_rates,
            failed_domains=tuple(failed_doms),
            offending_case_ids=tuple(offending_ids),
            failure_reasons=tuple(reasons),
        )

    def episode_ledger_bits(self) -> int:
        payload = [
            {
                "domain_id": dom,
                "case_id": c.case_id,
                "payload_hex": c.case_payload_bytes.hex(),
            }
            for dom in sorted(self._cases_by_domain.keys())
            for c in self._cases_by_domain[dom]
        ]
        if not payload:
            return 0
        return compressed_bit_length(canonical_json_bytes(payload))


class ContinualKnowledgeBase:
    """Unified CPU-native continual skill archive across relational, math/list, code, and UTF-8 domains."""

    def __init__(self, *, replay_manager: RegressionReplayManager | None = None) -> None:
        self.replay_manager = (
            replay_manager if replay_manager is not None else RegressionReplayManager()
        )
        # (domain_id, skill_id) -> append-only list[VersionedSkillEntry]
        self._histories: dict[tuple[str, str], list[VersionedSkillEntry]] = {}
        self._domain_order: list[str] = []
        # Dedicated OperatorVersionStore per relational domain
        self._relational_stores: dict[str, OperatorVersionStore] = {}
        # Auxiliary domain metadata (e.g. learned negation/condition/connector rules for UTF-8)
        self._domain_aux: dict[str, dict[str, Any]] = {}
        # Immutable cryptographic audit table: (domain_id, skill_id, version) -> (sha256, initial_phase)
        self._immutable_version_hashes: dict[tuple[str, str, int], tuple[str, int]] = {}
        self._initial_promoted_hash_by_key: dict[tuple[str, str], tuple[str, int]] = {}
        self._quarantine_log: list[QuarantineRecord] = []
        self._growth_timeline: list[LibraryGrowthSnapshot] = []
        self._shared_concepts_payload: list[dict[str, Any]] = []
        self._frozen: bool = False
        self._q_seq: int = 0

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True
        self.replay_manager.freeze()
        for store in self._relational_stores.values():
            if not store.is_frozen:
                store.freeze()

    def _require_mutable(self) -> None:
        if self._frozen:
            raise ContinualLearnerFrozenError(
                "ContinualKnowledgeBase is frozen; post-freeze mutations are forbidden"
            )

    def domain_ids(self) -> tuple[str, ...]:
        return tuple(self._domain_order)

    @property
    def quarantine_log(self) -> tuple[QuarantineRecord, ...]:
        return tuple(self._quarantine_log)

    @property
    def growth_timeline(self) -> tuple[LibraryGrowthSnapshot, ...]:
        return tuple(self._growth_timeline)

    def relational_store_for(self, domain_id: str) -> OperatorVersionStore | None:
        return self._relational_stores.get(domain_id)

    def set_domain_aux(self, domain_id: str, key: str, value: Any) -> None:
        self._require_mutable()
        self._domain_aux.setdefault(str(domain_id), {})[str(key)] = value

    def get_domain_aux(self, domain_id: str, key: str, default: Any = None) -> Any:
        return self._domain_aux.get(domain_id, {}).get(key, default)

    def register_shared_concepts_payload(self, concepts_payload: Sequence[Mapping[str, Any]]) -> None:
        self._require_mutable()
        self._shared_concepts_payload = [copy.deepcopy(dict(x)) for x in concepts_payload]

    def _next_version(self, domain_id: str, skill_id: str) -> int:
        hist = self._histories.get((domain_id, skill_id))
        if not hist:
            return 1
        return max(e.version for e in hist) + 1

    def register_initial_skill(
        self,
        *,
        domain_id: str,
        skill_id: str,
        kind: str,
        executable_dict: Mapping[str, Any],
        executable_object: Any,
        phase_index: int,
        notes: str = "initial_verified_promotion",
    ) -> VersionedSkillEntry:
        """Register and promote an initial verified skill in ``domain_id`` (non-destructive)."""
        self._require_mutable()
        dom = str(domain_id)
        sid = str(skill_id)
        if dom not in self._domain_order:
            self._domain_order.append(dom)
        key = (dom, sid)
        if key in self._histories:
            raise ValueError(
                f"Skill {dom}:{sid} already exists; use apply_candidate_update_with_replay()"
            )

        exec_copy = copy.deepcopy(dict(executable_dict))
        digest = sha256_hex(canonical_json_bytes(exec_copy))
        entry = VersionedSkillEntry(
            domain_id=dom,
            skill_id=sid,
            kind=str(kind),
            version=1,
            lifecycle_state="promoted",
            canonical_sha256=digest,
            executable_dict=exec_copy,
            executable_object=executable_object,
            phase_added=int(phase_index),
            parent_version=None,
            notes=str(notes),
        )
        self._histories[key] = [entry]
        self._immutable_version_hashes[(dom, sid, 1)] = (digest, int(phase_index))
        self._initial_promoted_hash_by_key[key] = (digest, int(phase_index))

        if kind == "relational_operator" and isinstance(executable_object, OperatorCard):
            store = self._relational_stores.setdefault(dom, OperatorVersionStore())
            card_v1 = (
                executable_object
                if executable_object.version == 1
                else OperatorCard.from_dict({**executable_object.to_dict(), "version": 1, "parent_version": None})
            )
            store.register_shadow(card_v1, notes=notes)
            store.promote(card_v1.action_id, 1, notes=notes)

        return entry

    def get_active_skill(self, domain_id: str, skill_id: str) -> VersionedSkillEntry | None:
        hist = self._histories.get((domain_id, skill_id), [])
        for entry in reversed(hist):
            if entry.lifecycle_state == "promoted":
                return entry
        return None

    def get_skill_history(self, domain_id: str, skill_id: str) -> tuple[VersionedSkillEntry, ...]:
        return tuple(self._histories.get((domain_id, skill_id), ()))

    def get_active_domain_objects(
        self,
        domain_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for (dom, sid), hist in sorted(self._histories.items()):
            if dom != domain_id:
                continue
            for entry in reversed(hist):
                if entry.lifecycle_state == "promoted":
                    if kind is None or entry.kind == kind:
                        out[sid] = entry.executable_object
                    break
        return out

    def apply_candidate_update_with_replay(
        self,
        *,
        domain_id: str,
        skill_id: str,
        kind: str,
        executable_dict: Mapping[str, Any],
        executable_object: Any,
        phase_index: int,
        notes: str = "",
    ) -> CandidateUpdateOutcome:
        """Stage a candidate skill update, run ALL archived cross-domain regression suites,
        and automatically roll back + quarantine on ANY regression failure."""
        self._require_mutable()
        dom = str(domain_id)
        sid = str(skill_id)
        if dom not in self._domain_order:
            self._domain_order.append(dom)
        key = (dom, sid)
        hist = self._histories.setdefault(key, [])

        prev_idx: int | None = None
        prev_entry: VersionedSkillEntry | None = None
        for idx, e in enumerate(hist):
            if e.lifecycle_state == "promoted":
                prev_idx = idx
                prev_entry = e
                break

        cand_ver = self._next_version(dom, sid)
        exec_copy = copy.deepcopy(dict(executable_dict))
        cand_sha = sha256_hex(canonical_json_bytes(exec_copy))

        # Provisionally retire previous promoted entry (non-destructively) and promote candidate
        if prev_idx is not None and prev_entry is not None:
            hist[prev_idx] = prev_entry.with_state("retired")

        cand_obj = executable_object
        rel_store: OperatorVersionStore | None = None
        if kind == "relational_operator" and isinstance(executable_object, OperatorCard):
            rel_store = self._relational_stores.setdefault(dom, OperatorVersionStore())
            store_ver = rel_store.next_version(executable_object.action_id)
            cand_obj = OperatorCard.from_dict(
                {
                    **executable_object.to_dict(),
                    "version": store_ver,
                    "parent_version": None if prev_entry is None else prev_entry.version,
                }
            )
            rel_store.register_shadow(cand_obj, notes=notes or "candidate_update")
            rel_store.promote(cand_obj.action_id, store_ver, notes=notes or "provisional_replay_check")

        cand_entry = VersionedSkillEntry(
            domain_id=dom,
            skill_id=sid,
            kind=str(kind),
            version=cand_ver,
            lifecycle_state="promoted",
            canonical_sha256=cand_sha,
            executable_dict=exec_copy,
            executable_object=cand_obj,
            phase_added=int(phase_index),
            parent_version=None if prev_entry is None else prev_entry.version,
            notes=str(notes),
        )
        hist.append(cand_entry)
        cand_hist_idx = len(hist) - 1
        self._immutable_version_hashes[(dom, sid, cand_ver)] = (cand_sha, int(phase_index))

        # Fail closed if any registered domain with promoted skills lacks archived regression cases!
        unprotected_domains = tuple(
            sorted(
                d_name
                for d_name in self._domain_order
                if any(k_dom == d_name for (k_dom, _s) in self._initial_promoted_hash_by_key)
                and not self.replay_manager.has_cases_for(d_name)
            )
        )
        replay_cand = self.replay_manager.run_all_prior_regressions(self)
        if unprotected_domains or replay_cand.total_cases_checked == 0:
            miss_reasons = tuple(
                f"{d_name}:missing_mandatory_regression_suite" for d_name in (unprotected_domains or (dom,))
            )
            replay_cand = ReplayVerificationReport(
                passed=False,
                total_cases_checked=replay_cand.total_cases_checked,
                total_failures=replay_cand.total_failures + len(miss_reasons),
                per_domain_pass_rate=replay_cand.per_domain_pass_rate,
                failed_domains=tuple(sorted(set(replay_cand.failed_domains) | set(unprotected_domains or (dom,)))),
                offending_case_ids=replay_cand.offending_case_ids,
                failure_reasons=replay_cand.failure_reasons + miss_reasons,
            )

        if replay_cand.passed:
            if key not in self._initial_promoted_hash_by_key:
                self._initial_promoted_hash_by_key[key] = (cand_sha, int(phase_index))
            return CandidateUpdateOutcome(
                accepted=True,
                rolled_back=False,
                quarantined=False,
                domain_id=dom,
                skill_id=sid,
                candidate_version=cand_ver,
                active_version_after=cand_ver,
                active_sha256_after=cand_sha,
                replay_on_candidate=replay_cand,
                replay_after_resolution=replay_cand,
                quarantine_record=None,
            )

        # Regression detected -> Automatic Rollback + Quarantine!
        hist[cand_hist_idx] = cand_entry.with_state(
            "quarantined",
            notes=f"quarantined_regression:{','.join(replay_cand.failed_domains)}",
        )
        restored_ver: int | None = None
        restored_sha: str | None = None
        if prev_idx is not None and prev_entry is not None:
            hist[prev_idx] = prev_entry.with_state("promoted")
            restored_ver = prev_entry.version
            restored_sha = prev_entry.canonical_sha256
            if rel_store is not None and isinstance(prev_entry.executable_object, OperatorCard):
                rel_store.rollback(
                    prev_entry.executable_object.action_id,
                    prev_entry.executable_object.version,
                    reason="auto_rollback_on_regression_replay_failure",
                )
        elif rel_store is not None and isinstance(cand_obj, OperatorCard):
            rel_store.retire_active(
                cand_obj.action_id,
                reason="quarantined_initial_candidate_on_regression",
            )

        self._q_seq += 1
        q_rec = QuarantineRecord(
            quarantine_id=f"quarantine_{self._q_seq:04d}",
            domain_id=dom,
            skill_id=sid,
            kind=str(kind),
            candidate_version=cand_ver,
            candidate_sha256=cand_sha,
            restored_version=restored_ver,
            restored_sha256=restored_sha,
            failed_domains=replay_cand.failed_domains,
            offending_case_ids=replay_cand.offending_case_ids,
            failure_reasons=replay_cand.failure_reasons,
            phase_index=int(phase_index),
            notes=str(notes),
        )
        self._quarantine_log.append(q_rec)

        replay_after = self.replay_manager.run_all_prior_regressions(self)
        return CandidateUpdateOutcome(
            accepted=False,
            rolled_back=True,
            quarantined=True,
            domain_id=dom,
            skill_id=sid,
            candidate_version=cand_ver,
            active_version_after=restored_ver,
            active_sha256_after=restored_sha,
            replay_on_candidate=replay_cand,
            replay_after_resolution=replay_after,
            quarantine_record=q_rec,
        )

    def rollback_skill(
        self,
        domain_id: str,
        skill_id: str,
        target_version: int,
        *,
        reason: str = "manual_rollback",
    ) -> VersionedSkillEntry:
        """Explicitly roll back ``(domain_id, skill_id)`` to ``target_version`` without deleting history."""
        self._require_mutable()
        dom = str(domain_id)
        sid = str(skill_id)
        hist = self._histories.get((dom, sid))
        if not hist:
            raise KeyError(f"Unknown skill {dom}:{sid}")

        target_idx: int | None = None
        for idx, e in enumerate(hist):
            if e.version == int(target_version):
                target_idx = idx
                break
        if target_idx is None:
            raise ValueError(f"Version {target_version} not found for {dom}:{sid}")

        target_entry = hist[target_idx]
        if target_entry.lifecycle_state in {"quarantined", "shadow"}:
            raise ValueError(
                f"Cannot roll back {dom}:{sid} to {target_entry.lifecycle_state!r} "
                f"version {target_version}; only verified promoted/retired versions may be restored"
            )

        for idx, e in enumerate(hist):
            if idx != target_idx and e.lifecycle_state == "promoted":
                hist[idx] = e.with_state("retired", notes=reason)
        restored = hist[target_idx].with_state("promoted", notes=reason)
        hist[target_idx] = restored

        rel_store = self._relational_stores.get(dom)
        if rel_store is not None and isinstance(restored.executable_object, OperatorCard):
            rel_store.rollback(
                restored.executable_object.action_id,
                restored.executable_object.version,
                reason=reason,
            )
        return restored

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_MANAGER_SCHEMA_VERSION,
            "is_frozen": self._frozen,
            "domains": list(self._domain_order),
            "histories": {
                f"{dom}:{sid}": [e.to_summary_dict() for e in hist]
                for (dom, sid), hist in sorted(self._histories.items())
            },
            "quarantine_log": [q.to_dict() for q in self._quarantine_log],
            "growth_timeline": [g.to_dict() for g in self._growth_timeline],
        }

    def audit_non_destructive_hashes(self, *, max_prior_phase: int | None = None) -> dict[str, Any]:
        """Cryptographically verify that ZERO historical versions were mutated or deleted and
        that all prior-phase promoted skills remain intact."""
        rewritten_versions = 0
        deleted_versions = 0
        displaced_prior_skills = 0
        total_versions = 0
        active_count = 0
        quarantined_count = 0

        for (dom, sid), hist in sorted(self._histories.items()):
            seen_vers: set[int] = set()
            for entry in hist:
                total_versions += 1
                seen_vers.add(entry.version)
                recomputed = sha256_hex(canonical_json_bytes(entry.executable_dict))
                rec_info = self._immutable_version_hashes.get((dom, sid, entry.version))
                if rec_info is None or rec_info[0] != recomputed or entry.canonical_sha256 != recomputed:
                    rewritten_versions += 1
                if entry.lifecycle_state == "promoted":
                    active_count += 1
                elif entry.lifecycle_state == "quarantined":
                    quarantined_count += 1

            expected_vers = {
                ver for (d_k, s_k, ver) in self._immutable_version_hashes if d_k == dom and s_k == sid
            }
            if expected_vers - seen_vers:
                deleted_versions += len(expected_vers - seen_vers)

            init_info = self._initial_promoted_hash_by_key.get((dom, sid))
            if init_info is not None:
                init_sha, init_phase = init_info
                if max_prior_phase is None or init_phase <= max_prior_phase:
                    cur_active = self.get_active_skill(dom, sid)
                    if cur_active is None or cur_active.canonical_sha256 != init_sha:
                        displaced_prior_skills += 1

        rewritten_total = rewritten_versions + deleted_versions + displaced_prior_skills
        return {
            "rewritten_prior_skills": rewritten_total,
            "rewritten_versions": rewritten_versions,
            "deleted_versions": deleted_versions,
            "displaced_prior_skills": displaced_prior_skills,
            "total_versions_archived": total_versions,
            "active_promoted_skills": active_count,
            "quarantined_candidates": quarantined_count,
            "non_destructive_verified": (rewritten_total == 0),
        }

    def active_library_compressed_bits(self) -> int:
        """Compute unified single-stream compressed description bits of all active skills + shared concepts."""
        active_entries: list[dict[str, Any]] = []
        for (dom, sid), hist in sorted(self._histories.items()):
            for entry in reversed(hist):
                if entry.lifecycle_state == "promoted":
                    active_entries.append(
                        {
                            "domain_id": dom,
                            "skill_id": sid,
                            "kind": entry.kind,
                            "payload": entry.executable_dict,
                        }
                    )
                    break
        archive = {
            "shared_concepts": self._shared_concepts_payload,
            "active_skills": active_entries,
            "domain_aux": {
                dom: {k: v for k, v in sorted(kv.items()) if k.endswith("_serializable")}
                for dom, kv in sorted(self._domain_aux.items())
            },
        }
        return compressed_bit_length(canonical_json_bytes(archive))

    def record_phase_growth_snapshot(
        self,
        *,
        phase_index: int,
        phase_name: str,
        cumulative_raw_training_bits: int,
        cumulative_compressed_training_bits: int,
    ) -> LibraryGrowthSnapshot:
        self._require_mutable()
        active_cnt = sum(
            1
            for (dom, sid) in self._histories
            if self.get_active_skill(dom, sid) is not None
        )
        snap = LibraryGrowthSnapshot(
            phase_index=int(phase_index),
            phase_name=str(phase_name),
            domains_registered=self.domain_ids(),
            active_skill_count=active_cnt,
            cumulative_raw_training_bits=int(cumulative_raw_training_bits),
            cumulative_compressed_training_bits=int(cumulative_compressed_training_bits),
            active_library_bits=self.active_library_compressed_bits(),
            episode_ledger_bits=self.replay_manager.episode_ledger_bits(),
        )
        self._growth_timeline.append(snap)
        return snap


# ---------------------------------------------------------------------------
# 3. Honest Baseline 1: Naive Overwrite Without Replay or Versioned Namespaces
# ---------------------------------------------------------------------------


class NaiveOverwriteNoReplayBaseline:
    """Mechanistic unversioned, fixed-capacity continual baseline without replay or namespace isolation.

    Stores active rules in a shared bounded slot table keyed only by ``slot_key`` (without domain
    namespace isolation or version history) and applies every incoming phase / candidate update
    directly without regression replay. As new phases arrive, shared slot collisions and FIFO
    capacity eviction genuinely overwrite earlier domain skills.
    """

    comparison_only: bool = False
    uses_backprop: bool = False

    def __init__(self, *, max_active_slots: int = 12) -> None:
        self.max_active_slots = int(max_active_slots)
        # shared_slot_key -> (domain_id, skill_id, kind, executable_object)
        self._table: OrderedDict[str, tuple[str, str, str, Any]] = OrderedDict()
        self._domain_aux: dict[str, dict[str, Any]] = {}
        self._frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def write_skill_unversioned(
        self,
        *,
        domain_id: str,
        skill_id: str,
        kind: str,
        shared_slot_key: str,
        executable_object: Any,
    ) -> None:
        if self._frozen:
            raise ContinualLearnerFrozenError("NaiveOverwriteNoReplayBaseline is frozen")
        # Overwrite existing shared_slot_key in-place without version backup or replay!
        if shared_slot_key in self._table:
            del self._table[shared_slot_key]
        self._table[shared_slot_key] = (
            str(domain_id),
            str(skill_id),
            str(kind),
            executable_object,
        )
        while len(self._table) > self.max_active_slots:
            self._table.popitem(last=False)

    def set_domain_aux(self, domain_id: str, key: str, value: Any) -> None:
        if self._frozen:
            raise ContinualLearnerFrozenError("NaiveOverwriteNoReplayBaseline is frozen")
        self._domain_aux.setdefault(str(domain_id), {})[str(key)] = value

    def get_domain_aux(self, domain_id: str, key: str, default: Any = None) -> Any:
        return self._domain_aux.get(domain_id, {}).get(key, default)

    def get_active_domain_objects(
        self,
        domain_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for _skey, (dom, sid, k_str, obj) in self._table.items():
            if dom == domain_id and (kind is None or k_str == kind):
                out[sid] = obj
        return out


# ---------------------------------------------------------------------------
# 4. Comparison-Only Baseline 2: Small CPU Neural Continual MLP (SGD, No Replay)
# ---------------------------------------------------------------------------


def _softmax_vec(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    inv_s = 1.0 / sum(exps)
    return [e * inv_s for e in exps]


def encode_hashed_feature_vector(tokens: Sequence[str], dim: int = 48) -> list[float]:
    """Deterministic signed feature-hashing projection onto unit-norm ``R^dim`` (Python 3.11)."""
    vec = [0.0] * dim
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = ((digest[0] << 8) | digest[1]) % dim
        sign = 1.0 if (digest[2] & 1) == 0 else -1.0
        vec[idx] += sign
        idx2 = ((digest[3] << 8) | digest[4]) % dim
        sign2 = 1.0 if (digest[5] & 1) == 0 else -1.0
        vec[idx2] += 0.5 * sign2
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-12:
        inv_n = 1.0 / norm
        vec = [v * inv_n for v in vec]
    return vec


class SmallCPUNeuralContinualRef:
    """Comparison-only pure-Python 2-layer shared-trunk MLP trained sequentially across
    chronological phases with analytical backpropagation + SGD and ZERO replay.

    STRICT ISOLATION TAG: ``comparison_only = True`` — NEVER imported or invoked by
    ``ContinualKnowledgeBase``, ``RegressionReplayManager``, or ``cross_domain.py``!
    """

    model_id: str = "small_cpu_neural_continual_ref"
    comparison_only: bool = True
    uses_backprop: bool = True

    def __init__(
        self,
        *,
        seed: int = 101,
        input_dim: int = 48,
        hidden_dim: int = 24,
        num_classes: int = 16,
        lr: float = 0.08,
    ) -> None:
        self.seed = int(seed)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.lr = float(lr)
        self._frozen: bool = False

        rng = random.Random(self.seed * 613 + 29)
        s1 = 1.0 / math.sqrt(float(self.input_dim))
        s2 = 1.0 / math.sqrt(float(self.hidden_dim))
        self.w1: list[list[float]] = [
            [rng.gauss(0.0, s1) for _ in range(self.input_dim)]
            for _ in range(self.hidden_dim)
        ]
        self.b1: list[float] = [0.0] * self.hidden_dim
        self.w2: list[list[float]] = [
            [rng.gauss(0.0, s2) for _ in range(self.hidden_dim)]
            for _ in range(self.num_classes)
        ]
        self.b2: list[float] = [0.0] * self.num_classes

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def _forward(self, x: Sequence[float]) -> tuple[list[float], list[float]]:
        h: list[float] = []
        for j in range(self.hidden_dim):
            w1j = self.w1[j]
            acc = self.b1[j]
            for i in range(self.input_dim):
                acc += w1j[i] * x[i]
            h.append(math.tanh(acc))

        logits: list[float] = []
        for k in range(self.num_classes):
            w2k = self.w2[k]
            acc = self.b2[k]
            for j in range(self.hidden_dim):
                acc += w2k[j] * h[j]
            logits.append(acc)

        return h, _softmax_vec(logits)

    def compute_step_grads(
        self,
        x: Sequence[float],
        target_class: int,
    ) -> tuple[float, dict[str, Any]]:
        """Exact analytical loss (nats) and gradients ``(dw1, db1, dw2, db2)`` for one example."""
        h, probs = self._forward(x)
        loss = -math.log(max(probs[target_class], 1e-15))

        dz2 = list(probs)
        dz2[target_class] -= 1.0

        dw2 = [[dz2[k] * h[j] for j in range(self.hidden_dim)] for k in range(self.num_classes)]
        db2 = list(dz2)

        dh = [0.0] * self.hidden_dim
        for k in range(self.num_classes):
            dzk = dz2[k]
            w2k = self.w2[k]
            for j in range(self.hidden_dim):
                dh[j] += dzk * w2k[j]

        dz1 = [dh[j] * (1.0 - h[j] * h[j]) for j in range(self.hidden_dim)]
        dw1 = [[dz1[j] * x[i] for i in range(self.input_dim)] for j in range(self.hidden_dim)]
        db1 = list(dz1)
        return (
            loss,
            {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2},
        )

    def fit_phase_sequential(
        self,
        examples: Sequence[tuple[Sequence[float], int]],
        *,
        epochs: int = 12,
        phase_seed_offset: int = 0,
    ) -> None:
        """Train shared weights sequentially on ONE phase's examples with SGD and NO replay."""
        if self._frozen:
            raise ContinualLearnerFrozenError("SmallCPUNeuralContinualRef is frozen")
        rng = random.Random(self.seed * 97 + phase_seed_offset)
        order = list(range(len(examples)))
        lr = self.lr
        for _ep in range(epochs):
            rng.shuffle(order)
            for idx in order:
                x, y = examples[idx]
                _, gr = self.compute_step_grads(x, y)
                dw2 = gr["dw2"]
                db2 = gr["db2"]
                for k in range(self.num_classes):
                    self.b2[k] -= lr * db2[k]
                    w2k = self.w2[k]
                    gw2k = dw2[k]
                    for j in range(self.hidden_dim):
                        w2k[j] -= lr * gw2k[j]
                dw1 = gr["dw1"]
                db1 = gr["db1"]
                for j in range(self.hidden_dim):
                    self.b1[j] -= lr * db1[j]
                    w1j = self.w1[j]
                    gw1j = dw1[j]
                    for i in range(self.input_dim):
                        w1j[i] -= lr * gw1j[i]

    def predict_class(self, x: Sequence[float]) -> int:
        _, probs = self._forward(x)
        best_k = 0
        best_p = probs[0]
        for k in range(1, self.num_classes):
            if probs[k] > best_p:
                best_p = probs[k]
                best_k = k
        return best_k

    def evaluate_accuracy(self, examples: Sequence[tuple[Sequence[float], int]]) -> float:
        if not examples:
            return 1.0
        hits = sum(1 for x, y in examples if self.predict_class(x) == y)
        return hits / float(len(examples))
