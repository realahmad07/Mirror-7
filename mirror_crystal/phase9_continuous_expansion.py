"""Phase 9: bounded, persistent and regression-gated capability expansion.

Each epoch consumes a passing Phase 8 transfer study, checks fresh task/execution
identities, replays a frozen regression suite, and appends an immutable hash-chained
decision. Promotion remains an explicit separate operation. This is a controlled
expansion mechanism, not unrestricted open-world learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .capability_model import CapabilityRegistry
from .execution import NativeOperationRegistry
from .operational_language import UOL
from .phase8_cross_domain_acceptance import (
    CrossDomainTransferBenchmarkReport,
    TransferBenchmarkStatus,
)
from .verified_program_learning import (
    UOLProgramLearningProposal,
    VerifiedProgramLearningEngine,
)


class ContinuousExpansionError(ValueError):
    """Raised when an expansion epoch or ledger operation is invalid."""


class ExpansionDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class FrozenRegressionCase:
    case_id: str
    capability_id: str
    inputs: Mapping[str, Any]
    expected: Any

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ContinuousExpansionError("regression case_id must be non-empty")
        if not isinstance(self.capability_id, str) or not self.capability_id.strip():
            raise ContinuousExpansionError("regression capability_id must be non-empty")
        if not isinstance(self.inputs, Mapping):
            raise TypeError("regression inputs must be a mapping")
        frozen_inputs = dict(self.inputs)
        _canonical_json({"inputs": frozen_inputs, "expected": self.expected})
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "capability_id", self.capability_id.strip())
        object.__setattr__(self, "inputs", frozen_inputs)


@dataclass(frozen=True)
class ExpansionLedgerEntry:
    sequence: int
    epoch_id: str
    decision: ExpansionDecision
    manifest_sha256: str
    payload: Mapping[str, Any]
    previous_digest: str
    digest: str


class ExpansionLedger:
    """Append-only SQLite hash-chain ledger for expansion decisions."""

    GENESIS = "0" * 64

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=5.0, isolation_level=None, check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=FULL")
            if self.path != ":memory:":
                self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS expansion_epochs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    epoch_id TEXT NOT NULL UNIQUE,
                    decision TEXT NOT NULL CHECK(decision IN ('accepted','rejected')),
                    manifest_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_digest TEXT NOT NULL,
                    digest TEXT NOT NULL UNIQUE
                )
                """
            )
            self._connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS expansion_epochs_no_update
                BEFORE UPDATE ON expansion_epochs
                BEGIN SELECT RAISE(ABORT, 'expansion ledger is append-only'); END
                """
            )
            self._connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS expansion_epochs_no_delete
                BEFORE DELETE ON expansion_epochs
                BEGIN SELECT RAISE(ABORT, 'expansion ledger is append-only'); END
                """
            )
            if not self.verify_chain():
                raise ContinuousExpansionError("expansion ledger hash chain is invalid")

    @staticmethod
    def _digest(
        sequence: int,
        epoch_id: str,
        decision: str,
        manifest_sha256: str,
        payload_json: str,
        previous_digest: str,
    ) -> str:
        encoded = _canonical_json({
            "sequence": sequence,
            "epoch_id": epoch_id,
            "decision": decision,
            "manifest_sha256": manifest_sha256,
            "payload_json": payload_json,
            "previous_digest": previous_digest,
        })
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def append(
        self,
        epoch_id: str,
        decision: ExpansionDecision,
        manifest_sha256: str,
        payload: Mapping[str, Any],
    ) -> ExpansionLedgerEntry:
        if not isinstance(epoch_id, str) or not epoch_id.strip():
            raise ContinuousExpansionError("epoch_id must be non-empty")
        if not isinstance(decision, ExpansionDecision):
            raise TypeError("decision must be ExpansionDecision")
        if not _is_digest(manifest_sha256):
            raise ContinuousExpansionError("manifest_sha256 must be a lowercase SHA-256 digest")
        payload_json = _canonical_json(dict(payload))
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                if not self.verify_chain():
                    raise ContinuousExpansionError("cannot append to an invalid ledger")
                duplicate = connection.execute(
                    "SELECT 1 FROM expansion_epochs WHERE epoch_id = ?",
                    (epoch_id.strip(),),
                ).fetchone()
                if duplicate:
                    raise ContinuousExpansionError("epoch_id already exists in the ledger")
                previous = connection.execute(
                    "SELECT sequence, digest FROM expansion_epochs ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = int(previous["sequence"]) + 1 if previous else 1
                previous_digest = str(previous["digest"]) if previous else self.GENESIS
                digest = self._digest(
                    sequence, epoch_id.strip(), decision.value, manifest_sha256,
                    payload_json, previous_digest,
                )
                connection.execute(
                    """
                    INSERT INTO expansion_epochs
                    (sequence, epoch_id, decision, manifest_sha256, payload_json, previous_digest, digest)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sequence, epoch_id.strip(), decision.value, manifest_sha256,
                        payload_json, previous_digest, digest,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            return ExpansionLedgerEntry(
                sequence, epoch_id.strip(), decision, manifest_sha256,
                json.loads(payload_json), previous_digest, digest,
            )

    def entries(self) -> tuple[ExpansionLedgerEntry, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM expansion_epochs ORDER BY sequence"
            ).fetchall()
        return tuple(
            ExpansionLedgerEntry(
                int(row["sequence"]), str(row["epoch_id"]),
                ExpansionDecision(row["decision"]), str(row["manifest_sha256"]),
                json.loads(row["payload_json"]), str(row["previous_digest"]),
                str(row["digest"]),
            )
            for row in rows
        )

    def verify_chain(self) -> bool:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM expansion_epochs ORDER BY sequence"
            ).fetchall()
        previous = self.GENESIS
        expected_sequence = 1
        for row in rows:
            if int(row["sequence"]) != expected_sequence or row["previous_digest"] != previous:
                return False
            calculated = self._digest(
                int(row["sequence"]), str(row["epoch_id"]), str(row["decision"]),
                str(row["manifest_sha256"]), str(row["payload_json"]),
                str(row["previous_digest"]),
            )
            if calculated != row["digest"]:
                return False
            previous = calculated
            expected_sequence += 1
        return True

    def seen_task_ids(self) -> frozenset[str]:
        values: set[str] = set()
        for entry in self.entries():
            items = entry.payload.get("task_ids", ())
            values.update(str(item) for item in items)
        return frozenset(values)

    def accepted_capability_ids(self) -> frozenset[str]:
        values: set[str] = set()
        for entry in self.entries():
            if entry.decision is ExpansionDecision.ACCEPTED:
                values.update(str(item) for item in entry.payload.get("capability_ids", ()))
        return frozenset(values)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


@dataclass(frozen=True)
class ExpansionEpochResult:
    epoch_id: str
    decision: ExpansionDecision
    benchmark_id: str
    manifest_sha256: str
    regression_manifest_sha256: str
    checked_regressions: int
    task_ids: tuple[str, ...]
    capability_ids: tuple[str, ...]
    proposals: tuple[UOLProgramLearningProposal, ...]
    blockers: tuple[str, ...]
    ledger_entry: ExpansionLedgerEntry

    @property
    def accepted(self) -> bool:
        return self.decision is ExpansionDecision.ACCEPTED


class ContinuousExpansionEngine:
    """Run repeatable expansion epochs without losing old capability behavior."""

    VERSION = "strategic-phase9-v1"
    MAX_REGRESSION_CASES = 10000

    def __init__(self, ledger: ExpansionLedger) -> None:
        if not isinstance(ledger, ExpansionLedger):
            raise TypeError("ledger must be an ExpansionLedger")
        self.ledger = ledger

    def run_epoch(
        self,
        epoch_id: str,
        benchmark: CrossDomainTransferBenchmarkReport,
        regression_cases: Iterable[FrozenRegressionCase],
        baseline_runner: Callable[[str, Mapping[str, Any]], Any],
    ) -> ExpansionEpochResult:
        if not isinstance(epoch_id, str) or not epoch_id.strip():
            raise ContinuousExpansionError("epoch_id must be non-empty")
        if not isinstance(benchmark, CrossDomainTransferBenchmarkReport):
            raise TypeError("benchmark must be a CrossDomainTransferBenchmarkReport")
        if not callable(baseline_runner):
            raise TypeError("baseline_runner must be callable")
        if not self.ledger.verify_chain():
            raise ContinuousExpansionError("expansion ledger hash chain is invalid")
        if any(item.epoch_id == epoch_id.strip() for item in self.ledger.entries()):
            raise ContinuousExpansionError("epoch_id already exists in the ledger")

        supplied_cases = tuple(regression_cases)
        if not 1 <= len(supplied_cases) <= self.MAX_REGRESSION_CASES:
            raise ContinuousExpansionError("each epoch requires 1..10000 frozen regression cases")
        if any(not isinstance(item, FrozenRegressionCase) for item in supplied_cases):
            raise TypeError("regression_cases must contain FrozenRegressionCase values")
        case_ids = tuple(item.case_id for item in supplied_cases)
        if len(set(case_ids)) != len(case_ids):
            raise ContinuousExpansionError("regression case identifiers must be unique")
        # Snapshot the complete JSON-compatible regression payload before executing
        # any callback. Each baseline call receives a fresh deep copy, so mutable
        # nested inputs cannot drift after the manifest is frozen.
        regression_snapshots = tuple(
            json.loads(_canonical_json({
                "case_id": item.case_id,
                "capability_id": item.capability_id,
                "inputs": dict(item.inputs),
                "expected": item.expected,
            }))
            for item in supplied_cases
        )

        proposals = benchmark.learning_proposals
        target_ids = tuple(
            task_id
            for target in benchmark.targets if target.passed
            for task_id in target.task_ids
        )
        capability_ids = tuple(item.capability.capability_id for item in proposals)
        blockers: list[str] = []
        checked = 0

        previous_tasks = self.ledger.seen_task_ids()
        if len(set(target_ids)) != len(target_ids):
            blockers.append("task identifiers are duplicated within this epoch")
        reused_tasks = sorted(set(target_ids) & set(previous_tasks))
        if reused_tasks:
            blockers.append("task identifiers were already consumed by an earlier epoch")
        if benchmark.status is not TransferBenchmarkStatus.PASS:
            blockers.append("Phase 8 transfer benchmark did not pass")
        if benchmark.minimum_target_domains < 2 or len(benchmark.passed_domains) < 2:
            blockers.append("Phase 9 requires a passing Phase 8 benchmark across at least two target domains")
        if not proposals or not target_ids:
            blockers.append("benchmark contains no independently learned target capabilities")
        if len(set(capability_ids)) != len(capability_ids):
            blockers.append("duplicate target capability identifiers were produced")
        accepted_before = self.ledger.accepted_capability_ids()
        reused_capabilities = sorted(set(capability_ids) & set(accepted_before))
        if reused_capabilities:
            blockers.append("target capability was already accepted in an earlier epoch")

        regression_results = []
        for case in regression_snapshots:
            try:
                actual = baseline_runner(
                    case["capability_id"],
                    json.loads(_canonical_json(case["inputs"])),
                )
                passed = _same_value(actual, case["expected"])
                regression_results.append({
                    "case_id": case["case_id"],
                    "capability_id": case["capability_id"],
                    "passed": passed,
                })
                checked += 1
                if not passed:
                    blockers.append(f"frozen regression failed: {case['case_id']}")
            except Exception as exc:
                regression_results.append({
                    "case_id": case["case_id"],
                    "capability_id": case["capability_id"],
                    "passed": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                })
                checked += 1
                blockers.append(f"frozen regression could not run: {case['case_id']}")

        regression_manifest_payload = {"cases": list(regression_snapshots)}
        regression_manifest_sha256 = hashlib.sha256(
            _canonical_json(regression_manifest_payload).encode("utf-8")
        ).hexdigest()
        decision = ExpansionDecision.ACCEPTED if not blockers else ExpansionDecision.REJECTED
        payload = {
            "version": self.VERSION,
            "benchmark_id": benchmark.benchmark_id,
            "benchmark_status": benchmark.status.value,
            "source_program_fingerprint": benchmark.source_program_fingerprint,
            "source_domain": benchmark.source_domain,
            "target_domains": list(benchmark.passed_domains),
            "task_ids": list(target_ids),
            "capability_ids": list(capability_ids),
            "capability_fingerprints": [item.capability.fingerprint for item in proposals],
            "regression_manifest_sha256": regression_manifest_sha256,
            "regression_results": regression_results,
            "checked_regressions": checked,
            "blockers": list(blockers),
        }
        entry = self.ledger.append(
            epoch_id.strip(), decision, benchmark.manifest_sha256, payload,
        )
        return ExpansionEpochResult(
            epoch_id.strip(), decision, benchmark.benchmark_id,
            benchmark.manifest_sha256, regression_manifest_sha256,
            checked, target_ids, capability_ids,
            proposals if decision is ExpansionDecision.ACCEPTED else (),
            tuple(blockers), entry,
        )

    def promote(
        self,
        result: ExpansionEpochResult,
        target_domain: str,
        capability_registry: CapabilityRegistry,
        operation_registry: NativeOperationRegistry,
    ) -> str:
        """Promote only a capability bound to an accepted, intact ledger entry."""
        if not isinstance(result, ExpansionEpochResult):
            raise TypeError("result must be ExpansionEpochResult")
        if not result.accepted:
            raise ContinuousExpansionError("a rejected expansion epoch cannot be promoted")
        if not self.ledger.verify_chain():
            raise ContinuousExpansionError("expansion ledger hash chain is invalid")
        if not _is_digest(result.ledger_entry.digest):
            raise ContinuousExpansionError("epoch has no valid ledger entry")
        ledger_entry = next(
            (
                item for item in self.ledger.entries()
                if item.epoch_id == result.epoch_id
            ),
            None,
        )
        if (
            ledger_entry is None
            or ledger_entry.digest != result.ledger_entry.digest
            or ledger_entry.decision is not ExpansionDecision.ACCEPTED
        ):
            raise ContinuousExpansionError("epoch is not an accepted entry in this ledger")
        if (
            ledger_entry.manifest_sha256 != result.manifest_sha256
            or ledger_entry.payload.get("regression_manifest_sha256")
                != result.regression_manifest_sha256
            or tuple(ledger_entry.payload.get("task_ids", ())) != result.task_ids
            or tuple(ledger_entry.payload.get("capability_ids", ())) != result.capability_ids
        ):
            raise ContinuousExpansionError("epoch result does not match its immutable ledger record")
        expected_fingerprints = tuple(
            ledger_entry.payload.get("capability_fingerprints", ())
        )
        actual_fingerprints = tuple(
            proposal.capability.fingerprint for proposal in result.proposals
        )
        if not expected_fingerprints or actual_fingerprints != expected_fingerprints:
            raise ContinuousExpansionError("epoch proposals do not match their recorded fingerprints")
        for proposal in result.proposals:
            try:
                UOL(proposal.handler.rules).validate(proposal.program)
            except Exception as exc:
                raise ContinuousExpansionError(
                    "epoch proposal has an invalid or mismatched UOL rule catalog"
                ) from exc
            if (
                proposal.handler.program != proposal.program
                or proposal.capability.provenance.source != "mirror_verified_uol_program_learning"
                or proposal.capability.provenance.source_ref != proposal.program.fingerprint
                or proposal.capability.domains != (proposal.program.domain,)
                or not proposal.capability.executable
                or not proposal.capability.verification.checks_passed
            ):
                raise ContinuousExpansionError(
                    "epoch proposal is not bound to its independently learned UOL program"
                )

        domain = str(target_domain).strip().casefold()
        matches = [
            item for item in result.proposals
            if item.capability.domains == (domain,)
        ]
        if len(matches) != 1:
            raise ContinuousExpansionError("target domain must identify exactly one epoch proposal")
        if not isinstance(capability_registry, CapabilityRegistry) or not isinstance(operation_registry, NativeOperationRegistry):
            raise TypeError("promotion requires capability and operation registries")
        return VerifiedProgramLearningEngine.promote(
            matches[0], capability_registry, operation_registry,
        )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContinuousExpansionError("ledger payload must contain finite JSON values") from exc


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=1e-9, abs_tol=1e-9,
        )
    return left == right
