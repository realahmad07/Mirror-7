"""Append-only, provenance-tracked SQLite EpisodeLedger (L3) with test-time audit hooks.

STRICT ISOLATION (PDF pp. 5-6, 10, 15, 16):
- Stores ONLY public before/action/after observations, raw UTF-8 instruction bytes,
  exact byte spans, candidate role frames, ObservationMasks, canonical SparseDeltas
  (with unobserved-is-not-false preservation accounting), and full provenance metadata.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Enforces append-only immutability via SQLite BEFORE UPDATE / BEFORE DELETE abort
  triggers PLUS a cryptographic SHA-256 record hash chain verified on reload.
- Enforces ``freeze()`` to forbid any test-time episode ingestion, and audits every
  read/lookup through composable ``MemoryAccessEvent`` hooks and ``audit_task()`` scopes.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research.cpu_learning.frames.frame_parser import BaseFrameParser, TypedRoleFrameParser
from research.cpu_learning.frames.state_frame import (
    CandidateActionFrame,
    FrameParseResult,
    MaskedDeltaReport,
    TypedStateFrame,
    assert_no_hidden_fields,
    compute_canonical_state_action_key,
    extract_masked_sparse_delta,
)
from research.cpu_learning.simulator.state_graph import (
    ObservationMask,
    SparseDelta,
    StateGraph,
    TrainingEpisode,
    canonical_json,
    canonical_json_bytes,
    compressed_bit_length,
    sha256_hex,
)

LEDGER_SCHEMA_VERSION: str = "s1-ledger-v1"
GENESIS_HASH: str = "0" * 64


class LedgerAppendError(ValueError):
    """Raised when an append violates uniqueness, schema, or provenance constraints."""


class LedgerFrozenError(RuntimeError):
    """Raised when a write is attempted on a frozen EpisodeLedger."""


class LedgerIntegrityError(RuntimeError):
    """Raised when cryptographic hash-chain, trigger, or invariant verification fails."""


@dataclass(frozen=True)
class MemoryAccessEvent:
    """Single audited read/query against the EpisodeLedger."""

    event_seq: int
    task_id: str | None
    operation: str
    query_summary: str
    matched_episode_ids: tuple[str, ...]
    exact_key_hit: bool
    frozen_mode: bool
    caller_tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "task_id": self.task_id,
            "operation": self.operation,
            "query_summary": self.query_summary,
            "matched_episode_ids": list(self.matched_episode_ids),
            "exact_key_hit": self.exact_key_hit,
            "frozen_mode": self.frozen_mode,
            "caller_tag": self.caller_tag,
        }


@dataclass(frozen=True)
class TaskMemoryAudit:
    """Summary of all EpisodeLedger reads recorded inside an evaluation task scope."""

    task_id: str
    access_count: int
    exact_key_hits: int
    matched_episode_ids: tuple[str, ...]
    events: tuple[MemoryAccessEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "access_count": self.access_count,
            "exact_key_hits": self.exact_key_hits,
            "matched_episode_ids": list(self.matched_episode_ids),
            "events": [e.to_dict() for e in self.events],
        }


class AuditScope:
    """Context manager capturing all EpisodeLedger memory accesses for one evaluation task."""

    def __init__(self, ledger: EpisodeLedger, task_id: str, caller_tag: str = "") -> None:
        self._ledger = ledger
        self.task_id = str(task_id)
        self.caller_tag = str(caller_tag)
        self._events: list[MemoryAccessEvent] = []
        self._prev_scope: AuditScope | None = None

    def _record(self, event: MemoryAccessEvent) -> None:
        self._events.append(event)

    def __enter__(self) -> AuditScope:
        self._prev_scope = self._ledger._active_scope
        self._ledger._active_scope = self
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._ledger._active_scope = self._prev_scope
        self._ledger._completed_task_audits.append(self.result())

    def result(self) -> TaskMemoryAudit:
        matched: set[str] = set()
        exact_hits = 0
        for ev in self._events:
            matched.update(ev.matched_episode_ids)
            if ev.exact_key_hit:
                exact_hits += 1
        return TaskMemoryAudit(
            task_id=self.task_id,
            access_count=len(self._events),
            exact_key_hits=exact_hits,
            matched_episode_ids=tuple(sorted(matched)),
            events=tuple(self._events),
        )


@dataclass(frozen=True)
class LedgerIntegrityReport:
    """Result of cryptographic hash-chain, SQLite trigger, and delta invariant verification."""

    ok: bool
    checked_records: int
    chain_head_sha256: str
    sqlite_integrity_check: str
    triggers_present: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_records": self.checked_records,
            "chain_head_sha256": self.chain_head_sha256,
            "sqlite_integrity_check": self.sqlite_integrity_check,
            "triggers_present": self.triggers_present,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class LedgerEpisodeRecord:
    """Immutable, provenance-tracked, hash-chained EpisodeLedger entry (PDF p. 6 schema)."""

    seq_id: int
    episode_id: str
    ledger_version: str
    record_version: int
    parent_record_sha256: str
    raw_instruction_bytes: bytes
    raw_instruction_sha256: str
    byte_spans: tuple[tuple[str, int, int], ...]
    before_frame: TypedStateFrame
    before_state_fp: str
    candidate_action_frames: tuple[CandidateActionFrame, ...]
    selected_action_id: str
    selected_roles: tuple[tuple[str, str], ...] | None
    is_action_ambiguous: bool
    observation_mask: ObservationMask
    after_observation_mask: ObservationMask
    after_frame: TypedStateFrame
    after_state_fp: str
    sparse_delta: SparseDelta
    delta_report: MaskedDeltaReport
    context: tuple[tuple[str, Any], ...]
    observation_source: str
    provenance: tuple[tuple[str, Any], ...]
    split_id: str
    generator_id: str
    generator_version: str
    seed: int
    timestamp: str
    state_action_key: str
    prev_record_sha256: str
    record_sha256: str

    def selected_role_dict(self) -> dict[str, str] | None:
        if self.selected_roles is None:
            return None
        return dict(self.selected_roles)

    def span_dict(self) -> dict[str, tuple[int, int]]:
        return {k: (s, e) for k, s, e in self.byte_spans}

    def context_dict(self) -> dict[str, Any]:
        return dict(self.context)

    def provenance_dict(self) -> dict[str, Any]:
        return dict(self.provenance)

    def canonical_hash_payload(self) -> dict[str, Any]:
        """Exact deterministic payload hashed into record_sha256."""
        return {
            "episode_id": self.episode_id,
            "ledger_version": self.ledger_version,
            "record_version": self.record_version,
            "parent_record_sha256": self.parent_record_sha256,
            "raw_instruction_sha256": self.raw_instruction_sha256,
            "raw_instruction_byte_length": len(self.raw_instruction_bytes),
            "byte_spans": {k: [s, e] for k, s, e in self.byte_spans},
            "before_frame": self.before_frame.to_dict(),
            "before_state_fp": self.before_state_fp,
            "candidate_action_frames": [c.to_dict() for c in self.candidate_action_frames],
            "selected_action_id": self.selected_action_id,
            "selected_roles": dict(self.selected_roles) if self.selected_roles is not None else None,
            "is_action_ambiguous": self.is_action_ambiguous,
            "observation_mask": self.observation_mask.to_dict(),
            "after_observation_mask": self.after_observation_mask.to_dict(),
            "after_frame": self.after_frame.to_dict(),
            "after_state_fp": self.after_state_fp,
            "sparse_delta": self.sparse_delta.to_dict(),
            "delta_occlusion_audit": {
                "preserved_facts": [list(f) for f in self.delta_report.preserved_facts],
                "occluded_in_after_facts": [list(f) for f in self.delta_report.occluded_in_after_facts],
                "newly_revealed_facts": [list(f) for f in self.delta_report.newly_revealed_facts],
                "occluded_in_after_entities": list(self.delta_report.occluded_in_after_entities),
                "newly_revealed_entities": list(self.delta_report.newly_revealed_entities),
            },
            "context": dict(self.context),
            "observation_source": self.observation_source,
            "provenance": dict(self.provenance),
            "split_id": self.split_id,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "state_action_key": self.state_action_key,
            "prev_record_sha256": self.prev_record_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        out = self.canonical_hash_payload()
        out["seq_id"] = self.seq_id
        out["record_sha256"] = self.record_sha256
        try:
            out["raw_instruction_utf8"] = self.raw_instruction_bytes.decode("utf-8")
        except UnicodeDecodeError:
            out["raw_instruction_utf8"] = self.raw_instruction_bytes.decode("utf-8", errors="replace")
        return out

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL UNIQUE,
    ledger_version TEXT NOT NULL,
    record_version INTEGER NOT NULL DEFAULT 1,
    parent_record_sha256 TEXT NOT NULL,
    raw_instruction_bytes BLOB NOT NULL,
    raw_instruction_sha256 TEXT NOT NULL,
    byte_spans_json TEXT NOT NULL,
    before_frame_json TEXT NOT NULL,
    before_state_fp TEXT NOT NULL,
    candidate_action_frames_json TEXT NOT NULL,
    selected_action_id TEXT NOT NULL,
    selected_roles_json TEXT NOT NULL,
    is_action_ambiguous INTEGER NOT NULL,
    observation_mask_json TEXT NOT NULL,
    after_observation_mask_json TEXT NOT NULL,
    after_frame_json TEXT NOT NULL,
    after_state_fp TEXT NOT NULL,
    sparse_delta_json TEXT NOT NULL,
    delta_occlusion_audit_json TEXT NOT NULL,
    context_json TEXT NOT NULL,
    observation_source TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    split_id TEXT NOT NULL,
    generator_id TEXT NOT NULL,
    generator_version TEXT NOT NULL,
    seed INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    state_action_key TEXT NOT NULL,
    prev_record_sha256 TEXT NOT NULL,
    record_sha256 TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_episodes_state_action_key ON episodes(state_action_key);
CREATE INDEX IF NOT EXISTS idx_episodes_action_id ON episodes(selected_action_id);
CREATE INDEX IF NOT EXISTS idx_episodes_split_seed ON episodes(split_id, seed);
CREATE INDEX IF NOT EXISTS idx_episodes_generator ON episodes(generator_id, generator_version);

CREATE TRIGGER IF NOT EXISTS trg_episodes_no_update
BEFORE UPDATE ON episodes
BEGIN
    SELECT RAISE(ABORT, 'EpisodeLedger is strictly append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS trg_episodes_no_delete
BEFORE DELETE ON episodes
BEGIN
    SELECT RAISE(ABORT, 'EpisodeLedger is strictly append-only: DELETE forbidden');
END;
"""


def _validate_spans(raw_bytes: bytes, spans: Sequence[tuple[str, int, int]]) -> tuple[tuple[str, int, int], ...]:
    n = len(raw_bytes)
    seen: set[str] = set()
    norm: list[tuple[str, int, int]] = []
    for k, s, e in spans:
        sk = str(k)
        si, ei = int(s), int(e)
        if sk in seen:
            raise LedgerAppendError(f"Duplicate byte span key {sk!r}")
        if not (0 <= si <= ei <= n):
            raise LedgerAppendError(f"Byte span {sk!r}=({si},{ei}) out of bounds for len={n}")
        try:
            raw_bytes[si:ei].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LedgerAppendError(f"Byte span {sk!r}=({si},{ei}) slices mid-UTF-8 codepoint") from exc
        seen.add(sk)
        norm.append((sk, si, ei))
    return tuple(sorted(norm))


class EpisodeLedger:
    """Append-only, provenance-tracked SQLite episode store with test-time memory-access auditing."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        journal_mode: str = "DELETE",
        read_only: bool = False,
    ) -> None:
        self._is_memory = str(db_path) == ":memory:"
        self.db_path: Path | None = None if self._is_memory else Path(db_path)
        self._read_only = bool(read_only)
        self._frozen: bool = bool(read_only)
        self._active_scope: AuditScope | None = None
        self._access_events: list[MemoryAccessEvent] = []
        self._completed_task_audits: list[TaskMemoryAudit] = []
        self._access_hooks: list[Callable[[MemoryAccessEvent], None]] = []
        self._event_counter: int = 0

        if self._read_only and not self._is_memory:
            assert self.db_path is not None
            uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
        else:
            if self.db_path is not None:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self.db_path))
            else:
                self._conn = sqlite3.connect(":memory:")
            self._conn.execute("PRAGMA foreign_keys = ON")
            if not self._is_memory:
                mode = "WAL" if journal_mode.upper() == "WAL" else "DELETE"
                self._conn.execute(f"PRAGMA journal_mode = {mode}")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.execute(
                "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES (?, ?)",
                ("schema_version", LEDGER_SCHEMA_VERSION),
            )
            self._conn.commit()
        self._conn.row_factory = sqlite3.Row

    def __enter__(self) -> EpisodeLedger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Lock ledger against all further writes (mandatory before held-out / temporal evaluation)."""
        self._frozen = True

    def _require_writable(self) -> None:
        if self._frozen or self._read_only:
            raise LedgerFrozenError(
                "EpisodeLedger is frozen/read-only: appending test-time episodes is forbidden"
            )

    def register_access_hook(self, callback: Callable[[MemoryAccessEvent], None]) -> None:
        """Register an external callback invoked synchronously on every ledger read."""
        self._access_hooks.append(callback)

    def clear_access_hooks(self) -> None:
        self._access_hooks.clear()

    def audit_task(self, task_id: str, caller_tag: str = "") -> AuditScope:
        """Create an audit scope recording all ledger reads performed for ``task_id``."""
        return AuditScope(self, task_id=task_id, caller_tag=caller_tag)

    @property
    def access_events(self) -> tuple[MemoryAccessEvent, ...]:
        return tuple(self._access_events)

    @property
    def completed_task_audits(self) -> tuple[TaskMemoryAudit, ...]:
        return tuple(self._completed_task_audits)

    def reset_audit_log(self) -> None:
        self._access_events.clear()
        self._completed_task_audits.clear()
        self._event_counter = 0

    def _emit_access(
        self,
        *,
        operation: str,
        query_summary: str,
        matched_ids: Sequence[str],
        exact_key_hit: bool,
        caller_tag: str = "",
    ) -> MemoryAccessEvent:
        self._event_counter += 1
        active_task = self._active_scope.task_id if self._active_scope is not None else None
        effective_tag = caller_tag or (
            self._active_scope.caller_tag if self._active_scope is not None else ""
        )
        ev = MemoryAccessEvent(
            event_seq=self._event_counter,
            task_id=active_task,
            operation=operation,
            query_summary=query_summary,
            matched_episode_ids=tuple(matched_ids),
            exact_key_hit=bool(exact_key_hit),
            frozen_mode=self._frozen,
            caller_tag=effective_tag,
        )
        self._access_events.append(ev)
        if self._active_scope is not None:
            self._active_scope._record(ev)
        for hook in self._access_hooks:
            hook(ev)
        return ev

    def audit_summary(self) -> dict[str, Any]:
        frozen_cnt = sum(1 for e in self._access_events if e.frozen_mode)
        exact_hits = sum(1 for e in self._access_events if e.exact_key_hit)
        touched = {eid for e in self._access_events for eid in e.matched_episode_ids}
        return {
            "total_accesses": len(self._access_events),
            "frozen_accesses": frozen_cnt,
            "exact_key_hits": exact_hits,
            "unique_episodes_touched": len(touched),
            "audited_tasks_count": len(self._completed_task_audits),
        }

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM episodes").fetchone()
        return int(row["c"]) if row is not None else 0

    def chain_head(self) -> str:
        row = self._conn.execute(
            "SELECT record_sha256 FROM episodes ORDER BY seq_id DESC LIMIT 1"
        ).fetchone()
        return str(row["record_sha256"]) if row is not None else GENESIS_HASH
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> LedgerEpisodeRecord:
        raw_bytes = bytes(row["raw_instruction_bytes"])
        spans_dict = json.loads(row["byte_spans_json"])
        byte_spans = tuple(sorted((str(k), int(v[0]), int(v[1])) for k, v in spans_dict.items()))
        before_frame = TypedStateFrame.from_dict(json.loads(row["before_frame_json"]))
        after_frame = TypedStateFrame.from_dict(json.loads(row["after_frame_json"]))
        obs_mask = ObservationMask.from_dict(json.loads(row["observation_mask_json"]))
        after_obs_mask = ObservationMask.from_dict(json.loads(row["after_observation_mask_json"]))
        cand_frames = tuple(
            CandidateActionFrame.from_dict(c) for c in json.loads(row["candidate_action_frames_json"])
        )
        sel_roles_raw = json.loads(row["selected_roles_json"])
        sel_roles = (
            tuple(sorted((str(k), str(v)) for k, v in sel_roles_raw.items()))
            if sel_roles_raw is not None
            else None
        )
        sparse_delta = SparseDelta.from_dict(json.loads(row["sparse_delta_json"]))
        occ_data = json.loads(row["delta_occlusion_audit_json"])
        occ_data_full = dict(occ_data)
        occ_data_full["sparse_delta"] = sparse_delta.to_dict()
        delta_report = MaskedDeltaReport.from_dict(occ_data_full)
        ctx = tuple(sorted((str(k), v) for k, v in json.loads(row["context_json"]).items()))
        prov = tuple(sorted((str(k), v) for k, v in json.loads(row["provenance_json"]).items()))
        return LedgerEpisodeRecord(
            seq_id=int(row["seq_id"]),
            episode_id=str(row["episode_id"]),
            ledger_version=str(row["ledger_version"]),
            record_version=int(row["record_version"]),
            parent_record_sha256=str(row["parent_record_sha256"]),
            raw_instruction_bytes=raw_bytes,
            raw_instruction_sha256=str(row["raw_instruction_sha256"]),
            byte_spans=byte_spans,
            before_frame=before_frame,
            before_state_fp=str(row["before_state_fp"]),
            candidate_action_frames=cand_frames,
            selected_action_id=str(row["selected_action_id"]),
            selected_roles=sel_roles,
            is_action_ambiguous=bool(row["is_action_ambiguous"]),
            observation_mask=obs_mask,
            after_observation_mask=after_obs_mask,
            after_frame=after_frame,
            after_state_fp=str(row["after_state_fp"]),
            sparse_delta=sparse_delta,
            delta_report=delta_report,
            context=ctx,
            observation_source=str(row["observation_source"]),
            provenance=prov,
            split_id=str(row["split_id"]),
            generator_id=str(row["generator_id"]),
            generator_version=str(row["generator_version"]),
            seed=int(row["seed"]),
            timestamp=str(row["timestamp"]),
            state_action_key=str(row["state_action_key"]),
            prev_record_sha256=str(row["prev_record_sha256"]),
            record_sha256=str(row["record_sha256"]),
        )

    def ingest_public_episode(
        self,
        *,
        episode_id: str,
        before_state: StateGraph | TypedStateFrame,
        after_state: StateGraph | TypedStateFrame,
        raw_instruction_bytes: bytes,
        observation_mask: ObservationMask | None = None,
        after_observation_mask: ObservationMask | None = None,
        parser: BaseFrameParser | None = None,
        parse_result: FrameParseResult | None = None,
        typed_action_id: str | None = None,
        typed_roles: Mapping[str, str] | None = None,
        typed_byte_spans: Sequence[tuple[str, int, int]] | None = None,
        context: Mapping[str, Any] | None = None,
        observation_source: str = "public_observation",
        provenance: Mapping[str, Any] | None = None,
        split_id: str = "train",
        generator_id: str = "unknown_generator",
        generator_version: str = "v1",
        seed: int = 0,
        timestamp: str = "2026-11-22T00:00:00Z",
        record_version: int = 1,
        parent_record_sha256: str = "",
    ) -> LedgerEpisodeRecord:
        """Ingest one public before/action/after observation into the append-only ledger."""
        self._require_writable()
        if not episode_id or not isinstance(episode_id, str):
            raise LedgerAppendError("episode_id must be a non-empty string")
        if record_version < 1:
            raise LedgerAppendError("record_version must be >= 1")

        ctx_dict = dict(context or {})
        prov_dict = dict(provenance or {})
        assert_no_hidden_fields(ctx_dict, path="context")
        assert_no_hidden_fields(prov_dict, path="provenance")

        raw = bytes(raw_instruction_bytes)
        raw_sha = sha256_hex(raw)

        if isinstance(before_state, TypedStateFrame):
            before_frame = before_state
            b_mask = before_frame.observation_mask
        else:
            b_mask = observation_mask or ObservationMask(fully_observed=True)
            before_frame = TypedStateFrame.from_state_graph(before_state, mask=b_mask)

        if isinstance(after_state, TypedStateFrame):
            after_frame = after_state
            a_mask = after_frame.observation_mask
        else:
            a_mask = after_observation_mask or b_mask
            after_frame = TypedStateFrame.from_state_graph(after_state, mask=a_mask)

        active_parser = parser or TypedRoleFrameParser(use_state_types=True)
        parsed = parse_result if parse_result is not None else active_parser.parse(raw, before_frame)
        if parsed.raw_instruction_bytes != raw:
            raise LedgerAppendError("parse_result.raw_instruction_bytes does not match raw_instruction_bytes")

        cand_frames = parsed.candidate_frames
        if typed_action_id is not None and typed_roles is not None:
            sel_action_id = str(typed_action_id)
            sel_roles: tuple[tuple[str, str], ...] | None = tuple(
                sorted((str(k), str(v)) for k, v in typed_roles.items())
            )
            is_amb = parsed.is_ambiguous
            if typed_byte_spans is not None:
                norm_spans = _validate_spans(raw, typed_byte_spans)
            elif parsed.top_frame is not None:
                norm_spans = parsed.top_frame.byte_spans
            else:
                norm_spans = ()
            ingest_mode = "typed_action_with_parser_frames"
        else:
            unamb = parsed.unambiguous_frame
            if unamb is not None:
                sel_action_id = unamb.action_id
                sel_roles = unamb.roles
                norm_spans = unamb.byte_spans
                is_amb = False
            else:
                sel_action_id = parsed.top_frame.action_id if parsed.top_frame is not None else ""
                sel_roles = None
                norm_spans = parsed.top_frame.byte_spans if parsed.top_frame is not None else ()
                is_amb = parsed.is_ambiguous
            ingest_mode = "raw_instruction_parser_only"

        delta_report = extract_masked_sparse_delta(before_frame, after_frame)
        before_state_fp = before_frame.to_observed_state_graph().fingerprint()
        after_state_fp = after_frame.to_observed_state_graph().fingerprint()
        if sel_action_id and sel_roles is not None:
            sa_key = compute_canonical_state_action_key(before_state_fp, sel_action_id, dict(sel_roles))
        else:
            sa_key = ""

        full_prov = dict(prov_dict)
        full_prov.setdefault("ingest_mode", ingest_mode)
        full_prov.setdefault("parser_name", parsed.parser_name)
        full_prov.setdefault("parser_version", parsed.parser_version)
        full_prov.setdefault("ledger_schema_version", LEDGER_SCHEMA_VERSION)
        assert_no_hidden_fields(full_prov, path="provenance")

        prev_hash = self.chain_head()
        provisional = LedgerEpisodeRecord(
            seq_id=0,
            episode_id=episode_id,
            ledger_version=LEDGER_SCHEMA_VERSION,
            record_version=record_version,
            parent_record_sha256=parent_record_sha256,
            raw_instruction_bytes=raw,
            raw_instruction_sha256=raw_sha,
            byte_spans=norm_spans,
            before_frame=before_frame,
            before_state_fp=before_state_fp,
            candidate_action_frames=cand_frames,
            selected_action_id=sel_action_id,
            selected_roles=sel_roles,
            is_action_ambiguous=is_amb,
            observation_mask=b_mask,
            after_observation_mask=a_mask,
            after_frame=after_frame,
            after_state_fp=after_state_fp,
            sparse_delta=delta_report.sparse_delta,
            delta_report=delta_report,
            context=tuple(sorted((str(k), v) for k, v in ctx_dict.items())),
            observation_source=str(observation_source),
            provenance=tuple(sorted((str(k), v) for k, v in full_prov.items())),
            split_id=str(split_id),
            generator_id=str(generator_id),
            generator_version=str(generator_version),
            seed=int(seed),
            timestamp=str(timestamp),
            state_action_key=sa_key,
            prev_record_sha256=prev_hash,
            record_sha256="",
        )
        rec_sha = sha256_hex(canonical_json_bytes(provisional.canonical_hash_payload()))

        occ_payload = {
            "preserved_facts": [list(f) for f in delta_report.preserved_facts],
            "occluded_in_after_facts": [list(f) for f in delta_report.occluded_in_after_facts],
            "newly_revealed_facts": [list(f) for f in delta_report.newly_revealed_facts],
            "occluded_in_after_entities": list(delta_report.occluded_in_after_entities),
            "newly_revealed_entities": list(delta_report.newly_revealed_entities),
        }

        try:
            cur = self._conn.execute(
                """
                INSERT INTO episodes (
                    episode_id, ledger_version, record_version, parent_record_sha256,
                    raw_instruction_bytes, raw_instruction_sha256, byte_spans_json,
                    before_frame_json, before_state_fp, candidate_action_frames_json,
                    selected_action_id, selected_roles_json, is_action_ambiguous,
                    observation_mask_json, after_observation_mask_json,
                    after_frame_json, after_state_fp, sparse_delta_json,
                    delta_occlusion_audit_json, context_json, observation_source,
                    provenance_json, split_id, generator_id, generator_version,
                    seed, timestamp, state_action_key, prev_record_sha256, record_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provisional.episode_id,
                    provisional.ledger_version,
                    provisional.record_version,
                    provisional.parent_record_sha256,
                    sqlite3.Binary(provisional.raw_instruction_bytes),
                    provisional.raw_instruction_sha256,
                    canonical_json({k: [s, e] for k, s, e in provisional.byte_spans}),
                    provisional.before_frame.canonical_json(),
                    provisional.before_state_fp,
                    canonical_json([c.to_dict() for c in provisional.candidate_action_frames]),
                    provisional.selected_action_id,
                    canonical_json(dict(provisional.selected_roles) if provisional.selected_roles is not None else None),
                    1 if provisional.is_action_ambiguous else 0,
                    canonical_json(provisional.observation_mask.to_dict()),
                    canonical_json(provisional.after_observation_mask.to_dict()),
                    provisional.after_frame.canonical_json(),
                    provisional.after_state_fp,
                    canonical_json(provisional.sparse_delta.to_dict()),
                    canonical_json(occ_payload),
                    canonical_json(dict(provisional.context)),
                    provisional.observation_source,
                    canonical_json(dict(provisional.provenance)),
                    provisional.split_id,
                    provisional.generator_id,
                    provisional.generator_version,
                    provisional.seed,
                    provisional.timestamp,
                    provisional.state_action_key,
                    provisional.prev_record_sha256,
                    rec_sha,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            raise LedgerAppendError(f"Failed to append episode {episode_id!r}: {exc}") from exc

        seq_id = int(cur.lastrowid)
        row = self._conn.execute("SELECT * FROM episodes WHERE seq_id = ?", (seq_id,)).fetchone()
        assert row is not None
        return self._row_to_record(row)

    def ingest_training_episode(
        self,
        ep: TrainingEpisode,
        *,
        parser: BaseFrameParser | None = None,
        use_typed_action: bool = True,
        extra_provenance: Mapping[str, Any] | None = None,
    ) -> LedgerEpisodeRecord:
        """Ingest a public S0 TrainingEpisode (preserving raw bytes, spans, mask, and provenance)."""
        prov = dict(extra_provenance or {})
        prov.setdefault("source_record_type", "TrainingEpisode")
        return self.ingest_public_episode(
            episode_id=ep.episode_id,
            before_state=ep.before_state,
            after_state=ep.after_state,
            raw_instruction_bytes=ep.action.raw_instruction_bytes,
            observation_mask=ep.observation_mask,
            after_observation_mask=ep.observation_mask,
            parser=parser,
            typed_action_id=ep.action.action_id if use_typed_action else None,
            typed_roles=ep.action.role_dict() if use_typed_action else None,
            typed_byte_spans=ep.action.byte_spans if use_typed_action else None,
            context=dict(ep.context),
            observation_source=ep.observation_source,
            provenance=prov,
            split_id=ep.split_id,
            generator_id=ep.generator_id,
            generator_version=ep.generator_version,
            seed=ep.seed,
            timestamp=ep.timestamp,
        )

    def append_revision(
        self,
        parent_episode_id: str,
        *,
        new_episode_id: str,
        parser: BaseFrameParser | None = None,
        extra_provenance: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> LedgerEpisodeRecord:
        """Append a new versioned record linked to an existing episode without mutating history."""
        self._require_writable()
        parent_row = self._conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (parent_episode_id,)
        ).fetchone()
        if parent_row is None:
            raise LedgerAppendError(f"Parent episode_id {parent_episode_id!r} not found")
        parent = self._row_to_record(parent_row)
        prov = dict(parent.provenance)
        prov.update(extra_provenance or {})
        prov["parent_episode_id"] = parent.episode_id
        return self.ingest_public_episode(
            episode_id=new_episode_id,
            before_state=parent.before_frame,
            after_state=parent.after_frame,
            raw_instruction_bytes=parent.raw_instruction_bytes,
            observation_mask=parent.observation_mask,
            after_observation_mask=parent.after_observation_mask,
            parser=parser,
            typed_action_id=parent.selected_action_id if parent.selected_roles is not None else None,
            typed_roles=parent.selected_role_dict(),
            typed_byte_spans=parent.byte_spans,
            context=parent.context_dict(),
            observation_source=parent.observation_source,
            provenance=prov,
            split_id=parent.split_id,
            generator_id=parent.generator_id,
            generator_version=parent.generator_version,
            seed=parent.seed,
            timestamp=timestamp or parent.timestamp,
            record_version=parent.record_version + 1,
            parent_record_sha256=parent.record_sha256,
        )

    # ---- Audited Read / Retrieval APIs ----

    def get_by_episode_id(self, episode_id: str, *, caller_tag: str = "") -> LedgerEpisodeRecord | None:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        rec = self._row_to_record(row) if row is not None else None
        self._emit_access(
            operation="get_by_episode_id",
            query_summary=f"episode_id={episode_id}",
            matched_ids=(rec.episode_id,) if rec is not None else (),
            exact_key_hit=False,
            caller_tag=caller_tag,
        )
        return rec

    def lookup_exact_state_action(
        self,
        state_action_key: str,
        *,
        caller_tag: str = "",
    ) -> tuple[LedgerEpisodeRecord, ...]:
        """Exact (before_state_fp, action_id, roles) lookup with mandatory retrieval-hit auditing."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE state_action_key = ? ORDER BY seq_id ASC",
            (state_action_key,),
        ).fetchall()
        recs = tuple(self._row_to_record(r) for r in rows)
        self._emit_access(
            operation="lookup_exact_state_action",
            query_summary=f"key={state_action_key[:16]}",
            matched_ids=[r.episode_id for r in recs],
            exact_key_hit=len(recs) > 0,
            caller_tag=caller_tag,
        )
        return recs

    def query_by_action_id(
        self,
        action_id: str,
        *,
        split_id: str | None = None,
        seed: int | None = None,
        caller_tag: str = "",
    ) -> tuple[LedgerEpisodeRecord, ...]:
        sql = "SELECT * FROM episodes WHERE selected_action_id = ?"
        params: list[Any] = [action_id]
        if split_id is not None:
            sql += " AND split_id = ?"
            params.append(split_id)
        if seed is not None:
            sql += " AND seed = ?"
            params.append(int(seed))
        sql += " ORDER BY seq_id ASC"
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        recs = tuple(self._row_to_record(r) for r in rows)
        self._emit_access(
            operation="query_by_action_id",
            query_summary=f"action_id={action_id},split={split_id},seed={seed}",
            matched_ids=[r.episode_id for r in recs],
            exact_key_hit=False,
            caller_tag=caller_tag,
        )
        return recs

    def query_by_split(
        self,
        split_id: str,
        *,
        seed: int | None = None,
        caller_tag: str = "",
    ) -> tuple[LedgerEpisodeRecord, ...]:
        if seed is None:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE split_id = ? ORDER BY seq_id ASC",
                (split_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE split_id = ? AND seed = ? ORDER BY seq_id ASC",
                (split_id, int(seed)),
            ).fetchall()
        recs = tuple(self._row_to_record(r) for r in rows)
        self._emit_access(
            operation="query_by_split",
            query_summary=f"split_id={split_id},seed={seed}",
            matched_ids=[r.episode_id for r in recs],
            exact_key_hit=False,
            caller_tag=caller_tag,
        )
        return recs

    def iter_records(self, *, caller_tag: str = "") -> tuple[LedgerEpisodeRecord, ...]:
        rows = self._conn.execute("SELECT * FROM episodes ORDER BY seq_id ASC").fetchall()
        recs = tuple(self._row_to_record(r) for r in rows)
        self._emit_access(
            operation="iter_records",
            query_summary="all",
            matched_ids=[r.episode_id for r in recs],
            exact_key_hit=False,
            caller_tag=caller_tag,
        )
        return recs

    # ---- Integrity & Bit-Budget Audits (Non-Query Administrative Verification) ----

    def verify_integrity(self, *, strict: bool = False) -> LedgerIntegrityReport:
        """Verify SQLite health, append-only triggers, SHA-256 chain, spans, and delta invariants."""
        errors: list[str] = []
        ic_row = self._conn.execute("PRAGMA integrity_check").fetchone()
        sqlite_ic = str(ic_row[0]) if ic_row is not None else "unknown"
        if sqlite_ic != "ok":
            errors.append(f"SQLite PRAGMA integrity_check failed: {sqlite_ic}")

        trig_rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='episodes'"
        ).fetchall()
        trig_names = {str(r["name"]) for r in trig_rows}
        triggers_ok = {"trg_episodes_no_update", "trg_episodes_no_delete"}.issubset(trig_names)
        if not triggers_ok:
            errors.append(f"Missing append-only triggers; found={sorted(trig_names)}")

        rows = self._conn.execute("SELECT * FROM episodes ORDER BY seq_id ASC").fetchall()
        expected_prev = GENESIS_HASH
        checked = 0
        for row in rows:
            checked += 1
            try:
                rec = self._row_to_record(row)
            except Exception as exc:
                errors.append(f"Row seq_id={row['seq_id']} failed deserialization: {exc}")
                continue
            if rec.prev_record_sha256 != expected_prev:
                errors.append(
                    f"Hash chain broken at seq_id={rec.seq_id} ({rec.episode_id}): "
                    f"expected prev={expected_prev}, got {rec.prev_record_sha256}"
                )
            recomputed_sha = sha256_hex(canonical_json_bytes(rec.canonical_hash_payload()))
            if recomputed_sha != rec.record_sha256:
                errors.append(
                    f"Record hash mismatch at seq_id={rec.seq_id} ({rec.episode_id}): "
                    f"expected {recomputed_sha}, got {rec.record_sha256}"
                )
            if sha256_hex(rec.raw_instruction_bytes) != rec.raw_instruction_sha256:
                errors.append(f"Raw instruction SHA-256 mismatch at {rec.episode_id}")
            try:
                _validate_spans(rec.raw_instruction_bytes, rec.byte_spans)
            except LedgerAppendError as exc:
                errors.append(f"Invalid byte spans at {rec.episode_id}: {exc}")
            for cf in rec.candidate_action_frames:
                if not cf.validate_spans_against(rec.raw_instruction_bytes):
                    errors.append(f"Invalid candidate frame spans at {rec.episode_id}")
            recomputed_delta = extract_masked_sparse_delta(rec.before_frame, rec.after_frame)
            if recomputed_delta != rec.delta_report or recomputed_delta.sparse_delta != rec.sparse_delta:
                errors.append(
                    f"Masked SparseDelta preservation invariant violated at {rec.episode_id}"
                )
            try:
                assert_no_hidden_fields(rec.context_dict(), path=f"{rec.episode_id}.context")
                assert_no_hidden_fields(rec.provenance_dict(), path=f"{rec.episode_id}.provenance")
            except ValueError as exc:
                errors.append(str(exc))
            expected_prev = rec.record_sha256

        report = LedgerIntegrityReport(
            ok=(len(errors) == 0),
            checked_records=checked,
            chain_head_sha256=expected_prev,
            sqlite_integrity_check=sqlite_ic,
            triggers_present=triggers_ok,
            errors=tuple(errors),
        )
        if strict and not report.ok:
            raise LedgerIntegrityError("; ".join(report.errors))
        return report

    def storage_audit(self) -> dict[str, Any]:
        """Compute explicit bit-budget and disk storage accounting for the ledger."""
        rows = self._conn.execute("SELECT * FROM episodes ORDER BY seq_id ASC").fetchall()
        recs = [self._row_to_record(r) for r in rows]
        payload_bytes = canonical_json_bytes([r.to_dict() for r in recs])
        raw_inst_bits = sum(len(r.raw_instruction_bytes) * 8 for r in recs)
        delta_bytes = canonical_json_bytes([r.sparse_delta.to_dict() for r in recs])
        file_bytes = (
            self.db_path.stat().st_size
            if (self.db_path is not None and self.db_path.is_file())
            else None
        )
        return {
            "episode_count": len(recs),
            "raw_payload_bits": len(payload_bytes) * 8,
            "compressed_payload_bits": compressed_bit_length(payload_bytes),
            "raw_instruction_bits": raw_inst_bits,
            "sparse_delta_raw_bits": len(delta_bytes) * 8,
            "sparse_delta_compressed_bits": compressed_bit_length(delta_bytes),
            "sqlite_file_bytes": file_bytes,
            "chain_head_sha256": self.chain_head(),
        }
