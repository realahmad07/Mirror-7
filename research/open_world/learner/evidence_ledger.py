"""Stage S11.1 Raw-Observation Evidence Extraction (Stages 1-2) & Append-Only Hash-Chained Ledger (Stage 9).

STRICT ISOLATION & GOVERNANCE:
- Pure Python 3.11 standard library ONLY (zero GPU, zero pretrained LLMs, zero external APIs).
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
- Every downstream entity, event, relation, causal hypothesis, or answer traces back to one or
  more immutable ``RawEvidenceUnit`` records anchored by exact UTF-8 byte offsets and SHA-256.
- ``RawEvidenceLedger`` enforces append-only immutability via SQLite BEFORE UPDATE / BEFORE DELETE
  abort triggers, cryptographic SHA-256 record hash chaining, and strict ``.freeze()`` write locks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
import sqlite3
from typing import Any, Sequence

from research.open_world.contracts import (
    OpenWorldObservation,
    SolverFrozenError,
    canonical_json_bytes,
    sha256_hex,
    split_sentences_with_offsets,
)

RAW_LEDGER_SCHEMA_VERSION: str = "s11.1-raw-evidence-ledger-v1"
GENESIS_HASH: str = "0" * 64

_ISO_DATE_RE = re.compile(r"\b((?:19|20)\d{2})[/-](\d{1,2})[/-](\d{1,2})\b")
_MONTH_MAP: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b("
    + "|".join(sorted(_MONTH_MAP.keys(), key=len, reverse=True))
    + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:18|19|20)\d{2})\b",
    re.I,
)
_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+("
    + "|".join(sorted(_MONTH_MAP.keys(), key=len, reverse=True))
    + r")\.?,?\s+((?:18|19|20)\d{2})\b",
    re.I,
)


def parse_timestamp_date(ts: str) -> date | None:
    """Deterministically parse a calendar date from an observation/session timestamp string."""
    if not ts:
        return None
    m = _ISO_DATE_RE.search(ts)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    m2 = _MONTH_DAY_YEAR_RE.search(ts)
    if m2:
        mo = _MONTH_MAP.get(m2.group(1).casefold(), 0)
        d, y = int(m2.group(2)), int(m2.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    m3 = _DAY_MONTH_YEAR_RE.search(ts)
    if m3:
        d = int(m3.group(1))
        mo = _MONTH_MAP.get(m3.group(2).casefold(), 0)
        y = int(m3.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    return None


_TIME_HM_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def timestamp_to_ordinal(ts: str) -> int:
    """Return Python calendar day ordinal for ``ts``, or 0 when no calendar date is present."""
    dt = parse_timestamp_date(ts)
    return dt.toordinal() if dt is not None else 0


def parse_timestamp_sort_key(ts: str) -> tuple[int, int]:
    """Return deterministic ``(date_ordinal, minute_of_day)`` key for ``ts``."""
    ord_day = timestamp_to_ordinal(ts)
    if ord_day <= 0:
        return (0, 0)
    m = _TIME_HM_RE.search(ts or "")
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return (ord_day, hh * 60 + mm)
    return (ord_day, 0)


@dataclass(frozen=True)
class RawEvidenceUnit:
    """Immutable provenance-anchored atomic UTF-8 observation segment (Stage 2)."""

    evidence_id: str
    source_id: str
    byte_start: int
    byte_end: int
    observed_timestamp: str
    observed_ordinal: int
    speaker_role: str
    raw_utf8_bytes: bytes
    text: str
    sha256: str
    turn_index: int = -1
    sent_index: int = 0
    citation_id: str = ""
    observed_minute: int = 0

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "citation_id": self.citation_id or self.source_id,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "observed_timestamp": self.observed_timestamp,
            "observed_ordinal": self.observed_ordinal,
            "speaker_role": self.speaker_role,
            "byte_length": len(self.raw_utf8_bytes),
            "sha256": self.sha256,
            "turn_index": self.turn_index,
            "sent_index": self.sent_index,
        }


def _utf8_byte_span(full_text: str, char_start: int, char_end: int) -> tuple[int, int]:
    b_start = len(full_text[:char_start].encode("utf-8"))
    b_end = b_start + len(full_text[char_start:char_end].encode("utf-8"))
    return b_start, b_end


def _split_paragraph_sentences(text: str) -> tuple[tuple[int, int, str], ...]:
    """Segment multi-line / bullet / dialogue text into clean sentence-level spans."""
    if not text or not text.strip():
        return ()
    results: list[tuple[int, int, str]] = []
    cursor = 0
    for block in text.splitlines(keepends=True):
        stripped = block.strip()
        if stripped:
            block_offset = text.find(stripped, cursor)
            if block_offset < 0:
                block_offset = cursor
            sub_spans = split_sentences_with_offsets(stripped)
            if sub_spans:
                for cs, ce, st in sub_spans:
                    results.append((block_offset + cs, block_offset + ce, st))
            else:
                results.append((block_offset, block_offset + len(stripped), stripped))
        cursor += len(block)
    return tuple(results)


def extract_evidence_units(
    observation: OpenWorldObservation,
    *,
    flatten_timestamps: bool = False,
) -> tuple[RawEvidenceUnit, ...]:
    """Deterministically segment any public ``OpenWorldObservation`` into immutable ``RawEvidenceUnit``s."""
    units: list[RawEvidenceUnit] = []

    if observation.task_kind == "long_memory_qa" and observation.sessions:
        for sess in observation.sessions:
            ts = "" if flatten_timestamps else sess.session_date
            ts_ord, ts_min = (0, 0) if flatten_timestamps else parse_timestamp_sort_key(ts)
            for turn in sess.turns:
                content = turn.content
                if not content or not content.strip():
                    continue
                sent_spans = _split_paragraph_sentences(content)
                if not sent_spans:
                    raw_b = content.encode("utf-8")
                    units.append(
                        RawEvidenceUnit(
                            evidence_id=f"{turn.turn_id}:s0",
                            source_id=sess.session_id,
                            byte_start=0,
                            byte_end=len(raw_b),
                            observed_timestamp=ts,
                            observed_ordinal=ts_ord,
                            speaker_role=turn.role,
                            raw_utf8_bytes=raw_b,
                            text=content.strip(),
                            sha256=sha256_hex(raw_b),
                            turn_index=turn.turn_index,
                            sent_index=0,
                            citation_id=sess.session_id,
                            observed_minute=ts_min,
                        )
                    )
                    continue
                prev_char = 0
                prev_byte = 0
                for s_idx, (cs, ce, stext) in enumerate(sent_spans):
                    prev_byte += len(content[prev_char:cs].encode("utf-8"))
                    raw_b = stext.encode("utf-8")
                    b_start = prev_byte
                    b_end = prev_byte + len(raw_b)
                    prev_char = cs
                    units.append(
                        RawEvidenceUnit(
                            evidence_id=f"{turn.turn_id}:s{s_idx}",
                            source_id=sess.session_id,
                            byte_start=b_start,
                            byte_end=b_end,
                            observed_timestamp=ts,
                            observed_ordinal=ts_ord,
                            speaker_role=turn.role,
                            raw_utf8_bytes=raw_b,
                            text=stext,
                            sha256=sha256_hex(raw_b),
                            turn_index=turn.turn_index,
                            sent_index=s_idx,
                            citation_id=sess.session_id,
                            observed_minute=ts_min,
                        )
                    )
        return tuple(units)

    if observation.evidence_units:
        base_text = observation.prompt_text or ""
        ts = "" if flatten_timestamps else observation.question_date
        ts_ord, ts_min = (0, 0) if flatten_timestamps else parse_timestamp_sort_key(ts)
        for idx, ev in enumerate(observation.evidence_units):
            stext = ev.text.strip()
            if not stext:
                continue
            raw_b = stext.encode("utf-8")
            if base_text and ev.char_end > ev.char_start and ev.char_end <= len(base_text):
                b_start, b_end = _utf8_byte_span(base_text, ev.char_start, ev.char_end)
            else:
                b_start, b_end = 0, len(raw_b)
            role = "document"
            if ev.evidence_id.endswith(":premise"):
                role = "premise"
            elif ":choice_" in ev.evidence_id:
                role = "choice"
            units.append(
                RawEvidenceUnit(
                    evidence_id=ev.evidence_id,
                    source_id=observation.task_id,
                    byte_start=b_start,
                    byte_end=b_end,
                    observed_timestamp=ts,
                    observed_ordinal=ts_ord,
                    speaker_role=role,
                    raw_utf8_bytes=raw_b,
                    text=stext,
                    sha256=sha256_hex(raw_b),
                    turn_index=-1,
                    sent_index=idx,
                    citation_id=ev.evidence_id,
                    observed_minute=ts_min,
                )
            )
        return tuple(units)

    raw_prompt = observation.prompt_text or ""
    for idx, (cs, ce, stext) in enumerate(_split_paragraph_sentences(raw_prompt)):
        b_start, b_end = _utf8_byte_span(raw_prompt, cs, ce)
        raw_b = stext.encode("utf-8")
        ev_id = f"{observation.task_id}:s{idx}"
        units.append(
            RawEvidenceUnit(
                evidence_id=ev_id,
                source_id=observation.task_id,
                byte_start=b_start,
                byte_end=b_end,
                observed_timestamp="",
                observed_ordinal=0,
                speaker_role="document",
                raw_utf8_bytes=raw_b,
                text=stext,
                sha256=sha256_hex(raw_b),
                turn_index=-1,
                sent_index=idx,
                citation_id=ev_id,
            )
        )
    return tuple(units)


class RawEvidenceLedger:
    """Append-only, hash-chained SQLite evidence ledger for raw UTF-8 spans (Stages 2 & 9)."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._frozen: bool = False
        self._post_freeze_writes: int = 0
        self._chain_head: str = GENESIS_HASH
        self._count: int = 0
        self._seen_ids: set[str] = set()
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_evidence_records (
                seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL,
                citation_id TEXT NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                observed_timestamp TEXT NOT NULL,
                observed_ordinal INTEGER NOT NULL,
                speaker_role TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                byte_length INTEGER NOT NULL,
                prev_record_sha256 TEXT NOT NULL,
                record_sha256 TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS trg_raw_evidence_no_update
            BEFORE UPDATE ON raw_evidence_records
            BEGIN
                SELECT RAISE(ABORT, 'RawEvidenceLedger is append-only: UPDATE forbidden');
            END;
            CREATE TRIGGER IF NOT EXISTS trg_raw_evidence_no_delete
            BEFORE DELETE ON raw_evidence_records
            BEGIN
                SELECT RAISE(ABORT, 'RawEvidenceLedger is append-only: DELETE forbidden');
            END;
            """
        )
        self._conn.commit()

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def post_freeze_writes(self) -> int:
        return self._post_freeze_writes

    @property
    def chain_head(self) -> str:
        return self._chain_head

    def __len__(self) -> int:
        return self._count

    def freeze(self) -> None:
        self._frozen = True

    def append_unit(self, unit: RawEvidenceUnit, *, commit: bool = True) -> str | None:
        """Append one immutable ``RawEvidenceUnit`` provenance record; returns record SHA-256."""
        if self._frozen:
            self._post_freeze_writes += 1
            raise SolverFrozenError("RawEvidenceLedger is frozen; post-freeze writes are forbidden")
        if unit.evidence_id in self._seen_ids:
            return None
        payload = {
            "schema_version": RAW_LEDGER_SCHEMA_VERSION,
            "evidence_id": unit.evidence_id,
            "source_id": unit.source_id,
            "citation_id": unit.citation_id or unit.source_id,
            "byte_start": unit.byte_start,
            "byte_end": unit.byte_end,
            "observed_timestamp": unit.observed_timestamp,
            "observed_ordinal": unit.observed_ordinal,
            "speaker_role": unit.speaker_role,
            "content_sha256": unit.sha256,
            "byte_length": len(unit.raw_utf8_bytes),
            "prev_record_sha256": self._chain_head,
        }
        rec_sha = sha256_hex(canonical_json_bytes(payload))
        self._conn.execute(
            """
            INSERT INTO raw_evidence_records (
                evidence_id, source_id, citation_id, byte_start, byte_end,
                observed_timestamp, observed_ordinal, speaker_role,
                content_sha256, byte_length, prev_record_sha256, record_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unit.evidence_id,
                unit.source_id,
                unit.citation_id or unit.source_id,
                unit.byte_start,
                unit.byte_end,
                unit.observed_timestamp,
                unit.observed_ordinal,
                unit.speaker_role,
                unit.sha256,
                len(unit.raw_utf8_bytes),
                self._chain_head,
                rec_sha,
            ),
        )
        self._seen_ids.add(unit.evidence_id)
        self._chain_head = rec_sha
        self._count += 1
        if commit:
            self._conn.commit()
        return rec_sha

    def append_batch(self, units: Sequence[RawEvidenceUnit]) -> int:
        if self._frozen:
            self._post_freeze_writes += 1
            raise SolverFrozenError("RawEvidenceLedger is frozen; post-freeze writes are forbidden")
        added = 0
        for u in units:
            if self.append_unit(u, commit=False) is not None:
                added += 1
        if added > 0:
            self._conn.commit()
        return added

    def verify_integrity(self) -> dict[str, Any]:
        """Cryptographically verify the entire SHA-256 hash chain and SQLite immutability triggers."""
        cur = self._conn.cursor()
        triggers = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        triggers_ok = (
            "trg_raw_evidence_no_update" in triggers
            and "trg_raw_evidence_no_delete" in triggers
        )
        rows = cur.execute(
            """
            SELECT evidence_id, source_id, citation_id, byte_start, byte_end,
                   observed_timestamp, observed_ordinal, speaker_role,
                   content_sha256, byte_length, prev_record_sha256, record_sha256
            FROM raw_evidence_records
            ORDER BY seq_id ASC
            """
        ).fetchall()
        prev = GENESIS_HASH
        errors: list[str] = []
        for idx, row in enumerate(rows):
            (
                ev_id, src_id, cit_id, b_start, b_end,
                obs_ts, obs_ord, role, c_sha, b_len, prev_sha, rec_sha,
            ) = row
            if prev_sha != prev:
                errors.append(f"Chain break at seq {idx + 1}: expected prev={prev}, got {prev_sha}")
                break
            payload = {
                "schema_version": RAW_LEDGER_SCHEMA_VERSION,
                "evidence_id": ev_id,
                "source_id": src_id,
                "citation_id": cit_id,
                "byte_start": b_start,
                "byte_end": b_end,
                "observed_timestamp": obs_ts,
                "observed_ordinal": obs_ord,
                "speaker_role": role,
                "content_sha256": c_sha,
                "byte_length": b_len,
                "prev_record_sha256": prev_sha,
            }
            expected = sha256_hex(canonical_json_bytes(payload))
            if expected != rec_sha:
                errors.append(f"Hash mismatch at seq {idx + 1} ({ev_id})")
                break
            prev = rec_sha
        return {
            "ok": triggers_ok and not errors,
            "checked_records": len(rows),
            "chain_head_sha256": prev,
            "triggers_present": triggers_ok,
            "post_freeze_writes": self._post_freeze_writes,
            "errors": errors,
        }

    def summary_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema": RAW_LEDGER_SCHEMA_VERSION,
                "count": self._count,
                "chain_head": self._chain_head,
            }
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass