"""Stage 22: trusted real-world execution and operational governance.

This module is a host-side gateway, not an unrestricted actuator. External effects
are possible only through explicitly registered, code-reviewed adapters. Reviewer
identity assertions and approval tokens are signed, one-use and scope-bound.
Adapter receipts are Ed25519-authenticated; durable audit events use a keyed hash
chain in SQLite. Unknown outcomes require recovery; failures never update Stage 20
evidence or trusted knowledge.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .stage21_grounded_action import (
    ActionRisk,
    ActionVerificationReport,
    GroundedActionEngine,
    GroundedActionError,
    GroundedActionProposal,
)


class Stage22Error(ValueError):
    """Invalid identity, adapter, authorization, ledger, execution or recovery."""


_SLUG = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SCOPE = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")
_MAX_JSON_BYTES = 64 * 1024


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise Stage22Error(f"value must be finite JSON data: {exc}") from exc


def _digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _text(value: Any, field: str, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage22Error(f"{field} must be non-empty text")
    value = value.strip()
    if len(value) > limit or any(ord(ch) < 32 for ch in value):
        raise Stage22Error(f"{field} exceeds its bound or contains control characters")
    return value


def _utc(value: str | None, field: str = "timestamp") -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str):
        raise Stage22Error(f"{field} must be timezone-aware ISO-8601 text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Stage22Error(f"{field} must be valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage22Error(f"{field} must include an explicit timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _timestamp(value: str, field: str = "timestamp") -> float:
    normalized = _utc(value, field)
    return datetime.fromisoformat(normalized).timestamp()


def _now_iso(clock: Callable[[], float]) -> str:
    return datetime.fromtimestamp(clock(), timezone.utc).isoformat()


def _freeze_json(value: Mapping[str, Any], field: str, limit: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage22Error(f"{field} must be a JSON object")
    raw = _canonical(dict(value)).encode("utf-8")
    if len(raw) > limit:
        raise Stage22Error(f"{field} exceeds {limit} bytes")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise Stage22Error(f"{field} must be a JSON object")
    return parsed


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64url(value: str) -> bytes:
    if not isinstance(value, str) or len(value) > 32_000 or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise Stage22Error("signed token has invalid base64url encoding")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as exc:
        raise Stage22Error("signed token encoding is malformed") from exc


def _token_sign(payload: Mapping[str, Any], key: bytes) -> str:
    body = _b64url(_canonical(dict(payload)).encode("utf-8"))
    mac = _b64url(hmac.new(key, body.encode("ascii"), hashlib.sha256).digest())
    return body + "." + mac


def _token_verify(token: str, key: bytes) -> dict[str, Any]:
    if not isinstance(token, str) or len(token) > 32_768 or token.count(".") != 1:
        raise Stage22Error("signed token is malformed")
    body, supplied = token.split(".", 1)
    expected = _b64url(hmac.new(key, body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, supplied):
        raise Stage22Error("signed token signature is invalid")
    try:
        payload = json.loads(_unb64url(body).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage22Error("signed token payload is malformed") from exc
    if not isinstance(payload, dict):
        raise Stage22Error("signed token payload must be an object")
    return payload


@dataclass(frozen=True)
class ReviewerIdentity:
    issuer: str
    subject: str
    scopes: tuple[str, ...]
    audience: str
    issued_at: str
    expires_at: str
    nonce: str


class ReviewerAuthenticator:
    """Verify host-issued reviewer assertions; it does not authenticate passwords."""

    def __init__(self, trusted_issuers: Mapping[str, bytes], *, audience: str = "mirror-stage22") -> None:
        if not isinstance(trusted_issuers, Mapping) or not trusted_issuers:
            raise Stage22Error("at least one trusted identity issuer is required")
        self.audience = _text(audience, "audience", 128)
        self._keys: dict[str, bytes] = {}
        for issuer, key in trusted_issuers.items():
            issuer_id = _text(issuer, "issuer", 128)
            if not _ID.fullmatch(issuer_id) or not isinstance(key, bytes) or len(key) < 32:
                raise Stage22Error("identity issuer IDs must be safe and HMAC keys at least 32 bytes")
            self._keys[issuer_id] = bytes(key)

    @staticmethod
    def sign_assertion(
        issuer: str,
        subject: str,
        scopes: tuple[str, ...] | list[str],
        signing_key: bytes,
        *,
        audience: str = "mirror-stage22",
        issued_at: str | None = None,
        expires_at: str | None = None,
        nonce: str | None = None,
    ) -> str:
        """Identity-provider helper. Keep signing_key inside the trusted IdP."""
        if not isinstance(signing_key, bytes) or len(signing_key) < 32:
            raise Stage22Error("identity signing key must contain at least 32 bytes")
        clean_scopes = sorted({_text(x, "scope", 128) for x in scopes})
        if any(not _SCOPE.fullmatch(x) for x in clean_scopes):
            raise Stage22Error("identity scopes contain an unsafe identifier")
        issued = _utc(issued_at or datetime.now(timezone.utc).isoformat(), "issued_at")
        expires = _utc(expires_at or datetime.fromtimestamp(time.time() + 300, timezone.utc).isoformat(), "expires_at")
        if _timestamp(expires) <= _timestamp(issued):
            raise Stage22Error("identity assertion expiry must follow issue time")
        payload = {
            "version": 1, "issuer": _text(issuer, "issuer", 128),
            "subject": _text(subject, "subject", 128), "scopes": clean_scopes,
            "audience": _text(audience, "audience", 128),
            "issued_at": issued, "expires_at": expires,
            "nonce": _text(nonce or secrets.token_urlsafe(18), "nonce", 128),
        }
        return _token_sign(payload, signing_key)

    def authenticate(self, assertion: str, *, now: float | None = None) -> ReviewerIdentity:
        if not isinstance(assertion, str) or assertion.count(".") != 1:
            raise Stage22Error("reviewer identity assertion is malformed")
        body = assertion.split(".", 1)[0]
        try:
            payload = json.loads(_unb64url(body).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage22Error("reviewer identity assertion payload is malformed") from exc
        if not isinstance(payload, dict):
            raise Stage22Error("reviewer identity assertion must contain an object")
        issuer = _text(payload.get("issuer"), "issuer", 128)
        key = self._keys.get(issuer)
        if key is None:
            raise Stage22Error("reviewer identity issuer is not host-trusted")
        verified = _token_verify(assertion, key)
        if verified.get("version") != 1 or verified.get("audience") != self.audience:
            raise Stage22Error("reviewer assertion version or audience is invalid")
        subject = _text(verified.get("subject"), "subject", 128)
        scopes = verified.get("scopes")
        if not isinstance(scopes, list) or len(scopes) > 128 or any(not isinstance(x, str) or not _SCOPE.fullmatch(x) for x in scopes):
            raise Stage22Error("reviewer scopes are malformed")
        if len(scopes) != len(set(scopes)):
            raise Stage22Error("reviewer scopes must be unique")
        issued = _utc(verified.get("issued_at"), "issued_at")
        expires = _utc(verified.get("expires_at"), "expires_at")
        current = time.time() if now is None else float(now)
        if _timestamp(issued) > current + 30:
            raise Stage22Error("reviewer assertion is not yet valid")
        if _timestamp(expires) <= current:
            raise Stage22Error("reviewer assertion has expired")
        if _timestamp(expires) - _timestamp(issued) > 900:
            raise Stage22Error("reviewer identity assertion lifetime exceeds 15 minutes")
        nonce = _text(verified.get("nonce"), "nonce", 128)
        return ReviewerIdentity(issuer, subject, tuple(scopes), self.audience, issued, expires, nonce)


class AdapterPhase(str, Enum):
    DRY_RUN = "dry_run"
    EXECUTE = "execute"
    OBSERVE = "observe"
    COMPENSATE = "compensate"


@dataclass(frozen=True)
class ActionAdapterRequest:
    adapter_id: str
    action_id: str
    proposal_id: str
    idempotency_key: str
    inputs: Mapping[str, Any]
    requested_at: str

    def __post_init__(self) -> None:
        for field in ("adapter_id", "action_id", "proposal_id", "idempotency_key"):
            _text(getattr(self, field), field, 128)
        if not _HEX.fullmatch(self.proposal_id):
            raise Stage22Error("proposal_id must be a SHA-256 fingerprint")
        if not _ID.fullmatch(self.idempotency_key):
            raise Stage22Error("idempotency_key must be a safe identifier")
        object.__setattr__(self, "inputs", _freeze_json(self.inputs, "adapter inputs"))
        object.__setattr__(self, "requested_at", _utc(self.requested_at, "requested_at"))


@dataclass(frozen=True)
class AdapterReceipt:
    adapter_id: str
    action_id: str
    phase: AdapterPhase
    idempotency_key: str
    operation_id: str
    ok: bool
    output: Mapping[str, Any]
    observed_at: str
    signature_b64: str

    def __post_init__(self) -> None:
        for field in ("adapter_id", "action_id", "idempotency_key", "operation_id"):
            _text(getattr(self, field), field, 128)
        if not _ID.fullmatch(self.idempotency_key) or not _ID.fullmatch(self.operation_id):
            raise Stage22Error("adapter receipt idempotency/operation ID is unsafe")
        object.__setattr__(self, "phase", AdapterPhase(self.phase))
        if not isinstance(self.ok, bool):
            raise Stage22Error("adapter receipt ok must be boolean")
        object.__setattr__(self, "output", _freeze_json(self.output, "adapter output", 64 * 1024))
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        if not isinstance(self.signature_b64, str) or len(self.signature_b64) > 256:
            raise Stage22Error("adapter receipt signature is malformed")

    def signing_payload(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id, "action_id": self.action_id,
            "phase": self.phase.value, "idempotency_key": self.idempotency_key,
            "operation_id": self.operation_id, "ok": self.ok,
            "output": dict(self.output), "observed_at": self.observed_at,
        }

    @staticmethod
    def signing_bytes(
        adapter_id: str, action_id: str, phase: AdapterPhase | str,
        idempotency_key: str, operation_id: str, ok: bool,
        output: Mapping[str, Any], observed_at: str,
    ) -> bytes:
        """Canonical Ed25519 message for a trusted host adapter to sign."""
        payload = {
            "adapter_id": _text(adapter_id, "adapter_id", 128),
            "action_id": _text(action_id, "action_id", 128),
            "phase": AdapterPhase(phase).value,
            "idempotency_key": _text(idempotency_key, "idempotency_key", 128),
            "operation_id": _text(operation_id, "operation_id", 128),
            "ok": bool(ok), "output": _freeze_json(output, "adapter output", 64 * 1024),
            "observed_at": _utc(observed_at, "observed_at"),
        }
        return _canonical(payload).encode("utf-8")

    @property
    def fingerprint(self) -> str:
        return _digest(_canonical(self.signing_payload()))


class TrustedActionAdapter:
    """Host-owned adapter base class. Implementations are trusted code-reviewed code."""

    adapter_id: str

    def dry_run(self, request: ActionAdapterRequest) -> AdapterReceipt:
        raise NotImplementedError

    def execute(self, request: ActionAdapterRequest) -> AdapterReceipt:
        raise NotImplementedError

    def observe(self, request: ActionAdapterRequest, operation_id: str | None) -> AdapterReceipt:
        raise NotImplementedError

    def compensate(self, request: ActionAdapterRequest, operation_id: str) -> AdapterReceipt:
        raise NotImplementedError


@dataclass(frozen=True)
class _AdapterBinding:
    adapter: TrustedActionAdapter
    public_key: bytes
    action_ids: frozenset[str]
    recovery_postconditions: Mapping[str, Any]


class TrustedActionAdapterRegistry:
    """Host-maintained allow-list; no user-provided URL or dynamic adapter loading."""

    def __init__(self) -> None:
        self._adapters: dict[str, _AdapterBinding] = {}
        self._action_to_adapter: dict[str, str] = {}

    def register(
        self,
        adapter: TrustedActionAdapter,
        *,
        verification_public_key: bytes,
        action_ids: tuple[str, ...] | list[str],
        recovery_postconditions: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(adapter, TrustedActionAdapter):
            raise Stage22Error("only a host-implemented TrustedActionAdapter may be registered")
        adapter_id = _text(getattr(adapter, "adapter_id", None), "adapter_id", 128)
        if not _SLUG.fullmatch(adapter_id):
            raise Stage22Error("adapter_id must be a lowercase safe identifier")
        if adapter_id in self._adapters:
            raise Stage22Error("adapter registration is immutable and cannot be replaced")
        for method_name in ("dry_run", "execute", "observe", "compensate"):
            if getattr(type(adapter), method_name, None) is getattr(TrustedActionAdapter, method_name):
                raise Stage22Error(f"adapter must implement host method: {method_name}")
        if not isinstance(verification_public_key, bytes) or len(verification_public_key) != 32:
            raise Stage22Error("adapter Ed25519 public key must be 32 bytes")
        try:
            Ed25519PublicKey.from_public_bytes(verification_public_key)
        except ValueError as exc:
            raise Stage22Error("adapter Ed25519 public key is invalid") from exc
        actions = frozenset(_text(x, "action_id", 64) for x in action_ids)
        if not actions or len(actions) > 64 or any(not _SLUG.fullmatch(x) for x in actions):
            raise Stage22Error("adapter requires 1-64 safe action IDs")
        if any(action in self._action_to_adapter for action in actions):
            raise Stage22Error("an action ID may be bound to only one trusted adapter")
        recovery = _freeze_json(recovery_postconditions or {}, "recovery_postconditions", 16 * 1024)
        self._adapters[adapter_id] = _AdapterBinding(adapter, bytes(verification_public_key), actions, recovery)
        for action in actions:
            self._action_to_adapter[action] = adapter_id

    def for_action(self, action_id: str) -> tuple[str, _AdapterBinding] | None:
        adapter_id = self._action_to_adapter.get(action_id)
        if adapter_id is None:
            return None
        return adapter_id, self._adapters[adapter_id]

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


@dataclass(frozen=True)
class LedgerVerification:
    valid: bool
    event_count: int
    head_hash: str
    error: str


class DurableAuditLedger:
    """SQLite WAL audit log with immutable event rows and an externally keyed hash chain.

    The HMAC key must be stored outside this database. For rollback/truncation
    detection, a host should also pin the returned head hash in an external system.
    """

    def __init__(self, path: str | Path, signing_key: bytes) -> None:
        if not isinstance(signing_key, bytes) or len(signing_key) < 32:
            raise Stage22Error("audit ledger signing key must contain at least 32 bytes")
        self.path = str(path)
        self._key = bytes(signing_key)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=20.0, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=FULL")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ledger_events (
                sequence INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE,
                event_mac TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS ledger_events_no_update
            BEFORE UPDATE ON ledger_events BEGIN
                SELECT RAISE(ABORT, 'audit ledger events are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS ledger_events_no_delete
            BEFORE DELETE ON ledger_events BEGIN
                SELECT RAISE(ABORT, 'audit ledger events are append-only');
            END;
            CREATE TABLE IF NOT EXISTS approvals (
                jti TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL UNIQUE,
                reviewer_id TEXT NOT NULL,
                expires_at REAL NOT NULL,
                consumed_at REAL
            );
            CREATE TABLE IF NOT EXISTS operations (
                proposal_id TEXT PRIMARY KEY,
                adapter_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                operation_id TEXT,
                approval_reviewer TEXT NOT NULL,
                request_json TEXT NOT NULL,
                receipt_json TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback_reviews (
                proposal_id TEXT PRIMARY KEY,
                reviewer_id TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                reason TEXT NOT NULL,
                reviewed_at TEXT NOT NULL
            );
        """)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _append_tx(self, event_type: str, aggregate_id: str, payload: Mapping[str, Any], occurred_at: str) -> int:
        last = self._conn.execute(
            "SELECT sequence, event_hash FROM ledger_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = int(last["sequence"]) + 1 if last else 1
        previous_hash = str(last["event_hash"]) if last else "0" * 64
        payload_json = _canonical(dict(payload))
        material = _canonical({
            "sequence": sequence, "event_type": event_type,
            "aggregate_id": aggregate_id, "occurred_at": occurred_at,
            "payload_json": payload_json, "previous_hash": previous_hash,
        }).encode("utf-8")
        event_hash = _digest(material)
        event_mac = hmac.new(self._key, event_hash.encode("ascii"), hashlib.sha256).hexdigest()
        self._conn.execute(
            "INSERT INTO ledger_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sequence, event_type, aggregate_id, occurred_at, payload_json, previous_hash, event_hash, event_mac),
        )
        return sequence

    def append(self, event_type: str, aggregate_id: str, payload: Mapping[str, Any]) -> int:
        event = _text(event_type, "event_type", 96)
        aggregate = _text(aggregate_id, "aggregate_id", 128)
        body = _freeze_json(payload, "event payload")
        stamp = _utc(None)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                sequence = self._append_tx(event, aggregate, body, stamp)
                self._conn.execute("COMMIT")
                return sequence
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def issue_approval(
        self, *, jti: str, proposal_id: str, reviewer_id: str,
        expires_at: float, payload: Mapping[str, Any],
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO approvals(jti, proposal_id, reviewer_id, expires_at) VALUES (?, ?, ?, ?)",
                    (jti, proposal_id, reviewer_id, float(expires_at)),
                )
                self._append_tx("approval.issued", proposal_id, payload, _utc(None))
                self._conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise Stage22Error("an approval token was already issued for this proposal or token ID") from exc
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def reserve_execution(
        self, *, jti: str, proposal_id: str, adapter_id: str, action_id: str,
        idempotency_key: str, reviewer_id: str, request: Mapping[str, Any], now: float,
        rate_limit_count: int, rate_limit_window_seconds: float,
    ) -> str:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                approval = self._conn.execute(
                    "SELECT * FROM approvals WHERE jti=? AND proposal_id=?", (jti, proposal_id)
                ).fetchone()
                if approval is None:
                    raise Stage22Error("approval token is unknown or bound to another proposal")
                if approval["reviewer_id"] != reviewer_id:
                    raise Stage22Error("approval token subject does not match the issued reviewer identity")
                if approval["consumed_at"] is not None:
                    raise Stage22Error("approval token has already been consumed")
                if float(approval["expires_at"]) <= now:
                    raise Stage22Error("approval token has expired")
                if self._conn.execute("SELECT 1 FROM operations WHERE proposal_id=?", (proposal_id,)).fetchone():
                    raise Stage22Error("proposal already has a reserved execution; replay is blocked")
                if self._conn.execute("SELECT 1 FROM operations WHERE idempotency_key=?", (idempotency_key,)).fetchone():
                    raise Stage22Error("idempotency key has already been used")
                rows = self._conn.execute(
                    "SELECT payload_json FROM ledger_events WHERE event_type='execution.started' AND occurred_at >= ?",
                    (_utc(datetime.fromtimestamp(now - rate_limit_window_seconds, timezone.utc).isoformat()),),
                ).fetchall()
                recent = 0
                for row in rows:
                    try:
                        event = json.loads(row["payload_json"])
                    except json.JSONDecodeError:
                        raise Stage22Error("audit ledger contains malformed rate-limit evidence")
                    if event.get("adapter_id") == adapter_id and event.get("action_id") == action_id:
                        recent += 1
                if recent >= rate_limit_count:
                    raise Stage22Error("adapter/action rate limit has been reached")
                self._conn.execute(
                    "UPDATE approvals SET consumed_at=? WHERE jti=? AND consumed_at IS NULL",
                    (now, jti),
                )
                self._conn.execute(
                    """INSERT INTO operations(
                        proposal_id, adapter_id, action_id, idempotency_key, status,
                        operation_id, approval_reviewer, request_json, receipt_json, updated_at
                    ) VALUES (?, ?, ?, ?, 'executing', NULL, ?, ?, '{}', ?)""",
                    (proposal_id, adapter_id, action_id, idempotency_key, approval["reviewer_id"], _canonical(dict(request)), now),
                )
                self._append_tx("approval.consumed", proposal_id, {
                    "jti": jti, "reviewer_id": approval["reviewer_id"],
                }, _utc(datetime.fromtimestamp(now, timezone.utc).isoformat()))
                self._append_tx("execution.started", proposal_id, {
                    "adapter_id": adapter_id, "action_id": action_id,
                    "idempotency_key": idempotency_key, "request_fingerprint": _digest(_canonical(dict(request))),
                }, _utc(datetime.fromtimestamp(now, timezone.utc).isoformat()))
                self._conn.execute("COMMIT")
                return str(approval["reviewer_id"])
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def set_operation(
        self, proposal_id: str, *, status: str, operation_id: str | None,
        receipt: Mapping[str, Any], event_type: str, payload: Mapping[str, Any],
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    "UPDATE operations SET status=?, operation_id=COALESCE(?, operation_id), receipt_json=?, updated_at=? WHERE proposal_id=?",
                    (status, operation_id, _canonical(dict(receipt)), now, proposal_id),
                )
                if cursor.rowcount != 1:
                    raise Stage22Error("durable operation record is missing")
                self._append_tx(event_type, proposal_id, payload, _utc(None))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def operation(self, proposal_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM operations WHERE proposal_id=?", (proposal_id,)).fetchone()
            if row is None:
                return None
            return {
                "proposal_id": row["proposal_id"], "adapter_id": row["adapter_id"],
                "action_id": row["action_id"], "idempotency_key": row["idempotency_key"],
                "status": row["status"], "operation_id": row["operation_id"],
                "approval_reviewer": row["approval_reviewer"],
                "request": json.loads(row["request_json"]),
                "receipt": json.loads(row["receipt_json"]),
            }

    def review_feedback(
        self, proposal_id: str, reviewer_id: str, accepted: bool, reason: str, reviewed_at: str,
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                operation = self._conn.execute(
                    "SELECT status, approval_reviewer FROM operations WHERE proposal_id=?", (proposal_id,)
                ).fetchone()
                if operation is None:
                    raise Stage22Error("feedback review requires a recorded operation")
                if operation["status"] not in {"succeeded", "postcondition_mismatch", "failed", "unknown", "rolled_back", "recovery_failed"}:
                    raise Stage22Error("feedback is not ready for independent review")
                if reviewer_id in {operation["approval_reviewer"]}:
                    raise Stage22Error("execution approver cannot independently review its feedback")
                self._conn.execute(
                    "INSERT INTO feedback_reviews VALUES (?, ?, ?, ?, ?)",
                    (proposal_id, reviewer_id, int(accepted), reason, reviewed_at),
                )
                self._append_tx("feedback.reviewed", proposal_id, {
                    "reviewer_id": reviewer_id, "accepted": accepted,
                    "reason": reason, "operation_status": operation["status"],
                    "trusted_knowledge_changed": False,
                }, reviewed_at)
                self._conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise Stage22Error("feedback already has an independent review decision") from exc
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def verify_chain(self, expected_head_hash: str | None = None) -> LedgerVerification:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM ledger_events ORDER BY sequence").fetchall()
        previous = "0" * 64
        count = 0
        for row in rows:
            count += 1
            if int(row["sequence"]) != count or row["previous_hash"] != previous:
                return LedgerVerification(False, count, previous, "ledger sequence or previous-hash link is invalid")
            material = _canonical({
                "sequence": count, "event_type": row["event_type"],
                "aggregate_id": row["aggregate_id"], "occurred_at": row["occurred_at"],
                "payload_json": row["payload_json"], "previous_hash": previous,
            }).encode("utf-8")
            event_hash = _digest(material)
            event_mac = hmac.new(self._key, event_hash.encode("ascii"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(event_hash, row["event_hash"]):
                return LedgerVerification(False, count, previous, "ledger event hash mismatch")
            if not hmac.compare_digest(event_mac, row["event_mac"]):
                return LedgerVerification(False, count, previous, "ledger event authentication failed")
            try:
                payload = json.loads(row["payload_json"])
                if not isinstance(payload, dict):
                    raise ValueError("not an object")
            except (json.JSONDecodeError, ValueError):
                return LedgerVerification(False, count, previous, "ledger event payload is malformed")
            previous = event_hash
        if expected_head_hash is not None:
            if not _HEX.fullmatch(expected_head_hash) or not hmac.compare_digest(previous, expected_head_hash):
                return LedgerVerification(False, count, previous, "ledger head does not match external anchor")
        return LedgerVerification(True, count, previous, "")

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM ledger_events ORDER BY sequence").fetchall()
        return tuple({
            "sequence": int(row["sequence"]), "event_type": row["event_type"],
            "aggregate_id": row["aggregate_id"], "occurred_at": row["occurred_at"],
            "payload": json.loads(row["payload_json"]), "previous_hash": row["previous_hash"],
            "event_hash": row["event_hash"], "event_mac": row["event_mac"],
        } for row in rows)

    @property
    def head_hash(self) -> str:
        with self._lock:
            row = self._conn.execute("SELECT event_hash FROM ledger_events ORDER BY sequence DESC LIMIT 1").fetchone()
            return str(row["event_hash"]) if row else "0" * 64


class GatewayExecutionStatus(str, Enum):
    DRY_RUN = "dry_run"
    SUCCEEDED = "succeeded"
    POSTCONDITION_MISMATCH = "postcondition_mismatch"
    FAILED = "failed"
    UNKNOWN = "unknown"
    ROLLED_BACK = "rolled_back"
    RECOVERY_FAILED = "recovery_failed"


@dataclass(frozen=True)
class GatewayExecutionReport:
    proposal_id: str
    status: GatewayExecutionStatus
    adapter_id: str
    idempotency_key: str
    operation_id: str
    observed_output: Mapping[str, Any]
    expected_postconditions: Mapping[str, Any]
    matched_conditions: tuple[str, ...]
    failed_conditions: tuple[str, ...]
    rollback_status: str
    error: str
    completed_at: str


@dataclass(frozen=True)
class FeedbackReviewReport:
    proposal_id: str
    reviewer_id: str
    accepted: bool
    reason: str
    reviewed_at: str
    trusted_knowledge_changed: bool = False


class TrustedActionGateway:
    """Authorize and execute exact Stage 21 proposals through pinned host adapters."""

    VERSION = "stage22-trusted-execution-governance-v1"

    def __init__(
        self,
        stage21_engine: GroundedActionEngine,
        adapter_registry: TrustedActionAdapterRegistry,
        authenticator: ReviewerAuthenticator,
        ledger: DurableAuditLedger,
        *,
        approval_signing_key: bytes,
        audience: str = "mirror-stage22",
        approval_ttl_seconds: int = 300,
        proposal_max_age_seconds: int = 3600,
        verification_max_age_seconds: int = 300,
        rate_limit_count: int = 10,
        rate_limit_window_seconds: int = 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(stage21_engine, GroundedActionEngine):
            raise Stage22Error("Stage 22 requires a Stage 21 GroundedActionEngine")
        if not isinstance(adapter_registry, TrustedActionAdapterRegistry):
            raise Stage22Error("adapter_registry must be TrustedActionAdapterRegistry")
        if not isinstance(authenticator, ReviewerAuthenticator) or not isinstance(ledger, DurableAuditLedger):
            raise Stage22Error("reviewer authenticator and durable audit ledger are required")
        if not isinstance(approval_signing_key, bytes) or len(approval_signing_key) < 32:
            raise Stage22Error("approval signing key must contain at least 32 bytes")
        if not 1 <= int(approval_ttl_seconds) <= 600:
            raise Stage22Error("approval token lifetime must be 1-600 seconds")
        if not 60 <= int(proposal_max_age_seconds) <= 86_400:
            raise Stage22Error("proposal maximum age must be 60-86400 seconds")
        if not 30 <= int(verification_max_age_seconds) <= 1800:
            raise Stage22Error("verification maximum age must be 30-1800 seconds")
        if not 1 <= int(rate_limit_count) <= 1000 or not 1 <= float(rate_limit_window_seconds) <= 86_400:
            raise Stage22Error("rate limit must be 1-1000 operations per 1-86400 seconds")
        self.stage21_engine = stage21_engine
        self.adapters = adapter_registry
        self.authenticator = authenticator
        self.ledger = ledger
        self._approval_key = bytes(approval_signing_key)
        self.audience = _text(audience, "audience", 128)
        self.approval_ttl_seconds = int(approval_ttl_seconds)
        self.proposal_max_age_seconds = int(proposal_max_age_seconds)
        self.verification_max_age_seconds = int(verification_max_age_seconds)
        self.rate_limit_count = int(rate_limit_count)
        self.rate_limit_window_seconds = float(rate_limit_window_seconds)
        self.clock = clock
        self._verified: dict[str, tuple[str, ActionVerificationReport]] = {}

    def _definition_and_adapter(self, proposal: GroundedActionProposal) -> tuple[Any, str, _AdapterBinding]:
        definition = self.stage21_engine.registry.get(proposal.action_id)
        if definition is None:
            raise Stage22Error("proposal action is not registered in Stage 21")
        found = self.adapters.for_action(proposal.action_id)
        if found is None:
            raise Stage22Error("action has no allow-listed host adapter")
        adapter_id, binding = found
        if definition.risk in {ActionRisk.REVERSIBLE, ActionRisk.CONSEQUENTIAL} and not binding.recovery_postconditions:
            raise Stage22Error("reversible/consequential adapter requires explicit recovery postconditions")
        return definition, adapter_id, binding

    def _check_freshness(self, proposal: GroundedActionProposal, verification: ActionVerificationReport) -> None:
        now = self.clock()
        created = _timestamp(proposal.created_at, "proposal.created_at")
        checked = _timestamp(verification.verified_at, "verification.verified_at")
        if created > now + 30 or now - created > self.proposal_max_age_seconds:
            raise Stage22Error("proposal has expired or has a future creation timestamp")
        if checked > now + 30 or now - checked > self.verification_max_age_seconds:
            raise Stage22Error("independent Stage 21 verification is stale or future-dated")

    def _require_verified(self, proposal: GroundedActionProposal, verification: ActionVerificationReport) -> None:
        if not isinstance(proposal, GroundedActionProposal) or not isinstance(verification, ActionVerificationReport):
            raise Stage22Error("exact Stage 21 proposal and verification objects are required")
        if not verification.passed or verification.proposal_id != proposal.proposal_id:
            raise Stage22Error("passing Stage 21 verification for this exact proposal is required")
        stored = self._verified.get(proposal.proposal_id)
        if stored is None or stored[0] != proposal.fingerprint or stored[1] != verification:
            raise Stage22Error("proposal must be verified by this gateway instance before authorization")
        self._check_freshness(proposal, verification)
        definition, _, _ = self._definition_and_adapter(proposal)
        if definition.fingerprint != proposal.definition_fingerprint or verification.definition_fingerprint != definition.fingerprint:
            raise Stage22Error("Stage 21 action definition changed after verification")

    def verify(self, proposal: GroundedActionProposal, report: Any) -> ActionVerificationReport:
        """Run the existing Stage 21 evidence/provenance and action-policy checkers."""
        verification = self.stage21_engine.verify(proposal, report)
        self._verified[proposal.proposal_id] = (proposal.fingerprint, verification)
        self.ledger.append(
            "proposal.verified" if verification.passed else "proposal.rejected",
            proposal.proposal_id,
            {
                "proposal_fingerprint": proposal.fingerprint,
                "verification_fingerprint": verification.fingerprint,
                "passed": verification.passed,
                "checks": [
                    {"check_id": item.check_id, "verifier_id": item.verifier_id,
                     "decision": item.decision.value, "detail": item.detail}
                    for item in verification.checks
                ],
            },
        )
        return verification

    def issue_approval_token(
        self,
        proposal: GroundedActionProposal,
        verification: ActionVerificationReport,
        *,
        reviewer_assertion: str,
        reason: str,
    ) -> str:
        self._require_verified(proposal, verification)
        _, adapter_id, _ = self._definition_and_adapter(proposal)
        reviewer = self.authenticator.authenticate(reviewer_assertion, now=self.clock())
        required_scope = f"action:approve:{proposal.action_id}"
        if required_scope not in reviewer.scopes:
            raise Stage22Error(f"reviewer lacks required scope {required_scope}")
        if reviewer.subject.casefold() == proposal.created_by.casefold():
            raise Stage22Error("proposal author cannot approve their own proposal")
        rationale = _text(reason, "approval reason", 2000)
        now = self.clock()
        expires = now + self.approval_ttl_seconds
        issued_at = datetime.fromtimestamp(now, timezone.utc).isoformat()
        expires_at = datetime.fromtimestamp(expires, timezone.utc).isoformat()
        jti = secrets.token_urlsafe(24)
        claims = {
            "version": 1, "issuer": "mirror-stage22",
            "audience": self.audience, "subject": reviewer.subject,
            "scopes": [required_scope, f"action:execute:{proposal.action_id}"],
            "proposal_id": proposal.proposal_id,
            "verification_fingerprint": verification.fingerprint,
            "adapter_id": adapter_id, "action_id": proposal.action_id,
            "reason_sha256": _digest(rationale), "jti": jti,
            "issued_at": issued_at, "expires_at": expires_at,
        }
        token = _token_sign(claims, self._approval_key)
        self.ledger.issue_approval(
            jti=jti, proposal_id=proposal.proposal_id, reviewer_id=reviewer.subject,
            expires_at=expires,
            payload={
                "reviewer_id": reviewer.subject, "adapter_id": adapter_id,
                "action_id": proposal.action_id, "verification_fingerprint": verification.fingerprint,
                "expires_at": expires_at, "reason_sha256": _digest(rationale),
            },
        )
        return token

    def _verify_approval_token(
        self, token: str, proposal: GroundedActionProposal,
        verification: ActionVerificationReport, adapter_id: str,
    ) -> dict[str, Any]:
        claims = _token_verify(token, self._approval_key)
        now = self.clock()
        if claims.get("version") != 1 or claims.get("issuer") != "mirror-stage22" or claims.get("audience") != self.audience:
            raise Stage22Error("approval token issuer, version or audience is invalid")
        if claims.get("proposal_id") != proposal.proposal_id or claims.get("verification_fingerprint") != verification.fingerprint:
            raise Stage22Error("approval token is not bound to this exact proposal and verification")
        if claims.get("action_id") != proposal.action_id or claims.get("adapter_id") != adapter_id:
            raise Stage22Error("approval token adapter/action scope mismatch")
        scopes = claims.get("scopes")
        required = f"action:execute:{proposal.action_id}"
        if not isinstance(scopes, list) or required not in scopes:
            raise Stage22Error("approval token lacks exact execution scope")
        issued = _timestamp(claims.get("issued_at"), "token.issued_at")
        expires = _timestamp(claims.get("expires_at"), "token.expires_at")
        if issued > now + 10 or expires <= now or expires - issued > 600:
            raise Stage22Error("approval token is expired, future-dated or exceeds its maximum lifetime")
        return claims

    @staticmethod
    def _verify_receipt(binding: _AdapterBinding, request: ActionAdapterRequest, receipt: AdapterReceipt,
                        phase: AdapterPhase, operation_id: str | None = None) -> None:
        if not isinstance(receipt, AdapterReceipt):
            raise Stage22Error("host adapter must return a typed signed AdapterReceipt")
        if receipt.adapter_id != request.adapter_id or receipt.action_id != request.action_id:
            raise Stage22Error("adapter receipt is not bound to the registered adapter/action")
        if receipt.phase is not phase or receipt.idempotency_key != request.idempotency_key:
            raise Stage22Error("adapter receipt phase or idempotency key mismatch")
        if operation_id is not None and receipt.operation_id != operation_id:
            raise Stage22Error("adapter receipt operation ID mismatch")
        try:
            signature = base64.b64decode(receipt.signature_b64, validate=True)
            Ed25519PublicKey.from_public_bytes(binding.public_key).verify(signature, _canonical(receipt.signing_payload()).encode("utf-8"))
        except (ValueError, InvalidSignature) as exc:
            raise Stage22Error("host adapter receipt signature verification failed") from exc

    @staticmethod
    def _condition_result(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        matched = tuple(sorted(key for key, value in expected.items() if key in observed and observed[key] == value))
        failed = tuple(sorted(set(expected) - set(matched)))
        return matched, failed

    def _request(self, proposal: GroundedActionProposal, adapter_id: str, idempotency_key: str) -> ActionAdapterRequest:
        return ActionAdapterRequest(
            adapter_id=adapter_id, action_id=proposal.action_id, proposal_id=proposal.proposal_id,
            idempotency_key=idempotency_key, inputs=dict(proposal.inputs), requested_at=_now_iso(self.clock),
        )

    def execute(
        self,
        proposal: GroundedActionProposal,
        verification: ActionVerificationReport,
        *,
        approval_token: str | None = None,
        idempotency_key: str | None = None,
        dry_run: bool = False,
    ) -> GatewayExecutionReport:
        self._require_verified(proposal, verification)
        definition, adapter_id, binding = self._definition_and_adapter(proposal)
        if not isinstance(dry_run, bool):
            raise Stage22Error("dry_run must be boolean")
        key = idempotency_key or _digest(f"mirror-stage22:{proposal.proposal_id}")
        key = _text(key, "idempotency_key", 128)
        if not _ID.fullmatch(key):
            raise Stage22Error("idempotency_key must use safe identifier characters")
        request = self._request(proposal, adapter_id, key)
        try:
            preview = binding.adapter.dry_run(request)
            self._verify_receipt(binding, request, preview, AdapterPhase.DRY_RUN)
            if not preview.ok:
                raise Stage22Error("adapter dry-run preflight rejected this action")
        except Exception as exc:
            self.ledger.append("execution.preflight_rejected", proposal.proposal_id, {
                "adapter_id": adapter_id, "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            })
            raise Stage22Error(f"adapter dry-run failed closed: {type(exc).__name__}: {str(exc)[:1000]}") from exc
        if dry_run:
            self.ledger.append("execution.dry_run", proposal.proposal_id, {
                "adapter_id": adapter_id, "idempotency_key": key,
                "preview_fingerprint": preview.fingerprint,
                "postconditions": dict(proposal.expected_postconditions),
            })
            matched, failed = self._condition_result(proposal.expected_postconditions, preview.output)
            return GatewayExecutionReport(
                proposal.proposal_id, GatewayExecutionStatus.DRY_RUN, adapter_id, key,
                preview.operation_id, preview.output, dict(proposal.expected_postconditions),
                matched, failed, "not_required", "", _now_iso(self.clock),
            )
        if not approval_token:
            raise Stage22Error("real-world execution requires a single-use signed human approval token")
        claims = self._verify_approval_token(approval_token, proposal, verification, adapter_id)
        reviewer_id = self.ledger.reserve_execution(
            jti=_text(claims.get("jti"), "approval jti", 128),
            proposal_id=proposal.proposal_id, adapter_id=adapter_id, action_id=proposal.action_id,
            idempotency_key=key, reviewer_id=_text(claims.get("subject"), "approval subject", 128), request={
                "proposal_fingerprint": proposal.fingerprint,
                "verification_fingerprint": verification.fingerprint,
                "inputs": dict(proposal.inputs),
                "expected_postconditions": dict(proposal.expected_postconditions),
            },
            now=self.clock(), rate_limit_count=self.rate_limit_count,
            rate_limit_window_seconds=self.rate_limit_window_seconds,
        )
        try:
            receipt = binding.adapter.execute(request)
            self._verify_receipt(binding, request, receipt, AdapterPhase.EXECUTE)
            if not receipt.ok:
                if receipt.operation_id:
                    return self._compensate_and_check(proposal, verification, request, binding, receipt.operation_id,
                                                       receipt.output, "adapter reported a partial/failed execution")
                raise Stage22Error("adapter reported failure without an operation identity")
            self.ledger.set_operation(
                proposal.proposal_id, status="executing", operation_id=receipt.operation_id,
                receipt={"execute": receipt.signing_payload(), "signature_b64": receipt.signature_b64},
                event_type="execution.adapter_accepted",
                payload={"adapter_id": adapter_id, "operation_id": receipt.operation_id,
                         "receipt_fingerprint": receipt.fingerprint},
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:1200]}"
            self.ledger.set_operation(
                proposal.proposal_id, status="unknown", operation_id=None, receipt={"error": message},
                event_type="execution.outcome_unknown",
                payload={"adapter_id": adapter_id, "error": message, "recovery_required": True},
            )
            return GatewayExecutionReport(
                proposal.proposal_id, GatewayExecutionStatus.UNKNOWN, adapter_id, key, "",
                {}, dict(proposal.expected_postconditions), (), tuple(sorted(proposal.expected_postconditions)),
                "recovery_required", message, _now_iso(self.clock),
            )
        try:
            observed = binding.adapter.observe(request, receipt.operation_id)
            self._verify_receipt(binding, request, observed, AdapterPhase.OBSERVE, receipt.operation_id)
            matched, failed = self._condition_result(proposal.expected_postconditions, observed.output)
            if not failed:
                self.ledger.set_operation(
                    proposal.proposal_id, status="succeeded", operation_id=receipt.operation_id,
                    receipt={"execute": receipt.signing_payload(), "observe": observed.signing_payload()},
                    event_type="execution.verified",
                    payload={"operation_id": receipt.operation_id, "observed_fingerprint": observed.fingerprint,
                             "matched_conditions": list(matched), "failed_conditions": []},
                )
                return GatewayExecutionReport(
                    proposal.proposal_id, GatewayExecutionStatus.SUCCEEDED, adapter_id, key,
                    receipt.operation_id, observed.output, dict(proposal.expected_postconditions),
                    matched, failed, "not_required", "", _now_iso(self.clock),
                )
            return self._compensate_and_check(
                proposal, verification, request, binding, receipt.operation_id,
                observed.output, "external telemetry did not satisfy predicted postconditions",
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:1200]}"
            self.ledger.set_operation(
                proposal.proposal_id, status="unknown", operation_id=receipt.operation_id,
                receipt={"execute": receipt.signing_payload(), "observe_error": message},
                event_type="execution.observation_unknown",
                payload={"operation_id": receipt.operation_id, "error": message, "recovery_required": True},
            )
            return GatewayExecutionReport(
                proposal.proposal_id, GatewayExecutionStatus.UNKNOWN, adapter_id, key,
                receipt.operation_id, {}, dict(proposal.expected_postconditions), (),
                tuple(sorted(proposal.expected_postconditions)), "recovery_required", message,
                _now_iso(self.clock),
            )

    def _compensate_and_check(
        self, proposal: GroundedActionProposal, verification: ActionVerificationReport,
        request: ActionAdapterRequest, binding: _AdapterBinding, operation_id: str,
        observed_output: Mapping[str, Any], reason: str,
    ) -> GatewayExecutionReport:
        definition = self.stage21_engine.registry.get(proposal.action_id)
        matched, failed = self._condition_result(proposal.expected_postconditions, observed_output)
        if definition is None or definition.risk is ActionRisk.READ_ONLY:
            self.ledger.set_operation(
                proposal.proposal_id, status="postcondition_mismatch", operation_id=operation_id,
                receipt={"observed_output": dict(observed_output), "error": reason},
                event_type="execution.postcondition_mismatch",
                payload={"operation_id": operation_id, "matched": list(matched), "failed": list(failed)},
            )
            return GatewayExecutionReport(
                proposal.proposal_id, GatewayExecutionStatus.POSTCONDITION_MISMATCH,
                request.adapter_id, request.idempotency_key, operation_id, dict(observed_output),
                dict(proposal.expected_postconditions), matched, failed, "not_supported",
                reason, _now_iso(self.clock),
            )
        try:
            rollback_key = _digest(f"{request.idempotency_key}:compensate")
            rollback_request = ActionAdapterRequest(
                request.adapter_id, request.action_id, request.proposal_id, rollback_key,
                dict(request.inputs), _now_iso(self.clock),
            )
            compensation = binding.adapter.compensate(rollback_request, operation_id)
            self._verify_receipt(binding, rollback_request, compensation, AdapterPhase.COMPENSATE, operation_id)
            if not compensation.ok:
                raise Stage22Error("adapter compensation receipt reports failure")
            recovered = binding.adapter.observe(rollback_request, operation_id)
            self._verify_receipt(binding, rollback_request, recovered, AdapterPhase.OBSERVE, operation_id)
            recovery_matched, recovery_failed = self._condition_result(binding.recovery_postconditions, recovered.output)
            if recovery_failed:
                raise Stage22Error("post-compensation observation failed recovery postconditions: " + ", ".join(recovery_failed))
            status = GatewayExecutionStatus.ROLLED_BACK
            ledger_status = "rolled_back"
            rollback_status = "verified"
            error = reason
        except Exception as exc:
            status = GatewayExecutionStatus.RECOVERY_FAILED
            ledger_status = "recovery_failed"
            rollback_status = "failed"
            error = f"{reason}; compensation/recovery failure: {type(exc).__name__}: {str(exc)[:1000]}"
            recovered = None
            recovery_matched, recovery_failed = (), tuple(sorted(binding.recovery_postconditions))
        self.ledger.set_operation(
            proposal.proposal_id, status=ledger_status, operation_id=operation_id,
            receipt={
                "observed_output": dict(observed_output),
                "compensation": compensation.signing_payload() if "compensation" in locals() else {},
                "recovery_observation": recovered.signing_payload() if recovered is not None else {},
                "recovery_matched": list(recovery_matched),
                "recovery_failed": list(recovery_failed),
                "error": error,
            },
            event_type="execution.compensated" if status is GatewayExecutionStatus.ROLLED_BACK else "execution.recovery_failed",
            payload={"operation_id": operation_id, "rollback_status": rollback_status,
                     "recovery_matched": list(recovery_matched), "recovery_failed": list(recovery_failed),
                     "error": error},
        )
        final_output = recovered.output if recovered is not None else dict(observed_output)
        return GatewayExecutionReport(
            proposal.proposal_id, status, request.adapter_id, request.idempotency_key,
            operation_id, final_output, dict(proposal.expected_postconditions),
            matched, failed, rollback_status, error, _now_iso(self.clock),
        )

    def recover(
        self,
        proposal: GroundedActionProposal,
        verification: ActionVerificationReport,
    ) -> GatewayExecutionReport:
        """Reconcile an unknown/partial operation by idempotency key; never blindly re-execute."""
        self._require_verified(proposal, verification)
        operation = self.ledger.operation(proposal.proposal_id)
        if operation is None or operation["status"] not in {"unknown", "executing", "recovery_failed"}:
            raise Stage22Error("only unknown, interrupted or recovery-failed operations can be reconciled")
        _, adapter_id, binding = self._definition_and_adapter(proposal)
        key = operation["idempotency_key"]
        request = self._request(proposal, adapter_id, key)
        try:
            observed = binding.adapter.observe(request, operation["operation_id"])
            self._verify_receipt(
                binding, request, observed, AdapterPhase.OBSERVE, operation.get("operation_id")
            )
            operation_id = observed.operation_id
            matched, failed = self._condition_result(proposal.expected_postconditions, observed.output)
            if not failed:
                self.ledger.set_operation(
                    proposal.proposal_id, status="succeeded", operation_id=operation_id,
                    receipt={"recovered_observation": observed.signing_payload()},
                    event_type="execution.recovered",
                    payload={"operation_id": operation_id, "observed_fingerprint": observed.fingerprint,
                             "matched_conditions": list(matched)},
                )
                return GatewayExecutionReport(
                    proposal.proposal_id, GatewayExecutionStatus.SUCCEEDED, adapter_id, key,
                    operation_id, observed.output, dict(proposal.expected_postconditions),
                    matched, failed, "not_required", "external state reconciled by signed observation",
                    _now_iso(self.clock),
                )
            return self._compensate_and_check(
                proposal, verification, request, binding, operation_id, observed.output,
                "recovery observation did not match the predicted postconditions",
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:1200]}"
            self.ledger.set_operation(
                proposal.proposal_id, status="recovery_failed",
                operation_id=operation.get("operation_id"),
                receipt={"recovery_error": message},
                event_type="execution.recovery_failed",
                payload={"error": message, "recovery_required": True},
            )
            return GatewayExecutionReport(
                proposal.proposal_id, GatewayExecutionStatus.RECOVERY_FAILED, adapter_id, key,
                operation.get("operation_id") or "", {}, dict(proposal.expected_postconditions), (),
                tuple(sorted(proposal.expected_postconditions)), "failed", message, _now_iso(self.clock),
            )

    def review_feedback(
        self,
        proposal: GroundedActionProposal,
        verification: ActionVerificationReport,
        *,
        reviewer_assertion: str,
        accepted: bool,
        reason: str,
    ) -> FeedbackReviewReport:
        self._require_verified(proposal, verification)
        if not isinstance(accepted, bool):
            raise Stage22Error("feedback decision must be boolean")
        identity = self.authenticator.authenticate(reviewer_assertion, now=self.clock())
        scope = f"action:feedback-review:{proposal.action_id}"
        if scope not in identity.scopes:
            raise Stage22Error(f"reviewer lacks required scope {scope}")
        operation = self.ledger.operation(proposal.proposal_id)
        if operation is None:
            raise Stage22Error("feedback review requires a durable execution record")
        if identity.subject.casefold() in {proposal.created_by.casefold(), operation["approval_reviewer"].casefold()}:
            raise Stage22Error("feedback reviewer must be independent of proposer and execution approver")
        rationale = _text(reason, "feedback review reason", 2000)
        reviewed_at = _now_iso(self.clock)
        self.ledger.review_feedback(proposal.proposal_id, identity.subject, accepted, rationale, reviewed_at)
        return FeedbackReviewReport(proposal.proposal_id, identity.subject, accepted, rationale, reviewed_at, False)

    def verify_audit_ledger(self, expected_head_hash: str | None = None) -> LedgerVerification:
        return self.ledger.verify_chain(expected_head_hash)
