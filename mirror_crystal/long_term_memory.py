"""Stage 15: Long-Term Memory & Learning Subsystem.

Provides:
- Multi-tier memory architecture (Working Episodic, Consolidated Semantic, and Immutable Trusted Vault).
- Ebbinghaus forgetting curve decay schedules and access-frequency consolidation.
- Memory correction with tombstoning and lineage auditing.
- Strict Trust Invariant Guard preventing corruption or mutation of trusted vault knowledge.
- SQLite-backed persistence with recency, relevance, and importance scoring.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import json
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
import os
import re
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .production_readiness import StructuredLogger


class MemoryTier(str, Enum):
    """Hierarchy of memory permanence and trust."""
    WORKING_EPISODIC = "working_episodic"        # Short-term, dialogue/turn-level
    CONSOLIDATED_SEMANTIC = "consolidated_semantic" # Learned experiences, cross-session
    IMMUTABLE_TRUSTED_VAULT = "immutable_trusted"   # Authoritative, gold-standard, read-only


class MemoryProtectionError(PermissionError):
    """Raised when an operation attempts to mutate or delete immutable trusted vault knowledge."""


class MemoryError(ValueError):
    """Raised when a memory contract or schema is violated."""


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MemoryNode:
    """An individual memory item within the multi-tier store."""
    memory_id: str
    content: str
    tier: MemoryTier
    importance_score: float = 0.5  # 0.0 to 1.0
    access_count: int = 1
    created_at: str = field(default_factory=_iso_utc_now)
    last_accessed_at: str = field(default_factory=_iso_utc_now)
    half_life_seconds: float = 86400.0  # 24 hours default stability
    metadata: Dict[str, Any] = field(default_factory=dict)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    superseded_by: Optional[str] = None  # Pointer to correcting memory if tombstoned
    is_active: bool = True
    sha256_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise MemoryError("memory_id must not be empty")
        if not isinstance(self.content, str) or not self.content.strip():
            raise MemoryError("content must not be empty")
        if not isinstance(self.tier, MemoryTier):
            raise MemoryError("tier must be a MemoryTier value")
        if not math.isfinite(float(self.importance_score)) or not (0.0 <= self.importance_score <= 1.0):
            raise MemoryError("importance_score must be between 0.0 and 1.0")
        if isinstance(self.access_count, bool) or not isinstance(self.access_count, int) or self.access_count < 0:
            raise MemoryError("access_count must be a non-negative integer")
        if not math.isfinite(float(self.half_life_seconds)) or self.half_life_seconds <= 0:
            raise MemoryError("half_life_seconds must be a positive finite number")
        if not isinstance(self.metadata, dict):
            raise MemoryError("metadata must be a JSON object")
        if not isinstance(self.feedback_history, list):
            raise MemoryError("feedback_history must be a JSON array")
        try:
            json.dumps(self.metadata, allow_nan=False)
            json.dumps(self.feedback_history, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise MemoryError(f"metadata and feedback_history must contain JSON-safe values: {exc}") from exc
        expected_hash = _compute_sha256(self.content)
        if not self.sha256_hash:
            self.sha256_hash = expected_hash
        elif not isinstance(self.sha256_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256_hash):
            raise MemoryError("sha256_hash must be a 64-character lowercase hex digest")
        elif self.sha256_hash != expected_hash:
            raise MemoryError("stored memory content SHA-256 hash mismatch")

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["tier"] = self.tier.value
        return res


class ForgettingCurveScheduler:
    """Implements Ebbinghaus forgetting dynamics: R = e^(-t / S)."""

    @classmethod
    def compute_retention(
        cls,
        last_accessed_iso: str,
        half_life_seconds: float,
        access_count: int,
        importance_score: float,
        current_time_iso: Optional[str] = None,
    ) -> float:
        """Calculates retention probability [0.0, 1.0] given elapsed time and memory strength."""
        try:
            last_dt = datetime.fromisoformat(last_accessed_iso)
            now_dt = datetime.fromisoformat(current_time_iso) if current_time_iso else datetime.now(timezone.utc)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=timezone.utc)
            elapsed_seconds = max(0.0, (now_dt - last_dt).total_seconds())
        except Exception:
            elapsed_seconds = 0.0

        # Memory stability scales with access frequency and intrinsic importance
        stability = half_life_seconds * (1.0 + math.log1p(access_count)) * (0.5 + importance_score)
        stability = max(60.0, stability)

        # Retention decay
        retention = math.exp(-elapsed_seconds / stability)
        return round(min(1.0, max(0.0, retention)), 4)


class MemoryConsolidationEngine:
    """Promotes working memories into consolidated semantic memory and prunes decayed items."""

    def __init__(
        self,
        consolidation_threshold_accesses: int = 3,
        consolidation_threshold_importance: float = 0.70,
        pruning_retention_threshold: float = 0.08,
    ) -> None:
        self.consolidation_threshold_accesses = consolidation_threshold_accesses
        self.consolidation_threshold_importance = consolidation_threshold_importance
        self.pruning_retention_threshold = pruning_retention_threshold

    def evaluate_node(
        self,
        node: MemoryNode,
        current_time_iso: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Evaluates node for 'retain', 'consolidate', or 'prune' along with its retention score."""
        # Immutable trusted vault memories are never pruned or decayed
        if node.tier == MemoryTier.IMMUTABLE_TRUSTED_VAULT:
            return "retain", 1.0

        retention = ForgettingCurveScheduler.compute_retention(
            last_accessed_iso=node.last_accessed_at,
            half_life_seconds=node.half_life_seconds,
            access_count=node.access_count,
            importance_score=node.importance_score,
            current_time_iso=current_time_iso,
        )

        if not node.is_active or retention < self.pruning_retention_threshold:
            return "prune", retention

        # Promote working episodic memory to consolidated semantic memory
        if node.tier == MemoryTier.WORKING_EPISODIC:
            if (
                node.access_count >= self.consolidation_threshold_accesses
                or node.importance_score >= self.consolidation_threshold_importance
            ):
                return "consolidate", retention

        return "retain", retention


class LongTermMemoryManager:
    """Persistent SQLite-backed multi-tier memory manager with feedback learning and immutable trust guards."""

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.db_path = Path(db_path or "artifacts/long_term_memory_store.sqlite")
        self.logger = logger or StructuredLogger(name="mirror-long-term-memory")
        self.consolidation_engine = MemoryConsolidationEngine()
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        tier TEXT NOT NULL,
                        importance_score REAL NOT NULL,
                        access_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        last_accessed_at TEXT NOT NULL,
                        half_life_seconds REAL NOT NULL,
                        metadata TEXT NOT NULL,
                        feedback_history TEXT NOT NULL,
                        superseded_by TEXT,
                        is_active INTEGER NOT NULL,
                        sha256_hash TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_tier ON memories(tier)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_active ON memories(is_active)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(sha256_hash)")
                # Trusted rows may be read/touched, but content, lineage, and trust cannot be changed.
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS protect_immutable_memory_update
                    BEFORE UPDATE OF memory_id, content, tier, importance_score, created_at,
                        half_life_seconds, metadata, feedback_history, superseded_by,
                        is_active, sha256_hash ON memories
                    WHEN OLD.tier = 'immutable_trusted'
                    BEGIN SELECT RAISE(ABORT, 'immutable trusted memory record'); END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS protect_immutable_memory_delete
                    BEFORE DELETE ON memories
                    WHEN OLD.tier = 'immutable_trusted'
                    BEGIN SELECT RAISE(ABORT, 'immutable trusted memory record'); END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS prevent_mutable_to_trusted_promotion
                    BEFORE UPDATE OF tier ON memories
                    WHEN NEW.tier = 'immutable_trusted' AND OLD.tier <> 'immutable_trusted'
                    BEGIN SELECT RAISE(ABORT, 'trusted tier requires a new canonical insert'); END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS prevent_memory_id_replacement
                    BEFORE INSERT ON memories
                    WHEN EXISTS (SELECT 1 FROM memories WHERE memory_id = NEW.memory_id)
                    BEGIN SELECT RAISE(ABORT, 'memory IDs are immutable and cannot be replaced'); END
                """)
                conn.commit()

    def store_memory(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.WORKING_EPISODIC,
        importance_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        half_life_seconds: float = 86400.0,
    ) -> MemoryNode:
        """Stores a new memory node in the specified tier."""
        mem_id = memory_id or f"mem_{_compute_sha256(content)[:12]}"
        now = _iso_utc_now()
        node = MemoryNode(
            memory_id=mem_id,
            content=content,
            tier=tier,
            importance_score=importance_score,
            access_count=1,
            created_at=now,
            last_accessed_at=now,
            half_life_seconds=half_life_seconds,
            metadata=metadata if metadata is not None else {},
        )

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO memories (
                            memory_id, content, tier, importance_score, access_count,
                            created_at, last_accessed_at, half_life_seconds, metadata,
                            feedback_history, superseded_by, is_active, sha256_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        node.memory_id,
                        node.content,
                        node.tier.value,
                        node.importance_score,
                        node.access_count,
                        node.created_at,
                        node.last_accessed_at,
                        node.half_life_seconds,
                        json.dumps(node.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False),
                        json.dumps(node.feedback_history, sort_keys=True, separators=(",", ":"), allow_nan=False),
                        node.superseded_by,
                        1 if node.is_active else 0,
                        node.sha256_hash,
                    ))
                    conn.commit()
            except sqlite3.IntegrityError as exc:
                raise MemoryProtectionError(
                    f"Cannot insert memory '{mem_id}': IDs are immutable and duplicate IDs cannot replace existing records."
                ) from exc

        return node

    def get_memory(self, memory_id: str, touch: bool = True) -> Optional[MemoryNode]:
        """Retrieves a memory by ID, optionally updating its access timestamp and count."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                node = self._row_to_node(row)
                if touch and node.is_active:
                    node.access_count += 1
                    node.last_accessed_at = _iso_utc_now()
                    conn.execute("""
                        UPDATE memories SET access_count = ?, last_accessed_at = ?
                        WHERE memory_id = ?
                    """, (node.access_count, node.last_accessed_at, node.memory_id))
                    conn.commit()
                return node

    def apply_feedback_correction(
        self,
        memory_id: str,
        correction_content: str,
        feedback_notes: str = "",
        feedback_score: float = -1.0,
    ) -> MemoryNode:
        """Corrects a memory by tombstoning the prior version and creating a superseding record.

        TRUST INVARIANT GUARD:
        Strictly forbids mutation or correction of IMMUTABLE_TRUSTED_VAULT records.
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
                row = cursor.fetchone()
                if not row:
                    raise MemoryError(f"Memory '{memory_id}' does not exist")

                old_node = self._row_to_node(row)

                # STRICT TRUST GUARD: Immutable vault cannot be mutated by feedback
                if old_node.tier == MemoryTier.IMMUTABLE_TRUSTED_VAULT:
                    self.logger.warning(
                        "Attempted forbidden modification of immutable trusted vault record",
                        extra={"memory_id": memory_id, "action": "apply_feedback_correction"},
                    )
                    raise MemoryProtectionError(
                        f"Cannot modify memory '{memory_id}': Records in IMMUTABLE_TRUSTED_VAULT "
                        "are cryptographically protected and cannot be altered by feedback."
                    )

                # Create the corrected superseding memory
                new_id = f"mem_corr_{_compute_sha256(correction_content)[:12]}_{uuid.uuid4().hex[:8]}"
                now = _iso_utc_now()
                meta = dict(old_node.metadata)
                meta["supersedes"] = old_node.memory_id
                meta["correction_reason"] = feedback_notes

                new_node = MemoryNode(
                    memory_id=new_id,
                    content=correction_content,
                    tier=old_node.tier,
                    importance_score=max(old_node.importance_score, 0.8),  # Boost importance of corrected facts
                    access_count=1,
                    created_at=now,
                    last_accessed_at=now,
                    metadata=meta,
                )

                # Tombstone old memory
                feedback_entry = {
                    "timestamp": now,
                    "score": feedback_score,
                    "notes": feedback_notes,
                    "superseded_by": new_id,
                }
                new_history = old_node.feedback_history + [feedback_entry]

                conn.execute("""
                    UPDATE memories
                    SET superseded_by = ?, is_active = 0, feedback_history = ?
                    WHERE memory_id = ?
                """, (new_id, json.dumps(new_history, sort_keys=True, separators=(',', ':'), allow_nan=False), old_node.memory_id))

                # Insert new memory
                conn.execute("""
                    INSERT INTO memories (
                        memory_id, content, tier, importance_score, access_count,
                        created_at, last_accessed_at, half_life_seconds, metadata,
                        feedback_history, superseded_by, is_active, sha256_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    new_node.memory_id,
                    new_node.content,
                    new_node.tier.value,
                    new_node.importance_score,
                    new_node.access_count,
                    new_node.created_at,
                    new_node.last_accessed_at,
                    new_node.half_life_seconds,
                    json.dumps(new_node.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False),
                    json.dumps(new_node.feedback_history, sort_keys=True, separators=(",", ":"), allow_nan=False),
                    None,
                    1,
                    new_node.sha256_hash,
                ))
                conn.commit()

        self.logger.info(
            "Memory feedback correction applied",
            extra={"old_id": memory_id, "new_id": new_node.memory_id, "tier": old_node.tier.value},
        )
        return new_node

    def run_consolidation_cycle(self, current_time_iso: Optional[str] = None) -> Dict[str, int]:
        """Runs periodic consolidation: promotes eligible episodic memories and prunes decayed ones."""
        promoted = 0
        pruned = 0
        retained = 0

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM memories WHERE is_active = 1")
                rows = cursor.fetchall()

                for row in rows:
                    node = self._row_to_node(row)
                    action, _ = self.consolidation_engine.evaluate_node(node, current_time_iso=current_time_iso)

                    if action == "consolidate":
                        conn.execute("""
                            UPDATE memories SET tier = ? WHERE memory_id = ?
                        """, (MemoryTier.CONSOLIDATED_SEMANTIC.value, node.memory_id))
                        promoted += 1
                    elif action == "prune":
                        conn.execute("""
                            UPDATE memories SET is_active = 0 WHERE memory_id = ?
                        """, (node.memory_id,))
                        pruned += 1
                    else:
                        retained += 1

                conn.commit()

        return {"promoted_to_semantic": promoted, "pruned_decayed": pruned, "retained": retained}

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        min_retention: float = 0.20,
        active_only: bool = True,
    ) -> List[Tuple[MemoryNode, float]]:
        """Searches memories with combined semantic token overlap, retention score, and importance ranking."""
        tokens = set(query.lower().split())
        results: List[Tuple[MemoryNode, float]] = []

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                query_sql = "SELECT * FROM memories"
                if active_only:
                    query_sql += " WHERE is_active = 1"
                cursor = conn.execute(query_sql)
                rows = cursor.fetchall()

                for row in rows:
                    node = self._row_to_node(row)
                    # Retention score
                    retention = ForgettingCurveScheduler.compute_retention(
                        last_accessed_iso=node.last_accessed_at,
                        half_life_seconds=node.half_life_seconds,
                        access_count=node.access_count,
                        importance_score=node.importance_score,
                    )
                    if retention < min_retention and node.tier != MemoryTier.IMMUTABLE_TRUSTED_VAULT:
                        continue

                    # Lexical overlap
                    content_tokens = set(node.content.lower().split())
                    overlap = len(tokens.intersection(content_tokens)) / float(max(1, len(tokens)))

                    # Tier trust weight: Immutable vault gets priority multiplier
                    tier_multiplier = 1.3 if node.tier == MemoryTier.IMMUTABLE_TRUSTED_VAULT else (
                        1.1 if node.tier == MemoryTier.CONSOLIDATED_SEMANTIC else 1.0
                    )

                    combined_score = (overlap * 0.50 + retention * 0.25 + node.importance_score * 0.25) * tier_multiplier
                    if overlap > 0.0 or node.tier == MemoryTier.IMMUTABLE_TRUSTED_VAULT:
                        results.append((node, round(combined_score, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    @staticmethod
    def _decode_persisted_json(raw_value: Any, expected_type: type, field_name: str) -> Any:
        """Decode JSON storage; safely read old str(dict/list) rows with literal_eval."""
        if not isinstance(raw_value, str) or len(raw_value) > 1_000_000:
            raise MemoryError(f"Persisted {field_name} is missing or exceeds the 1 MB safety limit")
        try:
            value = json.loads(raw_value)
        except (json.JSONDecodeError, RecursionError, MemoryError):
            try:
                value = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
                raise MemoryError(f"Persisted {field_name} is malformed") from exc
        if not isinstance(value, expected_type):
            raise MemoryError(f"Persisted {field_name} has an invalid data type")
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise MemoryError(f"Persisted {field_name} contains non-JSON data") from exc
        return value

    def _row_to_node(self, row: Tuple[Any, ...]) -> MemoryNode:
        metadata = self._decode_persisted_json(row[8], dict, "metadata")
        feedback_history = self._decode_persisted_json(row[9], list, "feedback_history")
        return MemoryNode(
            memory_id=row[0],
            content=row[1],
            tier=MemoryTier(row[2]),
            importance_score=float(row[3]),
            access_count=int(row[4]),
            created_at=row[5],
            last_accessed_at=row[6],
            half_life_seconds=float(row[7]),
            metadata=metadata,
            feedback_history=feedback_history,
            superseded_by=row[10],
            is_active=bool(row[11]),
            sha256_hash=row[12],
        )
