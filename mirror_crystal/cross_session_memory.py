"""
Mirror 7 — Cross-Session Memory Graph Engine.
Persistent SQLite-backed cognitive memory graph providing multi-turn entity tracking,
project environment anchors, relation links, and contradiction reconciliation across days.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple



class SecretScrubber:
    """Pre-commit privacy filter that scrubs sensitive credentials, API keys, and tokens."""
    PATTERNS = [
        (re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.I), "[REDACTED_API_KEY]"),
        (re.compile(r"ghp_[a-zA-Z0-9]{20,}", re.I), "[REDACTED_GH_TOKEN]"),
        (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I), "[REDACTED_BEARER_TOKEN]"),
        (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----", re.S), "[REDACTED_PRIVATE_KEY]"),
        (re.compile(r"(?:password|passwd|secret|api_key|token)\s*[:=]\s*([^\s,;]+)", re.I), r"secret: [REDACTED_SECRET]"),
    ]

    @classmethod
    def scrub(cls, text: str) -> str:
        s = str(text)
        for pat, repl in cls.PATTERNS:
            s = pat.sub(repl, s)
        return s


class StdlibCipher:
    """Pure Python stdlib AES/PBKDF2-equivalent stream cipher for encryption at rest."""
    MASTER_SALT = b"mirror7_memory_vault_salt_2026"

    @classmethod
    def _derive_key(cls, key_str: str) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", key_str.encode("utf-8"), cls.MASTER_SALT, 1000)

    @classmethod
    def encrypt(cls, plaintext: str, key_str: str = "mirror7_default_vault_key") -> str:
        data_bytes = plaintext.encode("utf-8")
        key = cls._derive_key(key_str)
        out = bytearray(len(data_bytes))
        block_idx = 0
        cur_block = b""
        block_pos = 32
        for i, b in enumerate(data_bytes):
            if block_pos >= 32:
                cur_block = hashlib.sha256(key + struct.pack("<I", block_idx)).digest()
                block_idx += 1
                block_pos = 0
            out[i] = b ^ cur_block[block_pos]
            block_pos += 1
        return "enc::" + out.hex()

    @classmethod
    def decrypt(cls, ciphertext: str, key_str: str = "mirror7_default_vault_key") -> str:
        if not ciphertext.startswith("enc::"):
            return ciphertext
        data_bytes = bytes.fromhex(ciphertext[5:])
        key = cls._derive_key(key_str)
        out = bytearray(len(data_bytes))
        block_idx = 0
        cur_block = b""
        block_pos = 32
        for i, b in enumerate(data_bytes):
            if block_pos >= 32:
                cur_block = hashlib.sha256(key + struct.pack("<I", block_idx)).digest()
                block_idx += 1
                block_pos = 0
            out[i] = b ^ cur_block[block_pos]
            block_pos += 1
        return out.decode("utf-8", errors="replace")

class CrossSessionMemoryGraph:
    """
    Persistent memory graph enabling Mirror 7 to remember user preferences,
    project profiles, server configurations, and relational entities across sessions and days.
    """

    DEFAULT_DB_PATH = Path("artifacts/mirror_memory_graph.sqlite")

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode
        )
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 10000")
        self._conn.execute("PRAGMA wal_autocheckpoint = 1000")
        self.max_entities_per_session = 200
        self.vault_encryption_key = os.getenv("MIRROR_MEMORY_KEY", "m7_memory_vault_secure_key")
        self._undo_stack: Dict[str, List[Dict[str, Any]]] = {}
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    value_text TEXT NOT NULL,
                    value_json TEXT,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    ttl_days INTEGER DEFAULT 365,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(session_id, entity_key)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    src_key TEXT NOT NULL,
                    rel_type TEXT NOT NULL,
                    dst_key TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    UNIQUE(session_id, src_key, rel_type, dst_key)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_profiles (
                    project_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    root_dir TEXT NOT NULL,
                    framework TEXT,
                    database TEXT,
                    port INTEGER,
                    constraints_json TEXT,
                    updated_at REAL NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT NOT NULL,
                    reason TEXT,
                    recorded_at REAL NOT NULL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_session ON memory_entities(session_id, entity_key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memory_entities(session_id, entity_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_session ON memory_relations(session_id, src_key)")

    # =========================================================================
    # Core Entity Memory Operations
    # =========================================================================
    def remember(
        self,
        session_id: str,
        key: str,
        value: Any,
        entity_type: str = "general_preference",
        confidence: float = 1.0,
        ttl_days: int = 365,
    ) -> Dict[str, Any]:
        """Stores or updates an entity memory with secret scrubbing, quota check, and encryption."""
        key_clean = key.strip().lower()
        # Privacy filter: scrub secrets & credentials before saving
        val_str = SecretScrubber.scrub(str(value).strip())
        val_json = json.dumps(value) if not isinstance(value, str) else json.dumps({"str": val_str})
        now = time.time()

        # Encryption at rest for sensitive categories
        is_sensitive = entity_type in ("credential", "secret", "auth", "private_key") or "password" in key_clean or "token" in key_clean
        db_val_text = StdlibCipher.encrypt(val_str, self.vault_encryption_key) if is_sensitive else val_str

        with self._lock:
            cur = self._conn.cursor()
            # Enforce per-session entity quota
            cur.execute("SELECT COUNT(*) FROM memory_entities WHERE session_id = ? AND is_active = 1", (session_id,))
            c_row = cur.fetchone()
            cur.execute(
                "SELECT value_text FROM memory_entities WHERE session_id = ? AND entity_key = ?",
                (session_id, key_clean),
            )
            row = cur.fetchone()

            if c_row and c_row[0] >= self.max_entities_per_session and row is None:
                # Evict lowest confidence / oldest active entity
                cur.execute(
                    """UPDATE memory_entities SET is_active = 0
                       WHERE id = (
                           SELECT id FROM memory_entities
                           WHERE session_id = ? AND is_active = 1
                           ORDER BY confidence ASC, updated_at ASC LIMIT 1
                       )""",
                    (session_id,),
                )

            if row is not None:
                old_val = row[0]
                if old_val != db_val_text:
                    cur.execute(
                        "INSERT INTO memory_history (session_id, entity_key, old_value, new_value, reason, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, key_clean, old_val, db_val_text, "user_update", now),
                    )
                cur.execute(
                    """UPDATE memory_entities
                       SET entity_type = ?, value_text = ?, value_json = ?, confidence = ?, updated_at = ?, ttl_days = ?, is_active = 1
                       WHERE session_id = ? AND entity_key = ?""",
                    (entity_type, db_val_text, val_json, confidence, now, ttl_days, session_id, key_clean),
                )
            else:
                cur.execute(
                    """INSERT INTO memory_entities
                       (session_id, entity_key, entity_type, value_text, value_json, confidence, created_at, updated_at, ttl_days, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                    (session_id, key_clean, entity_type, db_val_text, val_json, confidence, now, now, ttl_days),
                )

        return {
            "session_id": session_id,
            "entity_key": key_clean,
            "entity_type": entity_type,
            "value": val_str,
            "confidence": confidence,
            "status": "updated" if row is not None else "created",
        }

    def recall_entity(self, session_id: str, key: str) -> Optional[Dict[str, Any]]:
        """Recalls a specific entity by key."""
        key_clean = key.strip().lower()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """SELECT entity_key, entity_type, value_text, value_json, confidence, updated_at, ttl_days
                   FROM memory_entities
                   WHERE session_id = ? AND entity_key = ? AND is_active = 1""",
                (session_id, key_clean),
            )
            row = cur.fetchone()
            if not row:
                return None
            k, etype, val_text, val_json, conf, updated, ttl = row
            decrypted_val = StdlibCipher.decrypt(val_text, self.vault_encryption_key) if val_text.startswith("enc::") else val_text
            try:
                parsed_json = json.loads(val_json) if val_json else None
            except Exception:
                parsed_json = None
            return {
                "key": k,
                "type": etype,
                "value": decrypted_val,
                "json": parsed_json,
                "confidence": conf,
                "updated_at": updated,
                "ttl_days": ttl,
            }

    def forget_entity(self, session_id: str, key: str) -> bool:
        """Deactivates an entity from active memory."""
        key_clean = key.strip().lower()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE memory_entities SET is_active = 0 WHERE session_id = ? AND entity_key = ?",
                (session_id, key_clean),
            )
            return cur.rowcount > 0

    def list_entities(self, session_id: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all active entities for a session."""
        with self._lock:
            cur = self._conn.cursor()
            if entity_type:
                cur.execute(
                    """SELECT entity_key, entity_type, value_text, confidence, updated_at
                       FROM memory_entities
                       WHERE session_id = ? AND entity_type = ? AND is_active = 1
                       ORDER BY updated_at DESC""",
                    (session_id, entity_type),
                )
            else:
                cur.execute(
                    """SELECT entity_key, entity_type, value_text, confidence, updated_at
                       FROM memory_entities
                       WHERE session_id = ? AND is_active = 1
                       ORDER BY updated_at DESC""",
                    (session_id,),
                )
            rows = cur.fetchall()
            return [
                {"key": r[0], "type": r[1], "value": StdlibCipher.decrypt(r[2], self.vault_encryption_key) if r[2].startswith("enc::") else r[2], "confidence": r[3], "updated_at": r[4]}
                for r in rows
            ]

    # =========================================================================
    # Relational Knowledge Graph Operations
    # =========================================================================
    def link_entities(
        self,
        session_id: str,
        src_key: str,
        rel_type: str,
        dst_key: str,
        weight: float = 1.0,
    ) -> bool:
        """Creates a directional relationship between two entities."""
        s_k = src_key.strip().lower()
        r_t = rel_type.strip().lower()
        d_k = dst_key.strip().lower()
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO memory_relations (session_id, src_key, rel_type, dst_key, weight, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, src_key, rel_type, dst_key)
                   DO UPDATE SET weight = excluded.weight, created_at = excluded.created_at""",
                (session_id, s_k, r_t, d_k, weight, now),
            )
            return True

    def query_relations(self, session_id: str, key: str) -> List[Dict[str, Any]]:
        """Finds all relationships connected to an entity."""
        k = key.strip().lower()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """SELECT src_key, rel_type, dst_key, weight
                   FROM memory_relations
                   WHERE session_id = ? AND (src_key = ? OR dst_key = ?)""",
                (session_id, k, k),
            )
            rows = cur.fetchall()
            return [{"src": r[0], "rel": r[1], "dst": r[2], "weight": r[3]} for r in rows]

    # =========================================================================
    # Project Profile Management
    # =========================================================================
    def set_project_profile(
        self,
        session_id: str,
        project_name: str,
        root_dir: str = "",
        framework: str = "",
        database: str = "",
        port: Optional[int] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sets or updates the persistent project profile."""
        pid = f"{session_id}_{project_name.lower().replace(' ', '_')}"
        now = time.time()
        c_json = json.dumps(constraints or {})
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO project_profiles (project_id, session_id, name, root_dir, framework, database, port, constraints_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(project_id) DO UPDATE SET
                   name = excluded.name, root_dir = excluded.root_dir, framework = excluded.framework,
                   database = excluded.database, port = excluded.port, constraints_json = excluded.constraints_json,
                   updated_at = excluded.updated_at""",
                (pid, session_id, project_name, root_dir, framework, database, port, c_json, now),
            )
        return {
            "project_id": pid,
            "session_id": session_id,
            "name": project_name,
            "root_dir": root_dir,
            "framework": framework,
            "database": database,
            "port": port,
        }

    def get_project_profile(self, session_id: str, project_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the project profile for a session."""
        with self._lock:
            cur = self._conn.cursor()
            if project_name:
                cur.execute(
                    "SELECT project_id, name, root_dir, framework, database, port, constraints_json, updated_at FROM project_profiles WHERE session_id = ? AND name = ?",
                    (session_id, project_name),
                )
            else:
                cur.execute(
                    "SELECT project_id, name, root_dir, framework, database, port, constraints_json, updated_at FROM project_profiles WHERE session_id = ? ORDER BY updated_at DESC LIMIT 1",
                    (session_id,),
                )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "project_id": row[0],
                "name": row[1],
                "root_dir": row[2],
                "framework": row[3],
                "database": row[4],
                "port": row[5],
                "constraints": json.loads(row[6]) if row[6] else {},
                "updated_at": row[7],
            }

    # =========================================================================
    # Automated Extraction & Intelligent Memory Resolution
    # =========================================================================
    def auto_extract_and_remember(self, session_id: str, text: str) -> List[Dict[str, Any]]:
        """Context-aware scan of input text for explicit configuration anchors with undo tracking."""
        extracted: List[Dict[str, Any]] = []

        # 1. Context-aware Port Classification (resolves Bottleneck 11)
        port_match = re.search(r"\b(?:port|port\s+number)\s*(?:is|=|:)?\s*(\d{2,5})\b", text, re.I)
        if port_match:
            p_val = port_match.group(1).strip()
            p_start = port_match.start()
            prefix = text[max(0, p_start - 35):p_start].lower()
            if any(k in prefix for k in ("database", "db", "postgres", "mysql", "mongo", "redis", "sql")):
                k_label, e_type = "database_port", "config_database"
            elif any(k in prefix for k in ("server", "analytics", "web", "api", "backend", "service", "app")):
                k_label, e_type = "server_port", "service_config"
            else:
                k_label, e_type = "network_port", "config_port"

            res = self.remember(session_id, key=k_label, value=p_val, entity_type=e_type, confidence=1.0)
            extracted.append(res)
            self._record_undo(session_id, k_label, res)

        # 2. S3 Bucket
        b_match = re.search(r"\b(?:s3\s+bucket|bucket)\s+(?:named|is|=|:)?\s*([a-z0-9.\-_]{3,63})\b", text, re.I)
        if b_match:
            b_val = b_match.group(1).strip()
            res = self.remember(session_id, key="s3_bucket", value=b_val, entity_type="cloud_resource", confidence=1.0)
            extracted.append(res)
            self._record_undo(session_id, "s3_bucket", res)

        # 3. AWS Region
        r_match = re.search(r"\b(?:aws\s+region|region)\s*(?:is|=|:)\s*([a-z0-9-]+)\b", text, re.I)
        if r_match:
            r_val = r_match.group(1).strip()
            res = self.remember(session_id, key="aws_region", value=r_val, entity_type="cloud_resource", confidence=1.0)
            extracted.append(res)
            self._record_undo(session_id, "aws_region", res)

        # 4. Database Name
        db_match = re.search(r"\b(?:database|db)\s+(?:is|name|running|using)\s+([A-Za-z0-9_-]+)\b", text, re.I)
        if db_match:
            db_val = db_match.group(1).strip()
            res = self.remember(session_id, key="database_name", value=db_val, entity_type="config_database", confidence=1.0)
            extracted.append(res)
            self._record_undo(session_id, "database_name", res)

        # 5. Web Framework
        fw_match = re.search(r"\b(?:using|framework|backend\s+is)\s+(fastapi|django|flask|express|nextjs|spring\s*boot|gin)\b", text, re.I)
        if fw_match:
            fw_val = fw_match.group(1).strip()
            res = self.remember(session_id, key="web_framework", value=fw_val, entity_type="tech_stack", confidence=1.0)
            extracted.append(res)
            self._record_undo(session_id, "web_framework", res)

        return extracted

    def _record_undo(self, session_id: str, key: str, entity_info: Dict[str, Any]) -> None:
        if session_id not in self._undo_stack:
            self._undo_stack[session_id] = []
        self._undo_stack[session_id].append({"key": key, "info": entity_info, "timestamp": time.time()})
        if len(self._undo_stack[session_id]) > 20:
            self._undo_stack[session_id].pop(0)

    def undo_last(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Reverts the last auto-extracted or remembered entity for a session."""
        stack = self._undo_stack.get(session_id, [])
        if not stack:
            return None
        last_item = stack.pop()
        k = last_item["key"]
        self.forget_entity(session_id, k)
        return {"session_id": session_id, "undone_key": k, "previous_info": last_item["info"]}

    def confirm_entity(self, session_id: str, key: str) -> bool:
        """Confirms an entity, pinning its confidence at 1.0."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE memory_entities SET confidence = 1.0, updated_at = ? WHERE session_id = ? AND entity_key = ?",
                (time.time(), session_id, key.strip().lower()),
            )
            return cur.rowcount > 0

    # =========================================================================
    # GDPR Compliance & Compaction (Resolves Bottlenecks 12 & 13)
    # =========================================================================
    def export_user_data(self, session_id: str) -> Dict[str, Any]:
        """GDPR Right to Data Portability: Exports all user data in clean structured JSON."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT entity_key, entity_type, value_text, confidence, created_at, updated_at FROM memory_entities WHERE session_id = ?",
                (session_id,),
            )
            entities = [
                {
                    "key": r[0],
                    "type": r[1],
                    "value": StdlibCipher.decrypt(r[2], self.vault_encryption_key) if r[2].startswith("enc::") else r[2],
                    "confidence": r[3],
                    "created_at": r[4],
                    "updated_at": r[5],
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                "SELECT src_key, rel_type, dst_key, weight FROM memory_relations WHERE session_id = ?",
                (session_id,),
            )
            relations = [{"src": r[0], "rel": r[1], "dst": r[2], "weight": r[3]} for r in cur.fetchall()]

            cur.execute(
                "SELECT name, root_dir, framework, database, port, constraints_json, updated_at FROM project_profiles WHERE session_id = ?",
                (session_id,),
            )
            profiles = [
                {
                    "name": r[0],
                    "root_dir": r[1],
                    "framework": r[2],
                    "database": r[3],
                    "port": r[4],
                    "constraints": json.loads(r[5]) if r[5] else {},
                    "updated_at": r[6],
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                "SELECT entity_key, old_value, new_value, reason, recorded_at FROM memory_history WHERE session_id = ?",
                (session_id,),
            )
            history = [{"key": r[0], "old": r[1], "new": r[2], "reason": r[3], "recorded_at": r[4]} for r in cur.fetchall()]

            return {
                "session_id": session_id,
                "exported_at": time.time(),
                "entities": entities,
                "relations": relations,
                "profiles": profiles,
                "history": history,
            }

    def purge_user_data(self, session_id: str) -> Dict[str, int]:
        """GDPR Right to Be Forgotten: Permanently wipes all data for a session."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM memory_entities WHERE session_id = ?", (session_id,))
            e_count = cur.rowcount
            cur.execute("DELETE FROM memory_relations WHERE session_id = ?", (session_id,))
            r_count = cur.rowcount
            cur.execute("DELETE FROM project_profiles WHERE session_id = ?", (session_id,))
            p_count = cur.rowcount
            cur.execute("DELETE FROM memory_history WHERE session_id = ?", (session_id,))
            h_count = cur.rowcount
            if session_id in self._undo_stack:
                del self._undo_stack[session_id]

            return {
                "entities_purged": e_count,
                "relations_purged": r_count,
                "profiles_purged": p_count,
                "history_purged": h_count,
            }

    def decay_confidences(self, session_id: Optional[str] = None, decay_rate: float = 0.98) -> int:
        """Applies exponential time-decay to confidence scores of unreinforced entities."""
        now = time.time()
        updated_count = 0
        with self._lock:
            cur = self._conn.cursor()
            query = "SELECT id, confidence, updated_at FROM memory_entities WHERE is_active = 1"
            params: tuple = ()
            if session_id:
                query += " AND session_id = ?"
                params = (session_id,)

            rows = cur.execute(query, params).fetchall()
            for eid, conf, updated_at in rows:
                days_elapsed = max(0.0, (now - updated_at) / 86400.0)
                if days_elapsed >= 1.0:
                    decayed_conf = conf * (decay_rate ** days_elapsed)
                    if decayed_conf < 0.2:
                        cur.execute("UPDATE memory_entities SET is_active = 0 WHERE id = ?", (eid,))
                    else:
                        cur.execute("UPDATE memory_entities SET confidence = ? WHERE id = ?", (decayed_conf, eid))
                    updated_count += 1
        return updated_count

    def compact_memory(self, session_id: Optional[str] = None) -> Dict[str, int]:
        """Runs compaction job: purges inactive entities, removes dangling relations, and trims history."""
        now = time.time()
        cutoff_history = now - (90 * 86400.0)
        with self._lock:
            cur = self._conn.cursor()
            if session_id:
                cur.execute("DELETE FROM memory_entities WHERE session_id = ? AND is_active = 0", (session_id,))
            else:
                cur.execute("DELETE FROM memory_entities WHERE is_active = 0")
            entities_cleaned = cur.rowcount

            cur.execute("DELETE FROM memory_history WHERE recorded_at < ?", (cutoff_history,))
            history_trimmed = cur.rowcount

            cur.execute("""
                DELETE FROM memory_relations
                WHERE src_key NOT IN (SELECT entity_key FROM memory_entities WHERE is_active = 1)
                   OR dst_key NOT IN (SELECT entity_key FROM memory_entities WHERE is_active = 1)
            """)
            relations_cleaned = cur.rowcount

            try:
                cur.execute("PRAGMA incremental_vacuum(100)")
            except Exception:
                pass

            return {
                "inactive_entities_purged": entities_cleaned,
                "history_trimmed": history_trimmed,
                "dangling_relations_cleaned": relations_cleaned,
            }

    def resolve_memory_context(self, session_id: str, query: str) -> Optional[str]:
        """
        Synthesizes a relevant memory context block if the query references
        stored session variables or project configuration anchors.
        """
        entities = self.list_entities(session_id)
        if not entities:
            return None

        q_low = query.lower()
        matched_items: List[str] = []

        for e in entities:
            k = e["key"].replace("_", " ")
            if k in q_low or any(w in q_low for w in k.split()):
                matched_items.append(f"- **{e['key']}**: `{e['value']}` (type: {e['type']})")

        profile = self.get_project_profile(session_id)
        if profile and ("project" in q_low or "app" in q_low or "config" in q_low or "stack" in q_low):
            matched_items.append(
                f"- **Active Project Profile**: `{profile['name']}` (framework: `{profile['framework']}`, db: `{profile['database']}`, port: `{profile['port']}`)"
            )

        if not matched_items:
            return None

        return "### Active Cross-Session Memory Anchors\n" + "\n".join(matched_items)

    def close(self) -> None:
        """Closes the underlying SQLite connection cleanly."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
