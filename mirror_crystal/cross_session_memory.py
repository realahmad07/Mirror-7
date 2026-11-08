"""
Mirror 7 — Cross-Session Memory Graph Engine.
Persistent SQLite-backed cognitive memory graph providing multi-turn entity tracking,
project environment anchors, relation links, and contradiction reconciliation across days.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


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
        """Stores or updates an entity memory with version history."""
        key_clean = key.strip().lower()
        val_str = str(value).strip()
        val_json = json.dumps(value) if not isinstance(value, str) else json.dumps({"str": value})
        now = time.time()

        with self._lock:
            cur = self._conn.cursor()
            # Check existing value
            cur.execute(
                "SELECT value_text FROM memory_entities WHERE session_id = ? AND entity_key = ?",
                (session_id, key_clean),
            )
            row = cur.fetchone()

            if row is not None:
                old_val = row[0]
                if old_val != val_str:
                    # Record contradiction / state update history
                    cur.execute(
                        "INSERT INTO memory_history (session_id, entity_key, old_value, new_value, reason, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, key_clean, old_val, val_str, "user_update", now),
                    )
                cur.execute(
                    """UPDATE memory_entities
                       SET entity_type = ?, value_text = ?, value_json = ?, confidence = ?, updated_at = ?, ttl_days = ?, is_active = 1
                       WHERE session_id = ? AND entity_key = ?""",
                    (entity_type, val_str, val_json, confidence, now, ttl_days, session_id, key_clean),
                )
            else:
                cur.execute(
                    """INSERT INTO memory_entities
                       (session_id, entity_key, entity_type, value_text, value_json, confidence, created_at, updated_at, ttl_days, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                    (session_id, key_clean, entity_type, val_str, val_json, confidence, now, now, ttl_days),
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
            try:
                parsed_json = json.loads(val_json) if val_json else None
            except Exception:
                parsed_json = None
            return {
                "key": k,
                "type": etype,
                "value": val_text,
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
                {"key": r[0], "type": r[1], "value": r[2], "confidence": r[3], "updated_at": r[4]}
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
    AUTO_EXTRACT_PATTERNS = [
        (re.compile(r"\b(?:port|port\s+number)\s*(?:is|=|:)?\s*(\d{2,5})\b", re.I), "database_port", "config_port"),
        (re.compile(r"\b(?:database|db)\s+(?:is|name|running|using)\s+([A-Za-z0-9_-]+)\b", re.I), "database_name", "config_database"),
        (re.compile(r"\b(?:s3\s+bucket|bucket)\s+(?:named|is|=|:)?\s*([a-z0-9.\-_]{3,63})\b", re.I), "s3_bucket", "cloud_resource"),
        (re.compile(r"\b(?:aws\s+region|region)\s*(?:is|=|:)\s*([a-z0-9-]+)\b", re.I), "aws_region", "cloud_resource"),
        (re.compile(r"\b(?:pagination\s+limit|page\s+size|default\s+limit)\s*(?:is|=|:)?\s*(\d+)\b", re.I), "pagination_limit", "api_config"),
        (re.compile(r"\b(?:using|framework|backend\s+is)\s+(fastapi|django|flask|express|nextjs|spring\s*boot|gin)\b", re.I), "web_framework", "tech_stack"),
    ]

    def auto_extract_and_remember(self, session_id: str, text: str) -> List[Dict[str, Any]]:
        """Scans input text for explicit configuration anchors and saves them to memory."""
        extracted: List[Dict[str, Any]] = []
        for pat, key, etype in self.AUTO_EXTRACT_PATTERNS:
            m = pat.search(text)
            if m:
                val = m.group(1).strip()
                res = self.remember(session_id, key=key, value=val, entity_type=etype, confidence=1.0)
                extracted.append(res)
        return extracted

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
