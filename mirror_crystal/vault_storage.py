"""SQLite/WAL persistence and rule repository operations for DiskCrystalVault."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any, Optional

from .bit_signatures import ContextBitTable
from .blueprint_prompt import blueprint_prompt_fields
from .lsh_rule_store import CrystalRule


class VaultStorageMixin:

    def _open_thread_conn(self, enable_wal: bool = False) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA busy_timeout=15000;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA temp_store=MEMORY;")
        c.execute("PRAGMA mmap_size=268435456;")  # 256 MB OS kernel page cache map
        c.execute("PRAGMA cache_size=-64000;")   # 64 MB SQLite page cache
        with self._lock:
            self._conns.append(c)
        return c

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None or self._closed:
            if self._closed:
                raise RuntimeError("DiskCrystalVault is closed.")
            c = self._open_thread_conn(enable_wal=False)
            self._local.conn = c
        return c

    def cached_rules_in_ram(self) -> int:
        with self._lock:
            return len(self._rule_lru)

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                val BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rules (
                rule_id INTEGER PRIMARY KEY,
                prompt_hash BLOB UNIQUE NOT NULL,
                sig_bytes BLOB NOT NULL,
                num_slots INTEGER NOT NULL,
                pass_count INTEGER NOT NULL,
                fail_count INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS token_postings (
                tok TEXT PRIMARY KEY,
                rids BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS band_postings (
                band INTEGER PRIMARY KEY,
                rids BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS causal_edges (
                head TEXT PRIMARY KEY,
                edges_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS causal_tok_idx (
                tok TEXT PRIMARY KEY,
                heads_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blueprints (
                blueprint_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                intent TEXT NOT NULL,
                kind TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                prompt_key TEXT NOT NULL DEFAULT '',
                prompt_prefix TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_blueprints_domain_intent ON blueprints(domain, intent);
            CREATE TABLE IF NOT EXISTS blueprint_claim_evidence (
                blueprint_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (blueprint_id, claim_id, reviewer_id)
            );
            CREATE INDEX IF NOT EXISTS idx_blueprint_claim_evidence_id ON blueprint_claim_evidence(blueprint_id, claim_id);
            CREATE TABLE IF NOT EXISTS blueprint_claim_scopes (
                blueprint_id TEXT PRIMARY KEY,
                required_claim_ids_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rules_num_slots ON rules(num_slots) WHERE num_slots > 0;
            CREATE TABLE IF NOT EXISTS prompt_aliases (
                alias_hash BLOB PRIMARY KEY,
                rule_id INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prompt_aliases_rid ON prompt_aliases(rule_id);
            CREATE TABLE IF NOT EXISTS unanswered_queries (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                inferred_domain TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_unanswered_domain ON unanswered_queries(inferred_domain);
            """
        )
        columns = {str(row[1]) for row in cur.execute("PRAGMA table_info(blueprints)")}
        if "prompt_key" not in columns:
            cur.execute("ALTER TABLE blueprints ADD COLUMN prompt_key TEXT NOT NULL DEFAULT ''")
        if "prompt_prefix" not in columns:
            cur.execute("ALTER TABLE blueprints ADD COLUMN prompt_prefix TEXT NOT NULL DEFAULT ''")
        # One-time, additive index backfill for vaults created before prompt dispatch existed.
        legacy_blueprints = cur.execute(
            "SELECT blueprint_id, canonical_json FROM blueprints WHERE prompt_key = ''"
        ).fetchall()
        for blueprint_id, canonical_json in legacy_blueprints:
            try:
                pattern = str(json.loads(canonical_json).get("prompt_pattern", ""))
                prompt_key, prompt_prefix = blueprint_prompt_fields(pattern)
                cur.execute(
                    "UPDATE blueprints SET prompt_key = ?, prompt_prefix = ? WHERE blueprint_id = ?",
                    (prompt_key, prompt_prefix, blueprint_id),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        cur.execute("CREATE INDEX IF NOT EXISTS idx_blueprints_prompt_prefix ON blueprints(prompt_prefix)")
        self.conn.commit()

    def _load_lightweight_meta(self) -> None:
        """Load ONLY compact token bit-barcodes & counters — ZERO CrystalRules loaded into RAM!"""
        cur = self.conn.cursor()
        row_stats = cur.execute("SELECT val FROM meta WHERE key = 'stats'").fetchone()
        if row_stats is not None:
            loaded_stats = json.loads(zlib.decompress(row_stats[0]).decode("utf-8"))
            self.stats.update(loaded_stats)
            if "domain_counts" in loaded_stats and isinstance(loaded_stats["domain_counts"], dict):
                self._domain_counts_cache = dict(loaded_stats["domain_counts"])
            if "trust_tier_counts" in loaded_stats and isinstance(loaded_stats["trust_tier_counts"], dict):
                self._trust_tier_counts_cache = dict(loaded_stats["trust_tier_counts"])
        row_bt = cur.execute("SELECT val FROM meta WHERE key = 'bit_table'").fetchone()
        if row_bt is not None:
            self.bit_table = ContextBitTable.from_state_dict(
                json.loads(zlib.decompress(row_bt[0]).decode("utf-8"))
            )
        row_max = cur.execute("SELECT COALESCE(MAX(rule_id) + 1, 0) FROM rules").fetchone()
        self._next_rule_id = int(row_max[0]) if row_max else 0

    def _save_lightweight_meta(self) -> None:
        bt_blob = zlib.compress(
            json.dumps(self.bit_table.to_state_dict(), separators=(",", ":")).encode("utf-8"), level=6
        )
        st_blob = zlib.compress(json.dumps(self.stats, separators=(",", ":")).encode("utf-8"), level=6)
        self.conn.executemany(
            "INSERT OR REPLACE INTO meta(key, val) VALUES (?, ?)",
            [("bit_table", bt_blob), ("stats", st_blob)],
        )

    @staticmethod
    def _prompt_hash(norm_prompt: str) -> bytes:
        return hashlib.blake2b(norm_prompt.encode("utf-8", errors="replace"), digest_size=16).digest()

    @staticmethod
    def _pack_rule_payload(
        salient: tuple[str, ...],
        abs_prompt: str,
        abs_resp: str,
        abs_tool: str,
        derived_expr: str,
        domain: str,
        trust_tier: int = 2,
        updated_seq: int = 1,
    ) -> bytes:
        obj = {
            "toks": list(salient),
            "ap": abs_prompt,
            "ar": abs_resp,
            "at": abs_tool,
            "de": derived_expr,
            "dom": domain,
            "tt": int(trust_tier),
            "seq": int(updated_seq),
        }
        return zlib.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"), level=6)

    def _row_to_rule(self, row: tuple[Any, ...]) -> CrystalRule:
        rid, sig_bytes, num_slots, pass_c, fail_c, payload = row
        rid_int = int(rid)
        if hasattr(self, "_rule_lru") and self.lru_rule_cache_size > 0:
            cached = self._rule_lru.get(rid_int)
            if cached is not None:
                self._rule_lru.move_to_end(rid_int)
                return cached

        d = json.loads(zlib.decompress(payload).decode("utf-8"))
        rule = CrystalRule(
            rule_id=rid_int,
            signature=int.from_bytes(sig_bytes, "little"),
            salient_tokens=tuple(d["toks"]),
            abstract_prompt=d["ap"],
            abstract_response=d["ar"],
            abstract_tool=d.get("at", ""),
            num_slots=int(num_slots),
            derived_expr=d.get("de", ""),
            pass_count=int(pass_c),
            fail_count=int(fail_c),
            domain=d.get("dom", "general"),
            trust_tier=int(d.get("tt", 2)),
            updated_seq=int(d.get("seq", rid_int + 1)),
        )
        if hasattr(self, "_rule_lru") and self.lru_rule_cache_size > 0:
            self._rule_lru[rid_int] = rule
            if len(self._rule_lru) > self.lru_rule_cache_size:
                self._rule_lru.popitem(last=False)
        return rule

    def _fetch_rules_batch(self, rids: list[int]) -> list[CrystalRule]:
        """Fetch candidate rules using bounded O(1) LRU cache + a single batched SQLite query for misses."""
        out: list[CrystalRule] = []
        misses: list[int] = []
        with self._lock:
            for rid in rids:
                cached = self._rule_lru.get(rid)
                if cached is not None:
                    self._rule_lru.move_to_end(rid)
                    out.append(cached)
                else:
                    misses.append(rid)

        if misses:
            qmarks = ",".join("?" for _ in misses)
            fetched: list[CrystalRule] = []
            for row in self.conn.execute(
                f"SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules WHERE rule_id IN ({qmarks})",
                misses,
            ):
                fetched.append(self._row_to_rule(row))
            with self._lock:
                for rule in fetched:
                    out.append(rule)
                    if self.lru_rule_cache_size > 0:
                        self._rule_lru[rule.rule_id] = rule
                        if len(self._rule_lru) > self.lru_rule_cache_size:
                            self._rule_lru.popitem(last=False)
        return out

    def rule_count(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) FROM rules").fetchone()
            return int(row[0]) if row else 0

    def get_rule(self, rule_id: int) -> Optional[CrystalRule]:
        """Fetch a single rule by rule_id via LRU cache first, then single-row SQLite query."""
        with self._lock:
            cached = self._rule_lru.get(rule_id)
            if cached is not None:
                self._rule_lru.move_to_end(rule_id)
                return cached
            row = self.conn.execute(
                "SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()
            if row is not None:
                rule = self._row_to_rule(row)
                if self.lru_rule_cache_size > 0:
                    self._rule_lru[rule.rule_id] = rule
                    if len(self._rule_lru) > self.lru_rule_cache_size:
                        self._rule_lru.popitem(last=False)
                return rule
            return None

    def get_trust_tier_counts(self) -> dict[str, int]:
        """Return rule counts grouped by trust_tier (O(1) memory cached)."""
        with self._lock:
            if self._trust_tier_counts_cache is not None:
                return dict(self._trust_tier_counts_cache)
            counts: dict[str, int] = {}
            for (payload,) in self.conn.execute("SELECT payload FROM rules"):
                tier = str(json.loads(zlib.decompress(payload).decode("utf-8")).get("tt", 2))
                counts[tier] = counts.get(tier, 0) + 1
            self._trust_tier_counts_cache = dict(counts)
            self.stats["trust_tier_counts"] = dict(counts)
            return dict(counts)

    def get_domain_counts(self) -> dict[str, int]:
        """Return rule counts grouped by domain label (O(1) memory cached)."""
        with self._lock:
            if self._domain_counts_cache is not None:
                return dict(self._domain_counts_cache)
            counts: dict[str, int] = {}
            for (payload,) in self.conn.execute("SELECT payload FROM rules"):
                dom = str(json.loads(zlib.decompress(payload).decode("utf-8")).get("dom", "general"))
                counts[dom] = counts.get(dom, 0) + 1
            self._domain_counts_cache = dict(counts)
            self.stats["domain_counts"] = dict(counts)
            return dict(counts)

    def record_feedback(self, rule_id: int, is_positive: bool = True) -> dict[str, Any]:
        """Record user feedback (upvote / downvote). Auto-deactivates if downvoted."""
        with self._lock:
            cur = self.conn.cursor()
            row = cur.execute(
                "SELECT pass_count, fail_count, payload FROM rules WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()
            if not row:
                return {"ok": False, "error": f"Rule #{rule_id} not found"}
            pc, fc, payload = row
            if is_positive:
                pc += 1
            else:
                fc += 1
            active = (fc <= pc) and (fc < 3)
            cur.execute(
                "UPDATE rules SET pass_count = ?, fail_count = ? WHERE rule_id = ?",
                (pc, fc, rule_id),
            )
            self.conn.commit()
            self._rule_lru.pop(rule_id, None)
            return {
                "ok": True,
                "rule_id": rule_id,
                "pass_count": pc,
                "fail_count": fc,
                "active": active,
                "feedback": "upvote" if is_positive else "downvote",
            }

    def purge_by_domain(self, domain: str) -> dict[str, Any]:
        """Surgically remove all rules belonging to a specific domain."""
        with self._lock:
            cur = self.conn.cursor()
            rids_to_del = []
            for (rid, payload) in cur.execute("SELECT rule_id, payload FROM rules").fetchall():
                try:
                    d = json.loads(zlib.decompress(payload).decode("utf-8"))
                    if d.get("dom", "") == domain:
                        rids_to_del.append(rid)
                except Exception:
                    pass
            purged = 0
            if rids_to_del:
                qmarks = ",".join("?" for _ in rids_to_del)
                cur.execute(f"DELETE FROM rules WHERE rule_id IN ({qmarks})", rids_to_del)
                cur.execute(f"DELETE FROM prompt_aliases WHERE rule_id IN ({qmarks})", rids_to_del)
                for r in rids_to_del:
                    self._rule_lru.pop(r, None)
                purged = len(rids_to_del)
                self.conn.commit()
                self._domain_counts_cache = None
                self._trust_tier_counts_cache = None
            return {"ok": True, "domain": domain, "purged_rules": purged}

    def delete_rule(self, rule_id: int) -> bool:
        """Delete a single rule by ID."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
            cur.execute("DELETE FROM prompt_aliases WHERE rule_id = ?", (rule_id,))
            deleted = cur.rowcount > 0
            if deleted:
                self._rule_lru.pop(rule_id, None)
                self.conn.commit()
                self._domain_counts_cache = None
                self._trust_tier_counts_cache = None
            return deleted

    def compact(self) -> dict[str, Any]:
        """Compacts the SQLite database and executes VACUUM to reclaim disk space."""
        with self._lock:
            b_before = self.db_path.stat().st_size if self.db_path.exists() else 0
            cur = self.conn.cursor()
            cur.execute("PRAGMA auto_vacuum = INCREMENTAL;")
            cur.execute("PRAGMA incremental_vacuum;")
            self.conn.commit()
            cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            cur.execute("VACUUM;")
            b_after = self.db_path.stat().st_size if self.db_path.exists() else 0
            return {
                "bytes_before": b_before,
                "bytes_after": b_after,
                "bytes_reclaimed": max(0, b_before - b_after),
                "mb_before": round(b_before / (1024 * 1024), 2),
                "mb_after": round(b_after / (1024 * 1024), 2),
            }

    def optimize_disk(self) -> int:
        """Flush dirty metadata + WAL checkpoint so the .dcrystal file is a single portable file on disk."""
        with self._lock:
            if self._meta_dirty:
                with self.conn:
                    self._save_lightweight_meta()
                self._meta_dirty = False
            try:
                self.conn.commit()
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                self.conn.commit()
            except Exception:
                pass
            return self.db_path.stat().st_size

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._meta_dirty:
                    with self.conn:
                        self._save_lightweight_meta()
                    self._meta_dirty = False
            except Exception:
                pass
            self._closed = True
            for c in self._conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._conns.clear()
            try:
                if hasattr(self._local, "conn"):
                    delattr(self._local, "conn")
            except Exception:
                pass
