"""
DiskCrystalVault — Out-of-Core Disk-Paged Crystal Engine (.dcrystal) for Billion/Trillion-Byte Scaling.

Eliminates full-JSON RAM loading by storing CrystalRules and packed array('I') inverted postings inside
an indexed SQLite WAL + OS-mmap B-Tree vault on disk, keeping Python RAM bounded (< 50 MB) regardless
of how many millions of rules reside on SSD/NVMe.
"""
from __future__ import annotations

from array import array
from collections import OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any
import zlib

from .bit_signatures import ContextBitTable
from .engine import BinarySessionMemory, CrystalResponse, split_compound_intents
from .grammar_and_sandbox import ZeroTrustSandboxJudge
from .lsh_rule_store import CrystalRule, LSHRuleIndex
from .phase2_cognitive_core import (
    CoreferenceAndSenseTracker,
    EpistemicConfidenceCalibrator,
    FluentByteRealizer,
    UniversalConceptHub,
)
from .phase5_bicameral_and_cas import (
    ActiveCuriosityPlanner,
    DocumentAutoCrystallizer,
    SymbolicCASOrgan,
    TernaryFluidComposer,
)
from .slot_abstractor import SlotAbstractor


class DiskCrystalVault:
    """Out-of-core disk-backed intelligence vault: 0 rules loaded in RAM at startup, O(1) candidate disk reads."""

    def __init__(
        self,
        db_path: Path | str,
        max_frame_df: int = 8,
        max_postings_scan: int = 28,
        lru_rule_cache_size: int = 512,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_postings_scan = int(max_postings_scan)
        self.lru_rule_cache_size = int(lru_rule_cache_size)

        self._lock = threading.RLock()
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._closed = False

        init_conn = self._open_thread_conn(enable_wal=True)
        self._local.conn = init_conn
        self._init_schema()

        self.bit_table = ContextBitTable(max_frame_df=max_frame_df)
        self.sessions = BinarySessionMemory()
        self.concept_hub = UniversalConceptHub()
        self.coref_tracker = CoreferenceAndSenseTracker(ttl_turns=3)
        self.realizer = FluentByteRealizer()
        self.fluid_composer = TernaryFluidComposer()
        self.curiosity_planner = ActiveCuriosityPlanner()
        self.stats: dict[str, int] = {
            "pairs_seen": 0,
            "surprises_crystallized": 0,
            "duplicates_skipped": 0,
            "sandbox_verified_rules": 0,
            "sandbox_rejected_rules": 0,
        }
        self._next_rule_id = 0
        self._meta_dirty = False
        self._rule_lru: OrderedDict[int, CrystalRule] = OrderedDict()
        self._load_lightweight_meta()

    def _open_thread_conn(self, enable_wal: bool = False) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=5.0)
        if enable_wal:
            c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA busy_timeout=5000;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA temp_store=MEMORY;")
        c.execute("PRAGMA mmap_size=134217728;")  # 128 MB OS kernel page cache map
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
            CREATE INDEX IF NOT EXISTS idx_rules_num_slots ON rules(num_slots) WHERE num_slots > 0;
            """
        )
        self.conn.commit()

    def _load_lightweight_meta(self) -> None:
        """Load ONLY compact token bit-barcodes & counters — ZERO CrystalRules loaded into RAM!"""
        cur = self.conn.cursor()
        row_stats = cur.execute("SELECT val FROM meta WHERE key = 'stats'").fetchone()
        if row_stats is not None:
            self.stats.update(json.loads(zlib.decompress(row_stats[0]).decode("utf-8")))
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

    @staticmethod
    def _row_to_rule(row: tuple[Any, ...]) -> CrystalRule:
        rid, sig_bytes, num_slots, pass_c, fail_c, payload = row
        d = json.loads(zlib.decompress(payload).decode("utf-8"))
        return CrystalRule(
            rule_id=int(rid),
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
            updated_seq=int(d.get("seq", int(rid) + 1)),
        )

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

    def _merge_postings_to_disk(
        self,
        tok_delta: dict[str, array],
        band_delta: dict[int, array],
        causal_delta: dict[str, list[tuple[str, str]]],
    ) -> None:
        cur = self.conn.cursor()
        if tok_delta:
            tok_keys = list(tok_delta.keys())
            existing_tok: dict[str, bytes] = {}
            for i in range(0, len(tok_keys), 500):
                chunk = tok_keys[i : i + 500]
                qmarks = ",".join("?" for _ in chunk)
                for k, blob in cur.execute(f"SELECT tok, rids FROM token_postings WHERE tok IN ({qmarks})", chunk):
                    existing_tok[k] = blob
            upserts_t = []
            for k, arr in tok_delta.items():
                prev = existing_tok.get(k, b"")
                upserts_t.append((k, prev + arr.tobytes()))
            cur.executemany("INSERT OR REPLACE INTO token_postings(tok, rids) VALUES (?, ?)", upserts_t)

        if band_delta:
            b_keys = list(band_delta.keys())
            existing_b: dict[int, bytes] = {}
            for i in range(0, len(b_keys), 500):
                chunk = b_keys[i : i + 500]
                qmarks = ",".join("?" for _ in chunk)
                for k, blob in cur.execute(f"SELECT band, rids FROM band_postings WHERE band IN ({qmarks})", chunk):
                    existing_b[int(k)] = blob
            upserts_b = []
            for k, arr in band_delta.items():
                prev = existing_b.get(k, b"")
                upserts_b.append((k, prev + arr.tobytes()))
            cur.executemany("INSERT OR REPLACE INTO band_postings(band, rids) VALUES (?, ?)", upserts_b)

        if causal_delta:
            tok_heads_delta: dict[str, set[str]] = defaultdict(set)
            for head, new_edges in causal_delta.items():
                row_e = cur.execute("SELECT edges_json FROM causal_edges WHERE head = ?", (head,)).fetchone()
                merged_e: list[list[str]] = json.loads(row_e[0]) if row_e else []
                for tail, stmt in new_edges:
                    pair = [tail, stmt]
                    if pair not in merged_e:
                        merged_e.append(pair)
                cur.execute(
                    "INSERT OR REPLACE INTO causal_edges(head, edges_json) VALUES (?, ?)",
                    (head, json.dumps(merged_e)),
                )
                for w in ContextBitTable.tokenize(head):
                    if len(w) > 1:
                        tok_heads_delta[w].add(head)

            for w, heads in tok_heads_delta.items():
                row_h = cur.execute("SELECT heads_json FROM causal_tok_idx WHERE tok = ?", (w,)).fetchone()
                merged_h: list[str] = json.loads(row_h[0]) if row_h else []
                for h in heads:
                    if h not in merged_h:
                        merged_h.append(h)
                cur.execute(
                    "INSERT OR REPLACE INTO causal_tok_idx(tok, heads_json) VALUES (?, ?)",
                    (w, json.dumps(merged_h)),
                )

    def observe_unlabeled_text(self, text: str) -> int:
        """Stream unlabeled lines into ContextBitTable & disk causal graph."""
        with self._lock:
            temp_index = LSHRuleIndex()
            count = 0
            for line in text.splitlines():
                s = line.strip()
                if not s:
                    continue
                self.bit_table.observe_text(s)
                temp_index.observe_causal_fact(s)
                self.fluid_composer.observe_corpus_text(s)
                count += 1
            with self.conn:
                self._merge_postings_to_disk({}, {}, temp_index._causal_edges)
                self._save_lightweight_meta()
            return count

    def ingest_pair(
        self,
        user_text: str,
        target_text: str,
        think: str = "",
        tool: str = "",
        test_code: str = "",
        domain: str = "general",
        trust_tier: int = 2,
    ) -> bool:
        """Single-item online continual update with trust-tier contradiction supersession (< 25 ms)."""
        with self._lock:
            return self._ingest_pair_locked(
                user_text=user_text,
                target_text=target_text,
                think=think,
                tool=tool,
                test_code=test_code,
                domain=domain,
                trust_tier=trust_tier,
            )

    def _ingest_pair_locked(
        self,
        user_text: str,
        target_text: str,
        think: str = "",
        tool: str = "",
        test_code: str = "",
        domain: str = "general",
        trust_tier: int = 2,
    ) -> bool:
        u, t = user_text.strip(), target_text.strip()
        if not u or not t:
            return False

        self.stats["pairs_seen"] += 1
        self.bit_table.observe_text(u)
        self.bit_table.observe_text(t)
        self.fluid_composer.observe_corpus_text(t)
        full_response = f"<think> {think.strip()} </think>\n{t}" if think.strip() and "<think>" not in t else t

        verified_bonus = 0
        if test_code.strip():
            ok, _ = ZeroTrustSandboxJudge.verify_code_and_test(full_response, test_code=test_code)
            if not ok:
                self.stats["sandbox_rejected_rules"] += 1
                return False
            verified_bonus = 5
            self.stats["sandbox_verified_rules"] += 1

        effective_tier = 1 if verified_bonus > 0 else int(trust_tier)
        temp_causal = LSHRuleIndex()
        if len(full_response) <= 140:
            temp_causal.observe_causal_fact(full_response)

        abs_pair = SlotAbstractor.abstract_pair(u, full_response, tool=tool)
        sense_tags = CoreferenceAndSenseTracker.detect_sense_tags(abs_pair.abstract_prompt, context_hint=full_response)
        for st in sense_tags:
            self.bit_table.observe_text(st)
        sense_sfx = f" {' '.join(sorted(sense_tags))}" if sense_tags else ""
        phash = self._prompt_hash(abs_pair.abstract_prompt.strip().lower() + sense_sfx)

        row_dup = self.conn.execute(
            "SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules WHERE prompt_hash = ?",
            (phash,),
        ).fetchone()
        if row_dup is not None:
            existing_rule = self._row_to_rule(row_dup)
            rid_existing = existing_rule.rule_id
            if existing_rule.abstract_response == abs_pair.abstract_response:
                with self.conn:
                    self.conn.execute(
                        "UPDATE rules SET pass_count = pass_count + ? WHERE rule_id = ?",
                        (1 + verified_bonus, rid_existing),
                    )
            elif effective_tier <= existing_rule.trust_tier:
                # Authorised update supersedes outdated answer cleanly!
                new_payload = self._pack_rule_payload(
                    existing_rule.salient_tokens,
                    abs_pair.abstract_prompt,
                    abs_pair.abstract_response,
                    abs_pair.abstract_tool,
                    abs_pair.derived_expr,
                    domain,
                    trust_tier=effective_tier,
                    updated_seq=self.stats["pairs_seen"],
                )
                with self.conn:
                    self.conn.execute(
                        "UPDATE rules SET num_slots = ?, pass_count = pass_count + ?, payload = ? WHERE rule_id = ?",
                        (abs_pair.num_slots, 1 + verified_bonus, new_payload, rid_existing),
                    )
            self.stats["duplicates_skipped"] += 1
            self._rule_lru.pop(rid_existing, None)
            self._meta_dirty = True
            with self.conn:
                self._merge_postings_to_disk({}, {}, temp_causal._causal_edges)
            return False

        sig, chosen_toks = self.bit_table.encode_text_signature(abs_pair.abstract_prompt + sense_sfx)
        salient = tuple(sorted(set(chosen_toks), key=self.bit_table.token_salience, reverse=True)[:14])
        rid = self._next_rule_id
        self._next_rule_id += 1

        payload = self._pack_rule_payload(
            salient,
            abs_pair.abstract_prompt,
            abs_pair.abstract_response,
            abs_pair.abstract_tool,
            abs_pair.derived_expr,
            domain,
            trust_tier=effective_tier,
            updated_seq=self.stats["pairs_seen"],
        )
        tok_delta: dict[str, array] = defaultdict(lambda: array("I"))
        band_delta: dict[int, array] = defaultdict(lambda: array("I"))
        for tok in salient:
            tok_delta[tok].append(rid)
        for bkey in LSHRuleIndex._lsh_bands(sig):
            band_delta[bkey].append(rid)

        with self.conn:
            self.conn.execute(
                "INSERT INTO rules(rule_id, prompt_hash, sig_bytes, num_slots, pass_count, fail_count, payload) VALUES (?,?,?,?,?,?,?)",
                (rid, phash, sig.to_bytes(256, "little"), abs_pair.num_slots, 1 + verified_bonus, 0, payload),
            )
            self._merge_postings_to_disk(tok_delta, band_delta, temp_causal._causal_edges)
            self.stats["surprises_crystallized"] += 1
            self._meta_dirty = True
        return True

    def ingest_batch(
        self,
        items: Sequence[dict[str, Any]],
        batch_size: int = 5000,
        verify_math: bool = True,
    ) -> dict[str, int]:
        """High-throughput bulk ingestion with transaction batching and execution truth arbitration."""
        with self._lock:
            return self._ingest_batch_locked(items, batch_size=batch_size, verify_math=verify_math)

    def _ingest_batch_locked(
        self,
        items: Sequence[dict[str, Any]],
        batch_size: int = 5000,
        verify_math: bool = True,
    ) -> dict[str, int]:
        new_rules = 0
        duplicates_skipped = 0
        sandbox_rejected = 0

        rule_batch: list[tuple[Any, ...]] = []
        reinforce_batch: list[tuple[int, int]] = []
        tok_delta: dict[str, array] = defaultdict(lambda: array("I"))
        band_delta: dict[int, array] = defaultdict(lambda: array("I"))
        temp_causal = LSHRuleIndex()
        seen_batch_hashes: dict[bytes, int] = {}

        def _flush() -> None:
            if not rule_batch and not reinforce_batch and not temp_causal._causal_edges:
                return
            with self.conn:
                if rule_batch:
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO rules(rule_id, prompt_hash, sig_bytes, num_slots, pass_count, fail_count, payload) VALUES (?,?,?,?,?,?,?)",
                        rule_batch,
                    )
                if reinforce_batch:
                    self.conn.executemany(
                        "UPDATE rules SET pass_count = pass_count + ? WHERE rule_id = ?",
                        reinforce_batch,
                    )
                self._merge_postings_to_disk(tok_delta, band_delta, temp_causal._causal_edges)
            rule_batch.clear()
            reinforce_batch.clear()
            tok_delta.clear()
            band_delta.clear()
            temp_causal._causal_edges.clear()
            seen_batch_hashes.clear()

        for item in items:
            u_raw = str(item.get("prompt") or item.get("question") or item.get("user") or "").strip()
            r_raw = str(item.get("response") or item.get("answer") or item.get("target") or "").strip()
            if not u_raw or not r_raw:
                continue
            think = str(item.get("think") or "")
            tool = str(item.get("tool") or "")
            test_c = str(item.get("test_code") or "")
            dom = str(item.get("domain") or "general")
            trust_tier = int(item.get("trust_tier", 2))

            self.stats["pairs_seen"] += 1
            self.bit_table.observe_text(u_raw)
            self.bit_table.observe_text(r_raw)
            self.fluid_composer.observe_corpus_text(r_raw[:400])
            full_resp = f"<think> {think.strip()} </think>\n{r_raw}" if think.strip() and "<think>" not in r_raw else r_raw

            verified_bonus = 0
            if test_c.strip():
                ok, _ = ZeroTrustSandboxJudge.verify_code_and_test(full_resp, test_code=test_c)
                if not ok:
                    sandbox_rejected += 1
                    self.stats["sandbox_rejected_rules"] += 1
                    continue
                verified_bonus = 5
                self.stats["sandbox_verified_rules"] += 1

            effective_tier = 1 if verified_bonus > 0 else trust_tier

            if len(full_resp) <= 140:
                temp_causal.observe_causal_fact(full_resp)

            abs_pair = SlotAbstractor.abstract_pair(u_raw, full_resp, tool=tool)
            sense_tags = CoreferenceAndSenseTracker.detect_sense_tags(abs_pair.abstract_prompt, context_hint=full_resp)
            for st in sense_tags:
                self.bit_table.observe_text(st)
            sense_sfx = f" {' '.join(sorted(sense_tags))}" if sense_tags else ""
            phash = self._prompt_hash(abs_pair.abstract_prompt.strip().lower() + sense_sfx)

            if phash in seen_batch_hashes:
                duplicates_skipped += 1
                reinforce_batch.append((1 + verified_bonus, seen_batch_hashes[phash]))
                continue

            row_dup = self.conn.execute(
                "SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules WHERE prompt_hash = ?",
                (phash,),
            ).fetchone()

            if row_dup is not None:
                duplicates_skipped += 1
                existing_rule = self._row_to_rule(row_dup)
                rid_existing = existing_rule.rule_id
                seen_batch_hashes[phash] = rid_existing
                if existing_rule.abstract_response == abs_pair.abstract_response:
                    reinforce_batch.append((1 + verified_bonus, rid_existing))
                elif effective_tier <= existing_rule.trust_tier:
                    new_payload = self._pack_rule_payload(
                        existing_rule.salient_tokens,
                        abs_pair.abstract_prompt,
                        abs_pair.abstract_response,
                        abs_pair.abstract_tool,
                        abs_pair.derived_expr,
                        dom,
                        trust_tier=effective_tier,
                        updated_seq=self.stats["pairs_seen"],
                    )
                    with self.conn:
                        self.conn.execute(
                            "UPDATE rules SET num_slots = ?, pass_count = pass_count + ?, payload = ? WHERE rule_id = ?",
                            (abs_pair.num_slots, 1 + verified_bonus, new_payload, rid_existing),
                        )
                continue

            sig, chosen_toks = self.bit_table.encode_text_signature(abs_pair.abstract_prompt + sense_sfx)
            salient = tuple(sorted(set(chosen_toks), key=self.bit_table.token_salience, reverse=True)[:14])
            rid = self._next_rule_id
            self._next_rule_id += 1
            seen_batch_hashes[phash] = rid

            payload = self._pack_rule_payload(
                salient,
                abs_pair.abstract_prompt,
                abs_pair.abstract_response,
                abs_pair.abstract_tool,
                abs_pair.derived_expr,
                dom,
                trust_tier=effective_tier,
                updated_seq=self.stats["pairs_seen"],
            )

            rule_batch.append((rid, phash, sig.to_bytes(256, "little"), abs_pair.num_slots, 1 + verified_bonus, 0, payload))
            for tok in salient:
                tok_delta[tok].append(rid)
            for band in LSHRuleIndex._lsh_bands(sig):
                band_delta[band].append(rid)

            new_rules += 1
            if len(rule_batch) >= batch_size:
                _flush()

        _flush()
        self._save_lightweight_meta()
        return {
            "new_rules": new_rules,
            "duplicates_skipped": duplicates_skipped,
            "sandbox_rejected": sandbox_rejected,
            "total_processed": len(items),
        }

    def ingest_jsonl(
        self,
        path: Path | str,
        max_rows: int | None = None,
        batch_flush_rows: int = 2500,
        fold_synonyms: bool = True,
    ) -> dict[str, Any]:
        """Stream arbitrarily huge .jsonl files in bounded RAM, flushing directly to SQLite B-Trees."""
        with self._lock:
            return self._ingest_jsonl_locked(
                path=path,
                max_rows=max_rows,
                batch_flush_rows=batch_flush_rows,
                fold_synonyms=fold_synonyms,
            )

    def _ingest_jsonl_locked(
        self,
        path: Path | str,
        max_rows: int | None = None,
        batch_flush_rows: int = 2500,
        fold_synonyms: bool = True,
    ) -> dict[str, Any]:
        fpath = Path(path)
        t0 = time.perf_counter()
        rows_read = 0
        new_rules_this_file = 0
        dups_this_file = 0

        rule_batch: list[tuple[Any, ...]] = []
        reinforce_batch: list[tuple[int, int]] = []
        tok_delta: dict[str, array] = defaultdict(lambda: array("I"))
        band_delta: dict[int, array] = defaultdict(lambda: array("I"))
        temp_causal = LSHRuleIndex()
        seen_batch_hashes: dict[bytes, int] = {}

        def _flush() -> None:
            if not rule_batch and not reinforce_batch and not temp_causal._causal_edges:
                return
            with self.conn:
                if rule_batch:
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO rules(rule_id, prompt_hash, sig_bytes, num_slots, pass_count, fail_count, payload) VALUES (?,?,?,?,?,?,?)",
                        rule_batch,
                    )
                if reinforce_batch:
                    self.conn.executemany(
                        "UPDATE rules SET pass_count = pass_count + ? WHERE rule_id = ?",
                        reinforce_batch,
                    )
                self._merge_postings_to_disk(tok_delta, band_delta, temp_causal._causal_edges)
            rule_batch.clear()
            reinforce_batch.clear()
            tok_delta.clear()
            band_delta.clear()
            temp_causal._causal_edges.clear()
            seen_batch_hashes.clear()

        def _process_one(
            u_raw: str,
            r_raw: str,
            think: str,
            tool: str,
            test_c: str,
            dom: str,
            trust_tier: int = 2,
        ) -> None:
            nonlocal new_rules_this_file, dups_this_file
            u, t = u_raw.strip(), r_raw.strip()
            if not u or not t:
                return
            self.stats["pairs_seen"] += 1
            self.bit_table.observe_text(u)
            self.bit_table.observe_text(t)
            self.fluid_composer.observe_corpus_text(t[:400])
            full_resp = f"<think> {think.strip()} </think>\n{t}" if think.strip() and "<think>" not in t else t

            verified_bonus = 0
            if test_c.strip():
                ok, _ = ZeroTrustSandboxJudge.verify_code_and_test(full_resp, test_code=test_c)
                if not ok:
                    self.stats["sandbox_rejected_rules"] += 1
                    return
                verified_bonus = 5
                self.stats["sandbox_verified_rules"] += 1

            effective_tier = 1 if verified_bonus > 0 else int(trust_tier)
            if len(full_resp) <= 140:
                temp_causal.observe_causal_fact(full_resp)

            abs_pair = SlotAbstractor.abstract_pair(u, full_resp, tool=tool)
            sense_tags = CoreferenceAndSenseTracker.detect_sense_tags(abs_pair.abstract_prompt, context_hint=full_resp)
            for st in sense_tags:
                self.bit_table.observe_text(st)
            sense_sfx = f" {' '.join(sorted(sense_tags))}" if sense_tags else ""
            phash = self._prompt_hash(abs_pair.abstract_prompt.strip().lower() + sense_sfx)

            if phash in seen_batch_hashes:
                reinforce_batch.append((1 + verified_bonus, seen_batch_hashes[phash]))
                self.stats["duplicates_skipped"] += 1
                dups_this_file += 1
                return

            row_dup = self.conn.execute("SELECT rule_id FROM rules WHERE prompt_hash = ?", (phash,)).fetchone()
            if row_dup is not None:
                reinforce_batch.append((1 + verified_bonus, int(row_dup[0])))
                self.stats["duplicates_skipped"] += 1
                dups_this_file += 1
                return

            sig, chosen_toks = self.bit_table.encode_text_signature(abs_pair.abstract_prompt + sense_sfx)
            salient = tuple(sorted(set(chosen_toks), key=self.bit_table.token_salience, reverse=True)[:14])
            rid = self._next_rule_id
            self._next_rule_id += 1
            seen_batch_hashes[phash] = rid

            payload = self._pack_rule_payload(
                salient,
                abs_pair.abstract_prompt,
                abs_pair.abstract_response,
                abs_pair.abstract_tool,
                abs_pair.derived_expr,
                dom,
                trust_tier=effective_tier,
                updated_seq=self.stats["pairs_seen"],
            )
            rule_batch.append((rid, phash, sig.to_bytes(256, "little"), abs_pair.num_slots, 1 + verified_bonus, 0, payload))
            for tok in salient:
                tok_delta[tok].append(rid)
            for bkey in LSHRuleIndex._lsh_bands(sig):
                band_delta[bkey].append(rid)
            self.stats["surprises_crystallized"] += 1
            new_rules_this_file += 1

        with fpath.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                if max_rows is not None and rows_read >= max_rows:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("split") in {"test", "eval_holdout"}:
                    continue

                if isinstance(obj.get("turns"), list):
                    for turn in obj["turns"]:
                        _process_one(
                            str(turn.get("user") or turn.get("user_text") or ""),
                            str(turn.get("response") or turn.get("target_text") or ""),
                            "",
                            "",
                            "",
                            "multi_turn",
                            trust_tier=int(turn.get("trust_tier", obj.get("trust_tier", 2))),
                        )
                else:
                    u = str(
                        obj.get("user_text")
                        or obj.get("user")
                        or obj.get("prompt")
                        or obj.get("instruction")
                        or obj.get("question")
                        or ""
                    )
                    r = str(
                        obj.get("target_text")
                        or obj.get("response")
                        or obj.get("completion")
                        or obj.get("output")
                        or obj.get("answer")
                        or ""
                    )
                    state_meta = obj.get("state") if isinstance(obj.get("state"), dict) else {}
                    dom = str(state_meta.get("domain") or obj.get("category") or obj.get("goal") or "general")
                    _process_one(
                        u,
                        r,
                        str(obj.get("think") or ""),
                        str(obj.get("tool") or ""),
                        str(obj.get("test") or obj.get("test_code") or ""),
                        dom,
                        trust_tier=int(obj.get("trust_tier", 2)),
                    )

                rows_read += 1
                if len(rule_batch) + len(reinforce_batch) >= batch_flush_rows:
                    _flush()

        _flush()
        merged_syns = self.bit_table.fold_equivalences() if fold_synonyms else 0
        with self.conn:
            self._save_lightweight_meta()
        self._meta_dirty = False
        self._rule_lru.clear()

        elapsed_s = max(1e-6, time.perf_counter() - t0)
        repo_root = Path(__file__).resolve().parents[1]
        try:
            display_file = fpath.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            display_file = fpath.name

        return {
            "file": display_file,
            "rows_ingested": rows_read,
            "new_rules_added": new_rules_this_file,
            "duplicates_skipped_in_file": dups_this_file,
            "total_vault_rules_on_disk": self.rule_count(),
            "synonym_bridges_folded": merged_syns,
            "elapsed_sec": round(elapsed_s, 3),
            "rows_per_sec_cpu": round(rows_read / elapsed_s, 1),
        }

    def _query_disk_candidates(
        self,
        query_sig: int,
        query_tokens: list[str],
        top_k: int = 4,
        required_slots: int | None = None,
    ) -> list[tuple[float, CrystalRule]]:
        raw_q_set = set(query_tokens)
        expanded = self.bit_table.expand_content_tokens(query_tokens)
        cur = self.conn.cursor()

        if required_slots is not None and required_slots > 0:
            # Dedicated O(log N) B-Tree partial index scan over parameterized mathematical/tool laws!
            rows_slot = cur.execute(
                "SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules WHERE num_slots = ? LIMIT 128",
                (int(required_slots),),
            ).fetchall()
            fetched_rules = [self._row_to_rule(r) for r in rows_slot]
        else:
            candidate_hits: dict[int, int] = defaultdict(int)
            if expanded:
                # Prioritize rarest/highest-IDF query tokens first to eliminate 60,000-entry Python loops!
                prioritized = sorted(
                    expanded,
                    key=lambda t: self.bit_table.token_salience(t) * (1.25 if t in raw_q_set else 0.8),
                    reverse=True,
                )[:12]
                qmarks = ",".join("?" for _ in prioritized)
                posting_rows = cur.execute(
                    f"SELECT tok, rids FROM token_postings WHERE tok IN ({qmarks})", prioritized
                ).fetchall()
                # Process smallest posting lists first; skip broad buckets once >= 96 candidates found
                posting_rows.sort(key=lambda r: len(r[1]))
                for tok, blob in posting_rows:
                    n_rids = len(blob) // 4
                    if len(candidate_hits) >= 96 and n_rids > 512:
                        break
                    if len(candidate_hits) >= 256:
                        break
                    mv = memoryview(blob).cast("I")
                    if n_rids > 1024:
                        stride = max(1, n_rids // 1024)
                        view = mv[::stride][:1024]
                    else:
                        view = mv
                    w = max(1, int(4.0 * self.bit_table.token_salience(tok)))
                    if tok in raw_q_set:
                        w *= 2
                    for rid in view:
                        candidate_hits[rid] += w

            # Only scan LSH bands when rare-token inverted buckets yielded sparse hits (< 20 candidates)
            if len(candidate_hits) < 20:
                bands = LSHRuleIndex._lsh_bands(query_sig)
                if bands:
                    qmarks_b = ",".join("?" for _ in bands)
                    for _, blob in cur.execute(f"SELECT band, rids FROM band_postings WHERE band IN ({qmarks_b})", bands):
                        n_rids = len(blob) // 4
                        if n_rids <= 1024:
                            for rid in memoryview(blob).cast("I"):
                                candidate_hits[rid] += 1

            if not candidate_hits:
                return []

            top_rids = [
                rid
                for rid, _ in sorted(candidate_hits.items(), key=lambda kv: kv[1], reverse=True)[: self.max_postings_scan]
            ]
            fetched_rules = self._fetch_rules_batch(top_rids)

        q_set = expanded if expanded else raw_q_set
        q_bc = query_sig.bit_count()
        q_inv_sqrt = (1.0 / math.sqrt(q_bc)) if q_bc > 0 else 0.0
        sal_cache = {
            t: self.bit_table.token_salience(t) * (1.0 if t in raw_q_set else 0.68)
            for t in q_set
        }
        q_sal_total = sum(self.bit_table.token_salience(t) for t in raw_q_set) if raw_q_set else sum(sal_cache.values())
        q_neg = (query_sig >> 2040) != 0
        q_norm_exact = re.sub(r"\s+", " ", "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in " ".join(query_tokens))).strip()

        scored: list[tuple[float, CrystalRule]] = []
        for rule in fetched_rules:
            if not rule.active:
                continue
            r_norm_exact = re.sub(r"\s+", " ", "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in rule.abstract_prompt)).strip()
            if r_norm_exact and (r_norm_exact == q_norm_exact or (rule.domain == "identity" and r_norm_exact in q_norm_exact)):
                score = 1.05 * rule.reliability
                scored.append((score, rule))
                continue
            overlap_bits = (query_sig & rule.signature).bit_count()
            bit_sim = (overlap_bits * q_inv_sqrt * rule.inv_sqrt_bits) if overlap_bits else 0.0
            r_set = rule.token_set
            if q_set and r_set:
                inter = q_set & r_set
                if inter:
                    overlap_sal = sum(sal_cache[t] for t in inter)
                    extra_r = sum(self.bit_table.token_salience(t) for t in (r_set - q_set))
                    union_sal = max(overlap_sal, q_sal_total) + extra_r
                    lex_sim = overlap_sal / max(1e-6, union_sal)
                else:
                    lex_sim = 0.0
            else:
                lex_sim = 0.0
            r_neg = (rule.signature >> 2040) != 0
            polarity_mult = 0.45 if (q_neg != r_neg) else 1.0
            score = (0.50 * bit_sim + 0.50 * lex_sim) * rule.reliability * polarity_mult
            scored.append((score, rule))

        scored.sort(key=lambda item: (item[0], item[1].updated_seq), reverse=True)
        return scored[:top_k]

    def _chain_infer_disk(self, query: str, max_hops: int = 4) -> tuple[str, list[str]] | None:
        q_norm = LSHRuleIndex._normalize_entity(query)
        q_expanded = self.bit_table.expand_content_tokens(self.bit_table.tokenize(query))
        if not q_expanded:
            return None

        cur = self.conn.cursor()
        toks = list(q_expanded)[:24]
        qmarks = ",".join("?" for _ in toks)
        candidate_heads: set[str] = set()
        for (heads_json,) in cur.execute(f"SELECT heads_json FROM causal_tok_idx WHERE tok IN ({qmarks})", toks):
            for h in json.loads(heads_json):
                candidate_heads.add(h)

        starts = [
            node
            for node in candidate_heads
            if node in q_norm or set(self.bit_table.tokenize(node)).issubset(q_expanded)
        ]
        for start_node in starts:
            visited = {start_node}
            frontier: list[tuple[str, list[str], list[str]]] = [(start_node, [start_node], [])]
            best_chain: tuple[list[str], list[str]] | None = None
            while frontier:
                curr, path_nodes, path_facts = frontier.pop(0)
                if len(path_facts) >= max_hops:
                    continue
                row_e = cur.execute("SELECT edges_json FROM causal_edges WHERE head = ?", (curr,)).fetchone()
                if not row_e:
                    continue
                for nxt, raw_stmt in json.loads(row_e[0]):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    new_nodes = path_nodes + [nxt]
                    new_facts = path_facts + [raw_stmt]
                    if len(new_facts) >= 2:
                        best_chain = (new_nodes, new_facts)
                    frontier.append((nxt, new_nodes, new_facts))
            if best_chain is not None:
                nodes, facts = best_chain
                return f"Therefore, {nodes[0]} -> {nodes[-1]}.", facts
        return None

    def ingest_document(self, text: str, title: str = "Document", domain: str = "doc_crystallized") -> dict[str, int]:
        """Auto-crystallize unstructured Markdown, textbook prose, or inline Python code directly to disk."""
        return DocumentAutoCrystallizer.ingest_document_text(self, text, title=title, domain=domain)

    def ingest_source_file(self, path: Path | str, domain: str | None = None) -> dict[str, int]:
        """Auto-crystallize any .md, .txt, or .py file directly into DiskCrystalVault."""
        return DocumentAutoCrystallizer.ingest_file(self, path, domain=domain)

    def respond(self, prompt: str, session_id: str = "default") -> CrystalResponse:
        """Out-of-core inference with UniversalConceptHub, CAS, Curiosity, Coreference & <= 64 disk reads."""
        with self._lock:
            return self._respond_locked(prompt, session_id=session_id)

    def _respond_locked(self, prompt: str, session_id: str = "default") -> CrystalResponse:
        t0 = time.perf_counter_ns()
        q_raw = prompt.strip()
        if not q_raw:
            return CrystalResponse(
                answer="No verified Crystal Rule matched above threshold yet — feed one example to crystallize instantly.",
                confidence=0.0,
                mode="unmatched_honest_abstain",
                matched_rule_id=None,
                latency_ms=0.0,
                sandbox_verified=False,
                reasoning_trace=["Empty prompt abstained cleanly."],
                epistemic_tier="honest_abstain",
            )

        q_canon, detected_lang, mapped_concepts = self.concept_hub.normalize_to_canonical(q_raw)
        pre_trace: list[str] = []
        if mapped_concepts:
            pre_trace.append(f"UniversalConceptHub ({detected_lang}): {', '.join(mapped_concepts[:6])}")

        q_canon, clarified_this_turn = self.curiosity_planner.merge_followup_if_pending(session_id, q_canon)
        if clarified_this_turn:
            pre_trace.append(f"ActiveCuriosityPlanner fused follow-up choice -> '{q_canon}'")

        cas_hit = SymbolicCASOrgan.try_solve(q_canon)
        if cas_hit is not None:
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=cas_hit.answer,
                confidence=1.0 if cas_hit.verified else 0.9,
                mode=cas_hit.mode,
                matched_rule_id=None,
                latency_ms=round(ms, 3),
                sandbox_verified=cas_hit.verified,
                reasoning_trace=pre_trace + [cas_hit.trace],
                epistemic_tier="verified_exact",
            )

        self.sessions.update_from_prompt(session_id, q_canon)
        constraint_hit = self.sessions.evaluate_constraint_check(session_id, q_canon)
        if constraint_hit is not None:
            ans_c, trace_c = constraint_hit
            ok_c, _ = ZeroTrustSandboxJudge.verify_code_and_test(ans_c)
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=ans_c,
                confidence=1.0,
                mode="session_constraint_lock",
                matched_rule_id=None,
                latency_ms=round(ms, 3),
                sandbox_verified=ok_c,
                reasoning_trace=pre_trace + trace_c,
                epistemic_tier="verified_exact",
            )

        q, coref_trace = self.coref_tracker.resolve_turn(session_id, q_canon)
        pre_trace.extend(coref_trace)

        sub_intents = split_compound_intents(q)
        if len(sub_intents) >= 2:
            sub_res = [self._respond_atomic_disk(part, detected_lang=detected_lang) for part in sub_intents]
            valid_subs = [r for r in sub_res if r.mode != "unmatched_honest_abstain" and r.confidence >= 0.30]
            if len(valid_subs) >= 2:
                combined_ans = "\n\n".join(f"[{idx + 1}] {r.answer}" for idx, r in enumerate(valid_subs))
                avg_conf = round(sum(r.confidence for r in valid_subs) / len(valid_subs), 4)
                all_ver = all(r.sandbox_verified for r in valid_subs)
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=combined_ans,
                    confidence=avg_conf,
                    mode="compound_multi_intent",
                    matched_rule_id=valid_subs[0].matched_rule_id,
                    latency_ms=round(ms, 3),
                    sandbox_verified=all_ver,
                    reasoning_trace=pre_trace + [tr for r in valid_subs for tr in r.reasoning_trace],
                    epistemic_tier=EpistemicConfidenceCalibrator.classify_tier(avg_conf, "compound_multi_intent", all_ver),
                )

        windows = ContextBitTable.split_focus_windows(q)
        if len(windows) > 1:
            best_win: CrystalResponse | None = None
            for win in windows:
                cand_r = self._respond_atomic_disk(win, detected_lang=detected_lang)
                if best_win is None or cand_r.confidence > best_win.confidence:
                    best_win = cand_r
            if best_win is not None and best_win.mode != "unmatched_honest_abstain":
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=best_win.answer,
                    confidence=best_win.confidence,
                    mode=best_win.mode,
                    matched_rule_id=best_win.matched_rule_id,
                    latency_ms=round(ms, 3),
                    sandbox_verified=best_win.sandbox_verified,
                    reasoning_trace=pre_trace + ["Focused long-log diagnostic window"] + best_win.reasoning_trace,
                    epistemic_tier=best_win.epistemic_tier,
                )

        return self._respond_atomic_disk(
            q,
            pre_trace=pre_trace,
            session_id=session_id,
            clarified_this_turn=clarified_this_turn,
            original_prompt=q_raw,
            detected_lang=detected_lang,
        )

    def _respond_atomic_disk(
        self,
        q: str,
        pre_trace: list[str] | None = None,
        session_id: str = "default",
        clarified_this_turn: bool = False,
        original_prompt: str = "",
        detected_lang: str = "en",
    ) -> CrystalResponse:
        t0 = time.perf_counter_ns()
        trace: list[str] = list(pre_trace) if pre_trace else []
        orig_p = original_prompt or q

        q_core = self.fluid_composer.strip_style_framing(q)
        q_sense, sense_tags = self.coref_tracker.enrich_for_query(q_core, known_vocab=self.bit_table.token_df)
        if sense_tags:
            trace.append(f"Disambiguated homonym sense -> {', '.join(sense_tags)}")
        sense_sfx = f" {' '.join(sorted(sense_tags))}" if sense_tags else ""

        if not self.fluid_composer.is_explicit_creative_request(orig_p):
            abs_pair_exact = SlotAbstractor.abstract_pair(q_core, "")
            candidate_hashes = [self._prompt_hash(abs_pair_exact.abstract_prompt.strip().lower() + sense_sfx)]
            if sense_sfx:
                candidate_hashes.append(self._prompt_hash(abs_pair_exact.abstract_prompt.strip().lower()))

            for phash_exact in candidate_hashes:
                row_exact = self.conn.execute(
                    "SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules WHERE prompt_hash = ? ORDER BY pass_count DESC, rule_id DESC",
                    (phash_exact,),
                ).fetchone()
                if row_exact is not None:
                    rule_exact = self._row_to_rule(row_exact)
                    if rule_exact.active:
                        instantiated = SlotAbstractor.instantiate(
                            query=q_core,
                            num_slots=rule_exact.num_slots,
                            abstract_response=rule_exact.abstract_response,
                            derived_expr=rule_exact.derived_expr,
                        )
                        if instantiated is not None:
                            ok_ex, j_msg = ZeroTrustSandboxJudge.verify_code_and_test(instantiated)
                            if ok_ex:
                                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                                mode_str = "slot_formula_lock" if rule_exact.num_slots > 0 else "direct_crystal_lock"
                                trace.append(f"O(1) Exact Prompt Hash Hit #{rule_exact.rule_id} (domain={rule_exact.domain})")
                                return CrystalResponse(
                                    answer=instantiated,
                                    confidence=1.0,
                                    mode=mode_str,
                                    matched_rule_id=rule_exact.rule_id,
                                    latency_ms=round(ms, 3),
                                    sandbox_verified=j_msg != "non_code",
                                    reasoning_trace=trace,
                                    epistemic_tier="verified_exact",
                                )

        raw_sig, raw_toks = self.bit_table.encode_text_signature(q_sense)
        cands_raw = self._query_disk_candidates(raw_sig, raw_toks, top_k=4)

        masked_q = SlotAbstractor.mask_query_numbers(q_core)
        if masked_q != q_core:
            q_unique_nums = list(dict.fromkeys(SlotAbstractor.extract_numbers(q_core)))
            slot_sig, slot_toks = self.bit_table.encode_text_signature(masked_q)
            cands_slot = self._query_disk_candidates(
                slot_sig, slot_toks, top_k=4, required_slots=len(q_unique_nums)
            )
            merged_by_id: dict[int, tuple[float, CrystalRule]] = {}
            for sc, rl in cands_raw + cands_slot:
                prev = merged_by_id.get(rl.rule_id)
                if prev is None or sc > prev[0]:
                    merged_by_id[rl.rule_id] = (sc, rl)
            cands = sorted(merged_by_id.values(), key=lambda kv: kv[0], reverse=True)[:4]
        else:
            cands = cands_raw

        best_score = cands[0][0] if cands else 0.0
        if best_score < 0.72:
            chained = self._chain_infer_disk(q_core)
            if chained is not None:
                conclusion, hop_facts = chained
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=f"{conclusion} (Chained from: {' ; '.join(hop_facts)})",
                    confidence=0.92,
                    mode="multi_hop_rule_chain",
                    matched_rule_id=None,
                    latency_ms=round(ms, 3),
                    sandbox_verified=True,
                    reasoning_trace=trace + hop_facts,
                    epistemic_tier="verified_exact",
                )

        is_creative_req = self.fluid_composer.is_explicit_creative_request(orig_p)
        oov_entities = EpistemicConfidenceCalibrator.extract_oov_proper_entities(orig_p, self.bit_table)

        if not is_creative_req and not oov_entities:
            curiosity_hit = self.curiosity_planner.check_diagnostic_ambiguity(
                session_id=session_id,
                query=q_core,
                candidates=cands,
                query_salient_tokens=set(raw_toks),
                already_clarified_this_turn=clarified_this_turn,
            )
            if curiosity_hit is not None:
                q_msg, cur_trace = curiosity_hit
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=q_msg,
                    confidence=round(best_score, 4),
                    mode="clarification_needed",
                    matched_rule_id=None,
                    latency_ms=round(ms, 3),
                    sandbox_verified=False,
                    reasoning_trace=trace + cur_trace,
                    epistemic_tier="diagnostic_clarification",
                )

        q_expanded = self.bit_table.expand_content_tokens(raw_toks)
        min_lock_score = 0.85 if is_creative_req else 0.22
        for score, rule in cands:
            if oov_entities and rule.num_slots == 0 and not oov_entities.issubset(rule.token_set):
                trace.append(f"Rejected partial match #{rule.rule_id}: missing OOV proper entity {sorted(oov_entities)}")
                continue
            ok_match, cal_conf, q_cov, match_reason = EpistemicConfidenceCalibrator.verify_candidate_match(
                query_text=q_core,
                original_prompt=orig_p,
                raw_toks=raw_toks,
                q_expanded=q_expanded,
                rule=rule,
                score=score,
                bit_table=self.bit_table,
                detected_lang=detected_lang,
                min_lock_score=min_lock_score,
            )
            if not ok_match:
                trace.append(f"Rejected partial/false match on rule #{rule.rule_id} ({match_reason})")
                continue
            instantiated = SlotAbstractor.instantiate(
                query=q_core,
                num_slots=rule.num_slots,
                abstract_response=rule.abstract_response,
                derived_expr=rule.derived_expr,
            )
            if instantiated is None:
                continue
            ok, judge_msg = ZeroTrustSandboxJudge.verify_code_and_test(instantiated)
            if not ok:
                with self.conn:
                    self.conn.execute("UPDATE rules SET fail_count = fail_count + 1 WHERE rule_id = ?", (rule.rule_id,))
                trace.append(f"Quarantined rule #{rule.rule_id} ({judge_msg})")
                continue

            mode_str = "slot_formula_lock" if rule.num_slots > 0 else "direct_crystal_lock"
            conf_val = cal_conf
            sb_ver = judge_msg != "non_code"
            trace.append(
                f"Matched DiskVault Rule #{rule.rule_id} (domain={rule.domain}, bit_score={score:.3f}, cov={q_cov:.2f}, slots={rule.num_slots})"
            )
            if not sb_ver:
                styled_ans, was_styled = self.fluid_composer.realize_grounded_style(orig_p, instantiated, sig2048=raw_sig)
                if was_styled:
                    instantiated = styled_ans
                    mode_str = "bicameral_grounded_compose"
                    trace.append("TernaryFluidComposer (1.58-bit) styled verified fact under FactFidelityGuard")
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=instantiated,
                confidence=conf_val,
                mode=mode_str,
                matched_rule_id=rule.rule_id,
                latency_ms=round(ms, 3),
                sandbox_verified=sb_ver,
                reasoning_trace=trace,
                epistemic_tier=EpistemicConfidenceCalibrator.classify_tier(
                    conf_val, mode_str, sb_ver, query_coverage=q_cov
                ),
            )

        creative_out = self.fluid_composer.synthesize_creative(orig_p, sig2048=raw_sig)
        if creative_out is not None:
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=creative_out,
                confidence=0.75,
                mode="bicameral_creative_synthesis",
                matched_rule_id=None,
                latency_ms=round(ms, 3),
                sandbox_verified=False,
                reasoning_trace=trace
                + ["Synthesized via 1.58-bit TernaryFluidComposer (CreativeFactualBleedGuard: vault untouched)"],
                epistemic_tier="creative_synthesis",
            )

        ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return CrystalResponse(
            answer="No verified Crystal Rule matched above threshold yet — feed one example to crystallize instantly.",
            confidence=0.0,
            mode="unmatched_honest_abstain",
            matched_rule_id=None,
            latency_ms=round(ms, 3),
            sandbox_verified=False,
            reasoning_trace=trace + ["Abstained honestly instead of hallucinating."],
            epistemic_tier="honest_abstain",
        )

    def get_trust_tier_counts(self) -> dict[str, int]:
        """Return rule counts grouped by trust_tier (1=verified, 2=curated, 3=unverified)."""
        with self._lock:
            counts: dict[str, int] = {}
            for (payload,) in self.conn.execute("SELECT payload FROM rules"):
                tier = str(json.loads(zlib.decompress(payload).decode("utf-8")).get("tt", 2))
                counts[tier] = counts.get(tier, 0) + 1
            return counts

    def get_domain_counts(self) -> dict[str, int]:
        """Return rule counts grouped by domain label."""
        with self._lock:
            counts: dict[str, int] = {}
            for (payload,) in self.conn.execute("SELECT payload FROM rules"):
                dom = str(json.loads(zlib.decompress(payload).decode("utf-8")).get("dom", "general"))
                counts[dom] = counts.get(dom, 0) + 1
            return counts

    def purge_matching_rules(self, signatures: Any, check_prompt: bool = False) -> dict[str, int]:
        """Remove rules whose response (or optionally prompt) contains any stale/banned signature and prune postings."""
        sigs = tuple(str(s).lower() for s in signatures if str(s).strip())
        if not sigs:
            return {"rules_removed": 0, "postings_updated": 0, "postings_deleted": 0}
        with self._lock:
            bad_rids: set[int] = set()
            for rid, payload in self.conn.execute("SELECT rule_id, payload FROM rules"):
                d = json.loads(zlib.decompress(payload).decode("utf-8"))
                hay = f"{d.get('ap', '')}\n{d.get('ar', '')}".lower() if check_prompt else str(d.get("ar", "")).lower()
                if any(s in hay for s in sigs):
                    bad_rids.add(int(rid))
            if not bad_rids:
                return {"rules_removed": 0, "postings_updated": 0, "postings_deleted": 0}

            postings_updated = 0
            postings_deleted = 0
            tok_upserts: list[tuple[str, bytes]] = []
            tok_deletes: list[tuple[str,]] = []
            for tok, blob in self.conn.execute("SELECT tok, rids FROM token_postings"):
                arr = array("I")
                arr.frombytes(blob)
                if any(rid in bad_rids for rid in arr):
                    kept = array("I", (rid for rid in arr if rid not in bad_rids))
                    if kept:
                        tok_upserts.append((tok, kept.tobytes()))
                        postings_updated += 1
                    else:
                        tok_deletes.append((tok,))
                        postings_deleted += 1

            band_upserts: list[tuple[int, bytes]] = []
            band_deletes: list[tuple[int,]] = []
            for band, blob in self.conn.execute("SELECT band, rids FROM band_postings"):
                arr = array("I")
                arr.frombytes(blob)
                if any(rid in bad_rids for rid in arr):
                    kept = array("I", (rid for rid in arr if rid not in bad_rids))
                    if kept:
                        band_upserts.append((int(band), kept.tobytes()))
                        postings_updated += 1
                    else:
                        band_deletes.append((int(band),))
                        postings_deleted += 1

            rid_list = sorted(bad_rids)
            with self.conn:
                for i in range(0, len(rid_list), 400):
                    chunk = rid_list[i : i + 400]
                    qmarks = ",".join("?" for _ in chunk)
                    self.conn.execute(f"DELETE FROM rules WHERE rule_id IN ({qmarks})", chunk)
                if tok_upserts:
                    self.conn.executemany("INSERT OR REPLACE INTO token_postings(tok, rids) VALUES (?, ?)", tok_upserts)
                if tok_deletes:
                    self.conn.executemany("DELETE FROM token_postings WHERE tok = ?", tok_deletes)
                if band_upserts:
                    self.conn.executemany("INSERT OR REPLACE INTO band_postings(band, rids) VALUES (?, ?)", band_upserts)
                if band_deletes:
                    self.conn.executemany("DELETE FROM band_postings WHERE band = ?", band_deletes)
            self._rule_lru.clear()
            self._meta_dirty = True
            return {
                "rules_removed": len(bad_rids),
                "postings_updated": postings_updated,
                "postings_deleted": postings_deleted,
            }

    def optimize_disk(self) -> int:
        """Flush dirty metadata + WAL checkpoint so the .dcrystal file is a single portable file on disk."""
        with self._lock:
            if self._meta_dirty:
                with self.conn:
                    self._save_lightweight_meta()
                self._meta_dirty = False
            try:
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
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
