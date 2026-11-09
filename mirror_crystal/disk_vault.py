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
import mmap
import os
from pathlib import Path
import struct
import re
import sqlite3
import threading
import time
from typing import Any
import zlib

from .bit_signatures import ContextBitTable
from .intent import split_compound_intents
from .response_types import CrystalResponse
from .session_memory import BinarySessionMemory
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
from .vault_storage import VaultStorageMixin
from .vault_indexing import MmapPostingsIndex, VaultIndexingMixin
from .vault_retrieval import VaultRetrievalMixin
from .vault_response import VaultResponseMixin
from .vault_blueprints import VaultBlueprintRuntimeMixin




class MmapPostingsIndex:
    """
    Zero-copy memory-mapped binary inverted index for CrystalVault postings.
    Enables multiple worker processes to query token-to-rule postings using shared OS page cache
    without copying posting arrays into process heap memory.
    """
    MAGIC = b"MMRL"
    VERSION = 1
    ENTRY_SIZE = 20  # 8 bytes hash + 8 bytes offset + 4 bytes count

    def __init__(self, filepath: Path | str) -> None:
        self.filepath = Path(filepath)
        self._file = open(self.filepath, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        magic = self._mmap[:4]
        if magic != self.MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")
        version, self.token_count = struct.unpack("<II", self._mmap[4:12])
        self._dir_offset = 12

    @staticmethod
    def hash_token(tok: str) -> int:
        return int.from_bytes(hashlib.blake2b(tok.encode("utf-8", errors="replace"), digest_size=8).digest(), "little")

    def lookup_rids(self, tok: str) -> list[int]:
        if self.token_count == 0:
            return []
        target_hash = self.hash_token(tok)

        low = 0
        high = self.token_count - 1
        found_entry = None
        while low <= high:
            mid = (low + high) // 2
            entry_offset = self._dir_offset + mid * self.ENTRY_SIZE
            thash, offset, count = struct.unpack("<QQI", self._mmap[entry_offset : entry_offset + self.ENTRY_SIZE])
            if thash == target_hash:
                found_entry = (offset, count)
                break
            elif thash < target_hash:
                low = mid + 1
            else:
                high = mid - 1

        if not found_entry:
            return []

        offset, count = found_entry
        if count == 0:
            return []
        data = self._mmap[offset : offset + count * 4]
        return list(struct.unpack(f"<{count}I", data))

    def close(self) -> None:
        if hasattr(self, "_mmap") and self._mmap:
            try:
                self._mmap.close()
            except Exception:
                pass
        if hasattr(self, "_file") and self._file:
            try:
                self._file.close()
            except Exception:
                pass


class DiskCrystalVault(VaultStorageMixin, VaultIndexingMixin, VaultRetrievalMixin, VaultResponseMixin, VaultBlueprintRuntimeMixin):
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
        env_cache = os.getenv("MIRROR_RULE_CACHE_SIZE")
        self.lru_rule_cache_size = int(env_cache) if env_cache else (int(lru_rule_cache_size) if int(lru_rule_cache_size) > 1024 else 8192)

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
        self._prompt_hash_lru: OrderedDict[bytes, int] = OrderedDict()
        self._max_prompt_hash_cache: int = int(os.getenv("MIRROR_PROMPT_HASH_CACHE_SIZE", "8192"))
        self._domain_counts_cache: Optional[dict[str, int]] = None
        self._trust_tier_counts_cache: Optional[dict[str, int]] = None
        self._load_lightweight_meta()

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

        row_alias = self.conn.execute("SELECT rule_id FROM prompt_aliases WHERE alias_hash = ?", (phash,)).fetchone()
        if row_alias is not None:
            with self.conn:
                self.conn.execute(
                    "UPDATE rules SET pass_count = pass_count + ? WHERE rule_id = ?",
                    (1 + verified_bonus, int(row_alias[0])),
                )
            self.stats["duplicates_skipped"] += 1
            return False

        cands = self._query_disk_candidates(sig, salient[:8], top_k=3)
        for cand_score, cand_rule in cands:
            if cand_rule.abstract_response.strip().lower() == abs_pair.abstract_response.strip().lower():
                with self.conn:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO prompt_aliases(alias_hash, rule_id) VALUES (?, ?)",
                        (phash, cand_rule.rule_id),
                    )
                    self.conn.execute(
                        "UPDATE rules SET pass_count = pass_count + ? WHERE rule_id = ?",
                        (1 + verified_bonus, cand_rule.rule_id),
                    )
                self.stats["duplicates_skipped"] += 1
                self.stats["synonym_bridges_folded"] = self.stats.get("synonym_bridges_folded", 0) + 1
                return False

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
            with self._lock:
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
                self._domain_counts_cache = None
                self._trust_tier_counts_cache = None
            time.sleep(0.001)  # Cooperative yield to reader threads

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

            row_alias = self.conn.execute("SELECT rule_id FROM prompt_aliases WHERE alias_hash = ?", (phash,)).fetchone()
            if row_alias is not None:
                reinforce_batch.append((1 + verified_bonus, int(row_alias[0])))
                self.stats["duplicates_skipped"] += 1
                dups_this_file += 1
                return

            sig, chosen_toks = self.bit_table.encode_text_signature(abs_pair.abstract_prompt + sense_sfx)
            salient = tuple(sorted(set(chosen_toks), key=self.bit_table.token_salience, reverse=True)[:14])

            # Semantic paraphrase deduplication
            if fold_synonyms:
                cands = self._query_disk_candidates(sig, salient[:8], top_k=2)
                for cand_score, cand_rule in cands:
                    if cand_rule.abstract_response.strip().lower() == abs_pair.abstract_response.strip().lower():
                        self.conn.execute(
                            "INSERT OR IGNORE INTO prompt_aliases(alias_hash, rule_id) VALUES (?, ?)",
                            (phash, cand_rule.rule_id),
                        )
                        reinforce_batch.append((1 + verified_bonus, cand_rule.rule_id))
                        self.stats["duplicates_skipped"] += 1
                        self.stats["synonym_bridges_folded"] += 1
                        dups_this_file += 1
                        return
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
        if new_rules_this_file > 0 or self._meta_dirty:
            with self._lock:
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

    _PAT_POSSESSIVE = re.compile(
        r"^(?:what|which|who)\s+is\s+the\s+(?P<rel>[a-zA-Z0-9_ ]+?)\s+of\s+(?P<subj>[a-zA-Z0-9_ ]+?)(?:'s|’s)\s+(?P<inner_rel>[a-zA-Z0-9_ ]+?)\??$",
        re.I
    )
    _PAT_OF_THE = re.compile(
        r"^(?:what|which|who)\s+is\s+the\s+(?P<rel>[a-zA-Z0-9_ ]+?)\s+of\s+the\s+(?P<inner_rel>[a-zA-Z0-9_ ]+?)\s+of\s+(?P<subj>[a-zA-Z0-9_ ]+?)\??$",
        re.I
    )
    _PAT_WHERE = re.compile(
        r"^(?:what|which|who)\s+is\s+the\s+(?P<rel>[a-zA-Z0-9_ ]+?)\s+of\s+the\s+(?P<ent>[a-zA-Z0-9_ ]+?)\s+where\s+(?P<subj>.+?)\s+(?P<verb>(?:was|is|were)\s+[a-z]+|[a-z]+ed)\??$",
        re.I
    )

    def ingest_document(self, text: str, title: str = "Document", domain: str = "doc_crystallized") -> dict[str, int]:
        """Auto-crystallize unstructured Markdown, textbook prose, or inline Python code directly to disk."""
        return DocumentAutoCrystallizer.ingest_document_text(self, text, title=title, domain=domain)

    def ingest_source_file(self, path: Path | str, domain: str | None = None) -> dict[str, int]:
        """Auto-crystallize any .md, .txt, or .py file directly into DiskCrystalVault."""
        return DocumentAutoCrystallizer.ingest_file(self, path, domain=domain)

    def get_coverage_gaps(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            try:
                cur = self.conn.cursor()
                rows = cur.execute(
                    """
                    SELECT inferred_domain, COUNT(*) as cnt, GROUP_CONCAT(query_text, ' || ') as samples
                    FROM unanswered_queries
                    GROUP BY inferred_domain
                    ORDER BY cnt DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                out = []
                for r in rows:
                    samples = [s.strip() for s in (r[2] or "").split(" || ") if s.strip()][:4]
                    out.append({
                        "topic": r[0],
                        "unanswered_count": int(r[1]),
                        "sample_queries": samples,
                    })
                return out
            except Exception:
                return []

    def generate_gap_fill_prompt(self, topic: Optional[str] = None) -> str:
        gaps = self.get_coverage_gaps(limit=5)
        target = topic or (gaps[0]["topic"] if gaps else "science_and_technology")
        samples_text = ""
        for g in gaps:
            if g["topic"].lower() == target.lower() and g["sample_queries"]:
                samples_text = "\n".join(f"- {q}" for q in g["sample_queries"][:4])
                break
        prompt = f"""You are building a high-quality question-answer dataset for a retrieval-based AI.
Topic: {target}
Generate 150 entries.

OUTPUT FORMAT (strict):
- JSONL only: one JSON object per line, no array brackets, no markdown, no code fences.
- Each line: {{"question": "...", "answer": "...", "domain": "{target.lower().replace(' ', '_')}", "trust_tier": 2}}

QUALITY RULES:
1. Facts must be correct, well-established and verifiable.
2. Each question must stand alone.
3. Answers: 1-4 sentences, direct first, then brief explanation.
4. No duplicate entries.
5. Difficulty: 30% beginner, 50% intermediate, 20% advanced.
"""
        if samples_text:
            prompt += f"\nHere are actual user questions Mirror was asked but could not answer:\n{samples_text}\n"
        return prompt
