"""Posting index implementation and memory-mapped index reader."""
from __future__ import annotations

from array import array
from collections import defaultdict
import hashlib
import json
import math
import mmap
import re
import struct
from pathlib import Path
from typing import Any

from .bit_signatures import ContextBitTable
from .lsh_rule_store import CrystalRule, LSHRuleIndex

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




class VaultIndexingMixin:

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
                        tail_len = min(256, n_rids)
                        view = list(mv[::stride][:768]) + list(mv[-tail_len:])
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

    def export_mmap_index(self, output_path: Path | str) -> dict[str, Any]:
        """
        Exports all token postings into a portable, zero-copy memory-mapped binary (.mmrules) file.
        Enables multiple worker processes to query postings from OS page cache with zero heap duplication.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cur = self.conn.cursor()

        hashed: list[tuple[int, str, list[int]]] = []
        total_postings = 0
        for tok, rids_blob in cur.execute("SELECT tok, rids FROM token_postings"):
            arr = array("I")
            arr.frombytes(rids_blob)
            rids = list(arr)
            total_postings += len(rids)
            h = MmapPostingsIndex.hash_token(tok)
            hashed.append((h, tok, sorted(rids)))

        hashed.sort(key=lambda x: x[0])
        dir_size = len(hashed) * MmapPostingsIndex.ENTRY_SIZE
        data_offset = 12 + dir_size

        with open(out_p, "wb") as f:
            f.write(MmapPostingsIndex.MAGIC)
            f.write(struct.pack("<II", MmapPostingsIndex.VERSION, len(hashed)))

            cur_off = data_offset
            entries = []
            data_blobs = []
            for h, tok, rids in hashed:
                blob = struct.pack(f"<{len(rids)}I", *rids)
                entries.append((h, cur_off, len(rids)))
                data_blobs.append(blob)
                cur_off += len(blob)

            for h, off, cnt in entries:
                f.write(struct.pack("<QQI", h, off, cnt))

            for blob in data_blobs:
                f.write(blob)

        return {
            "token_count": len(hashed),
            "total_postings": total_postings,
            "bytes_written": out_p.stat().st_size,
            "path": str(out_p),
        }

    def shard_by_domain(self, output_dir: Path | str) -> dict[str, int]:
        """
        Partitions the CrystalVault into domain-isolated database shards.
        Enables multi-worker domain routing and prevents large monolithic database locks.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = defaultdict(int)

        cur = self.conn.cursor()
        domain_rules: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cur.execute("SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules"):
            rule = self._row_to_rule(row)
            dom = rule.domain or "general"
            counts[dom] += 1
            domain_rules[dom].append({
                "prompt": rule.abstract_prompt,
                "response": rule.abstract_response,
                "tool": rule.abstract_tool,
                "domain": dom,
                "trust_tier": rule.trust_tier,
            })

        for dom, items in domain_rules.items():
            shard_db = out_path / f"vault_{dom}.dcrystal"
            shard_vault = type(self)(shard_db)
            shard_vault.ingest_batch(items, batch_size=500)
            shard_vault.optimize_disk()
            shard_vault.close()

        return dict(counts)
