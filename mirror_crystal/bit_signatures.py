"""
Hardware-efficient 2,048-bit Context Barcode Signatures using native C-int POPCNT (`int.bit_count()`).
Zero floating-point weights, zero gradient descent, zero GPU requirement.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any

SIGNATURE_BITS = 2048
MORPH_BITS = 1024
FRAME_BITS = 1024
MASK_2048 = (1 << SIGNATURE_BITS) - 1

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|\d+(?:\.\d+)?|[+|><=^*()-]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "as",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "how",
        "what",
        "why",
        "do",
        "does",
        "can",
        "you",
        "please",
        "explain",
        "tell",
        "me",
        "write",
        "give",
        "show",
    }
)


_NEGATION_TOKENS: frozenset[str] = frozenset(
    {"not", "never", "no", "without", "cannot", "dont", "don't", "neither", "nor", "forbid", "disable"}
)
# Dedicated 8-bit polarity signature in bits 2040..2047 so negated & affirmative queries never collide
_NEGATION_POLARITY_BITS: int = 0b10100101 << 2040


def _fast_hash64(raw: bytes) -> int:
    """Deterministic cross-run 64-bit Blake2b digest (immune to Python hash randomization)."""
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def _rotl_2048(val: int, shift: int) -> int:
    s = shift % SIGNATURE_BITS
    return ((val << s) & MASK_2048) | (val >> (SIGNATURE_BITS - s))


def morph_byte_signature(token: str) -> int:
    """Compute deterministic 1,024-bit morphological byte n-gram barcode (bits 0..1023)."""
    b = b"<" + token.lower().encode("utf-8") + b">"
    bits = 0
    n = len(b)
    # Whole-token anchor bits
    h_full = _fast_hash64(b)
    bits |= 1 << (h_full & 1023)
    bits |= 1 << ((h_full >> 11) & 1023)
    # Sliding 2-byte sub-grams (catches transposition & 1-char edits)
    for i in range(max(0, n - 1)):
        gram = b[i : i + 2]
        h = _fast_hash64(gram)
        bits |= 1 << (h & 1023)
    # Sliding 3-byte morphological sub-grams so inflections ("entangle"/"entangled") share bits
    for i in range(max(0, n - 2)):
        gram = b[i : i + 3]
        h = _fast_hash64(gram)
        bits |= 1 << ((h ^ 0x5555) & 1023)
    # Sliding 4-byte sub-grams for root stem alignment
    for i in range(max(0, n - 3)):
        gram = b[i : i + 4]
        h = _fast_hash64(gram)
        bits |= 1 << ((h ^ 0xAAAA) & 1023)
    return bits


def _damerau_levenshtein_bounded(a: str, b: str, max_dist: int = 2) -> int:
    """Optimal String Alignment Damerau-Levenshtein distance with early row abort (transposition = 1 edit)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if a == b:
        return 0

    prev2: list[int] = []
    prev1 = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ai = a[i - 1]
        row_min = i
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            val = min(
                prev1[j] + 1,          # deletion
                curr[j - 1] + 1,       # insertion
                prev1[j - 1] + cost,   # substitution
            )
            if i > 1 and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1]:
                val = min(val, prev2[j - 2] + 1)  # adjacent transposition
            curr[j] = val
            if val < row_min:
                row_min = val
        if row_min > max_dist:
            return max_dist + 1
        prev2 = prev1
        prev1 = curr
    return prev1[lb]


class ContextBitTable:
    """Learns 2,048-bit semantic barcodes from raw UTF-8 byte neighbor frames in 1 pass on CPU."""

    def __init__(
        self,
        max_frame_df: int = 8,
        max_synonyms_per_token: int = 4,
        max_tracked_frames: int = 65_536,
        max_vocab_entries: int = 100_000,
    ):
        self.max_frame_df = int(max_frame_df)
        self.max_synonyms_per_token = int(max_synonyms_per_token)
        self.max_tracked_frames = int(max_tracked_frames)
        self.max_vocab_entries = int(max_vocab_entries)
        self.doc_count: int = 0
        self.token_df: dict[str, int] = defaultdict(int)
        self.morph_cache: dict[str, int] = {}
        self.frame_bits: dict[str, int] = defaultdict(int)
        # frame_hash -> set of distinct tokens seen inside that exact (left, right) neighbor frame
        self._frame_to_tokens: dict[int, set[str]] = defaultdict(set)
        self._synonyms: dict[str, tuple[str, ...]] = {}
        # Phase-3 MorphologicalFuzzySpellBridge: (first_char, length) -> known alphabetic tokens
        self._spell_buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        self._ngram_to_tokens: dict[str, set[str]] = defaultdict(set)
        self._typo_cache: dict[str, str] = {}

    def _register_spell_token(self, tok: str) -> None:
        if 4 <= len(tok) <= 24 and tok.isalpha() and tok not in _STOPWORDS:
            bkey = (tok[0], len(tok))
            bucket = self._spell_buckets[bkey]
            if len(bucket) < 256:
                bucket.append(tok)
                if self._typo_cache:
                    self._typo_cache.clear()
            if len(self._ngram_to_tokens) < 131_072:
                for i in range(len(tok) - 2):
                    ng = tok[i : i + 3]
                    if len(self._ngram_to_tokens[ng]) < 64:
                        self._ngram_to_tokens[ng].add(tok)

    def resolve_typo_token(self, tok: str) -> str:
        """Bridge OOV keyboard typos ('grvoer', 'quantm', 'serach') to canonical vocab in < 0.1 ms."""
        if (
            self.doc_count == 0
            or tok in self.token_df
            or len(tok) < 5
            or len(tok) > 24
            or not tok.isalpha()
            or tok in _STOPWORDS
        ):
            return tok
        cached = self._typo_cache.get(tok)
        if cached is not None:
            return cached

        tlen = len(tok)
        max_ed = 1 if tlen <= 7 else 2
        c0 = tok[0]
        candidates: list[str] = []
        for dl in (0, -1, 1):
            candidates.extend(self._spell_buckets.get((c0, tlen + dl), ()))
        if tok[0] != tok[1]:
            for c_tr in self._spell_buckets.get((tok[1], tlen), ()):
                if len(c_tr) >= 2 and c_tr[0] == tok[1] and c_tr[1] == tok[0]:
                    candidates.append(c_tr)
        if hasattr(self, "_ngram_to_tokens") and self._ngram_to_tokens:
            cand_counts: dict[str, int] = defaultdict(int)
            for i in range(len(tok) - 2):
                ng = tok[i : i + 3]
                for c in self._ngram_to_tokens.get(ng, ()):
                    if abs(len(c) - tlen) <= 2:
                        cand_counts[c] += 1
            for c, cnt in cand_counts.items():
                if cnt >= 2:
                    candidates.append(c)

        best_key: tuple[int, int, int, str] | None = None
        tok_chars = frozenset(tok)
        max_sym_diff = max_ed * 2
        m_tok: int | None = None
        for cand in candidates:
            if len(tok_chars.symmetric_difference(cand)) > max_sym_diff:
                continue
            ed = _damerau_levenshtein_bounded(tok, cand, max_dist=max_ed)
            if ed <= max_ed:
                if m_tok is None:
                    m_tok = self._get_morph(tok)
                m_overlap = (m_tok & self._get_morph(cand)).bit_count()
                if m_overlap < 2:
                    continue
                rank = (ed, -m_overlap, -self.token_df.get(cand, 0), cand)
                if best_key is None or rank < best_key:
                    best_key = rank

        resolved = best_key[3] if best_key is not None else tok
        if len(self._typo_cache) >= 4096:
            self._typo_cache.clear()
        self._typo_cache[tok] = resolved
        return resolved

    @staticmethod
    def tokenize(text: str) -> list[str]:
        out: list[str] = []
        for m in _TOKEN_RE.finditer(text):
            t = m.group(0).lower()
            if len(t) > 3 and t.endswith("'s"):
                t = t[:-2]
            out.append(t)
        return out

    def _get_morph(self, tok: str) -> int:
        cached = self.morph_cache.get(tok)
        if cached is None:
            cached = morph_byte_signature(tok)
            if len(self.morph_cache) >= self.max_vocab_entries:
                self.morph_cache.clear()
            self.morph_cache[tok] = cached
        return cached

    def observe_text(self, text: str) -> None:
        """Single-pass O(L) scan updating token document frequencies and neighbor context frames."""
        toks = self.tokenize(text)
        if not toks:
            return
        self.doc_count += 1
        allow_new_vocab = len(self.token_df) < self.max_vocab_entries
        for t in set(toks):
            if allow_new_vocab or t in self.token_df:
                is_first = t not in self.token_df
                self.token_df[t] += 1
                if is_first:
                    self._register_spell_token(t)
            self._get_morph(t)

        padded = ["<bos>"] + toks + ["<eos>"]
        frames_full = len(self._frame_to_tokens) >= self.max_tracked_frames
        for i in range(1, len(padded) - 1):
            center = padded[i]
            if center in _STOPWORDS or len(center) <= 1 or center[0].isdigit() or center.startswith("-"):
                continue
            left_tok = padded[i - 1]
            right_tok = padded[i + 1]
            # Specific 2-sided neighbor frame
            frame_key = f"{left_tok}|{right_tok}".encode("utf-8")
            fh = _fast_hash64(frame_key)
            if frames_full and fh not in self._frame_to_tokens:
                bit_idx = 1024 + (fh & 1023)
                if allow_new_vocab or center in self.frame_bits:
                    self.frame_bits[center] |= 1 << bit_idx
                continue
            bucket = self._frame_to_tokens[fh]
            if len(bucket) <= self.max_frame_df:
                bucket.add(center)
                bit_idx = 1024 + (fh & 1023)
                if allow_new_vocab or center in self.frame_bits:
                    self.frame_bits[center] |= 1 << bit_idx

    def fold_equivalences(self, min_shared_specific_frames: int = 2) -> int:
        """Cross-pollinate context bits between distinct words sharing rare neighbor frames (corpus-scale guarded)."""
        eff_min = int(min_shared_specific_frames)
        if self.doc_count >= 500:
            eff_min = max(eff_min, int(3 + math.log10(self.doc_count) * 1.6))
        max_generic_df = max(64, self.doc_count // 35) if self.doc_count >= 500 else 1_000_000

        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for tokens in self._frame_to_tokens.values():
            if len(tokens) < 2 or len(tokens) > self.max_frame_df:
                continue
            sorted_toks = [t for t in sorted(tokens) if self.token_df.get(t, 0) <= max_generic_df]
            for i in range(len(sorted_toks)):
                ti = sorted_toks[i]
                for j in range(i + 1, len(sorted_toks)):
                    pair_counts[(ti, sorted_toks[j])] += 1

        adj: dict[str, list[tuple[int, str]]] = defaultdict(list)
        merged = 0
        for (a, b), count in pair_counts.items():
            if count >= eff_min:
                adj[a].append((count, b))
                adj[b].append((count, a))
                merged += 1

        new_syns: dict[str, tuple[str, ...]] = {}
        for tok, peers in adj.items():
            peers.sort(key=lambda x: x[0], reverse=True)
            top_peers = tuple(p for _, p in peers[: self.max_synonyms_per_token])
            new_syns[tok] = top_peers

        for tok, top_peers in new_syns.items():
            union_f = self.frame_bits[tok]
            shared_m = self._get_morph(tok) & ((1 << 256) - 1)
            for p in top_peers:
                union_f |= self.frame_bits[p]
                shared_m |= self._get_morph(p) & ((1 << 256) - 1)
            self.frame_bits[tok] = union_f | (shared_m << 1024)

        self._synonyms = new_syns
        return merged

    def token_signature(self, tok: str) -> int:
        t = tok.lower()
        return self._get_morph(t) | self.frame_bits.get(t, 0)

    def token_salience(self, tok: str) -> float:
        if tok in _NEGATION_TOKENS:
            return 2.6
        if tok in _STOPWORDS:
            return 0.15
        df = self.token_df.get(tok, 0)
        if df == 0 or self.doc_count == 0:
            return 2.0
        return max(0.25, math.log((self.doc_count + 2.0) / (df + 1.0)) + 0.5)

    @classmethod
    def split_focus_windows(cls, text: str, max_windows: int = 6) -> list[str]:
        """Prevent 2,048-bit saturation on 500-line logs/docs by extracting sharpest diagnostic lines & windows."""
        s = text.strip()
        if len(s) <= 420 and s.count("\n") <= 5:
            return [s]

        diag_cues = ("error", "exception", "traceback", "fail", "fatal", "compute", "what", "how", "why", "?")
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]

        # Extract individual high-priority diagnostic lines (stripping leading ISO timestamps) first!
        diag_lines: list[str] = []
        for ln in lines:
            if any(c in ln.lower() for c in diag_cues):
                cleaned = re.sub(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*", "", ln)
                diag_lines.append(cleaned[:280])

        if len(lines) <= 3:
            chunks = [s[i : i + 280] for i in range(0, len(s), 220)]
        else:
            chunks = ["\n".join(lines[i : i + 2])[:280] for i in range(0, len(lines), 2)]

        def _window_priority(w: str) -> float:
            wl = w.lower()
            bonus = 4.0 if any(c in wl for c in diag_cues) else 0.0
            toks = cls.tokenize(w)
            return bonus + min(len(set(toks)), 12) * 0.1

        combined = diag_lines[:3] + sorted(chunks, key=_window_priority, reverse=True)
        deduped = list(dict.fromkeys(combined))
        return deduped[:max_windows] if deduped else [s[:280]]

    def expand_content_tokens(self, tokens: list[str]) -> set[str]:
        out: set[str] = set()
        for t in tokens:
            if t not in _STOPWORDS and len(t) > 1:
                canon = self.resolve_typo_token(t)
                out.add(t)
                out.add(canon)
                if canon not in _NEGATION_TOKENS:
                    for syn in self._synonyms.get(canon, ()):
                        out.add(syn)
        return out

    def encode_text_signature(self, text: str) -> tuple[int, list[str]]:
        """Encode any UTF-8 string into a 2,048-bit integer signature + salient content tokens."""
        toks = self.tokenize(text)
        if not toks:
            raw_h = _fast_hash64(text.encode("utf-8", errors="replace"))
            return (1 << (raw_h & 2039)), []

        content = [t for t in toks if t not in _STOPWORDS and len(t) > 1]
        raw_chosen = content if content else toks
        chosen = [self.resolve_typo_token(t) for t in raw_chosen] if self.doc_count > 0 else raw_chosen

        # Keep top-12 most informative tokens + ordered bigram shifts so order matters
        scored = sorted(set(chosen), key=self.token_salience, reverse=True)[:12]
        sig = 0
        has_negation = any(t in _NEGATION_TOKENS for t in toks)
        for t in scored:
            sig |= self.token_signature(t) & ((1 << 2040) - 1)
            if t not in _NEGATION_TOKENS:
                for syn in self._synonyms.get(t, ()):
                    sig |= self.token_signature(syn) & ((1 << 2040) - 1)

        # Ordered 2-token bindings (captures order without float positional embeddings)
        for i in range(min(10, len(chosen) - 1)):
            s1 = self._get_morph(chosen[i])
            s2 = self._get_morph(chosen[i + 1])
            bigram_h = _fast_hash64(f"{chosen[i]}->{chosen[i+1]}".encode("utf-8"))
            sig |= 1 << (1024 + (bigram_h & 1015))
            sig |= (_rotl_2048(s1, 3) & _rotl_2048(s2, 7)) & ((1 << 2040) - 1)

        if has_negation:
            sig |= _NEGATION_POLARITY_BITS

        return sig, chosen

    @staticmethod
    def bit_cosine(sig_a: int, sig_b: int) -> float:
        """Exact binary cosine similarity in compiled C using hardware POPCNT (`int.bit_count()`)."""
        if sig_a == 0 or sig_b == 0:
            return 0.0
        overlap = (sig_a & sig_b).bit_count()
        if overlap == 0:
            return 0.0
        return overlap / math.sqrt(sig_a.bit_count() * sig_b.bit_count())

    def to_state_dict(self) -> dict[str, Any]:
        prune_hapax = len(self.token_df) > 20_000
        kept_df = {k: v for k, v in self.token_df.items() if (v >= 2 or not prune_hapax)}
        return {
            "max_frame_df": self.max_frame_df,
            "doc_count": self.doc_count,
            "token_df": kept_df,
            "frame_bits": {k: hex(v) for k, v in self.frame_bits.items() if v != 0 and (not prune_hapax or k in kept_df or k in self._synonyms)},
            "synonyms": {k: sorted(v) for k, v in self._synonyms.items() if v},
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> ContextBitTable:
        obj = cls(max_frame_df=int(data.get("max_frame_df", 24)))
        obj.doc_count = int(data.get("doc_count", 0))
        loaded_df = {k: int(v) for k, v in data.get("token_df", {}).items()}
        obj.token_df.update(loaded_df)
        for k, _ in sorted(loaded_df.items(), key=lambda kv: kv[1], reverse=True):
            obj._register_spell_token(k)
        for k, hex_v in data.get("frame_bits", {}).items():
            obj.frame_bits[k] = int(hex_v, 16)
        for k, syns in data.get("synonyms", {}).items():
            obj._synonyms[k] = tuple(syns)
        return obj
