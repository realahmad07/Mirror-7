"""Byte-level BPE tokenizer for mirror_lm (pure Python, stdlib only).

Design
------
* Ids 0..255 are the raw bytes (byte fallback: any UTF-8 string is encodable).
* Ids 256..262 are the special tokens (fixed ids, independent of merges).
* Ids 263.. are learned merges, in merge order (rank == id - FIRST_MERGE_ID).
* Pre-tokenization follows the GPT-2 idea (contractions / letters / digits /
  punctuation / whitespace) but is expressed with the stdlib ``re`` module.
* Training is deterministic: ties between equally frequent pairs are broken by
  the smallest ``(left_id, right_id)`` tuple.

A tokenizer trained with ``vocab_size == 263`` has no merges and is therefore a
pure byte-level tokenizer; ``ByteTokenizer()`` is a convenience for that.
"""
from __future__ import annotations

import heapq
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
)
NUM_BYTES = 256
FIRST_SPECIAL_ID = NUM_BYTES
FIRST_MERGE_ID = NUM_BYTES + len(SPECIAL_TOKENS)  # 263
MIN_VOCAB_SIZE = FIRST_MERGE_ID

# GPT-2 style pre-tokenizer using only the stdlib ``re`` module.
#   [^\W\d_]  == "letter" (unicode word char that is not a digit/underscore)
#   (?:[^\s\w]|_) == punctuation / symbols / emoji / underscore
# Every character falls in exactly one class, so the concatenation of all
# chunks always reproduces the input (asserted in tests).
PRETOKENIZE_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?[^\W\d_]+"
    r"| ?\d{1,3}"
    r"| ?(?:[^\s\w]|_)+"
    r"|\s+(?!\S)"
    r"|\s+"
)

FORMAT_VERSION = 1


class BPETokenizer:
    """Byte-level BPE tokenizer with special tokens and lossless round-trip."""

    def __init__(
        self,
        merges: Sequence[tuple[int, int]] | None = None,
        pattern: str = PRETOKENIZE_PATTERN,
    ) -> None:
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]  # type: ignore[misc]
        self.special_tokens: dict[str, int] = {
            tok: FIRST_SPECIAL_ID + i for i, tok in enumerate(SPECIAL_TOKENS)
        }
        self._special_re = re.compile(
            "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
        )
        self._rebuild()

    # ------------------------------------------------------------------ setup
    def _rebuild(self) -> None:
        self.ranks: dict[tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        vocab: list[bytes] = [bytes([i]) for i in range(NUM_BYTES)]
        vocab += [t.encode("utf-8") for t in SPECIAL_TOKENS]
        for a, b in self.merges:
            vocab.append(vocab[a] + vocab[b])
        self.vocab_bytes = vocab
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    # Convenience ids
    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    @property
    def end_id(self) -> int:
        return self.special_tokens["<|end|>"]

    def token_id(self, token: str) -> int:
        return self.special_tokens[token]

    def token_byte_lengths(self) -> list[int]:
        """UTF-8 byte length of each token's text; special tokens count as 0 bytes.

        Used to compute tokenizer-independent bits-per-byte.
        """
        lengths = [len(b) for b in self.vocab_bytes]
        for sid in self.id_to_special:
            lengths[sid] = 0
        return lengths

    # --------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        pattern: str = PRETOKENIZE_PATTERN,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn ``vocab_size - 263`` merges from an iterator of strings.

        Uses a word-frequency dict (pre-tokenized chunks), incremental pair-count
        updates restricted to the words containing the merged pair, and a lazy
        max-heap.  Deterministic for a given input order-independent multiset of
        chunks (chunk counts, not order, determine the result).
        """
        if vocab_size < MIN_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be >= {MIN_VOCAB_SIZE}")
        regex = re.compile(pattern)
        word_freq: Counter[bytes] = Counter()
        for text in texts:
            for chunk in regex.findall(text):
                word_freq[chunk.encode("utf-8")] += 1

        # Deterministic word ordering.
        items = sorted(word_freq.items())
        words: list[list[int]] = [list(w) for w, _ in items]
        freqs: list[int] = [f for _, f in items]

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                pair_where[pair].add(wi)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        num_merges = vocab_size - FIRST_MERGE_ID
        next_id = FIRST_MERGE_ID
        while len(merges) < num_merges and heap:
            neg_c, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current != -neg_c:
                continue  # stale heap entry
            if current < min_frequency:
                break
            merges.append(pair)
            new_id = next_id
            next_id += 1
            a, b = pair
            changed: dict[tuple[int, int], int] = defaultdict(int)
            for wi in sorted(pair_where.pop(pair, ())):
                w = words[wi]
                f = freqs[wi]
                # remove old pair contributions of this word
                for p in zip(w, w[1:]):
                    changed[p] -= f
                out: list[int] = []
                i = 0
                n = len(w)
                while i < n:
                    if i < n - 1 and w[i] == a and w[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):
                    changed[p] += f
                    pair_where[p].add(wi)
            pair_counts.pop(pair, None)
            for p, delta in changed.items():
                if p == pair or delta == 0:
                    continue
                c = pair_counts.get(p, 0) + delta
                if c <= 0:
                    pair_counts.pop(p, None)
                else:
                    pair_counts[p] = c
                    heapq.heappush(heap, (-c, p))
            if verbose and len(merges) % 500 == 0:
                print(f"[bpe] {len(merges)}/{num_merges} merges")
        return cls(merges=merges, pattern=pattern)

    # ---------------------------------------------------------------- encode
    def _bpe(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk)
        ranks = self.ranks
        while len(ids) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            pair = (ids[best_i], ids[best_i + 1])
            new_id = FIRST_MERGE_ID + best_rank
            out: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        if not self.merges:
            return list(text.encode("utf-8"))
        for chunk in self._regex.findall(text):
            out.extend(self._bpe(chunk.encode("utf-8")))
        return out

    def encode(
        self,
        text: str,
        allow_special: bool = True,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text. If ``allow_special``, literal special-token strings map to their ids;
        otherwise they are encoded as ordinary bytes (safe for untrusted text)."""
        ids: list[int] = [self.bos_id] if add_bos else []
        if allow_special:
            for part in self._special_re.split(text):
                if not part:
                    continue
                sid = self.special_tokens.get(part)
                if sid is not None:
                    ids.append(sid)
                else:
                    ids.extend(self._encode_ordinary(part))
        else:
            ids.extend(self._encode_ordinary(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def pretokenize(self, text: str) -> list[str]:
        return self._regex.findall(text)

    # ---------------------------------------------------------------- decode
    def decode_bytes(self, ids: Iterable[int], skip_special: bool = False) -> bytes:
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if skip_special and i in self.id_to_special:
                continue
            parts.append(self.vocab_bytes[i])
        return b"".join(parts)

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        return self.decode_bytes(ids, skip_special=skip_special).decode(
            "utf-8", errors="replace"
        )

    # ------------------------------------------------------------- save/load
    def to_dict(self) -> dict:
        return {
            "format": "mirror_lm.bpe",
            "version": FORMAT_VERSION,
            "pattern": self.pattern,
            "special_tokens": list(SPECIAL_TOKENS),
            "merges": [list(m) for m in self.merges],
        }

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        if data.get("format") != "mirror_lm.bpe":
            raise ValueError("not a mirror_lm BPE tokenizer file")
        if list(data.get("special_tokens", [])) != list(SPECIAL_TOKENS):
            raise ValueError("special token set mismatch")
        return cls(merges=[tuple(m) for m in data["merges"]], pattern=data["pattern"])

    @classmethod
    def load(cls, path: str | os.PathLike) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def ByteTokenizer() -> BPETokenizer:
    """Pure byte-level tokenizer (256 bytes + 7 specials, no merges)."""
    return BPETokenizer(merges=[])


def iter_texts(paths: Iterable[str | os.PathLike]) -> Iterator[str]:
    for p in paths:
        yield Path(p).read_text(encoding="utf-8", errors="replace")
