"""Stage S7 Byte-Level Language Model Bits-Per-Byte Benchmark (PDF p. 13 Mandatory Gate).

STRICT ISOLATION & SCIENTIFIC FAIRNESS:
- Contains ZERO hardcoded S0/D2/D3 domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Generates a deterministic, strictly zero-leakage multi-domain UTF-8 byte corpus per seed
  (~30 KB train / ~9 KB held-out) spanning 4 text registers (structured action & conditional
  directives with multi-byte UTF-8 identifiers, grounded state verification prose, multi-digit
  arithmetic/sequence step traces, and Python code definitions) with 100% disjoint entity roots,
  identifier names, and numeric literals between train and held-out splits.
- Evaluates ALL 5 models under the EXACT SAME train/held-out split, strictly frozen at test time
  (zero test-time count or weight updates), verifying exact probability normalization
  (``len(probs) == 256``, ``all(p > 0)``, ``abs(sum(probs) - 1.0) <= 1e-6``):
  1. ``uniform_byte_floor`` (exact ``8.0000 bpb``)
  2. ``byte_3gram_kneser_ney`` (interpolated Modified Kneser-Ney order-3 byte LM)
  3. ``ppm_c_order5_byte_lm`` (classic PPM-C order-5 byte LM with full exclusion)
  4. ``mirror_ctw_construction_byte_lm`` (Mirror's CPU-native non-neural deep KT/PPM-D +
     construction-skeleton context-tree mixture — ``0`` backprop, ``0`` GPU, ``0`` neural matrices!)
  5. ``small_cpu_byte_transformer_ref`` (``comparison_only = True``: genuine pure-Python 1-layer
     causal self-attention byte transformer trained from scratch with analytical backprop + Adam)
"""
from __future__ import annotations

import math
import random
import re
import struct
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from research.cpu_learning.simulator.state_graph import canonical_json_bytes

BYTE_LM_SCHEMA_VERSION: str = "s7-byte-lm-v1"

# Strictly disjoint train vs held-out lexical pools (zero overlap with S0/D2/D3 forbidden literals!)
_TRAIN_OPERATORS: tuple[str, ...] = ("relocate", "transfer", "unseal", "fasten")
_HELDOUT_OPERATORS: tuple[str, ...] = ("reposition", "convey", "unlatch", "secure")

_TRAIN_AGENTS: tuple[str, ...] = ("élan", "renée", "marta", "lukas", "søren", "ines")
_HELDOUT_AGENTS: tuple[str, ...] = ("noël", "chloé", "henrik", "freja", "tomas", "claire")

_TRAIN_ITEMS: tuple[str, ...] = ("token", "cylinder", "spool", "module", "beaker", "sensor")
_HELDOUT_ITEMS: tuple[str, ...] = ("wafer", "cartridge", "lens", "capsule", "rotor", "probe")

_TRAIN_SITES: tuple[str, ...] = ("rack", "bench", "cradle", "tray", "hopper", "slot")
_HELDOUT_SITES: tuple[str, ...] = ("gantry", "stage", "platen", "trough", "chute", "rail")

_TRAIN_CODE_FN: tuple[str, ...] = (" tally_span", "clamp_band", "fold_delta", "align_pair")
_HELDOUT_CODE_FN: tuple[str, ...] = ("sum_window", "bound_range", "scan_offset", "merge_step")

_ID_TOKEN_RE = re.compile(r"\b[^\W\d_]+_[0-9]+\b", flags=re.UNICODE)


class ByteLMFrozenError(RuntimeError):
    """Raised when attempting to fit an already-frozen ByteLM or evaluate an unfrozen ByteLM."""


@dataclass(frozen=True)
class DisjointByteCorpus:
    """Deterministic multi-domain UTF-8 byte corpus with verified train/held-out disjointness."""

    seed: int
    train_bytes: bytes
    heldout_bytes: bytes
    train_entity_ids: tuple[str, ...]
    heldout_entity_ids: tuple[str, ...]
    disjoint_entities_verified: bool

    def summary_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "train_byte_len": len(self.train_bytes),
            "heldout_byte_len": len(self.heldout_bytes),
            "train_entity_count": len(self.train_entity_ids),
            "heldout_entity_count": len(self.heldout_entity_ids),
            "disjoint_entities_verified": self.disjoint_entities_verified,
        }


def _render_corpus_section(
    rng: random.Random,
    *,
    agents: tuple[str, ...],
    items: tuple[str, ...],
    sites: tuple[str, ...],
    ops: tuple[str, ...],
    code_fns: tuple[str, ...],
    id_range: tuple[int, int],
    num_range: tuple[int, int],
    n_blocks: int,
) -> bytes:
    lines: list[str] = []
    lo_id, hi_id = id_range
    lo_num, hi_num = num_range

    for idx in range(n_blocks):
        ag1 = f"{rng.choice(agents)}_{rng.randint(lo_id, hi_id)}"
        ag2 = f"{rng.choice(agents)}_{rng.randint(lo_id, hi_id)}"
        it = f"{rng.choice(items)}_{rng.randint(lo_id, hi_id)}"
        s1 = f"{rng.choice(sites)}_{rng.randint(lo_id, hi_id)}"
        s2 = f"{rng.choice(sites)}_{rng.randint(lo_id, hi_id)}"
        op_move, op_give, op_open, op_shut = ops

        # Register 1: Multi-byte UTF-8 directives (active, conditional, negated, chained)
        mode = idx % 4
        if mode == 0:
            lines.append(
                f"[DIRECTIVE] {ag1} {op_move}s {it} from {s1} → {s2}; then {ag1} {op_shut}s {s1}."
            )
        elif mode == 1:
            lines.append(
                f"[DIRECTIVE] if {s2} is clear, {ag1} {op_move}s {it} from {s1} → {s2}."
            )
        elif mode == 2:
            lines.append(
                f"[DIRECTIVE] do not {op_give} {it} from {ag1} → {ag2}; instead {ag1} {op_open}s {s2}."
            )
        else:
            lines.append(
                f"[DIRECTIVE] unless {s1} is sealed, {ag2} {op_give}s {it} to {ag1}."
            )

        # Register 2: Grounded verification prose
        if idx % 2 == 0:
            lines.append(
                f"[REPORT] In this step, the action by {ag1} on {it} between {s1} and {s2} "
                f"succeeded and updated 2 verified world relations without collateral edits."
            )
        else:
            lines.append(
                f"[REPORT] Because a required guard failed on {s1}, the step involving {ag2} "
                f"and {it} was blocked, preserving all existing world relations."
            )

        # Register 3: Multi-digit arithmetic & sequence trace with UTF-8 math symbols
        n_a = rng.randint(lo_num, hi_num)
        n_b = rng.randint(lo_num, hi_num)
        tot = n_a + n_b
        diff = n_a - n_b
        lines.append(
            f"[TRACE] ∑({n_a}, {n_b}) = {tot} | Δ± = {diff} | carry_chain = [{(n_a % 10 + n_b % 10) // 10}, {(n_a // 10 % 10 + n_b // 10 % 10) // 10}]"
        )

        # Register 4: Structured Python helper snippet (every 2nd block)
        if idx % 2 == 0:
            fn_name = code_fns[idx % len(code_fns)].strip()
            k_const = rng.randint(lo_id, hi_id)
            lines.append(
                f"def {fn_name}_{k_const}(xs: list[int], limit: int = {n_a}) -> int:\n"
                f"    acc = 0\n"
                f"    for val in xs:\n"
                f"        if val <= limit:\n"
                f"            acc += val\n"
                f"    return acc"
            )

    text = "\n".join(lines) + "\n"
    return text.encode("utf-8")


def generate_disjoint_byte_lm_corpus(
    seed: int,
    *,
    train_blocks: int = 76,
    heldout_blocks: int = 24,
) -> DisjointByteCorpus:
    """Generate deterministic UTF-8 train (~30 KB) and held-out (~9 KB) corpora with zero entity overlap."""
    rng_train = random.Random(seed * 131 + 17)
    rng_eval = random.Random(seed * 131 + 991)

    train_bytes = _render_corpus_section(
        rng_train,
        agents=_TRAIN_AGENTS,
        items=_TRAIN_ITEMS,
        sites=_TRAIN_SITES,
        ops=_TRAIN_OPERATORS,
        code_fns=_TRAIN_CODE_FN,
        id_range=(10, 49),
        num_range=(100, 499),
        n_blocks=train_blocks,
    )
    heldout_bytes = _render_corpus_section(
        rng_eval,
        agents=_HELDOUT_AGENTS,
        items=_HELDOUT_ITEMS,
        sites=_HELDOUT_SITES,
        ops=_HELDOUT_OPERATORS,
        code_fns=_HELDOUT_CODE_FN,
        id_range=(60, 99),
        num_range=(600, 999),
        n_blocks=heldout_blocks,
    )

    train_ids = tuple(sorted(set(_ID_TOKEN_RE.findall(train_bytes.decode("utf-8")))))
    heldout_ids = tuple(sorted(set(_ID_TOKEN_RE.findall(heldout_bytes.decode("utf-8")))))
    overlap = set(train_ids) & set(heldout_ids)
    if overlap:
        raise AssertionError(f"Train/held-out Byte-LM entity leak detected: {sorted(overlap)}")

    return DisjointByteCorpus(
        seed=seed,
        train_bytes=train_bytes,
        heldout_bytes=heldout_bytes,
        train_entity_ids=train_ids,
        heldout_entity_ids=heldout_ids,
        disjoint_entities_verified=True,
    )


# ---------------------------------------------------------------------------
# Base Interface & Shared Exact Bits-Per-Byte Evaluator
# ---------------------------------------------------------------------------


class BaseByteLM:
    """Abstract base class for frozen byte-level language models."""

    model_id: str = "base_byte_lm"
    comparison_only: bool = False
    uses_backprop: bool = False

    def __init__(self) -> None:
        self._frozen: bool = False
        self.train_seconds: float = 0.0

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def _check_unfrozen(self) -> None:
        if self._frozen:
            raise ByteLMFrozenError(f"{self.model_id} is frozen; cannot mutate parameters")

    def fit(self, train_bytes: bytes) -> float:
        raise NotImplementedError

    def predict_probs(self, context: bytes) -> list[float]:
        """Return an exact normalized 256-float distribution P(b | context) for b in 0..255."""
        raise NotImplementedError

    def byte_prob(self, context: bytes, target_byte: int) -> float:
        """Return P(target_byte | context) consistent with ``predict_probs(context)[target_byte]``."""
        return self.predict_probs(context)[target_byte]

    def compressed_model_bits(self) -> int:
        raise NotImplementedError


def evaluate_byte_lm_bpb(
    model: BaseByteLM,
    eval_bytes: bytes,
    *,
    norm_check_positions: int = 24,
) -> dict[str, Any]:
    """Evaluate exact cross-entropy ``bits_per_byte`` on ``eval_bytes`` for a strictly frozen model,
    explicitly verifying probability normalization and ``byte_prob == predict_probs[b]`` consistency."""
    if not model.is_frozen:
        raise ByteLMFrozenError(f"Model {model.model_id} must be frozen before evaluation!")
    if not eval_bytes:
        raise ValueError("eval_bytes must be non-empty")

    n = len(eval_bytes)
    stride = max(1, n // max(1, norm_check_positions))
    check_indices = {min(n - 1, k * stride) for k in range(norm_check_positions)} | {0, n - 1}

    max_norm_err = 0.0
    max_consistency_err = 0.0
    total_bits = 0.0

    t0 = time.perf_counter()
    for i in range(n):
        b = eval_bytes[i]
        ctx = eval_bytes[max(0, i - 32) : i]
        p_b = model.byte_prob(ctx, b)
        if p_b <= 0.0 or not math.isfinite(p_b):
            raise ValueError(f"Invalid byte probability {p_b} at index {i} in {model.model_id}")
        if i in check_indices:
            dist = model.predict_probs(ctx)
            if len(dist) != 256:
                raise AssertionError(f"{model.model_id} returned {len(dist)} probs != 256")
            if any((x <= 0.0 or not math.isfinite(x)) for x in dist):
                raise AssertionError(f"{model.model_id} returned non-positive probability")
            s_err = abs(sum(dist) - 1.0)
            c_err = abs(dist[b] - p_b)
            if s_err > 1e-6:
                raise AssertionError(
                    f"{model.model_id} unnormalized distribution at idx={i}: sum={sum(dist)}"
                )
            if c_err > 1e-9:
                raise AssertionError(
                    f"{model.model_id} byte_prob mismatch vs predict_probs at idx={i}: {c_err}"
                )
            if s_err > max_norm_err:
                max_norm_err = s_err
            if c_err > max_consistency_err:
                max_consistency_err = c_err
        total_bits -= math.log2(p_b)

    eval_sec = time.perf_counter() - t0
    bpb = total_bits / float(n)
    return {
        "model_id": model.model_id,
        "comparison_only": model.comparison_only,
        "uses_backprop": model.uses_backprop,
        "is_frozen": model.is_frozen,
        "eval_byte_count": n,
        "bits_per_byte": round(bpb, 6),
        "total_codelength_bits": round(total_bits, 3),
        "model_bits": model.compressed_model_bits(),
        "train_seconds": round(model.train_seconds, 4),
        "eval_seconds": round(eval_sec, 4),
        "normalization_verified": True,
        "max_sum_prob_error": round(max_norm_err, 12),
        "max_consistency_error": round(max_consistency_err, 12),
    }


# ---------------------------------------------------------------------------
# 1. Uniform Byte Floor (8.0000 bpb)
# ---------------------------------------------------------------------------


class UniformByteFloorLM(BaseByteLM):
    """Zero-context uniform byte baseline: exact 1/256 per byte = 8.000000 bpb."""

    model_id = "uniform_byte_floor"
    comparison_only = False
    uses_backprop = False

    def fit(self, train_bytes: bytes) -> float:
        self._check_unfrozen()
        t0 = time.perf_counter()
        _ = len(train_bytes)
        self.train_seconds = time.perf_counter() - t0
        return self.train_seconds

    def predict_probs(self, context: bytes) -> list[float]:
        _ = context
        return [1.0 / 256.0] * 256

    def byte_prob(self, context: bytes, target_byte: int) -> float:
        _ = (context, target_byte)
        return 1.0 / 256.0

    def compressed_model_bits(self) -> int:
        return 8


# ---------------------------------------------------------------------------
# 2. Interpolated Modified Kneser-Ney 3-Gram Byte LM
# ---------------------------------------------------------------------------


class Byte3GramKneserNeyLM(BaseByteLM):
    """Interpolated Kneser-Ney order-3 byte LM (context length up to 2 bytes)."""

    model_id = "byte_3gram_kneser_ney"
    comparison_only = False
    uses_backprop = False

    def __init__(self, discount: float = 0.75) -> None:
        super().__init__()
        self.discount = float(discount)
        self._cont_unigram: list[float] = [1.0 / 256.0] * 256
        self._c1: dict[bytes, dict[int, int]] = {}
        self._tot1: dict[bytes, int] = {}
        self._c2: dict[bytes, dict[int, int]] = {}
        self._tot2: dict[bytes, int] = {}

    def fit(self, train_bytes: bytes) -> float:
        self._check_unfrozen()
        t0 = time.perf_counter()
        c1: dict[bytes, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        tot1: dict[bytes, int] = defaultdict(int)
        c2: dict[bytes, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        tot2: dict[bytes, int] = defaultdict(int)
        left_contexts_by_byte: list[set[int]] = [set() for _ in range(256)]
        unigram_counts = [0] * 256

        n = len(train_bytes)
        for i, b in enumerate(train_bytes):
            unigram_counts[b] += 1
            if i >= 1:
                prev1 = train_bytes[i - 1]
                left_contexts_by_byte[b].add(prev1)
                h1 = train_bytes[i - 1 : i]
                c1[h1][b] += 1
                tot1[h1] += 1
            if i >= 2:
                h2 = train_bytes[i - 2 : i]
                c2[h2][b] += 1
                tot2[h2] += 1

        # Kneser-Ney continuation probability interpolated with uniform 1/256 for unseen bytes
        total_bigram_types = sum(len(s) for s in left_contexts_by_byte)
        if total_bigram_types > 0:
            d0 = self.discount
            u_seen = sum(1 for s in left_contexts_by_byte if len(s) > 0)
            lam0 = (d0 * u_seen) / float(total_bigram_types)
            self._cont_unigram = [
                max(0.0, len(left_contexts_by_byte[b]) - d0) / float(total_bigram_types)
                + lam0 * (1.0 / 256.0)
                for b in range(256)
            ]
        elif n > 0:
            self._cont_unigram = [(unigram_counts[b] + 1.0) / float(n + 256.0) for b in range(256)]

        self._c1 = {k: dict(v) for k, v in c1.items()}
        self._tot1 = dict(tot1)
        self._c2 = {k: dict(v) for k, v in c2.items()}
        self._tot2 = dict(tot2)
        self.train_seconds = time.perf_counter() - t0
        return self.train_seconds

    def predict_probs(self, context: bytes) -> list[float]:
        probs = list(self._cont_unigram)
        d = self.discount
        if len(context) >= 1:
            h1 = context[-1:]
            t1 = self._tot1.get(h1, 0)
            if t1 > 0:
                d1 = self._c1[h1]
                lam1 = (d * len(d1)) / float(t1)
                inv_t1 = 1.0 / float(t1)
                probs = [
                    max(0.0, d1.get(b, 0) - d) * inv_t1 + lam1 * probs[b]
                    for b in range(256)
                ]
        if len(context) >= 2:
            h2 = context[-2:]
            t2 = self._tot2.get(h2, 0)
            if t2 > 0:
                d2 = self._c2[h2]
                lam2 = (d * len(d2)) / float(t2)
                inv_t2 = 1.0 / float(t2)
                probs = [
                    max(0.0, d2.get(b, 0) - d) * inv_t2 + lam2 * probs[b]
                    for b in range(256)
                ]
        return probs

    def byte_prob(self, context: bytes, target_byte: int) -> float:
        p = self._cont_unigram[target_byte]
        d = self.discount
        if len(context) >= 1:
            h1 = context[-1:]
            t1 = self._tot1.get(h1, 0)
            if t1 > 0:
                d1 = self._c1[h1]
                lam1 = (d * len(d1)) / float(t1)
                p = max(0.0, d1.get(target_byte, 0) - d) / float(t1) + lam1 * p
        if len(context) >= 2:
            h2 = context[-2:]
            t2 = self._tot2.get(h2, 0)
            if t2 > 0:
                d2 = self._c2[h2]
                lam2 = (d * len(d2)) / float(t2)
                p = max(0.0, d2.get(target_byte, 0) - d) / float(t2) + lam2 * p
        return p

    def compressed_model_bits(self) -> int:
        payload = {
            "c1": {k.hex(): v for k, v in sorted(self._c1.items())},
            "c2": {k.hex(): v for k, v in sorted(self._c2.items())},
        }
        return len(zlib.compress(canonical_json_bytes(payload), level=9)) * 8


# ---------------------------------------------------------------------------
# 3. Classic PPM-C Order-5 Byte LM with Full Exclusion
# ---------------------------------------------------------------------------


class _PPMContextTable:
    """Shared compact context-to-counts structure supporting both PPM-C and KT/PPM-D exclusion."""

    def __init__(self, max_order: int, method: str = "C") -> None:
        if method not in {"C", "D"}:
            raise ValueError(f"Unsupported PPM escape method: {method}")
        self.max_order = int(max_order)
        self.method = method
        self.tables: list[dict[bytes, dict[int, int]]] = [
            {} for _ in range(self.max_order + 1)
        ]

    def fit_contexts(
        self,
        raw_bytes: bytes,
        *,
        context_transform: Any = None,
    ) -> None:
        tables = [defaultdict(lambda: defaultdict(int)) for _ in range(self.max_order + 1)]
        max_k = self.max_order
        for i, b in enumerate(raw_bytes):
            if context_transform is None:
                hist = raw_bytes[max(0, i - max_k) : i]
            else:
                hist = context_transform(raw_bytes[max(0, i - 28) : i])
            hlen = len(hist)
            upto = max_k if hlen >= max_k else hlen
            for k in range(upto + 1):
                ctx = b"" if k == 0 else hist[-k:]
                tables[k][ctx][b] += 1
        self.tables = [{ctx: dict(cnts) for ctx, cnts in tbl.items()} for tbl in tables]

    def predict_probs(self, hist: bytes) -> list[float]:
        probs = [0.0] * 256
        excluded: set[int] = set()
        rem_mass = 1.0
        start_k = min(self.max_order, len(hist))

        for k in range(start_k, -1, -1):
            ctx = b"" if k == 0 else hist[-k:]
            cnt_map = self.tables[k].get(ctx)
            if not cnt_map:
                continue
            # Filter out bytes already scored in longer contexts (full exclusion)
            active_items = [(s, c) for s, c in cnt_map.items() if s not in excluded]
            if not active_items:
                continue
            u_rem = len(active_items)
            c_rem = sum(c for _s, c in active_items)
            if self.method == "C":
                denom = float(c_rem + u_rem)
                for s, c in active_items:
                    probs[s] += rem_mass * (float(c) / denom)
                    excluded.add(s)
                rem_mass *= float(u_rem) / denom
            else:
                # Krichevsky-Trofimov / PPM-D half-count estimator:
                # sum((c - 0.5)/c_rem) + (0.5 * u_rem)/c_rem == 1.0 identically!
                denom = float(c_rem)
                for s, c in active_items:
                    probs[s] += rem_mass * ((float(c) - 0.5) / denom)
                    excluded.add(s)
                rem_mass *= (0.5 * float(u_rem)) / denom

        # Order -1 uniform escape over remaining unexcluded bytes
        n_unseen = 256 - len(excluded)
        if n_unseen > 0:
            share = rem_mass / float(n_unseen)
            for b in range(256):
                if b not in excluded:
                    probs[b] += share
        return probs

    def byte_prob(self, hist: bytes, target_byte: int) -> float:
        excluded: set[int] = set()
        rem_mass = 1.0
        start_k = min(self.max_order, len(hist))

        for k in range(start_k, -1, -1):
            ctx = b"" if k == 0 else hist[-k:]
            cnt_map = self.tables[k].get(ctx)
            if not cnt_map:
                continue
            active_items = [(s, c) for s, c in cnt_map.items() if s not in excluded]
            if not active_items:
                continue
            u_rem = len(active_items)
            c_rem = sum(c for _s, c in active_items)
            if self.method == "C":
                denom = float(c_rem + u_rem)
                for s, c in active_items:
                    if s == target_byte:
                        return rem_mass * (float(c) / denom)
                    excluded.add(s)
                rem_mass *= float(u_rem) / denom
            else:
                denom = float(c_rem)
                for s, c in active_items:
                    if s == target_byte:
                        return rem_mass * ((float(c) - 0.5) / denom)
                    excluded.add(s)
                rem_mass *= (0.5 * float(u_rem)) / denom

        n_unseen = 256 - len(excluded)
        return rem_mass / float(n_unseen)

    def matched_depth(self, hist: bytes) -> int:
        """Return the longest suffix length ``k`` (``0..max_order``) present in ``self.tables``."""
        start_k = min(self.max_order, len(hist))
        for k in range(start_k, 0, -1):
            if hist[-k:] in self.tables[k]:
                return k
        return 0

    def to_compact_bytes(self) -> bytes:
        """Exact lossless binary serialization of context counts for fast compressed bit measurement."""
        buf = bytearray()
        buf.extend(struct.pack("<BB", self.max_order, ord(self.method[0])))
        for k, tbl in enumerate(self.tables):
            buf.extend(struct.pack("<BI", k, len(tbl)))
            for ctx in sorted(tbl.keys()):
                cnts = tbl[ctx]
                buf.extend(ctx)
                buf.extend(struct.pack("<H", len(cnts)))
                for s, c in sorted(cnts.items()):
                    buf.extend(struct.pack("<BH", s, min(65535, c)))
        return bytes(buf)


class PPMCOrder5ByteLM(BaseByteLM):
    """Classic Moffat PPM-C order-5 byte LM with full exclusion and order -1 uniform escape."""

    model_id = "ppm_c_order5_byte_lm"
    comparison_only = False
    uses_backprop = False

    def __init__(self, max_order: int = 5) -> None:
        super().__init__()
        self.max_order = int(max_order)
        self._tree = _PPMContextTable(max_order=self.max_order, method="C")

    def fit(self, train_bytes: bytes) -> float:
        self._check_unfrozen()
        t0 = time.perf_counter()
        self._tree.fit_contexts(train_bytes)
        self.train_seconds = time.perf_counter() - t0
        return self.train_seconds

    def predict_probs(self, context: bytes) -> list[float]:
        return self._tree.predict_probs(context[-self.max_order :] if self.max_order > 0 else b"")

    def byte_prob(self, context: bytes, target_byte: int) -> float:
        return self._tree.byte_prob(
            context[-self.max_order :] if self.max_order > 0 else b"",
            target_byte,
        )

    def compressed_model_bits(self) -> int:
        return len(zlib.compress(self._tree.to_compact_bytes(), level=6)) * 8


# ---------------------------------------------------------------------------
# 4. Mirror CTW + Construction-Skeleton Byte LM (0 Backprop, 0 GPU, 0 Neural)
# ---------------------------------------------------------------------------

_DIGIT_RUN_BYTES_RE = re.compile(rb"[0-9]+")
# Matches ASCII or UTF-8 word stem immediately preceding "_#" (entity/function slot canonicalizer)
_SLOT_STEM_BYTES_RE = re.compile(rb"[A-Za-z\x80-\xff]+(?=_#)")


def canonicalize_construction_history(context_bytes: bytes) -> bytes:
    """Deterministic finite-state construction history canonicalizer (applied ONLY to past context!).

    Collapses digit runs ``[0-9]+ -> #`` and ``<stem>_# -> @_#`` so unseen held-out entity IDs
    (e.g. ``chloé_84``) and unseen multi-digit numbers share exact construction slot boundaries
    while still predicting the exact 256-way raw next byte!
    """
    if not context_bytes:
        return b""
    step1 = _DIGIT_RUN_BYTES_RE.sub(b"#", context_bytes)
    step2 = _SLOT_STEM_BYTES_RE.sub(b"@", step1)
    return step2


class MirrorCTWConstructionByteLM(BaseByteLM):
    """Mirror's CPU-native non-neural Byte LM (0 backprop, 0 GPU, 0 dense float matrices).

    Combines two Krichevsky-Trofimov / PPM-D exclusion context trees:
    1. Deep Order-10 Raw UTF-8 KT Context Tree (exact lexical & multi-byte subword contexts).
    2. Order-12 Construction-Skeleton KT Context Tree (canonicalizing completed entity/digit slots
       in the conditioning history so unseen held-out entities inherit construction continuations).
    Mixture weight ``raw_weight`` is selected strictly on the last 10% validation slice of
    ``train_bytes`` during ``fit()``, after which both trees are refit on ``train_bytes`` and frozen.
    """

    model_id = "mirror_ctw_construction_byte_lm"
    comparison_only = False
    uses_backprop = False

    def __init__(self, raw_max_order: int = 10, skel_max_order: int = 12) -> None:
        super().__init__()
        self.raw_max_order = int(raw_max_order)
        self.skel_max_order = int(skel_max_order)
        self._raw_tree = _PPMContextTable(max_order=self.raw_max_order, method="D")
        self._skel_tree = _PPMContextTable(max_order=self.skel_max_order, method="D")
        # Context-depth-gated convex weights (depend ONLY on past context -> exact sum(P)=1.0!)
        self.w_raw_deeper: float = 0.90
        self.w_tie: float = 0.70
        self.w_skel_deeper: float = 0.25

    def _gate_weight(self, raw_hist: bytes, skel_hist: bytes) -> float:
        d_raw = self._raw_tree.matched_depth(raw_hist)
        d_skel = self._skel_tree.matched_depth(skel_hist)
        if d_skel >= d_raw + 2:
            return self.w_skel_deeper
        if d_raw >= d_skel + 1:
            return self.w_raw_deeper
        return self.w_tie

    def fit(self, train_bytes: bytes) -> float:
        self._check_unfrozen()
        t0 = time.perf_counter()
        n = len(train_bytes)
        split_at = int(n * 0.88)
        if split_at >= 256 and (n - split_at) >= 128:
            dev_train = train_bytes[:split_at]
            dev_val = train_bytes[split_at:]
            tmp_raw = _PPMContextTable(max_order=self.raw_max_order, method="D")
            tmp_skel = _PPMContextTable(max_order=self.skel_max_order, method="D")
            tmp_raw.fit_contexts(dev_train)
            tmp_skel.fit_contexts(dev_train, context_transform=canonicalize_construction_history)

            buckets: dict[str, list[tuple[float, float]]] = {
                "raw_deeper": [],
                "tie": [],
                "skel_deeper": [],
            }
            for i, b in enumerate(dev_val):
                ctx = dev_val[max(0, i - 28) : i]
                r_h = ctx[-self.raw_max_order :]
                s_h = canonicalize_construction_history(ctx)[-self.skel_max_order :]
                d_r = tmp_raw.matched_depth(r_h)
                d_s = tmp_skel.matched_depth(s_h)
                pr = tmp_raw.byte_prob(r_h, b)
                ps = tmp_skel.byte_prob(s_h, b)
                if d_s >= d_r + 2:
                    buckets["skel_deeper"].append((pr, ps))
                elif d_r >= d_s + 1:
                    buckets["raw_deeper"].append((pr, ps))
                else:
                    buckets["tie"].append((pr, ps))

            def _opt_bucket(pairs: list[tuple[float, float]], default_w: float) -> float:
                if len(pairs) < 16:
                    return default_w
                best_w = default_w
                best_l = float("inf")
                for w_int in range(10, 96, 5):
                    w = w_int / 100.0
                    mw = 1.0 - w
                    loss = -sum(math.log2(w * pr + mw * ps) for pr, ps in pairs)
                    if loss < best_l:
                        best_l = loss
                        best_w = w
                return best_w

            self.w_raw_deeper = _opt_bucket(buckets["raw_deeper"], 0.90)
            self.w_tie = _opt_bucket(buckets["tie"], 0.70)
            self.w_skel_deeper = _opt_bucket(buckets["skel_deeper"], 0.25)

        self._raw_tree.fit_contexts(train_bytes)
        self._skel_tree.fit_contexts(
            train_bytes, context_transform=canonicalize_construction_history
        )
        self.train_seconds = time.perf_counter() - t0
        return self.train_seconds

    def predict_probs(self, context: bytes) -> list[float]:
        ctx_win = context[-28:]
        r_h = ctx_win[-self.raw_max_order :]
        s_h = canonicalize_construction_history(ctx_win)[-self.skel_max_order :]
        w = self._gate_weight(r_h, s_h)
        mw = 1.0 - w
        p_raw = self._raw_tree.predict_probs(r_h)
        p_skel = self._skel_tree.predict_probs(s_h)
        return [w * p_raw[b] + mw * p_skel[b] for b in range(256)]

    def byte_prob(self, context: bytes, target_byte: int) -> float:
        ctx_win = context[-28:]
        r_h = ctx_win[-self.raw_max_order :]
        s_h = canonicalize_construction_history(ctx_win)[-self.skel_max_order :]
        w = self._gate_weight(r_h, s_h)
        pr = self._raw_tree.byte_prob(r_h, target_byte)
        ps = self._skel_tree.byte_prob(s_h, target_byte)
        return w * pr + (1.0 - w) * ps

    def compressed_model_bits(self) -> int:
        raw_blob = (
            struct.pack("<ddd", self.w_raw_deeper, self.w_tie, self.w_skel_deeper)
            + self._raw_tree.to_compact_bytes()
            + self._skel_tree.to_compact_bytes()
        )
        return len(zlib.compress(raw_blob, level=6)) * 8


# ---------------------------------------------------------------------------
# 5. Comparison-Only Small CPU Causal Byte Transformer (Genuine Backprop + Adam)
# ---------------------------------------------------------------------------


def _softmax_list(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    inv_s = 1.0 / sum(exps)
    return [e * inv_s for e in exps]


class SmallCPUByteTransformerRef(BaseByteLM):
    """Comparison-only tiny 1-layer causal self-attention byte transformer trained from scratch
    on CPU with exact analytical backpropagation + mini-batch Adam under the same corpus & budget.

    STRICT ISOLATION TAG: ``comparison_only = True`` — NEVER imported or used by Mirror learners!
    """

    model_id = "small_cpu_byte_transformer_ref"
    comparison_only: bool = True
    uses_backprop: bool = True

    def __init__(
        self,
        *,
        seed: int = 101,
        d_model: int = 6,
        ctx_len: int = 8,
        lr: float = 0.035,
        train_stride: int = 16,
        batch_size: int = 16,
        epochs: int = 1,
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        self.d_model = int(d_model)
        self.ctx_len = int(ctx_len)
        self.lr = float(lr)
        self.train_stride = int(train_stride)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)

        rng = random.Random(self.seed * 977 + 31)
        scale = 1.0 / math.sqrt(float(self.d_model))

        def _mat(rows: int, cols: int) -> list[list[float]]:
            return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]

        self.e_tok: list[list[float]] = _mat(257, self.d_model)
        self.e_pos: list[list[float]] = _mat(self.ctx_len, self.d_model)
        self.w_q: list[list[float]] = _mat(self.d_model, self.d_model)
        self.w_k: list[list[float]] = _mat(self.d_model, self.d_model)
        self.w_v: list[list[float]] = _mat(self.d_model, self.d_model)
        self.w_out: list[list[float]] = _mat(256, self.d_model)
        self.b_out: list[float] = [0.0] * 256

    def _context_tokens(self, context: bytes) -> list[int]:
        if not context:
            return [256]
        return list(context[-self.ctx_len :])

    def _forward_cache(self, tok_ids: list[int]) -> dict[str, Any]:
        L = len(tok_ids)
        d = self.d_model
        inv_sqrt_d = 1.0 / math.sqrt(float(d))

        embeds: list[list[float]] = []
        e_tok = self.e_tok
        e_pos = self.e_pos
        for i, tid in enumerate(tok_ids):
            et = e_tok[tid]
            ep = e_pos[L - 1 - i]
            embeds.append([et[m] + ep[m] for m in range(d)])

        e_last = embeds[L - 1]
        w_q = self.w_q
        w_k = self.w_k
        w_v = self.w_v
        sumprod = math.sumprod

        q = [sumprod(w_q[r], e_last) for r in range(d)]

        keys: list[list[float]] = []
        vals: list[list[float]] = []
        scores: list[float] = []
        for j in range(L):
            ej = embeds[j]
            kj = [sumprod(w_k[r], ej) for r in range(d)]
            vj = [sumprod(w_v[r], ej) for r in range(d)]
            keys.append(kj)
            vals.append(vj)
            scores.append(inv_sqrt_d * sumprod(q, kj))

        alpha = _softmax_list(scores)
        h: list[float] = []
        for r in range(d):
            acc = e_last[r]
            for j in range(L):
                acc += alpha[j] * vals[j][r]
            h.append(acc)

        w_out = self.w_out
        b_out = self.b_out
        sumprod = math.sumprod
        logits = [b_out[v] + sumprod(w_out[v], h) for v in range(256)]

        probs = _softmax_list(logits)
        return {
            "tok_ids": tok_ids,
            "embeds": embeds,
            "q": q,
            "keys": keys,
            "vals": vals,
            "alpha": alpha,
            "h": h,
            "probs": probs,
        }

    def compute_step_grads(
        self, context: bytes, target_byte: int
    ) -> tuple[float, dict[str, Any]]:
        """Compute exact analytical loss (nats) and parameter gradients for a single causal step."""
        tok_ids = self._context_tokens(context)
        cache = self._forward_cache(tok_ids)
        probs: list[float] = cache["probs"]
        loss_nats = -math.log(max(probs[target_byte], 1e-15))

        L = len(tok_ids)
        d = self.d_model
        inv_sqrt_d = 1.0 / math.sqrt(float(d))
        embeds = cache["embeds"]
        q = cache["q"]
        keys = cache["keys"]
        vals = cache["vals"]
        alpha = cache["alpha"]
        h = cache["h"]

        dz = list(probs)
        dz[target_byte] -= 1.0

        dw_out: list[list[float]] = []
        dh = [0.0] * d
        w_out = self.w_out
        for v in range(256):
            dzv = dz[v]
            wov = w_out[v]
            dw_out.append([dzv * h[r] for r in range(d)])
            for r in range(d):
                dh[r] += dzv * wov[r]
        db_out = dz

        de = [[0.0] * d for _ in range(L)]
        for r in range(d):
            de[L - 1][r] += dh[r]

        dw_v = [[0.0] * d for _ in range(d)]
        d_alpha = [0.0] * L
        for j in range(L):
            aj = alpha[j]
            ej = embeds[j]
            vj = vals[j]
            da_j = 0.0
            for r in range(d):
                da_j += dh[r] * vj[r]
                dv_jr = aj * dh[r]
                w_vr = self.w_v[r]
                dw_vr = dw_v[r]
                dej = de[j]
                for m in range(d):
                    dw_vr[m] += dv_jr * ej[m]
                    dej[m] += w_vr[m] * dv_jr
            d_alpha[j] = da_j

        g_bar = 0.0
        for j in range(L):
            g_bar += alpha[j] * d_alpha[j]
        da = [alpha[j] * (d_alpha[j] - g_bar) for j in range(L)]

        dq = [0.0] * d
        dw_k = [[0.0] * d for _ in range(d)]
        for j in range(L):
            s_da = inv_sqrt_d * da[j]
            kj = keys[j]
            ej = embeds[j]
            dej = de[j]
            for r in range(d):
                dq[r] += s_da * kj[r]
                dk_jr = s_da * q[r]
                w_kr = self.w_k[r]
                dw_kr = dw_k[r]
                for m in range(d):
                    dw_kr[m] += dk_jr * ej[m]
                    dej[m] += w_kr[m] * dk_jr

        e_last = embeds[L - 1]
        dw_q = [[dq[r] * e_last[m] for m in range(d)] for r in range(d)]
        de_last = de[L - 1]
        for r in range(d):
            dqr = dq[r]
            w_qr = self.w_q[r]
            for m in range(d):
                de_last[m] += w_qr[m] * dqr

        de_tok: dict[int, list[float]] = {}
        de_pos: dict[int, list[float]] = {}
        for j, tid in enumerate(tok_ids):
            rel_p = L - 1 - j
            gt = de_tok.setdefault(tid, [0.0] * d)
            gp = de_pos.setdefault(rel_p, [0.0] * d)
            dej = de[j]
            for m in range(d):
                gt[m] += dej[m]
                gp[m] += dej[m]

        return (
            loss_nats,
            {
                "dw_out": dw_out,
                "db_out": db_out,
                "dw_q": dw_q,
                "dw_k": dw_k,
                "dw_v": dw_v,
                "de_tok": de_tok,
                "de_pos": de_pos,
            },
        )

    def fit(self, train_bytes: bytes) -> float:
        self._check_unfrozen()
        t0 = time.perf_counter()
        n = len(train_bytes)
        d = self.d_model

        counts = [1.0] * 256
        for b in train_bytes:
            counts[b] += 1.0
        tot = sum(counts)
        self.b_out = [math.log(counts[v] / tot) for v in range(256)]

        m_wout = [[0.0] * d for _ in range(256)]
        v_wout = [[0.0] * d for _ in range(256)]
        m_bout = [0.0] * 256
        v_bout = [0.0] * 256
        m_wq = [[0.0] * d for _ in range(d)]
        v_wq = [[0.0] * d for _ in range(d)]
        m_wk = [[0.0] * d for _ in range(d)]
        v_wk = [[0.0] * d for _ in range(d)]
        m_wv = [[0.0] * d for _ in range(d)]
        v_wv = [[0.0] * d for _ in range(d)]
        m_etok = [[0.0] * d for _ in range(257)]
        v_etok = [[0.0] * d for _ in range(257)]
        m_epos = [[0.0] * d for _ in range(self.ctx_len)]
        v_epos = [[0.0] * d for _ in range(self.ctx_len)]

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step_t = 0
        bsz = max(1, self.batch_size)

        def _adam_mat(
            param: list[list[float]],
            grad: list[list[float]],
            mm: list[list[float]],
            vv: list[list[float]],
            lr_eff: float,
            inv_b: float,
        ) -> None:
            for r in range(len(param)):
                pr = param[r]
                gr = grad[r]
                mr = mm[r]
                vr = vv[r]
                for c in range(len(pr)):
                    g = gr[c] * inv_b
                    mc = beta1 * mr[c] + (1.0 - beta1) * g
                    vc = beta2 * vr[c] + (1.0 - beta2) * (g * g)
                    mr[c] = mc
                    vr[c] = vc
                    pr[c] -= lr_eff * mc / (math.sqrt(vc) + eps)

        rng = random.Random(self.seed * 313 + 7)
        positions = list(range(0, n, max(1, self.train_stride)))
        for _ep in range(self.epochs):
            rng.shuffle(positions)
            for b_start in range(0, len(positions), bsz):
                batch_pos = positions[b_start : b_start + bsz]
                inv_b = 1.0 / float(len(batch_pos))
                acc_wout = [[0.0] * d for _ in range(256)]
                acc_bout = [0.0] * 256
                acc_wq = [[0.0] * d for _ in range(d)]
                acc_wk = [[0.0] * d for _ in range(d)]
                acc_wv = [[0.0] * d for _ in range(d)]
                acc_etok: dict[int, list[float]] = {}
                acc_epos: dict[int, list[float]] = {}

                for idx in batch_pos:
                    ctx = train_bytes[max(0, idx - self.ctx_len) : idx]
                    _, gr = self.compute_step_grads(ctx, train_bytes[idx])
                    gw = gr["dw_out"]
                    gb = gr["db_out"]
                    for v_i in range(256):
                        acc_bout[v_i] += gb[v_i]
                        aw = acc_wout[v_i]
                        gwv = gw[v_i]
                        for c in range(d):
                            aw[c] += gwv[c]
                    for r in range(d):
                        for c in range(d):
                            acc_wq[r][c] += gr["dw_q"][r][c]
                            acc_wk[r][c] += gr["dw_k"][r][c]
                            acc_wv[r][c] += gr["dw_v"][r][c]
                    for tid, gt in gr["de_tok"].items():
                        at = acc_etok.setdefault(tid, [0.0] * d)
                        for c in range(d):
                            at[c] += gt[c]
                    for pid, gp in gr["de_pos"].items():
                        ap = acc_epos.setdefault(pid, [0.0] * d)
                        for c in range(d):
                            ap[c] += gp[c]

                step_t += 1
                lr_eff = (
                    self.lr
                    * math.sqrt(1.0 - (beta2 ** step_t))
                    / (1.0 - (beta1 ** step_t))
                )
                _adam_mat(self.w_out, acc_wout, m_wout, v_wout, lr_eff, inv_b)
                for v_i in range(256):
                    g = acc_bout[v_i] * inv_b
                    m_bout[v_i] = beta1 * m_bout[v_i] + (1.0 - beta1) * g
                    v_bout[v_i] = beta2 * v_bout[v_i] + (1.0 - beta2) * (g * g)
                    self.b_out[v_i] -= lr_eff * m_bout[v_i] / (math.sqrt(v_bout[v_i]) + eps)
                _adam_mat(self.w_q, acc_wq, m_wq, v_wq, lr_eff, inv_b)
                _adam_mat(self.w_k, acc_wk, m_wk, v_wk, lr_eff, inv_b)
                _adam_mat(self.w_v, acc_wv, m_wv, v_wv, lr_eff, inv_b)
                for tid, gt in acc_etok.items():
                    pr = self.e_tok[tid]
                    mr = m_etok[tid]
                    vr = v_etok[tid]
                    for c in range(d):
                        g = gt[c] * inv_b
                        mr[c] = beta1 * mr[c] + (1.0 - beta1) * g
                        vr[c] = beta2 * vr[c] + (1.0 - beta2) * (g * g)
                        pr[c] -= lr_eff * mr[c] / (math.sqrt(vr[c]) + eps)
                for pid, gp in acc_epos.items():
                    pr = self.e_pos[pid]
                    mr = m_epos[pid]
                    vr = v_epos[pid]
                    for c in range(d):
                        g = gp[c] * inv_b
                        mr[c] = beta1 * mr[c] + (1.0 - beta1) * g
                        vr[c] = beta2 * vr[c] + (1.0 - beta2) * (g * g)
                        pr[c] -= lr_eff * mr[c] / (math.sqrt(vr[c]) + eps)

        self.train_seconds = time.perf_counter() - t0
        return self.train_seconds

    def predict_probs(self, context: bytes) -> list[float]:
        tok_ids = self._context_tokens(context)
        return self._forward_cache(tok_ids)["probs"]

    def byte_prob(self, context: bytes, target_byte: int) -> float:
        tok_ids = self._context_tokens(context)
        return self._forward_cache(tok_ids)["probs"][target_byte]

    def compressed_model_bits(self) -> int:
        n_params = (
            257 * self.d_model
            + self.ctx_len * self.d_model
            + 3 * self.d_model * self.d_model
            + 256 * self.d_model
            + 256
        )
        return n_params * 32


def run_byte_lm_benchmark_for_seed(seed: int) -> dict[str, Any]:
    """Run all 5 Byte-LM models on the identical zero-leakage UTF-8 corpus for ``seed``."""
    corpus = generate_disjoint_byte_lm_corpus(seed)
    models: list[BaseByteLM] = [
        UniformByteFloorLM(),
        Byte3GramKneserNeyLM(discount=0.75),
        PPMCOrder5ByteLM(max_order=5),
        MirrorCTWConstructionByteLM(raw_max_order=10, skel_max_order=12),
        SmallCPUByteTransformerRef(
            seed=seed, d_model=6, ctx_len=8, lr=0.035, train_stride=16, batch_size=16, epochs=1
        ),
    ]

    results_by_model: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    for m in models:
        m.fit(corpus.train_bytes)
        m.freeze()
        results_by_model[m.model_id] = evaluate_byte_lm_bpb(m, corpus.heldout_bytes)
    total_sec = time.perf_counter() - t0

    return {
        "seed": seed,
        "corpus": corpus.summary_dict(),
        "models": results_by_model,
        "total_benchmark_seconds": round(total_sec, 4),
    }
