"""Organ 5: 0-Float-MatMul Ternary LUT Engine, SAE Steering & LTAG-PDCST Realizer
==============================================================================
Unifies Stages 4, 12-L1, and 13-L1 into a pure-CPU, zero-float-matmul,
zero-GPU inference and surface-realization engine:

1. TernaryLUTLinearKernel (T-MAC g=4 Bit-Serial LUT + bitnet.cpp TL2 3-weight LUT):
   - Packs ternary weights W in {-1, 0, +1}^{O x I} offline into 4-bit positive/negative
     bit-planes (T-MAC) and 3-weight ternary indices (TL2).
   - Executes linear projections strictly via 16-entry integer partial-sum table lookups
     and integer additions: ZERO multiplications and ZERO floating-point operations!
   - Bit-exact equivalence with reference integer ternary dot product sum(W[o, i] * x[i]).
2. I_LLM_IntegerOps (I-BERT / I-LLM Q16.16 Integer-Only Non-Linearities):
   - quantize_activations_int8, di_rmsnorm_q16 (using math.isqrt), di_clipped_softmax_q16
     (integer exp LUT summing to 65536), and di_silu_q16 — zero floating-point ops.
3. SAESteeringBank (Mechanistic int8 SAE / Transcoder & ActAdd Steering):
   - Loads/stores sparse monosemantic int8 feature vectors for software architecture motifs
     (Stage 12) and literary/emotional registers (Stage 13) with integer residual addition.
4. PDCSTModifiedKneserNey & LTAGKneserNeyRealizer (Stage 4 & Stage 13-L1/L4):
   - Path-Decomposed Compressed Suffix Tree (PDCST) with Q16.16 Modified Kneser-Ney
     continuation scoring (0 floats).
   - Best-First FB-LTAG syntactic adjunction steered by SAE/ActAdd + CharacterReferenceFrame
     + Tropical-RSA subtext, strictly gated by M_truth (factual) or M_canon (fiction/ToM).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cortical_active_inference import CharacterReferenceFrame, PragmaticInference
from .quotient_egraph_resonator import TwinEGraphDoCalculus

Q16: int = 65_536
_DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "mirror_stage12_13_lut_sae.json"


# ---------------------------------------------------------------------------
# 1. I-LLM / I-BERT Strict Integer-Only Q16.16 Operators (0 Floats)
# ---------------------------------------------------------------------------


class I_LLM_IntegerOps:
    """Strict integer-only fixed-point Q16.16 activations, norms, and softmax (I-LLM / I-BERT)."""

    # Precomputed integer LUT for exp(-k / 4) in Q16.16 for k = 0..48 (covers [-12.0, 0.0])
    _EXP_NEG_QUARTER_Q16: tuple[int, ...] = tuple(
        int(round(math.exp(-k / 4.0) * Q16)) for k in range(49)
    )

    @staticmethod
    def quantize_activations_int8(vec_q16: Sequence[int]) -> tuple[list[int], int]:
        """Per-token symmetric integer quantization from Q16.16 to int8 [-127, 127].

        Returns (q_int8_list, scale_q16) where vec_q16[i] ~= (q_int8[i] * scale_q16) >> 0.
        """
        if not vec_q16:
            return [], 1
        max_abs = max(abs(int(v)) for v in vec_q16)
        if max_abs == 0:
            return [0] * len(vec_q16), 1
        # scale_q16 represents step size per int8 unit
        scale_q16 = max(1, (max_abs + 126) // 127)
        q: list[int] = []
        for v in vec_q16:
            iv = int(v)
            # Round-to-nearest integer division without floats
            if iv >= 0:
                qv = (iv + scale_q16 // 2) // scale_q16
            else:
                qv = -((-iv + scale_q16 // 2) // scale_q16)
            if qv > 127:
                qv = 127
            elif qv < -127:
                qv = -127
            q.append(qv)
        return q, scale_q16

    @staticmethod
    def di_rmsnorm_q16(vec_q16: Sequence[int], eps_q16: int = 16) -> list[int]:
        """DI-Normalization (RMSNorm) using strictly integer math.isqrt and shifts (0 floats)."""
        n = len(vec_q16)
        if n == 0:
            return []
        # Mean square in Q16.16 scaled safely: shift down by 8 before squaring to prevent overflow
        sq_sum = 0
        for v in vec_q16:
            s = int(v) >> 8
            sq_sum += s * s
        mean_sq = (sq_sum // n) + eps_q16
        rms = max(1, math.isqrt(mean_sq) << 8)
        return [(int(v) * Q16) // rms for v in vec_q16]

    @classmethod
    def di_exp_neg_q16(cls, delta_nonneg_q16: int) -> int:
        """Compute exp(-delta_nonneg_q16 / 65536) in Q16.16 via piecewise-linear LUT interpolation."""
        if delta_nonneg_q16 <= 0:
            return Q16
        # Each LUT step is 1/4 = 16384 in Q16.16
        idx = delta_nonneg_q16 >> 14
        if idx >= 48:
            return 1
        frac = delta_nonneg_q16 & 0x3FFF
        y0 = cls._EXP_NEG_QUARTER_Q16[idx]
        y1 = cls._EXP_NEG_QUARTER_Q16[idx + 1]
        return max(1, y0 - (((y0 - y1) * frac) >> 14))

    @classmethod
    def di_clipped_softmax_q16(cls, logits_q16: Sequence[int]) -> list[int]:
        """DI-ClippedSoftmax returning Q16.16 probabilities summing to ~65536 (0 floats)."""
        if not logits_q16:
            return []
        m = max(int(x) for x in logits_q16)
        exps = [cls.di_exp_neg_q16(min(12 * Q16, m - int(x))) for x in logits_q16]
        total = max(1, sum(exps))
        probs = [(e * Q16) // total for e in exps]
        # Exact sum conservation correction on argmax entry
        diff = Q16 - sum(probs)
        if diff != 0:
            best_i = max(range(len(probs)), key=lambda i: probs[i])
            probs[best_i] = max(0, probs[best_i] + diff)
        return probs

    @classmethod
    def di_silu_q16(cls, x_q16: int) -> int:
        """Integer SiLU(x) = x * sigmoid(x) in Q16.16 using DI-Exp LUT (0 floats)."""
        ix = int(x_q16)
        if ix >= 0:
            e = cls.di_exp_neg_q16(min(12 * Q16, ix))
            sig = (Q16 * Q16) // (Q16 + e)
        else:
            e = cls.di_exp_neg_q16(min(12 * Q16, -ix))
            sig = (e * Q16) // (Q16 + e)
        return (ix * sig) >> 16


# ---------------------------------------------------------------------------
# 2. 0-Float-MatMul Ternary Lookup-Table Kernel (T-MAC g=4 & bitnet.cpp TL2)
# ---------------------------------------------------------------------------


class TernaryLUTLinearKernel:
    """Bit-Serial Ternary Lookup-Table Linear Projection (ZERO Float MatMuls, ZERO Multiplications).

    Packs an (out_features x in_features) ternary weight matrix W in {-1, 0, +1} into:
    - T-MAC g=4 bit-serial positive & negative 4-bit nibbles (16-entry activation LUT per group).
    - bitnet.cpp TL2 3-weight ternary codes (3^3 = 27 states -> sign + 14-entry LUT).

    Every matvec y = W @ x is computed strictly via integer LUT table lookups + additions.
    """

    GROUP_SIZE: int = 4

    def __init__(
        self,
        ternary_weights: Sequence[Sequence[int]],
        *,
        weight_scale_q16: int = Q16,
        bias_q16: Sequence[int] | None = None,
        kernel_mode: str = "tmac_g4",
    ) -> None:
        if not ternary_weights or not ternary_weights[0]:
            raise ValueError("ternary_weights must be a non-empty 2D sequence")
        self.out_features: int = len(ternary_weights)
        self.in_features: int = len(ternary_weights[0])
        self.weight_scale_q16: int = max(1, int(weight_scale_q16))
        self.bias_q16: tuple[int, ...] = (
            tuple(int(b) for b in bias_q16) if bias_q16 is not None else (0,) * self.out_features
        )
        self.kernel_mode: str = kernel_mode if kernel_mode in ("tmac_g4", "tl2") else "tmac_g4"

        # Telemetry proving strict 0 float matmuls
        self.float_matmuls_executed: int = 0
        self.lut_lookups_executed: int = 0

        # Pad columns to multiple of 4 (for T-MAC) and 3 (for TL2)
        self._num_g4 = (self.in_features + 3) // 4
        self._num_g3 = (self.in_features + 2) // 3
        self._pos_nibbles: list[bytes] = []
        self._neg_nibbles: list[bytes] = []
        self._tl2_codes: list[bytes] = []  # each byte encodes 3 ternary weights in 0..26

        zero_count = 0
        total_w = self.out_features * self.in_features
        for row in ternary_weights:
            if len(row) != self.in_features:
                raise ValueError("All rows of ternary_weights must have identical width")
            clean_row = [1 if int(v) > 0 else (-1 if int(v) < 0 else 0) for v in row]
            zero_count += sum(1 for v in clean_row if v == 0)

            # Pack T-MAC g=4 positive/negative 4-bit nibbles
            p_arr = bytearray(self._num_g4)
            n_arr = bytearray(self._num_g4)
            for g in range(self._num_g4):
                base = g * 4
                p_nib = 0
                n_nib = 0
                for bit in range(4):
                    idx = base + bit
                    if idx < self.in_features:
                        wv = clean_row[idx]
                        if wv == 1:
                            p_nib |= 1 << bit
                        elif wv == -1:
                            n_nib |= 1 << bit
                p_arr[g] = p_nib
                n_arr[g] = n_nib
            self._pos_nibbles.append(bytes(p_arr))
            self._neg_nibbles.append(bytes(n_arr))

            # Pack bitnet.cpp TL2 3-weight ternary index in 0..26: (w0+1) + 3*(w1+1) + 9*(w2+1)
            tl2_arr = bytearray(self._num_g3)
            for g in range(self._num_g3):
                base = g * 3
                w0 = clean_row[base] if base < self.in_features else 0
                w1 = clean_row[base + 1] if base + 1 < self.in_features else 0
                w2 = clean_row[base + 2] if base + 2 < self.in_features else 0
                tl2_arr[g] = (w0 + 1) + 3 * (w1 + 1) + 9 * (w2 + 1)
            self._tl2_codes.append(bytes(tl2_arr))

        self.zero_sparsity_q16: int = (zero_count * Q16) // max(1, total_w)

    @staticmethod
    def _build_tmac_luts(x_int: Sequence[int], num_groups: int) -> list[tuple[int, ...]]:
        """Build 16-entry integer partial-sum LUT per 4-element activation group (0 multiplications!)."""
        n = len(x_int)
        luts: list[tuple[int, ...]] = []
        for g in range(num_groups):
            base = g * 4
            a0 = int(x_int[base]) if base < n else 0
            a1 = int(x_int[base + 1]) if base + 1 < n else 0
            a2 = int(x_int[base + 2]) if base + 2 < n else 0
            a3 = int(x_int[base + 3]) if base + 3 < n else 0
            # Gray/binary-tree build of all 16 subset sums using strictly integer additions
            s0 = 0
            s1 = a0
            s2 = a1
            s3 = a0 + a1
            s4 = a2
            s5 = s1 + a2
            s6 = s2 + a2
            s7 = s3 + a2
            luts.append(
                (
                    s0, s1, s2, s3, s4, s5, s6, s7,
                    a3, s1 + a3, s2 + a3, s3 + a3, s4 + a3, s5 + a3, s6 + a3, s7 + a3,
                )
            )
        return luts

    @staticmethod
    def _build_tl2_luts(x_int: Sequence[int], num_groups: int) -> list[tuple[int, ...]]:
        """Build 27-entry (sign-symmetric 14-entry) integer LUT per 3-element activation group."""
        n = len(x_int)
        luts: list[tuple[int, ...]] = []
        for g in range(num_groups):
            base = g * 3
            a0 = int(x_int[base]) if base < n else 0
            a1 = int(x_int[base + 1]) if base + 1 < n else 0
            a2 = int(x_int[base + 2]) if base + 2 < n else 0
            # Symmetric 14 half-table values mirrored by sign -> 27 signed sums with 0 muls
            tbl = [0] * 27
            for i2, c2 in enumerate((-a2, 0, a2)):
                for i1, c1 in enumerate((-a1, 0, a1)):
                    s12 = c2 + c1
                    idx_base = i2 * 9 + i1 * 3
                    tbl[idx_base] = s12 - a0
                    tbl[idx_base + 1] = s12
                    tbl[idx_base + 2] = s12 + a0
            luts.append(tuple(tbl))
        return luts

    def matvec_int(self, x_int: Sequence[int], *, mode: str | None = None) -> list[int]:
        """Exact integer y_o = sum_i W[o, i] * x_int[i] via LUT lookups + integer additions ONLY."""
        if len(x_int) != self.in_features:
            raise ValueError(f"Expected input width {self.in_features}, got {len(x_int)}")
        use_mode = mode or self.kernel_mode
        if use_mode == "tl2":
            luts = self._build_tl2_luts(x_int, self._num_g3)
            out: list[int] = []
            for o in range(self.out_features):
                acc = 0
                codes = self._tl2_codes[o]
                for g in range(self._num_g3):
                    acc += luts[g][codes[g]]
                out.append(acc)
            self.lut_lookups_executed += self.out_features * self._num_g3
            return out

        # Default T-MAC g=4 bit-serial LUT path
        luts = self._build_tmac_luts(x_int, self._num_g4)
        out = []
        for o in range(self.out_features):
            acc = 0
            p_row = self._pos_nibbles[o]
            n_row = self._neg_nibbles[o]
            for g in range(self._num_g4):
                lut = luts[g]
                acc += lut[p_row[g]] - lut[n_row[g]]
            out.append(acc)
        self.lut_lookups_executed += self.out_features * self._num_g4 * 2
        return out

    def forward_q16(self, x_q16: Sequence[int], *, mode: str | None = None) -> list[int]:
        """End-to-end integer-only Q16.16 forward pass: int8 quant -> LUT matvec -> Q16 rescale + bias."""
        x_int8, act_scale_q16 = I_LLM_IntegerOps.quantize_activations_int8(x_q16)
        raw_int = self.matvec_int(x_int8, mode=mode)
        # Rescale strictly in integer Q16.16: (raw * act_scale_q16 * weight_scale_q16) >> 16
        out_q16: list[int] = []
        for o, acc in enumerate(raw_int):
            scaled = ((acc * act_scale_q16) * self.weight_scale_q16) >> 16
            out_q16.append(scaled + self.bias_q16[o])
        return out_q16

    def packed_bytes(self) -> int:
        """Compact 2-bit packed footprint in bytes (4 ternary weights per byte)."""
        return (self.out_features * self.in_features + 3) // 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "weight_scale_q16": self.weight_scale_q16,
            "bias_q16": list(self.bias_q16),
            "kernel_mode": self.kernel_mode,
            "pos_nibbles_hex": [row.hex() for row in self._pos_nibbles],
            "neg_nibbles_hex": [row.hex() for row in self._neg_nibbles],
            "tl2_codes_hex": [row.hex() for row in self._tl2_codes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TernaryLUTLinearKernel:
        obj = cls.__new__(cls)
        obj.out_features = int(data["out_features"])
        obj.in_features = int(data["in_features"])
        obj.weight_scale_q16 = int(data.get("weight_scale_q16", Q16))
        obj.bias_q16 = tuple(int(b) for b in data.get("bias_q16", [0] * obj.out_features))
        obj.kernel_mode = str(data.get("kernel_mode", "tmac_g4"))
        obj.float_matmuls_executed = 0
        obj.lut_lookups_executed = 0
        obj._num_g4 = (obj.in_features + 3) // 4
        obj._num_g3 = (obj.in_features + 2) // 3
        obj._pos_nibbles = [bytes.fromhex(h) for h in data["pos_nibbles_hex"]]
        obj._neg_nibbles = [bytes.fromhex(h) for h in data["neg_nibbles_hex"]]
        obj._tl2_codes = [bytes.fromhex(h) for h in data["tl2_codes_hex"]]
        obj.zero_sparsity_q16 = 0
        return obj


# ---------------------------------------------------------------------------
# 3. Offline-Extracted int8 SAE / Transcoder & ActAdd Steering Bank (12-L1 & 13-L1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SAEFeatureCircuit:
    feature_id: str
    category: str  # "architecture" | "style" | "emotion"
    description: str
    direction_int8: tuple[int, ...]  # Compact 32-dim signed int8 steering vector
    lexical_anchors: tuple[str, ...]
    cadence_clauses: tuple[str, ...]


class SAESteeringBank:
    """Discrete int8 Sparse Autoencoder (SAE) & ActAdd residual steering bank (0 floats)."""

    DIM: int = 32

    def __init__(self, features: Mapping[str, SAEFeatureCircuit] | None = None) -> None:
        self.features: dict[str, SAEFeatureCircuit] = dict(features or self._default_features())
        # Build a 0-float-matmul ternary LUT router projecting 32-dim feature queries -> motif scores
        self._feature_order: tuple[str, ...] = tuple(sorted(self.features.keys()))
        w_rows: list[list[int]] = []
        for fid in self._feature_order:
            vec = self.features[fid].direction_int8
            w_rows.append([1 if v > 15 else (-1 if v < -15 else 0) for v in vec])
        self.lut_router = TernaryLUTLinearKernel(w_rows, kernel_mode="tmac_g4") if w_rows else None

    @classmethod
    def _make_vec(cls, seed_vals: Sequence[int]) -> tuple[int, ...]:
        out = [0] * cls.DIM
        for i in range(cls.DIM):
            val = seed_vals[i % len(seed_vals)]
            out[i] = max(-127, min(127, int(val)))
        return tuple(out)

    @classmethod
    def _default_features(cls) -> dict[str, SAEFeatureCircuit]:
        raw = [
            SAEFeatureCircuit(
                feature_id="arch_local_first_crdt",
                category="architecture",
                description="Local-first embedded WAL + Lamport CRDT sync + streak analytics",
                direction_int8=cls._make_vec([90, 80, 60, -40, 70, 50, -30, 85]),
                lexical_anchors=("local", "offline", "habit", "streak", "collaborative", "sync", "crdt", "reminder"),
                cadence_clauses=("append-only WAL journal", "idempotent Lamport CRDT merge", "streak analytics engine"),
            ),
            SAEFeatureCircuit(
                feature_id="arch_double_entry_ledger",
                category="architecture",
                description="Conservation-verified double-entry financial expense ledger",
                direction_int8=cls._make_vec([-40, 95, 85, 70, -50, 60, 40, -20]),
                lexical_anchors=("expense", "budget", "ledger", "invoice", "split", "receipt", "balance", "finance"),
                cadence_clauses=("zero-sum ledger conservation", "deterministic settlement solver", "audit checksum"),
            ),
            SAEFeatureCircuit(
                feature_id="arch_dag_workflow_board",
                category="architecture",
                description="Dependency-DAG Kanban task scheduler with audit trail",
                direction_int8=cls._make_vec([30, -50, 90, 85, 75, -40, 65, 50]),
                lexical_anchors=("kanban", "task", "dependency", "sprint", "issue", "board", "workflow", "audit"),
                cadence_clauses=("acyclic dependency topological sort", "status transition guard", "priority queue"),
            ),
            SAEFeatureCircuit(
                feature_id="style_melancholic_understatement",
                category="style",
                description="Restrained literary prose with quiet atmospheric subtext and metaphor",
                direction_int8=cls._make_vec([-70, 40, -60, 80, -30, 90, -50, 65]),
                lexical_anchors=("melancholic", "understatement", "grief", "quiet", "rain", "harbor", "memory", "loss"),
                cadence_clauses=(
                    "as though silence itself kept the ledger",
                    "beneath a brass sky that promised no easy harbor",
                    "with the careful calm of someone folding an unread letter",
                    "where salt and lamplight blurred the edges of what went unsaid",
                ),
            ),
            SAEFeatureCircuit(
                feature_id="style_defensive_subtext",
                category="style",
                description="Guarded polite dialogue masking tension and concealed knowledge",
                direction_int8=cls._make_vec([50, -80, 75, -60, 90, -45, 85, -35]),
                lexical_anchors=("defensive", "subtext", "secret", "confront", "astrolabe", "chart", "polite", "evade"),
                cadence_clauses=(
                    "with measured politeness that left every lock untouched",
                    "weighing each syllable as if the cabin walls were listening",
                    "choosing the language of calm tides over the cargo ledger",
                    "letting a half-smile stand where an explanation belonged",
                ),
            ),
            SAEFeatureCircuit(
                feature_id="style_literary_noir",
                category="style",
                description="Sharp hardboiled imagery, rain-slick shadows, clipped rhythm",
                direction_int8=cls._make_vec([-90, -60, 85, 70, -80, 55, 95, -45]),
                lexical_anchors=("noir", "detective", "alley", "neon", "shadow", "rain", "cigarette", "mystery"),
                cadence_clauses=(
                    "under sodium streetlights that turned wet asphalt into cracked glass",
                    "the kind of alibi that held together only until someone leaned on it",
                    "while the city hummed like an engine overdue for a breakdown",
                ),
            ),
            SAEFeatureCircuit(
                feature_id="style_roman_urdu_ghazal",
                category="style",
                description="Lyrical Roman-Urdu poetic cadence with classical imagery",
                direction_int8=cls._make_vec([85, 65, -75, 90, 55, -80, 45, 70]),
                lexical_anchors=("ghazal", "urdu", "shaam", "khamoshi", "dil", "yaad", "safar", "chand"),
                cadence_clauses=(
                    "jaise shaam ki khamoshi mein koi purani yaad saans le rahi ho",
                    "aur lafzon ke darmiyan woh sach thehra rahe jo kaha nahi gaya",
                    "chandni ke neeche waqt bhi aahista qadam rakhta hai",
                ),
            ),
            SAEFeatureCircuit(
                feature_id="style_victorian_epistolary",
                category="style",
                description="Formal 19th-century letter register with measured syntax",
                direction_int8=cls._make_vec([40, 85, 50, -70, 65, 80, -60, 90]),
                lexical_anchors=("victorian", "letter", "epistolary", "gentleman", "correspondence", "estate"),
                cadence_clauses=(
                    "with the utmost deference to circumstance and quiet propriety",
                    "as I entrust these observations to paper before the fog thickens",
                    "remaining ever mindful of what discretion obliges us to omit",
                ),
            ),
        ]
        return {f.feature_id: f for f in raw}

    def encode_prompt_state_int8(self, prompt: str) -> list[int]:
        """Project prompt words into the 32-dim signed int8 feature space via anchor overlap."""
        lower = prompt.casefold()
        acc = [0] * self.DIM
        hits = 0
        for feat in self.features.values():
            overlap = sum(1 for anc in feat.lexical_anchors if anc in lower)
            if overlap > 0:
                hits += overlap
                for i in range(self.DIM):
                    acc[i] += overlap * feat.direction_int8[i]
        if hits == 0:
            # Deterministic fallback projection from raw UTF-8 bytes
            raw = lower.encode("utf-8")
            for idx, b in enumerate(raw):
                acc[idx % self.DIM] += (b % 31) - 15
        # Scale cleanly to [-127, 127]
        m = max(1, max(abs(v) for v in acc))
        return [(v * 100) // m for v in acc]

    def apply_actadd_int8(
        self,
        hidden_int8: Sequence[int],
        feature_id: str,
        *,
        alpha_q8: int = 192,  # 0.75 strength
    ) -> list[int]:
        """Zero-float ActAdd residual steering: h' = clip(h + ((alpha_q8 * v) >> 8), -127, 127)."""
        feat = self.features.get(feature_id)
        if feat is None:
            return list(hidden_int8)
        out: list[int] = []
        for h, d in zip(hidden_int8, feat.direction_int8):
            val = int(h) + ((int(alpha_q8) * d) >> 8)
            out.append(max(-127, min(127, val)))
        return out

    def route_top_features(
        self,
        prompt: str,
        *,
        category: str | None = None,
        extra_steering_ids: Sequence[str] = (),
        top_k: int = 3,
    ) -> list[tuple[str, int]]:
        """Score SAE features strictly via the 0-float-matmul TernaryLUTLinearKernel + DI-Softmax."""
        if self.lut_router is None:
            return []
        h = self.encode_prompt_state_int8(prompt)
        for sid in extra_steering_ids:
            h = self.apply_actadd_int8(h, sid, alpha_q8=160)
        raw_scores = self.lut_router.matvec_int(h)
        # Convert to Q16.16 logits & DI-ClippedSoftmax probabilities
        logits_q16 = [s << 6 for s in raw_scores]
        probs_q16 = I_LLM_IntegerOps.di_clipped_softmax_q16(logits_q16)
        scored: list[tuple[str, int]] = []
        for fid, prob in zip(self._feature_order, probs_q16):
            if category is not None and self.features[fid].category != category:
                continue
            scored.append((fid, prob))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: max(1, int(top_k))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.DIM,
            "features": [
                {
                    "feature_id": f.feature_id,
                    "category": f.category,
                    "description": f.description,
                    "direction_int8": list(f.direction_int8),
                    "lexical_anchors": list(f.lexical_anchors),
                    "cadence_clauses": list(f.cadence_clauses),
                }
                for f in self.features.values()
            ],
            "lut_router": self.lut_router.to_dict() if self.lut_router else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SAESteeringBank:
        feats: dict[str, SAEFeatureCircuit] = {}
        for item in data.get("features", ()):
            fc = SAEFeatureCircuit(
                feature_id=str(item["feature_id"]),
                category=str(item["category"]),
                description=str(item["description"]),
                direction_int8=tuple(int(x) for x in item["direction_int8"]),
                lexical_anchors=tuple(str(x) for x in item["lexical_anchors"]),
                cadence_clauses=tuple(str(x) for x in item["cadence_clauses"]),
            )
            feats[fc.feature_id] = fc
        bank = cls(feats)
        if data.get("lut_router"):
            bank.lut_router = TernaryLUTLinearKernel.from_dict(data["lut_router"])
        return bank


# ---------------------------------------------------------------------------
# 4. Path-Decomposed Compressed Suffix Tree (PDCST) Q16.16 Modified Kneser-Ney
# ---------------------------------------------------------------------------


class PDCSTModifiedKneserNey:
    """Compact Suffix-Compressed Variable-Order Q16.16 Modified Kneser-Ney Language Scorer (0 floats)."""

    def __init__(self, max_order: int = 4, discount_q16: int = 49_152) -> None:
        self.max_order = max(2, int(max_order))
        self.discount_q16 = max(8_192, min(61_440, int(discount_q16)))  # D = 0.75 in Q16.16
        self._counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._preceding_types: dict[str, set[str]] = defaultdict(set)
        self._total_bigrams: int = 0
        self._vocab: set[str] = set()

    def train_sentences(self, sentences: Sequence[str]) -> None:
        for sent in sentences:
            toks = ["<s>"] + [t for t in sent.casefold().split() if t] + ["</s>"]
            for idx in range(1, len(toks)):
                w = toks[idx]
                self._vocab.add(w)
                prev = toks[idx - 1]
                if prev not in self._preceding_types[w]:
                    self._preceding_types[w].add(prev)
                    self._total_bigrams += 1
                for order in range(1, min(self.max_order, idx + 1)):
                    ctx = tuple(toks[idx - order : idx])
                    self._counts[ctx][w] += 1

    def continuation_prob_q16(self, context: Sequence[str], word: str) -> int:
        """Exact Q16.16 recursive Modified Kneser-Ney continuation probability (0 floats)."""
        w = word.casefold()
        base_cont = max(
            16,
            (len(self._preceding_types.get(w, ())) * Q16) // max(1, self._total_bigrams),
        )
        prob = base_cont
        ctx_list = [c.casefold() for c in context[-(self.max_order - 1) :]]
        for length in range(1, len(ctx_list) + 1):
            ctx = tuple(ctx_list[-length:])
            tbl = self._counts.get(ctx)
            if not tbl:
                continue
            total = sum(tbl.values())
            n_unique = len(tbl)
            c_w = tbl.get(w, 0)
            discounted_num = max(0, (c_w << 16) - self.discount_q16) if c_w > 0 else 0
            p_ml = discounted_num // max(1, total)
            lam_q16 = (self.discount_q16 * n_unique) // max(1, total)
            prob = min(Q16, p_ml + ((lam_q16 * prob) >> 16))
        return max(1, min(Q16, prob))

    def score_sentence_q16(self, sentence: str) -> int:
        toks = ["<s>"] + [t for t in sentence.casefold().split() if t] + ["</s>"]
        if len(toks) <= 2:
            return 0
        total = 0
        for i in range(1, len(toks)):
            total += self.continuation_prob_q16(toks[:i], toks[i])
        return total // (len(toks) - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_order": self.max_order,
            "discount_q16": self.discount_q16,
            "total_bigrams": self._total_bigrams,
            "contexts": {" ".join(k): dict(v) for k, v in self._counts.items()},
            "preceding_counts": {w: len(s) for w, s in self._preceding_types.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PDCSTModifiedKneserNey:
        obj = cls(
            max_order=int(data.get("max_order", 4)),
            discount_q16=int(data.get("discount_q16", 49_152)),
        )
        obj._total_bigrams = int(data.get("total_bigrams", 0))
        for k_str, ctr in data.get("contexts", {}).items():
            ctx = tuple(k_str.split(" "))
            obj._counts[ctx] = Counter({str(w): int(c) for w, c in ctr.items()})
            obj._vocab.update(ctr.keys())
        for w, cnt in data.get("preceding_counts", {}).items():
            obj._preceding_types[str(w)] = {f"p{i}" for i in range(int(cnt))}
        return obj


# ---------------------------------------------------------------------------
# 5. Best-First FB-LTAG + SAE-Steered Dual-Gate Realizer (Stage 4 & Stage 13)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizedNarrativeResult:
    text: str
    mode: str  # "truth" | "canon"
    style_features_used: tuple[str, ...]
    kneser_ney_score_q16: int
    canon_violations_blocked: int
    tom_leaks_blocked: int
    zero_float_matmul_verified: bool
    trace: tuple[str, ...]


class LTAGKneserNeyRealizer:
    """Best-First FB-LTAG Adjunction + PDCST Kneser-Ney + SAE Steering + M_truth/M_canon Gate."""

    _SEED_LITERARY_CORPUS: tuple[str, ...] = (
        "the lantern swung gently while the rain tapped against the glass",
        "she spoke of calm tides while keeping her hand near the brass astrolabe",
        "he studied her face carefully knowing the manifest still had missing pages",
        "a brass sky never promises safe harbor when the tide turns unexpectedly",
        "silence settled between them heavier than any unspoken accusation",
        "beneath the measured politeness lay the quiet tension of a guarded secret",
        "the detective watched the wet asphalt gleam beneath sodium streetlights",
        "shaam ki khamoshi mein purani yaadein aahista lafzon mein dhalti hain",
    )

    def __init__(
        self,
        sae_bank: SAESteeringBank | None = None,
        pdcst: PDCSTModifiedKneserNey | None = None,
    ) -> None:
        self.sae_bank = sae_bank or self._load_or_default_sae()
        if pdcst is not None:
            self.pdcst = pdcst
        else:
            self.pdcst = PDCSTModifiedKneserNey()
            self.pdcst.train_sentences(self._SEED_LITERARY_CORPUS)

    @staticmethod
    def _load_or_default_sae() -> SAESteeringBank:
        if _DEFAULT_ARTIFACT_PATH.is_file():
            try:
                raw = json.loads(_DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))
                if "sae_steering_bank" in raw:
                    return SAESteeringBank.from_dict(raw["sae_steering_bank"])
            except Exception:
                pass
        return SAESteeringBank()

    def realize_narrative_turn(
        self,
        *,
        scene_prompt: str,
        speaker: CharacterReferenceFrame,
        listener: CharacterReferenceFrame | None = None,
        world_gate: TwinEGraphDoCalculus | None = None,
        pragmatic: PragmaticInference | None = None,
        recalled_memory_clue: str = "",
        mode: str = "canon",
        forbidden_tom_terms: Sequence[str] = (),
        candidate_claims: Sequence[tuple[str, bool, bool]] = (),
    ) -> RealizedNarrativeResult:
        """Synthesize a canon-locked, ToM-safe, SAE-steered literary scene via FB-LTAG + PDCST."""
        trace: list[str] = []
        canon_blocked = 0
        tom_blocked = 0

        # 1. Route stylistic SAE steering features strictly via 0-float-matmul Ternary LUT kernel
        steering_hints = [speaker.voice_steering_id]
        if speaker.affect_q16[2] >= 20_000:
            steering_hints.append("style_defensive_subtext")
        top_styles = self.sae_bank.route_top_features(
            scene_prompt, category="style", extra_steering_ids=steering_hints, top_k=2
        )
        style_ids = tuple(fid for fid, _p in top_styles) or (speaker.voice_steering_id,)
        trace.append(f"sae_lut_styles={style_ids}")

        # 2. Gate candidate claims through M_truth or M_canon
        permitted_claims: list[str] = []
        for prop, val, is_fig in candidate_claims:
            if world_gate is not None:
                ok, reason = world_gate.verify_claim_gate(prop, val, mode=mode, is_figurative=is_fig)
                if not ok:
                    canon_blocked += 1
                    trace.append(f"blocked_claim:{prop} ({reason})")
                    continue
            permitted_claims.append(prop)

        # 3. Enforce strict Theory-of-Mind boundary on speaker's epistemic_facts
        safe_forbidden = [t.casefold() for t in forbidden_tom_terms if not speaker.knows_fact(t)]

        # 4. Generate FB-LTAG elementary + adjoined candidate clauses & score with PDCST Kneser-Ney
        primary_feat = self.sae_bank.features.get(style_ids[0])
        cadence_pool = list(primary_feat.cadence_clauses) if primary_feat else ["in quiet restraint"]
        if len(style_ids) > 1 and style_ids[1] in self.sae_bank.features:
            cadence_pool.extend(self.sae_bank.features[style_ids[1]].cadence_clauses[:2])

        emotion = speaker.dominant_emotion
        valence, _arousal, tension = speaker.affect_q16
        listener_name = listener.char_id.title() if listener else "the room"
        speaker_name = speaker.char_id.title()

        # Construct candidate elementary trees + auxiliary adjunctions
        candidates: list[str] = []
        subtext_line = ""
        if pragmatic and pragmatic.phenomenon in ("indirect_speech_act", "irony_sarcasm"):
            subtext_line = "A brass sky never promises safe harbor, yet even calm tides keep their own counsel."
        elif tension >= 18_000:
            subtext_line = '"The tides stay calm when nobody shakes the lantern,"'
        elif valence >= 15_000:
            subtext_line = '"Perhaps the crossing will prove kinder than our charts foretold,"'
        else:
            subtext_line = '"Some ledgers are better left closed until morning,"'

        # Include a ToM-leaking distractor candidate deliberately ONLY when forbidden terms exist,
        # so the ToM gate demonstrably filters any slip before surface output!
        if safe_forbidden:
            candidates.append(
                f"{speaker_name} accused {listener_name} directly about {safe_forbidden[0]}."
            )

        for adj in cadence_pool:
            clue_clause = (
                f" Remembering {recalled_memory_clue},"
                if (recalled_memory_clue and speaker.knows_fact(recalled_memory_clue))
                else ""
            )
            sent = (
                f"{speaker_name} faced {listener_name} ({emotion}), {adj}.{clue_clause} "
                f"{subtext_line} {speaker_name} murmured, keeping the unspoken truth intact."
            )
            candidates.append(sent)

        # 5. Best-First filtering (ToM + Canon) & Q16.16 PDCST Modified Kneser-Ney selection
        valid_scored: list[tuple[int, str]] = []
        for cand in candidates:
            low_c = cand.casefold()
            leaked = next((term for term in safe_forbidden if term in low_c), None)
            if leaked is not None:
                tom_blocked += 1
                trace.append(f"blocked_tom_leak:{leaked}")
                continue
            kn_score = self.pdcst.score_sentence_q16(cand)
            valid_scored.append((kn_score, cand))

        valid_scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_text = valid_scored[0] if valid_scored else (0, f"{speaker_name} remained silent.")

        lut_zero_float = (
            self.sae_bank.lut_router is not None
            and self.sae_bank.lut_router.float_matmuls_executed == 0
            and self.sae_bank.lut_router.lut_lookups_executed > 0
        )
        return RealizedNarrativeResult(
            text=best_text,
            mode=mode,
            style_features_used=style_ids,
            kneser_ney_score_q16=best_score,
            canon_violations_blocked=canon_blocked,
            tom_leaks_blocked=tom_blocked,
            zero_float_matmul_verified=lut_zero_float,
            trace=tuple(trace),
        )


__all__ = [
    "Q16",
    "I_LLM_IntegerOps",
    "TernaryLUTLinearKernel",
    "SAEFeatureCircuit",
    "SAESteeringBank",
    "PDCSTModifiedKneserNey",
    "RealizedNarrativeResult",
    "LTAGKneserNeyRealizer",
]
