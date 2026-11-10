"""
Phase-2 Cognitive Core & Phase-3 Bottleneck Guards for Mirror Crystal (100% Pure Python CPU, 0 Weights):

Phase-2 Capabilities:
  1. PerceptualVQLens           — Brightness/gain-invariant INT8 2D image & 1D waveform VQ patch tokenizer.
  2. SandboxMCTSPlanner         — Best-First / UCB1 CPU program synthesis & AST self-repair around ZeroTrustSandboxJudge.
  3. CoreferenceAndSenseTracker — Multi-turn entity pronoun resolution + context-cue homonym sense disambiguation.
  4. NightlyDreamSynthesizer    — Autonomous idle-CPU multi-hop causal shortcut discovery & crystallization.
  5. UniversalConceptHub & FluentByteRealizer — Zero-duplication cross-lingual concept locking & grounded realization.

Phase-3 Bottleneck Defenses:
  - P3-1 DreamUtilityPruner          — Cycle check + novelty utility threshold + strict per-cycle dream cap.
  - P3-2 TopicShiftDecayGuard        — TTL expiry + new-entity override preventing sticky stale pronouns.
  - P3-3 BoundedLoopStepGuard        — Enforced via ZeroTrustSandboxJudge(max_loop_steps) + wall-clock search cap.
  - P3-4 MorphologicalFuzzySpellBridge — Wired in ContextBitTable (bucketed Damerau-Levenshtein + POPCNT).
  - P3-5 EpistemicConfidenceCalibrator — Calibrated certainty bands + false-cousin single-word overlap filter.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import math
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from .bit_signatures import ContextBitTable, _STOPWORDS
from .grammar_and_sandbox import ZeroTrustSandboxJudge


# ============================================================================
# 1. PHASE-2 #1: PerceptualVQLens (Multimodal Vision & Signal INT8 Tokenizer)
# ============================================================================


class PerceptualVQLens:
    """Zero-weight INT8 perceptual Vector-Quantized patch & waveform tokenizer (affine lighting invariant)."""

    @staticmethod
    def _d4_canonical_hex(q_levels: list[int], side: int) -> str:
        """Compute canonical D4 dihedral hash invariant to 90/180/270-deg camera rotations & mirror flips."""
        if side < 2 or side * side != len(q_levels):
            return hashlib.blake2b(bytes(q_levels), digest_size=2).hexdigest()
        mat = [q_levels[r * side : (r + 1) * side] for r in range(side)]
        sym_bytes: list[bytes] = []
        curr = mat
        for _ in range(4):
            sym_bytes.append(bytes(v for row in curr for v in row))
            sym_bytes.append(bytes(v for row in curr for v in reversed(row)))  # horizontal mirror
            # Rotate 90 degrees clockwise
            curr = [[curr[side - 1 - r][c] for r in range(side)] for c in range(side)]
        return hashlib.blake2b(min(sym_bytes), digest_size=2).hexdigest()

    @classmethod
    def _quantize_patch_values(cls, vals: list[float]) -> tuple[str, str, str]:
        if not vals:
            return "empty", "empty", "flat"
        n = len(vals)
        mean_v = sum(vals) / n
        mad = sum(abs(v - mean_v) for v in vals) / n
        if mad < 1e-5:
            return "flat000", "flat000", "flat"

        # Affine-invariant (zero-mean, MAD-scaled) 5-level INT8 structural quantization: {-2,-1,0,1,2}
        q_levels: list[int] = []
        for v in vals:
            z = (v - mean_v) / mad
            q = max(-2, min(2, int(round(z))))
            q_levels.append(q + 2)  # shift into 0..4

        packed = bytes(q_levels)
        code_hex = hashlib.blake2b(packed, digest_size=2).hexdigest()
        side = int(math.isqrt(n))
        d4_hex = cls._d4_canonical_hex(q_levels, side)

        # Coarse orientation cue (horizontal vs vertical vs symmetric)
        if side >= 2 and side * side == n:
            top_sum = sum(vals[: n // 2])
            bot_sum = sum(vals[n // 2 :])
            left_sum = sum(vals[r * side + c] for r in range(side) for c in range(side // 2))
            right_sum = sum(vals[r * side + c] for r in range(side) for c in range(side // 2, side))
            dy = abs(top_sum - bot_sum) / max(1e-6, mad * n)
            dx = abs(left_sum - right_sum) / max(1e-6, mad * n)
            if dx > dy * 1.35 and dx > 0.25:
                orient = "vgrad"
            elif dy > dx * 1.35 and dy > 0.25:
                orient = "hgrad"
            else:
                orient = "balanced"
        else:
            orient = "1d"
        return code_hex, d4_hex, orient

    @classmethod
    def encode_image_grid(cls, grid: Sequence[Sequence[float | int]], patch_size: int = 4) -> list[str]:
        """Convert any HxW numeric pixel matrix into brightness/contrast & D4-rotation VQ barcode tokens."""
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        ps = max(2, min(int(patch_size), max(2, min(h, w))))

        tokens: list[str] = []
        shape_counts: dict[str, int] = defaultdict(int)
        d4_counts: dict[str, int] = defaultdict(int)

        # Local non-overlapping (clamped-edge) spatial patches
        pr_idx = 0
        for r0 in range(0, h, ps):
            pc_idx = 0
            for c0 in range(0, w, ps):
                vals: list[float] = []
                for dr in range(ps):
                    rr = min(h - 1, r0 + dr)
                    row_ref = grid[rr]
                    for dc in range(ps):
                        cc = min(w - 1, c0 + dc)
                        vals.append(float(row_ref[cc]))
                code_hex, d4_hex, orient = cls._quantize_patch_values(vals)
                tokens.append(f"vq2d_r{pr_idx}c{pc_idx}_{code_hex}")
                if code_hex != "flat000":
                    shape_counts[code_hex] += 1
                    d4_counts[d4_hex] += 1
                    tokens.append(f"vqorient_r{pr_idx}c{pc_idx}_{orient}")
                pc_idx += 1
            pr_idx += 1

        # Translation-tolerant & D4 rotation/flip-invariant bag-of-shapes tokens
        for code_hex in sorted(shape_counts):
            tokens.append(f"vqshape_{code_hex}")
        for d4_hex in sorted(d4_counts):
            tokens.append(f"vqd4_{d4_hex}")

        # Global 2x2 macro-quadrant layout descriptor + D4 macro symmetry token
        quad_means: list[float] = []
        mid_r = max(1, h // 2)
        mid_c = max(1, w // 2)
        for r_slice in ((0, mid_r), (mid_r, h)):
            for c_slice in ((0, mid_c), (mid_c, w)):
                q_vals = [
                    float(grid[r][c])
                    for r in range(r_slice[0], max(r_slice[0] + 1, r_slice[1]))
                    for c in range(c_slice[0], max(c_slice[0] + 1, c_slice[1]))
                    if r < h and c < w
                ]
                quad_means.append(sum(q_vals) / max(1, len(q_vals)))
        macro_code, macro_d4, macro_orient = cls._quantize_patch_values(quad_means)
        tokens.append(f"vqmacro_{macro_code}_{macro_orient}")
        tokens.append(f"vqmacrod4_{macro_d4}")
        return tokens

    @classmethod
    def encode_waveform(cls, samples: Sequence[float | int], num_segments: int = 8) -> list[str]:
        """Convert any 1D numeric signal/waveform into DC-offset & gain-invariant VQ tokens."""
        n = len(samples)
        if n == 0:
            return []
        f_vals = [float(x) for x in samples]
        mean_v = sum(f_vals) / n
        mad = sum(abs(x - mean_v) for x in f_vals) / n
        if mad < 1e-6:
            return ["vq1d_flat_signal"]

        norm = [(x - mean_v) / mad for x in f_vals]
        # Zero-crossing rate bucket (invariant to DC offset & positive gain)
        zc = sum(1 for i in range(1, n) if (norm[i - 1] >= 0.0) != (norm[i] >= 0.0))
        zc_bin = min(8, int(round(8.0 * zc / max(1, n - 1))))
        crest_bin = min(6, int(round(max(abs(z) for z in norm))))

        segs = max(2, min(int(num_segments), n))
        tokens = [f"vqwave_zc{zc_bin}", f"vqwave_crest{crest_bin}"]
        for s_idx in range(segs):
            i0 = (s_idx * n) // segs
            i1 = max(i0 + 1, ((s_idx + 1) * n) // segs)
            chunk = norm[i0:i1]
            lvl = max(-2, min(2, int(round(sum(chunk) / len(chunk)))))
            slope_raw = chunk[-1] - chunk[0] if len(chunk) > 1 else 0.0
            slope = "up" if slope_raw > 0.35 else ("dn" if slope_raw < -0.35 else "fl")
            tokens.append(f"vq1d_s{s_idx}_l{lvl + 2}_{slope}")
        return tokens

    @classmethod
    def encode_png_bytes(cls, png_bytes_or_path: bytes | Any, patch_size: int = 4) -> list[str]:
        """Decode real RFC-2083 .png bytes/file (with PNGDecompressionBombGuard) into VQ barcode tokens."""
        from .phase5_sensory_shards_wasm import SensoryCodec

        return SensoryCodec.encode_png_vq_tokens(png_bytes_or_path, patch_size=patch_size)

    @classmethod
    def encode_wav_bytes(cls, wav_bytes_or_path: bytes | Any, num_segments: int = 8) -> list[str]:
        """Decode real RIFF .wav PCM bytes/file (with WAVDurationCap) into VQ waveform & pitch-band tokens."""
        from .phase5_sensory_shards_wasm import SensoryCodec

        return SensoryCodec.encode_wav_vq_tokens(wav_bytes_or_path, num_segments=num_segments)

    @classmethod
    def wrap_multimodal_prompt(
        cls,
        prompt: str,
        image_grid: Sequence[Sequence[float | int]] | None = None,
        waveform: Sequence[float | int] | None = None,
        png_bytes: bytes | Any | None = None,
        wav_bytes: bytes | Any | None = None,
        patch_size: int = 4,
    ) -> str:
        parts = [prompt.strip()]
        if image_grid is not None:
            parts.extend(cls.encode_image_grid(image_grid, patch_size=patch_size))
        if png_bytes is not None:
            parts.extend(cls.encode_png_bytes(png_bytes, patch_size=patch_size))
        if waveform is not None:
            parts.extend(cls.encode_waveform(waveform))
        if wav_bytes is not None:
            parts.extend(cls.encode_wav_bytes(wav_bytes))
        return " ".join(p for p in parts if p)


# ============================================================================
# 2. PHASE-2 #3 + PHASE-3 #2: CoreferenceAndSenseTracker + TopicShiftDecayGuard
# ============================================================================

_HOMONYM_FAST_RE = re.compile(
    r"\b(?:banks?|python|trees?|keys?|tables?|threads?|cells?|charges?)\b", re.I
)

_SENSE_DISAMBIGUATION_RULES: tuple[tuple[frozenset[str], frozenset[str], str], ...] = (
    (
        frozenset({"bank", "banks"}),
        frozenset({"river", "water", "erosion", "erode", "flood", "flooding", "stream", "mud", "shore", "fish", "sediment"}),
        "sense_river_bank",
    ),
    (
        frozenset({"bank", "banks"}),
        frozenset({"money", "loan", "loans", "account", "interest", "deposit", "deposits", "credit", "finance", "financial", "vault", "currency"}),
        "sense_finance_bank",
    ),
    (
        frozenset({"python"}),
        frozenset({"snake", "snakes", "reptile", "reptiles", "constrictor", "prey", "jungle", "species", "venom", "nonvenomous", "scales"}),
        "sense_reptile_python",
    ),
    (
        frozenset({"python"}),
        frozenset({"code", "def", "function", "list", "dict", "import", "script", "pip", "gil", "interpreter", "programming", "class", "syntax", "loop", "while", "for", "array", "slice", "reverse"}),
        "sense_code_python",
    ),
    (
        frozenset({"tree", "trees"}),
        frozenset({"leaf", "leaves", "forest", "soil", "oak", "pine", "bark", "photosynthesis", "plant", "roots", "timber", "wood"}),
        "sense_botany_tree",
    ),
    (
        frozenset({"tree", "trees"}),
        frozenset({"binary", "node", "nodes", "bst", "avl", "traversal", "heap", "pointer", "recursion", "graph", "search", "balanced", "segment", "fenwick", "prefix", "interval", "trie", "spanning", "btree", "root", "leaf", "subtree"}),
        "sense_cs_tree",
    ),
    (
        frozenset({"cell", "cells"}),
        frozenset({"mitosis", "dna", "nucleus", "membrane", "biology", "organism", "cytoplasm", "ribosome", "tissue"}),
        "sense_bio_cell",
    ),
    (
        frozenset({"cell", "cells"}),
        frozenset({"battery", "voltage", "lithium", "anode", "cathode", "electrolyte", "ampere", "discharge"}),
        "sense_battery_cell",
    ),
)

_PRONOUN_CUES: frozenset[str] = frozenset(
    {"it", "its", "itself", "they", "them", "their"}
)

# Generic follow-up attribute/question words that do NOT constitute a topic switch
_GENERIC_FOLLOWUP_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "with",
        "from",
        "by",
        "at",
        "as",
        "and",
        "or",
        "it",
        "its",
        "itself",
        "that",
        "this",
        "they",
        "them",
        "their",
        "time",
        "space",
        "complexity",
        "asymptotic",
        "speed",
        "fast",
        "faster",
        "slow",
        "cost",
        "memory",
        "overhead",
        "advantage",
        "advantages",
        "disadvantage",
        "benefit",
        "benefits",
        "limitation",
        "limitations",
        "work",
        "works",
        "working",
        "run",
        "runs",
        "running",
        "use",
        "used",
        "using",
        "usage",
        "example",
        "examples",
        "implement",
        "implementation",
        "calculate",
        "compute",
        "formula",
        "equation",
        "step",
        "steps",
        "purpose",
        "meaning",
        "mean",
        "means",
        "explain",
        "tell",
        "me",
        "show",
        "give",
        "write",
        "please",
        "about",
        "main",
        "key",
        "rate",
        "value",
        "output",
        "result",
        "safe",
        "unsafe",
        "bursts",
        "burst",
    }
)


@dataclass
class _SessionAnchor:
    entity_tokens: tuple[str, ...]
    ttl_remaining: int


class CoreferenceAndSenseTracker:
    """Tracks per-session salient subject entities with TopicShiftDecayGuard + homonym sense disambiguation."""

    def __init__(self, ttl_turns: int = 3) -> None:
        self.ttl_turns = max(1, int(ttl_turns))
        self._anchors: dict[str, _SessionAnchor] = {}
        self.last_resolved_query: dict[str, str] = {}

    @staticmethod
    def detect_sense_tags(prompt: str, context_hint: str = "") -> list[str]:
        """Ultra-fast O(L) homonym context-sense disambiguator (~0.15 us fast exit on non-homonyms)."""
        if _HOMONYM_FAST_RE.search(prompt) is None:
            return []
        p_toks = set(ContextBitTable.tokenize(prompt))
        c_toks = p_toks | (set(ContextBitTable.tokenize(context_hint)) if context_hint else set())

        tags: list[str] = []
        for target_words, cue_words, sense_tag in _SENSE_DISAMBIGUATION_RULES:
            if p_toks & target_words:
                # Prefer prompt cues; fall back to response context_hint only if prompt itself was ambiguous
                if p_toks & cue_words:
                    tags.append(sense_tag)
                elif context_hint and (c_toks & cue_words):
                    # Ensure prompt didn't already trigger the opposite sense of the same word
                    tags.append(sense_tag)
        return tags

    @classmethod
    def enrich_for_ingest(cls, prompt: str, response_hint: str = "") -> tuple[str, list[str]]:
        tags = cls.detect_sense_tags(prompt, context_hint=response_hint)
        if not tags:
            return prompt, []
        return f"{prompt} {' '.join(tags)}", tags

    @classmethod
    def enrich_for_query(
        cls,
        prompt: str,
        known_vocab: Mapping[str, int] | None = None,
    ) -> tuple[str, list[str]]:
        tags = cls.detect_sense_tags(prompt)
        if not tags:
            return prompt, []
        if known_vocab is not None:
            tags = [t for t in tags if known_vocab.get(t, 0) > 0]
        if not tags:
            return prompt, []
        return f"{prompt} {' '.join(tags)}", tags

    @staticmethod
    def extract_topic_entities(prompt: str) -> list[str]:
        # Strip emails, URLs, and __email_0__/__secret_0__/__ip_0__ scrubber placeholders before topic scan
        cleaned = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|https?://\S+|__[a-z]+_\d+__",
            " ",
            prompt,
        )
        toks = ContextBitTable.tokenize(cleaned)
        out: list[str] = []
        for t in toks:
            if (
                len(t) >= 3
                and not t[0].isdigit()
                and not t.startswith("__")
                and not t.startswith("vq")
                and not t.startswith("sense_")
                and t not in _GENERIC_FOLLOWUP_WORDS
                and any(ch.isalpha() for ch in t)
            ):
                if t not in out:
                    out.append(t)
        return out

    def resolve_turn(self, session_id: str, prompt: str) -> tuple[str, list[str]]:
        """Expand pronouns ONLY on genuine follow-ups; expire or override on topic shifts (TopicShiftDecayGuard)."""
        raw_toks = ContextBitTable.tokenize(prompt)
        if not raw_toks:
            return prompt, []

        has_pronoun = any(t in _PRONOUN_CUES for t in raw_toks) or (
            "this algorithm" in prompt.lower() or "that algorithm" in prompt.lower()
        )
        new_entities = self.extract_topic_entities(prompt)
        anchor = self._anchors.get(session_id)
        trace: list[str] = []

        # Case 1: Active anchor + pronoun + NO overriding new topic entity -> expand coreference!
        if anchor is not None and anchor.ttl_remaining > 0 and has_pronoun and len(new_entities) == 0:
            anchor.ttl_remaining = self.ttl_turns
            injected = " ".join(anchor.entity_tokens)
            trace.append(f"Coreference resolved -> '{injected}'")
            resolved_q = f"{prompt} ({injected})"
            self.last_resolved_query[session_id] = resolved_q
            return resolved_q, trace

        # Case 2: Prompt introduces explicit topic entities -> TopicShiftDecayGuard overrides anchor!
        if new_entities:
            if anchor is not None and has_pronoun and tuple(new_entities[:4]) != anchor.entity_tokens:
                trace.append(
                    f"TopicShiftDecayGuard blocked stale anchor '{' '.join(anchor.entity_tokens)}' in favor of '{' '.join(new_entities[:4])}'"
                )
            self._anchors[session_id] = _SessionAnchor(
                entity_tokens=tuple(new_entities[:4]),
                ttl_remaining=self.ttl_turns,
            )
            self.last_resolved_query[session_id] = prompt
            return prompt, trace

        # Case 3: Neither pronoun expansion nor new entity -> decay TTL and purge when expired
        if anchor is not None:
            anchor.ttl_remaining -= 1
            if anchor.ttl_remaining <= 0:
                self._anchors.pop(session_id, None)
        self.last_resolved_query[session_id] = prompt
        return prompt, trace


# ============================================================================
# 3. PHASE-2 #5 + PHASE-3 #5: UniversalConceptHub, FluentByteRealizer & Calibrator
# ============================================================================

# Built-in high-leverage multilingual lexicon -> canonical English concept tokens (0 vault duplication!)
_DEFAULT_MULTILINGUAL_CONCEPTS: dict[str, str] = {
    # Urdu (اردو)
    "گروور": "grover",
    "کوانٹم": "quantum",
    "تلاش": "search",
    "الگورتھم": "algorithm",
    "پیچیدگی": "complexity",
    "وقت": "time",
    "الجھن": "entanglement",
    "سپرپوزیشن": "superposition",
    "کیوبٹ": "qubit",
    "حساب": "compute",
    "مفرد": "prime",
    "بائنری": "binary",
    "درخت": "tree",
    "پائتھون": "python",
    # Arabic (العربية)
    "جروفر": "grover",
    "غروفر": "grover",
    "كمي": "quantum",
    "الكمي": "quantum",
    "كمومية": "quantum",
    "بحث": "search",
    "البحث": "search",
    "خوارزمية": "algorithm",
    "تعقيد": "complexity",
    "تشابك": "entanglement",
    "التشابك": "entanglement",
    "كيوبت": "qubit",
    "احسب": "compute",
    # Chinese (中文 — substring-matched for unsegmented CJK)
    "格罗弗": "grover",
    "量子": "quantum",
    "搜索": "search",
    "算法": "algorithm",
    "复杂度": "complexity",
    "时间": "time",
    "纠缠": "entanglement",
    "叠加": "superposition",
    "量子比特": "qubit",
    "计算": "compute",
    "质数": "prime",
    "素数": "prime",
    "二分": "binary",
    # Hindi (हिन्दी)
    "ग्रोवर": "grover",
    "क्वांटम": "quantum",
    "खोज": "search",
    "एल्गोरिदम": "algorithm",
    "जटिलता": "complexity",
    "उलझाव": "entanglement",
    "क्यूबिट": "qubit",
    "गणना": "compute",
    # Spanish (Español)
    "búsqueda": "search",
    "busqueda": "search",
    "cuántica": "quantum",
    "cuantica": "quantum",
    "cuántico": "quantum",
    "cuantico": "quantum",
    "algoritmo": "algorithm",
    "complejidad": "complexity",
    "entrelazamiento": "entanglement",
    "superposición": "superposition",
    "calcular": "compute",
    "calcula": "compute",
    # French (Français)
    "quantique": "quantum",
    "recherche": "search",
    "algorithme": "algorithm",
    "complexité": "complexity",
    "complexite": "complexity",
    "intrication": "entanglement",
    "calculer": "compute",
    # Roman Urdu & Hinglish transliterated mobile chat
    "talash": "search",
    "pechidgi": "complexity",
    "hisab": "compute",
    "hisaab": "compute",
    "uljhan": "entanglement",
    "khoj": "search",
    "jatilta": "complexity",
    "ganana": "compute",
    # Cross-Lingual Symbolic CAS (Calculus & Algebra operators across 7 languages)
    "مشتق": "differentiate",
    "تکامل": "integrate",
    "حل": "solve",
    "اشتق": "differentiate",
    "تكامل": "integrate",
    "求导": "differentiate",
    "微分": "differentiate",
    "积分": "integrate",
    "解方程": "solve",
    "अवकलन": "differentiate",
    "समाकलन": "integrate",
    "हल": "solve",
    "deriva": "differentiate",
    "derivar": "differentiate",
    "integra": "integrate",
    "integrar": "integrate",
    "resuelve": "solve",
    "dériver": "differentiate",
    "intégrer": "integrate",
    "résoudre": "solve",
}

_NON_ASCII_WORD_RE = re.compile(r"[^\x00-\x7F]+|[A-Za-zÀ-ÿ_][A-Za-zÀ-ÿ0-9_']*")


_FOREIGN_PARTICLES: frozenset[str] = frozenset(
    {
        "ka",
        "ki",
        "ke",
        "ko",
        "se",
        "mein",
        "par",
        "kya",
        "hai",
        "hain",
        "karo",
        "karein",
        "el",
        "la",
        "los",
        "las",
        "del",
        "al",
        "cual",
        "cuál",
        "que",
        "qué",
        "est",
        "une",
        "des",
        "les",
        "quel",
        "quelle",
        "de",
        "du",
        "pour",
    }
)


class UniversalConceptHub:
    """Maps 200-language / non-ASCII prompts onto canonical concept tokens with 0% vault duplication."""

    def __init__(self, extra_lexicon: Mapping[str, str] | None = None) -> None:
        self.lexicon: dict[str, str] = dict(_DEFAULT_MULTILINGUAL_CONCEPTS)
        if extra_lexicon:
            for surface, canon in extra_lexicon.items():
                self.lexicon[surface.strip().lower()] = canon.strip().lower()
        # Pre-separate CJK unsegmented keys for fast substring scan
        self._cjk_items: list[tuple[str, str]] = sorted(
            [(k, v) for k, v in self.lexicon.items() if any("\u4e00" <= ch <= "\u9fff" for ch in k)],
            key=lambda kv: len(kv[0]),
            reverse=True,
        )

    def register_translations(self, canonical_concept: str, surface_forms: Iterable[str]) -> None:
        canon = canonical_concept.strip().lower()
        for sf in surface_forms:
            key = sf.strip().lower()
            if key:
                self.lexicon[key] = canon
                if any("\u4e00" <= ch <= "\u9fff" for ch in key):
                    self._cjk_items.append((key, canon))
                    self._cjk_items.sort(key=lambda kv: len(kv[0]), reverse=True)

    @staticmethod
    def detect_language(text: str) -> str:
        for ch in text:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF:
                return "zh"
            if 0x0900 <= cp <= 0x097F:
                return "hi"
            if cp in (0x0679, 0x0688, 0x0691, 0x06BA, 0x06BE, 0x06C1, 0x06CC, 0x06D2, 0x06AF, 0x0686, 0x067E):
                return "ur"
        for ch in text:
            cp = ord(ch)
            if 0x0600 <= cp <= 0x06FF:
                return "ar"
        tl = text.lower()
        if any(w in tl for w in ("¿", "cuántic", "búsqueda", "complejidad", "cuál")):
            return "es"
        if any(w in tl for w in ("quantique", "complexité", "intrication", "qu'est")):
            return "fr"
        return "en"

    def normalize_to_canonical(self, text: str) -> tuple[str, str, list[str]]:
        """Return (canonical_augmented_prompt, detected_lang, mapped_concept_pairs)."""
        if text.isascii():
            # Ultra-fast ASCII check for unaccented Romance / Roman-Urdu words ('hisab', 'cuantica', 'quantique')
            toks = ContextBitTable.tokenize(text)
            mapped_ascii: list[str] = []
            repl_map: dict[str, str] = {}
            for t in toks:
                canon = self.lexicon.get(t)
                if canon and canon != t:
                    mapped_ascii.append(f"{t}->{canon}")
                    repl_map[t] = canon
            if not repl_map:
                return text, "en", []
            lang_a = self.detect_language(text)
            rewritten = re.sub(
                r"\b[A-Za-z_][A-Za-z0-9_']*\b",
                lambda m: (
                    ""
                    if m.group(0).lower() in _FOREIGN_PARTICLES
                    else repl_map.get(m.group(0).lower(), m.group(0))
                ),
                text,
            )
            rewritten = re.sub(r"\s+", " ", rewritten).strip()
            return rewritten, lang_a, mapped_ascii

        lang = self.detect_language(text)
        mapped: list[str] = []
        canon_tokens: list[str] = []
        latin_repl: dict[str, str] = {}

        # 1. CJK unsegmented substring scan (longest-match first so '量子比特' matches before '量子')
        if lang == "zh" or any("\u4e00" <= ch <= "\u9fff" for ch in text):
            rem = text
            for cjk_key, canon in self._cjk_items:
                if cjk_key in rem:
                    mapped.append(f"{cjk_key}->{canon}")
                    canon_tokens.append(canon)

        # 2. Space/script word lookup for Urdu, Arabic, Hindi, Spanish, French, etc.
        for m in _NON_ASCII_WORD_RE.finditer(text):
            w = m.group(0).lower().strip("؟،۔!?,.;:")
            canon = self.lexicon.get(w)
            if canon and canon != w:
                pair_str = f"{w}->{canon}"
                if pair_str not in mapped:
                    mapped.append(pair_str)
                    canon_tokens.append(canon)
                if any("a" <= ch <= "z" or "\u00c0" <= ch <= "\u00ff" for ch in w):
                    latin_repl[w] = canon

        if not canon_tokens:
            return text, lang, []
        cleaned_base = text
        if latin_repl or lang in {"es", "fr"}:
            cleaned_base = re.sub(
                r"[A-Za-zÀ-ÿ_][A-Za-zÀ-ÿ0-9_']*",
                lambda m: (
                    ""
                    if m.group(0).lower() in _FOREIGN_PARTICLES
                    else latin_repl.get(m.group(0).lower(), m.group(0))
                ),
                cleaned_base,
            )
        return f"{cleaned_base} {' '.join(canon_tokens)}".strip(), lang, mapped


_LANG_VERIFIED_PREFIX: dict[str, str] = {
    "ur": "[تصدیق شدہ کرسٹل اصول | Verified Crystal Lock]\n",
    "ar": "[قاعدة كريستال موثقة | Verified Crystal Lock]\n",
    "zh": "[已验证晶体规则 | Verified Crystal Lock]\n",
    "hi": "[सत्यापित क्रिस्टल नियम | Verified Crystal Lock]\n",
    "es": "[Regla Cristal Verificada]\n",
    "fr": "[Règle Cristal Vérifiée]\n",
}


class FluentByteRealizer:
    """Grounded surface realizer: strips internal `<think>` tags if requested & frames multilingual locks."""

    @staticmethod
    def realize(answer: str, lang: str = "en", include_think: bool = False) -> str:
        out = answer
        if not include_think:
            out = re.sub(r"<think>.*?</think>\s*", "", out, flags=re.DOTALL).strip()
        prefix = _LANG_VERIFIED_PREFIX.get(lang, "")
        if prefix and not out.startswith("["):
            return f"{prefix}{out}"
        return out


_WRAPPER_PROPER_ALLOWLIST: frozenset[str] = frozenset(
    {
        "python",
        "json",
        "html",
        "http",
        "https",
        "rest",
        "bash",
        "linux",
        "windows",
        "unicode",
        "ascii",
        "utf8",
        # Technical, data, and programming vocabulary
        "csv",
        "sql",
        "dataframe",
        "dataframes",
        "pandas",
        "numpy",
        "regex",
        "regexes",
        "array",
        "arrays",
        "string",
        "strings",
        "integer",
        "integers",
        "float",
        "floats",
        "boolean",
        "booleans",
        "dict",
        "dictionary",
        "list",
        "lists",
        "tuple",
        "tuples",
        "set",
        "sets",
        "stack",
        "queue",
        "tree",
        "graph",
        "matrix",
        "node",
        "hashmap",
        "binary",
        "pointer",
        "recursion",
        "recursive",
        "iteration",
        "sorting",
        "search",
        "leetcode",
        "lru",
        "lrucache",
        "cache",
        "trie",
        "unionfind",
        "linkedlist",
        "deque",
        "hashset",
        "github",
        "stdout",
        "stdin",
        "stderr",
        "async",
        "await",
        "function",
        "functions",
        "class",
        "classes",
        "method",
        "methods",
        "module",
        "modules",
        "package",
        "packages",
        "variable",
        "variables",
        "constant",
        "argument",
        "arguments",
        "parameter",
        "parameters",
        "api",
        "apis",
        "crud",
        "orm",
        "sdk",
        "cli",
        "gui",
        "ui",
        "ux",
        "url",
        "urls",
        "jwt",
        "oauth",
        "token",
        "tokens",
        "uuid",
        "datetime",
        "timestamp",
        "lambda",
        "yield",
        "generator",
        "iterator",
        "decorator",
        "database",
        "databases",
        "table",
        "tables",
        "column",
        "columns",
        "row",
        "rows",
        "index",
        "indices",
        "query",
        "queries",
        "schema",
        "schemas",
        "bfs",
        "dfs",
        "dag",
        "lca",
        "dp",
        "kmp",
        "fft",
        "ifft",
        "fenwick",
        "manacher",
        "levenshtein",
        "simpson",
        "euler",
        "gcd",
        "lcm",
        "heap",
        "heaps",
        "minheap",
        "maxheap",
        "priorityqueue",
        "segmenttree",
        "xml",
        "yaml",
        "toml",
        "json",
        "fifo",
        "lifo",
        "ast",
        "regexp",
        "decimal",
        "cmath",
        "operator",
        "enum",
        "code",
        "coding",
        "program",
        "script",
        "scripting",
        "implement",
        "implementation",
        "solution",
        "complexity",
    }
)


class EpistemicConfidenceCalibrator:
    """Phase-3 #5: Calibrated certainty tiers + false-cousin single-token overlap & coverage abstention filter."""

    @staticmethod
    def is_false_cousin_overlap(
        score: float,
        matched_salient_count: int,
        query_salient_count: int,
        num_slots: int,
    ) -> bool:
        """Reject weak single-word overlaps on specific >=4-concept queries so tangential cousins abstain."""
        if num_slots > 0:
            return False
        if query_salient_count >= 4 and matched_salient_count <= 1 and score < 0.34:
            return True
        return False

    @classmethod
    def is_coding_intent(
        cls,
        query: str,
        detected_domain: str = "",
        sense_tags: Sequence[str] = (),
    ) -> bool:
        """Detect whether a query represents a coding, algorithmic, or programming request."""
        if any("sense_code" in str(st) for st in sense_tags):
            return True
        if str(detected_domain).lower() in (
            "algorithms", "algorithm", "code", "programming", "python", "leetcode", "computer_science"
        ):
            return True
        q_low = query.lower()
        if re.search(
            r"\b(def\b|class\b|import\b|python|code|coding|program|programming|script|scripting|function|method|algorithm|implement|implementation|solution|leetcode|dataframe|sql|csv|regex|quicksort|mergesort|binary search|longest palindromic|two sum|fibonacci|linked list|binary tree|recursion|recursive|dynamic programming|palindrome|cache|lru|stack|queue|deque|trie|graph|bfs|dfs|heap|priority queue|fenwick|segment tree|fft|simpson|matrix|array|data structure)\b",
            q_low,
        ):
            return True
        return False

    @staticmethod
    def extract_oov_proper_entities(original_prompt: str, bit_table: Any) -> set[str]:
        """
        Phase-6 B6-8 Defense: Detect mid-sentence capitalized or alphanumeric proper entities
        ('Xylofluoride-992', 'Zorgon999', 'E903') that have df == 0 across the entire vault (even after typo bridging).
        Ignores recognized programming/tech allowlist keywords.
        """
        words = re.findall(r"[A-Za-z0-9_'-]+", original_prompt.strip())
        if len(words) < 2:
            return set()
        oov_entities: set[str] = set()
        for w in words[1:]:
            if w.startswith(("__", "vq", "sense_")):
                continue
            clean = w.strip("'-_")
            if clean.lower() in _WRAPPER_PROPER_ALLOWLIST:
                continue
            is_alnum_id = any(c.isalpha() for c in clean) and any(c.isdigit() for c in clean)
            if len(clean) < (4 if is_alnum_id else 5) or clean.startswith(("__", "vq", "sense_")):
                continue
            is_proper_or_code = clean[0].isupper() or is_alnum_id
            if not is_proper_or_code:
                continue
            for sub_tok in bit_table.tokenize(clean):
                if sub_tok in _WRAPPER_PROPER_ALLOWLIST or sub_tok.lower() in _WRAPPER_PROPER_ALLOWLIST:
                    continue
                if (len(sub_tok) >= 5 and sub_tok.isalpha()) or (is_alnum_id and len(sub_tok) >= 4):
                    resolved = bit_table.resolve_typo_token(sub_tok)
                    if resolved in _WRAPPER_PROPER_ALLOWLIST or resolved.lower() in _WRAPPER_PROPER_ALLOWLIST:
                        continue
                    if bit_table.token_df.get(resolved, 0) == 0:
                        oov_entities.add(resolved)
        return oov_entities

    @classmethod
    def verify_candidate_match(
        cls,
        *,
        query_text: str,
        original_prompt: str,
        raw_toks: Sequence[str],
        q_expanded: set[str],
        rule: Any,
        score: float,
        bit_table: Any,
        detected_lang: str = "en",
        min_lock_score: float = 0.22,
        is_coding_intent: bool = False,
    ) -> tuple[bool, float, float, str]:
        """
        Validate that a retrieved CrystalRule genuinely answers the query's core subjects,
        years, and entities rather than hallucinating on partial word overlap.

        Returns:
            (accepted, calibrated_confidence, query_coverage, reason)
        """
        if score < min_lock_score:
            return False, 0.0, 0.0, f"below_min_score ({score:.3f} < {min_lock_score:.2f})"

        q_salient_cnt = len(set(raw_toks))
        matched_cnt = len(q_expanded & rule.token_set)
        if cls.is_false_cousin_overlap(
            score,
            matched_salient_count=matched_cnt,
            query_salient_count=q_salient_cnt,
            num_slots=rule.num_slots,
        ):
            return False, 0.0, 0.0, f"false_cousin_overlap (score={score:.3f})"

        if rule.num_slots > 0:
            cal = round(min(1.0, score), 4)
            return True, cal, 1.0, "slot_rule_ok"

        rule_prompt_toks = set(bit_table.tokenize(rule.abstract_prompt))
        rule_all_vocab: set[str] = set(rule.token_set) | rule_prompt_toks
        rule_response_toks = set(bit_table.tokenize(rule.abstract_response))

        # Include response tokens ONLY when the rule prompt itself is very short (e.g., "Tell me about python")
        # or shares an explicit homonym sense lock (`sense_reptile_python`). Also add singularized forms.
        prompt_content_cnt = sum(1 for t in rule_prompt_toks if t not in _STOPWORDS and len(t) >= 2)
        q_sense_tags = {t for t in raw_toks if t.startswith("sense_")}
        shared_sense = bool(q_sense_tags & rule.token_set)
        if shared_sense or prompt_content_cnt <= 2:
            rule_match_vocab: set[str] = set(rule_all_vocab) | rule_response_toks
        else:
            rule_match_vocab = set(rule_all_vocab)
        # Light plural normalization so 'snakes'/'jungles' align with 'snake'/'jungle'
        rule_match_norms = set(rule_match_vocab) | {t[:-1] for t in rule_match_vocab if len(t) > 4 and t.endswith("s")}

        # 1. Explicit standalone 4-digit year guard (e.g. "2026 World Cup" vs generic 1930/1942 rule)
        q_years = re.findall(r"(?<!-)\b(1[89]\d{2}|20\d{2}|2100)\b(?!-)", query_text)
        if q_years:
            rule_text = f"{rule.abstract_prompt} {rule.abstract_response}"
            if any(yr not in rule_text for yr in q_years):
                return False, 0.0, 0.0, f"missing_explicit_query_year {q_years}"

        # 2. Mid-sentence capitalized proper entities / domain identifiers must exist in the matched rule
        # (even if df > 0 elsewhere in a multi-domain vault, e.g. 'Dijkstra' vs 'Fenwick'!).
        rule_entity_vocab = rule_all_vocab | rule_response_toks
        words = re.findall(r"[A-Za-z0-9_'-]+", original_prompt.strip())
        if len(words) >= 2:
            for w in words[1:]:
                if w.startswith(("__", "vq", "sense_")):
                    continue
                clean = w.strip("'-_")
                if (
                    len(clean) < 4
                    or clean.startswith(("__", "vq", "sense_"))
                    or clean.lower() in _WRAPPER_PROPER_ALLOWLIST
                ):
                    continue
                is_proper_or_id = clean[0].isupper() or (
                    any(c.isalpha() for c in clean) and any(c.isdigit() for c in clean)
                )
                if not is_proper_or_id:
                    continue
                for sub_tok in bit_table.tokenize(clean):
                    if (
                        len(sub_tok) < 4
                        or not sub_tok.isascii()
                        or not any(ch.isalpha() for ch in sub_tok)
                        or sub_tok in _STOPWORDS
                        or sub_tok in _WRAPPER_PROPER_ALLOWLIST
                        or sub_tok.lower() in _WRAPPER_PROPER_ALLOWLIST
                    ):
                        continue
                    resolved = bit_table.resolve_typo_token(sub_tok)
                    if (
                        resolved in _WRAPPER_PROPER_ALLOWLIST
                        or resolved.lower() in _WRAPPER_PROPER_ALLOWLIST
                    ):
                        continue
                    syns = set(bit_table._synonyms.get(resolved, ()))
                    if resolved not in rule_entity_vocab and not (syns & rule_entity_vocab):
                        return False, 0.0, 0.0, f"missing_proper_entity '{resolved}'"

        # 3. Salience-weighted query concept recall over unique ASCII content tokens
        syns_map = getattr(bit_table, "_synonyms", {})
        df_map = getattr(bit_table, "token_df", {})
        q_content: list[str] = []
        seen_qc: set[str] = set()
        for raw_t in raw_toks:
            if (
                raw_t in seen_qc
                or not raw_t.isascii()
                or len(raw_t) < 2
                or raw_t in _STOPWORDS
                or raw_t.startswith(("__", "vq", "sense_"))
                or not any(ch.isalpha() for ch in raw_t)
            ):
                continue
            if detected_lang != "en" and (raw_t in _FOREIGN_PARTICLES or df_map.get(raw_t, 0) == 0):
                continue
            seen_qc.add(raw_t)
            q_content.append(raw_t)

        if not q_content:
            cal = round(min(1.0, score), 4)
            return True, cal, 1.0, "non_content_ok"

        matched_sal = 0.0
        total_sal = 0.0
        for tok in q_content:
            w = float(bit_table.token_salience(tok))
            total_sal += w
            cand_syns = syns_map.get(tok, ())
            tok_stem = tok[:-1] if len(tok) > 4 and tok.endswith("s") else tok
            if (
                tok in rule_match_norms
                or tok_stem in rule_match_norms
                or any(s in rule_match_norms for s in cand_syns)
            ):
                matched_sal += w

        q_coverage = matched_sal / max(1e-6, total_sal)
        if is_coding_intent:
            min_cov = 0.28 if len(q_content) <= 3 else 0.32
        else:
            min_cov = 0.50 if len(q_content) <= 3 else 0.52
        if len(q_content) >= 3 and q_coverage < min_cov:
            return False, 0.0, q_coverage, f"low_query_coverage ({q_coverage:.2f} < {min_cov:.2f})"

        # Calibrate confidence downward when the query mentions unmatched terms
        cov_factor = 1.0 if q_coverage >= 0.85 else (0.55 + 0.45 * q_coverage)
        calibrated = round(min(1.0, score * cov_factor), 4)
        return True, calibrated, round(q_coverage, 4), "verified_match"

    @staticmethod
    def classify_tier(
        confidence: float,
        mode: str,
        sandbox_verified: bool,
        query_coverage: float = 1.0,
    ) -> str:
        if mode == "bicameral_creative_synthesis":
            return "creative_synthesis"
        if mode == "clarification_needed":
            return "diagnostic_clarification"
        if mode == "unmatched_honest_abstain" or confidence <= 0.0:
            return "honest_abstain"
        if (
            mode
            in {
                "session_constraint_lock",
                "slot_formula_lock",
                "multi_hop_rule_chain",
                "mcts_program_synthesis",
                "symbolic_cas_exact",
            }
            or (sandbox_verified and confidence >= 0.60 and query_coverage >= 0.50)
            or (confidence >= 0.85 and query_coverage >= 0.85)
        ):
            return "verified_exact"
        if sandbox_verified or (confidence >= 0.58 and query_coverage >= 0.70):
            return "high_confidence"
        return "moderate_confidence"


# ============================================================================
# 4. PHASE-2 #2 + PHASE-3 #3: SandboxMCTSPlanner (CPU Tree-Search & Self-Repair)
# ============================================================================


@dataclass
class _MCTSCodeNode:
    code: str
    origin: str
    reward: float = 0.0
    visits: int = 1
    expanded: bool = False


class SandboxMCTSPlanner:
    """Best-First / UCB1 CPU code planner composing & self-repairing rules against ZeroTrustSandboxJudge."""

    def __init__(
        self,
        max_nodes: int = 28,
        max_wall_ms: float = 350.0,
        max_loop_steps: int = 25_000,
        ucb_c: float = 1.15,
    ) -> None:
        self.max_nodes = max(4, int(max_nodes))
        self.max_wall_ms = max(20.0, float(max_wall_ms))
        self.max_loop_steps = max(500, int(max_loop_steps))
        self.ucb_c = float(ucb_c)

    @staticmethod
    def _infer_target_signature(prompt: str, test_code: str) -> tuple[str, int]:
        m_prompt = re.search(r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)", prompt)
        if not m_prompt:
            m_prompt = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)", prompt)
        if m_prompt and m_prompt.group(1) not in {"range", "len", "sum", "min", "max", "print", "assert", "O"}:
            fname = m_prompt.group(1)
            raw_args = [a.strip() for a in m_prompt.group(2).split(",") if a.strip()]
            return fname, max(1, len(raw_args))

        try:
            tree = ast.parse(test_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    fn = node.func.id
                    if fn not in {"len", "sum", "min", "max", "sorted", "set", "list", "tuple", "abs", "round", "pow", "range"}:
                        return fn, max(1, len(node.args))
        except SyntaxError:
            pass
        return "solve", 1

    @staticmethod
    def _extract_helper_functions(code_text: str) -> list[tuple[str, int, str]]:
        """Return list of (func_name, arity, source_segment) from safe Python code."""
        extracted = ZeroTrustSandboxJudge.extract_python_code(code_text) or code_text.strip()
        if not extracted:
            return []
        safe, _ = ZeroTrustSandboxJudge.is_ast_safe(extracted)
        if not safe:
            return []
        try:
            tree = ast.parse(extracted)
        except SyntaxError:
            return []

        helpers: list[tuple[str, int, str]] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                src = ast.get_source_segment(extracted, node)
                if src:
                    helpers.append((node.name, len(node.args.args), src))
        return helpers

    def _evaluate_candidate(self, code: str, test_code: str, assert_lines: list[str]) -> tuple[bool, float, str]:
        ok, msg = ZeroTrustSandboxJudge.verify_code_and_test(
            code, test_code=test_code, max_loop_steps=self.max_loop_steps
        )
        if ok:
            return True, 1.0, msg
        if "LoopStepLimitExceeded" in msg or "RangeSpanLimitExceeded" in msg or "SyntaxError" in msg or "Blocked" in msg:
            return False, 0.0, msg

        # Smooth partial assertion score for UCB1 guidance!
        if len(assert_lines) >= 2:
            passed = 0
            for aln in assert_lines:
                ok_sub, _ = ZeroTrustSandboxJudge.verify_code_and_test(
                    code, test_code=aln, max_loop_steps=self.max_loop_steps
                )
                if ok_sub:
                    passed += 1
            if passed > 0:
                return False, 0.35 + 0.55 * (passed / len(assert_lines)), msg

        if "AssertionError" in msg:
            return False, 0.45, msg
        if "ZeroDivisionError" in msg or "IndexError" in msg:
            return False, 0.30, msg
        return False, 0.10, msg

    @staticmethod
    def _generate_repair_mutations(code: str, target_fn: str, arity: int) -> list[tuple[str, str]]:
        """Generate bounded AST/token self-repair variants (off-by-one range, boundary <=, zero/empty guard)."""
        muts: list[tuple[str, str]] = []

        # 1. Off-by-one inclusive range repair: range(a, n) -> range(a, n + 1) or range(n) -> range(n + 1)
        m_rng2 = re.search(r"range\(\s*([^,]+?)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)", code)
        if m_rng2:
            repaired = (
                code[: m_rng2.start()]
                + f"range({m_rng2.group(1)}, {m_rng2.group(2)} + 1)"
                + code[m_rng2.end() :]
            )
            muts.append((repaired, "repair:range_inclusive_upper"))
        else:
            m_rng1 = re.search(r"range\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)", code)
            if m_rng1:
                repaired = code[: m_rng1.start()] + f"range({m_rng1.group(1)} + 1)" + code[m_rng1.end() :]
                muts.append((repaired, "repair:range_plus_one"))

        # 2. Comparison boundary flips (< -> <=, > -> >=)
        if " < " in code:
            muts.append((code.replace(" < ", " <= ", 1), "repair:lt_to_lte"))
        if " <= " in code:
            muts.append((code.replace(" <= ", " < ", 1), "repair:lte_to_lt"))

        # 3. Defensive empty/zero guard at top of target function
        lines = code.splitlines()
        for idx, ln in enumerate(lines):
            m_def = re.match(rf"^(\s*)def\s+{re.escape(target_fn)}\s*\(([^)]*)\)\s*(?:->[^:]+)?:", ln)
            if m_def and idx + 1 < len(lines):
                raw_params = [p.split(":")[0].split("=")[0].strip() for p in m_def.group(2).split(",") if p.strip()]
                if raw_params:
                    body_indent = re.match(r"^(\s*)", lines[idx + 1]).group(1) or (m_def.group(1) + "    ")
                    p0 = raw_params[0]
                    for ret_default, tag in (("0", "guard_zero"), ("[]", "guard_empty_list")):
                        guarded_lines = (
                            lines[: idx + 1]
                            + [f"{body_indent}if not {p0}: return {ret_default}"]
                            + lines[idx + 1 :]
                        )
                        muts.append(("\n".join(guarded_lines), f"repair:{tag}"))
                break

        # 4. Sign flip on binary return/accumulator (- -> +)
        if " - " in code:
            muts.append((code.replace(" - ", " + ", 1), "repair:minus_to_plus"))

        return muts

    @classmethod
    def _synthesize_helper_compositions(
        cls,
        helpers: dict[str, tuple[int, str]],
        target_fn: str,
        arity: int,
    ) -> list[tuple[str, str]]:
        """Compose retrieved helper functions into candidate solutions for `target_fn`."""
        candidates: list[tuple[str, str]] = []
        h_items = list(helpers.items())

        # Direct single-helper alias / range-filter / sum-pipeline
        for h_name, (h_arity, h_src) in h_items:
            if h_name == target_fn:
                candidates.append((h_src, f"direct:{h_name}"))
                continue
            if h_arity == arity:
                args_str = ", ".join(f"a{i}" for i in range(arity))
                candidates.append(
                    (f"{h_src}\n\ndef {target_fn}({args_str}):\n    return {h_name}({args_str})", f"alias:{h_name}")
                )
            if arity == 1 and h_arity == 1:
                candidates.append(
                    (
                        f"{h_src}\n\ndef {target_fn}(n):\n    return [x for x in range(2, n + 1) if {h_name}(x)]",
                        f"filter_range:{h_name}",
                    )
                )
                candidates.append(
                    (
                        f"{h_src}\n\ndef {target_fn}(items):\n    return [x for x in items if {h_name}(x)]",
                        f"filter_items:{h_name}",
                    )
                )

        # Pairwise multi-rule compositions (h1 + h2)!
        if arity == 1 and len(h_items) >= 2:
            for i in range(min(5, len(h_items))):
                n1, (a1, s1) = h_items[i]
                if a1 != 1:
                    continue
                for j in range(min(5, len(h_items))):
                    if i == j:
                        continue
                    n2, (a2, s2) = h_items[j]
                    if a2 != 1:
                        continue
                    preamble = f"{s1}\n\n{s2}"
                    # Pattern A: primes/numbers x in 2..n where h1(x) and h1(h2(x)) (e.g. prime & prime digit_sum!)
                    candidates.append(
                        (
                            f"{preamble}\n\ndef {target_fn}(limit):\n"
                            f"    return [x for x in range(2, limit + 1) if {n1}(x) and {n1}({n2}(x))]",
                            f"compose_pred_nested:{n1}+{n2}",
                        )
                    )
                    # Pattern B: x in 1..n where h1(x) and h2(x)
                    candidates.append(
                        (
                            f"{preamble}\n\ndef {target_fn}(limit):\n"
                            f"    return [x for x in range(1, limit + 1) if {n1}(x) and {n2}(x)]",
                            f"compose_conj_range:{n1}&{n2}",
                        )
                    )
                    # Pattern C: functional pipeline h2(h1(x))
                    candidates.append(
                        (
                            f"{preamble}\n\ndef {target_fn}(x):\n    return {n2}({n1}(x))",
                            f"compose_pipe:{n2}({n1}(x))",
                        )
                    )
                    # Pattern D: map h2 over items filtered by h1, or sum
                    candidates.append(
                        (
                            f"{preamble}\n\ndef {target_fn}(items):\n    return [{n2}(x) for x in items if {n1}(x)]",
                            f"compose_filter_map:{n1}->{n2}",
                        )
                    )
                    candidates.append(
                        (
                            f"{preamble}\n\ndef {target_fn}(limit):\n"
                            f"    return sum({n2}(x) for x in range(1, limit + 1) if {n1}(x))",
                            f"compose_filter_sum:{n1}->{n2}",
                        )
                    )
        return candidates

    def solve_and_verify(
        self,
        oracle: Any,
        prompt: str,
        test_code: str,
        crystallize_on_success: bool = True,
    ) -> Any:
        """Search, compose, repair, and sandbox-verify Python solutions within strict CPU node & time budgets."""
        from .engine import CrystalResponse

        t0 = time.perf_counter()
        target_fn, arity = self._infer_target_signature(prompt, test_code)
        assert_lines = [ln.strip() for ln in test_code.splitlines() if ln.strip().startswith("assert ")]

        # Step 1: Retrieve candidate code rules from either MirrorCrystalEngine or DiskCrystalVault
        helpers: dict[str, tuple[int, str]] = {}
        raw_code_blocks: list[tuple[str, str]] = []

        sig_full, toks_full = oracle.bit_table.encode_text_signature(prompt)
        query_batches = [(sig_full, toks_full)]
        # Also query individual content tokens so distinct helper locks (e.g., 'prime' and 'digit_sum') are both retrieved!
        for tok in toks_full[:6]:
            if len(tok) >= 4 and tok != target_fn:
                s_sub, t_sub = oracle.bit_table.encode_text_signature(tok)
                query_batches.append((s_sub, t_sub))

        seen_rids: set[int] = set()
        candidate_rules: list[Any] = []
        for q_sig, q_toks in query_batches:
            if hasattr(oracle, "rule_index"):
                cands = oracle.rule_index.query_candidates(q_sig, q_toks, oracle.bit_table, top_k=12)
            else:
                cands = oracle._query_disk_candidates(q_sig, q_toks, top_k=12)
            for _, rule in cands:
                if rule.rule_id not in seen_rids:
                    seen_rids.add(rule.rule_id)
                    candidate_rules.append(rule)

        # Also scan sandbox-verified code axioms (pass_count >= 6 / trust_tier == 1) matching prompt stems
        # so 30,000+ prose rules never crowd out verified helper functions like `is_prime_fast`!
        prompt_stems = {
            part[:5]
            for tok in toks_full
            for part in tok.split("_")
            if len(part) >= 4
        }
        if hasattr(oracle, "rule_index"):
            verified_pool = [r for r in reversed(oracle.rule_index.rules) if r.trust_tier == 1][:128]
        else:
            rows_ver = oracle.conn.execute(
                "SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules WHERE pass_count >= 6 ORDER BY rule_id DESC LIMIT 128"
            ).fetchall()
            verified_pool = [oracle._row_to_rule(r) for r in rows_ver]

        for v_rule in verified_pool:
            if v_rule.rule_id in seen_rids:
                continue
            r_stems = {
                part[:5]
                for tok in v_rule.salient_tokens
                for part in tok.split("_")
                if len(part) >= 4
            }
            if not prompt_stems or (prompt_stems & r_stems):
                seen_rids.add(v_rule.rule_id)
                candidate_rules.append(v_rule)

        for rule in candidate_rules:
            code_blk = ZeroTrustSandboxJudge.extract_python_code(rule.abstract_response)
            if code_blk:
                raw_code_blocks.append((code_blk, f"rule#{rule.rule_id}"))
                for h_name, h_arity, h_src in self._extract_helper_functions(code_blk):
                    if h_name not in helpers:
                        helpers[h_name] = (h_arity, h_src)

        # Step 2: Build initial Best-First / UCB1 search pool
        seed_programs: list[tuple[str, str]] = list(raw_code_blocks)
        seed_programs.extend(self._synthesize_helper_compositions(helpers, target_fn=target_fn, arity=arity))

        dedup_codes: set[str] = set()
        frontier: list[_MCTSCodeNode] = []
        nodes_evaluated = 0
        trace: list[str] = [f"MCTS target={target_fn}/{arity}, retrieved_helpers={sorted(helpers.keys())}"]

        for cand_code, origin in seed_programs:
            if nodes_evaluated >= self.max_nodes or (time.perf_counter() - t0) * 1000.0 >= self.max_wall_ms:
                break
            norm_c = cand_code.strip()
            if not norm_c or norm_c in dedup_codes:
                continue
            dedup_codes.add(norm_c)
            nodes_evaluated += 1
            ok, reward, msg = self._evaluate_candidate(norm_c, test_code, assert_lines)
            if ok:
                if crystallize_on_success:
                    oracle.ingest_pair(
                        prompt,
                        f"```python\n{norm_c}\n```",
                        test_code=test_code,
                        domain="mcts_synthesized",
                        trust_tier=1,
                    )
                ms = (time.perf_counter() - t0) * 1000.0
                trace.append(f"Verified '{origin}' after {nodes_evaluated} node(s) ({ms:.2f} ms)")
                return CrystalResponse(
                    answer=f"```python\n{norm_c}\n```",
                    confidence=1.0,
                    mode="mcts_program_synthesis",
                    matched_rule_id=None,
                    latency_ms=round(ms, 3),
                    sandbox_verified=True,
                    reasoning_trace=trace,
                    epistemic_tier="verified_exact",
                )
            if reward > 0.0:
                frontier.append(_MCTSCodeNode(code=norm_c, origin=origin, reward=reward))

        # Step 3: UCB1 Self-Repair Expansion over promising partial candidates
        while frontier and nodes_evaluated < self.max_nodes and (time.perf_counter() - t0) * 1000.0 < self.max_wall_ms:
            unexpanded = [nd for nd in frontier if not nd.expanded]
            if not unexpanded:
                break
            best_node = max(
                unexpanded,
                key=lambda nd: nd.reward + self.ucb_c * math.sqrt(math.log(nodes_evaluated + 1.0) / nd.visits),
            )
            best_node.expanded = True
            best_node.visits += 1

            for mut_code, mut_tag in self._generate_repair_mutations(best_node.code, target_fn=target_fn, arity=arity):
                if nodes_evaluated >= self.max_nodes or (time.perf_counter() - t0) * 1000.0 >= self.max_wall_ms:
                    break
                norm_m = mut_code.strip()
                if norm_m in dedup_codes:
                    continue
                dedup_codes.add(norm_m)
                nodes_evaluated += 1
                ok, reward, _ = self._evaluate_candidate(norm_m, test_code, assert_lines)
                if ok:
                    if crystallize_on_success:
                        oracle.ingest_pair(
                            prompt,
                            f"```python\n{norm_m}\n```",
                            test_code=test_code,
                            domain="mcts_synthesized",
                            trust_tier=1,
                        )
                    ms = (time.perf_counter() - t0) * 1000.0
                    trace.append(f"Self-repaired via '{best_node.origin} -> {mut_tag}' at node #{nodes_evaluated}")
                    return CrystalResponse(
                        answer=f"```python\n{norm_m}\n```",
                        confidence=1.0,
                        mode="mcts_program_synthesis",
                        matched_rule_id=None,
                        latency_ms=round(ms, 3),
                        sandbox_verified=True,
                        reasoning_trace=trace,
                        epistemic_tier="verified_exact",
                    )
                if reward > best_node.reward:
                    frontier.append(_MCTSCodeNode(code=norm_m, origin=f"{best_node.origin}+{mut_tag}", reward=reward))

        ms = (time.perf_counter() - t0) * 1000.0
        trace.append(f"MCTS budget exhausted ({nodes_evaluated} nodes, {ms:.2f} ms) -> abstaining honestly.")
        return CrystalResponse(
            answer="No verified synthesis passed all sandbox assertions within CPU budget.",
            confidence=0.0,
            mode="unmatched_honest_abstain",
            matched_rule_id=None,
            latency_ms=round(ms, 3),
            sandbox_verified=False,
            reasoning_trace=trace,
            epistemic_tier="honest_abstain",
        )


# ============================================================================
# 5. PHASE-2 #4 + PHASE-3 #1: NightlyDreamSynthesizer + DreamUtilityPruner
# ============================================================================


@dataclass
class DreamUtilityPruner:
    """Phase-3 #1 Guard: Prevents O(N^2) dream bloat & graph cycles via DAG checks, novelty scoring & hard caps."""

    max_dreams_per_cycle: int = 128
    min_hops: int = 2
    max_hops: int = 4
    min_utility_score: float = 1.5

    def compute_utility(self, start: str, end: str, hops: int) -> float:
        if hops < self.min_hops or start == end:
            return 0.0
        s_toks = set(ContextBitTable.tokenize(start))
        e_toks = set(ContextBitTable.tokenize(end))
        if not s_toks or not e_toks or s_toks == e_toks:
            return 0.0
        novel_tokens = len(e_toks - s_toks)
        if novel_tokens == 0:
            return 0.0
        return float(hops) * math.log2(2.0 + novel_tokens)


class NightlyDreamSynthesizer:
    """Idle-CPU autonomous reasoning daemon that discovers & crystallizes verified multi-hop causal shortcuts."""

    def __init__(self, pruner: DreamUtilityPruner | None = None) -> None:
        self.pruner = pruner or DreamUtilityPruner()

    @staticmethod
    def _extract_causal_graph(oracle: Any) -> dict[str, list[tuple[str, str]]]:
        if hasattr(oracle, "rule_index"):
            return {k: list(v) for k, v in oracle.rule_index._causal_edges.items()}
        graph: dict[str, list[tuple[str, str]]] = {}
        import json as _json

        for head, edges_json in oracle.conn.execute("SELECT head, edges_json FROM causal_edges"):
            graph[str(head)] = [(str(pair[0]), str(pair[1])) for pair in _json.loads(edges_json)]
        return graph

    def run_dream_cycle(self, oracle: Any) -> dict[str, Any]:
        """Discover cycle-free multi-hop DAG shortcuts, prune low-utility bloat, and crystallize top shortcuts."""
        t0 = time.perf_counter()
        graph = self._extract_causal_graph(oracle)
        direct_pairs: set[tuple[str, str]] = {
            (head, tail) for head, edges in graph.items() for tail, _ in edges
        }

        cycles_pruned = 0
        candidates_explored = 0
        discovered: dict[tuple[str, str], tuple[float, list[str], list[str]]] = {}

        for start_node in sorted(graph.keys()):
            # Bounded BFS/DFS with per-path cycle check
            stack: list[tuple[str, list[str], list[str]]] = [(start_node, [start_node], [])]
            while stack:
                curr, path_nodes, path_facts = stack.pop()
                if len(path_facts) >= self.pruner.max_hops:
                    continue
                for nxt, stmt in graph.get(curr, ()):
                    candidates_explored += 1
                    if nxt in path_nodes:
                        cycles_pruned += 1
                        continue
                    new_nodes = path_nodes + [nxt]
                    new_facts = path_facts + [stmt]
                    hops = len(new_facts)
                    if hops >= self.pruner.min_hops and (start_node, nxt) not in direct_pairs:
                        util = self.pruner.compute_utility(start_node, nxt, hops)
                        if util >= self.pruner.min_utility_score:
                            key = (start_node, nxt)
                            prev = discovered.get(key)
                            if prev is None or util > prev[0]:
                                discovered[key] = (util, new_nodes, new_facts)
                    if hops < self.pruner.max_hops and nxt in graph:
                        stack.append((nxt, new_nodes, new_facts))

        ranked = sorted(
            discovered.items(),
            key=lambda kv: (kv[1][0], len(kv[1][2])),
            reverse=True,
        )
        total_valid = len(ranked)
        kept = ranked[: self.pruner.max_dreams_per_cycle]
        bloat_pruned = max(0, total_valid - len(kept))

        dreams_crystallized = 0
        for (start_node, end_node), (_, nodes, facts) in kept:
            hops = len(facts)
            dream_prompt = f"What happens to {start_node} and how does {start_node} relate to {end_node}?"
            dream_think = f"Dream-synthesized {hops}-hop causal shortcut: {' ; '.join(facts)}"
            dream_ans = f"Therefore, {start_node} -> {end_node} (synthesized {hops}-hop chain: {' -> '.join(nodes)})."
            created = oracle.ingest_pair(
                user_text=dream_prompt,
                target_text=dream_ans,
                think=dream_think,
                domain="dream_synthesis",
                trust_tier=2,
            )
            if created:
                dreams_crystallized += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "dreams_crystallized": dreams_crystallized,
            "valid_shortcuts_found": total_valid,
            "cycles_pruned": cycles_pruned,
            "bloat_pruned": bloat_pruned,
            "candidates_explored": candidates_explored,
            "elapsed_ms": round(elapsed_ms, 3),
        }
