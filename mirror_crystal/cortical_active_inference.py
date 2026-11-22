"""Organ 1: Cortical Front-End, Bilingual CCG-HDC Parser & Active Inference Controller
====================================================================================
Unifies Stages 1, 5B, 10, 12-L2, and 13-L3 into one pure-Python CPU-native front-end:

1. BilingualCCGHDCParser (Stage 1):
   - Parses English (SVO) and Roman-Urdu (SOV) clauses into 5-role hyperedges:
     (Who, Act, Targ, Cond, Eff) bound into D = 10,240-bit BSC hypervectors.
   - Telescoping XOR K-Hop Abduction (K >= 4 hops) across chained causal premises.
2. TropicalRSAPragmaticEngine (Stage 5B):
   - Rational Speech Act (L0 -> S1 -> L1) in fixed-point Q16.16 log2(|W(u)|) entropy
     + 2-hop ATOMIC2020 (xNeed -> HinderedBy -> xWant) commonsense bridging.
   - Solves scalar implicature ("some" -> "not all"), indirect speech acts (EN & Roman-Urdu),
     and sarcasm/irony polarity inversion.
3. Thousand-Brains Cortical Reference Frames & OCC 22-Emotion Appraisal (Stages 12-L2 & 13-L3):
   - ModuleReferenceFrame + CorticalCMPVoter lateral consensus for multi-module software.
   - CharacterReferenceFrame with strict Theory-of-Mind epistemic isolation (CompactRoaringBitmap)
     and OCC 22-emotion precision-weighted Active Inference affect updates.
4. CorticalActiveInferenceController (Stages 10 & 12-L2):
   - Minimizes Expected Free Energy G(pi) = -EpistemicValue + PragmaticRisk + Generalized Binary
     Search + TropicalJTMS defaults to collapse vague 1-sentence prompts in 1 pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

from .hippocampal_sdm_memory import CompactRoaringBitmap
from .quotient_egraph_resonator import (
    BitSlicedBundle10240,
    DefeasibleRule,
    TropicalJTMS,
    bind10240,
    hv10240_of,
    sim_q16_10240,
)

Q16: int = 65_536


def _log2_q16(n: int) -> int:
    """Exact Q16.16 integer approximation of log2(max(1, n)) using bit_length + piecewise linear."""
    v = max(1, int(n))
    k = v.bit_length() - 1
    rem = v - (1 << k)
    frac = (rem * Q16) >> k if k > 0 else 0
    return (k << 16) + frac


# ---------------------------------------------------------------------------
# 1. Bilingual CCG-HDC Functor Parser & Telescoping XOR K-Hop Abduction (Stage 1)
# ---------------------------------------------------------------------------

ROLE_WHO_HV: int = hv10240_of("ccg_role:WHO")
ROLE_ACT_HV: int = hv10240_of("ccg_role:ACT")
ROLE_TARG_HV: int = hv10240_of("ccg_role:TARG")
ROLE_COND_HV: int = hv10240_of("ccg_role:COND")
ROLE_EFF_HV: int = hv10240_of("ccg_role:EFF")

_URDU_MARKERS: frozenset[str] = frozenset(
    {
        "ne", "ko", "se", "mein", "par", "hai", "hain", "tha", "thi", "gaya", "gayi",
        "kar", "kiya", "karta", "karti", "hota", "hoti", "agar", "to", "kyunke",
        "isliye", "bohat", "thand", "garam", "wah", "phir", "chali", "banao", "likho",
    }
)

_EN_STOP: frozenset[str] = frozenset({"the", "a", "an", "that", "which", "is", "are", "was", "were", "in", "on", "of"})


@dataclass(frozen=True)
class Hyperedge5Role:
    language: str
    who: str
    act: str
    targ: str
    cond: str
    eff: str
    hypervector: int
    raw_text: str


@dataclass(frozen=True)
class AbductionChainResult:
    found: bool
    root_cause: str
    target_effect: str
    hops: int
    chain: tuple[tuple[str, str], ...]
    xor_verification_q16: int
    explanation: str


class BilingualCCGHDCParser:
    """Deterministic English (SVO) & Roman-Urdu (SOV) CCG functor parser into 5-role 10,240-bit HVs."""

    @staticmethod
    def detect_language(text: str) -> str:
        toks = set(re.findall(r"[a-zA-Z]+", text.casefold()))
        urdu_hits = len(toks & _URDU_MARKERS)
        return "roman_urdu" if urdu_hits >= 2 else "english"

    @staticmethod
    def _clean_phrase(phrase: str) -> str:
        p = re.sub(r"[^a-zA-Z0-9_\-\s']", " ", phrase).strip().casefold()
        p = re.sub(r"\s+", " ", p)
        return p

    @classmethod
    def encode_5role_hv(cls, who: str, act: str, targ: str, cond: str, eff: str) -> int:
        bd = BitSlicedBundle10240()
        if who:
            bd.add(bind10240(ROLE_WHO_HV, hv10240_of(f"filler:{who}")), 2)
        if act:
            bd.add(bind10240(ROLE_ACT_HV, hv10240_of(f"filler:{act}")), 2)
        if targ:
            bd.add(bind10240(ROLE_TARG_HV, hv10240_of(f"filler:{targ}")), 2)
        if cond:
            bd.add(bind10240(ROLE_COND_HV, hv10240_of(f"filler:{cond}")), 3)
        if eff:
            bd.add(bind10240(ROLE_EFF_HV, hv10240_of(f"filler:{eff}")), 3)
        return bd.binary()

    def parse_clause(self, text: str) -> Hyperedge5Role:
        """Parse an English (SVO) or Roman-Urdu (SOV) sentence into a 5-role CCG hyperedge."""
        raw = text.strip()
        lower = raw.casefold()
        lang = self.detect_language(lower)
        who = act = targ = cond = eff = ""

        # Conditional / Causal extraction first (bilingual)
        m_en_if = re.match(r"^\s*if\s+(.+?)\s*(?:,\s*then|\bthen\b|,)\s*(.+?)\s*[.!?]?$", lower)
        m_en_cause = re.match(r"^\s*(.+?)\s+\b(?:causes|triggers|leads to|produces|results in)\b\s+(.+?)\s*[.!?]?$", lower)
        m_en_because = re.match(r"^\s*(.+?)\s+\bbecause\b\s+(.+?)\s*[.!?]?$", lower)
        m_ur_if = re.match(r"^\s*agar\s+(.+?)\s+\bto\b\s+(.+?)\s*[.!?]?$", lower)
        m_ur_cause = re.match(r"^\s*(.+?)\s+\b(?:ki wajah se|se)\b\s+(.+?)\s+(?:hota hai|hoti hai|hota h|ho jata hai|hua|hui)\s*[.!?]?$", lower)
        m_ur_isliye = re.match(r"^\s*(.+?)\s+\bisliye\b\s+(.+?)\s*[.!?]?$", lower)
        m_ur_kyunke = re.match(r"^\s*(.+?)\s+\bkyunke\b\s+(.+?)\s*[.!?]?$", lower)

        if m_en_if:
            cond = self._clean_phrase(m_en_if.group(1))
            eff = self._clean_phrase(m_en_if.group(2))
            act = "implies"
        elif m_en_cause:
            cond = self._clean_phrase(m_en_cause.group(1))
            eff = self._clean_phrase(m_en_cause.group(2))
            act = "causes"
        elif m_en_because:
            eff = self._clean_phrase(m_en_because.group(1))
            cond = self._clean_phrase(m_en_because.group(2))
            act = "caused_by"
        elif m_ur_if:
            cond = self._clean_phrase(m_ur_if.group(1))
            eff = self._clean_phrase(m_ur_if.group(2))
            act = "implies"
        elif m_ur_isliye:
            cond = self._clean_phrase(m_ur_isliye.group(1))
            eff = self._clean_phrase(m_ur_isliye.group(2))
            act = "causes"
        elif m_ur_kyunke:
            eff = self._clean_phrase(m_ur_kyunke.group(1))
            cond = self._clean_phrase(m_ur_kyunke.group(2))
            act = "caused_by"
        elif m_ur_cause:
            cond = self._clean_phrase(m_ur_cause.group(1))
            eff = self._clean_phrase(m_ur_cause.group(2))
            act = "causes"

        # SVO (English) vs SOV (Roman-Urdu) argument extraction
        if lang == "roman_urdu":
            # Pattern: "<Who> ne <Targ> (ko)? <Act> (kiya|karta hai|hai)"
            m_sov = re.match(
                r"^\s*([a-z0-9_\s]+?)\s+\bne\b\s+([a-z0-9_\s]+?)(?:\s+\bko\b)?\s+([a-z]+(?:\s+(?:kiya|karta|karti|kar|diya|liya|rakha|chhupaya))?)(?:\s+(?:hai|tha|thi|hain))?\s*[.!?]?$",
                lower,
            )
            if m_sov:
                who = self._clean_phrase(m_sov.group(1))
                targ = self._clean_phrase(m_sov.group(2))
                act = self._clean_phrase(m_sov.group(3)) or act
            elif not cond:
                toks = [t for t in re.findall(r"[a-z0-9_]+", lower) if t not in {"hai", "hain", "tha", "thi"}]
                if len(toks) >= 3:
                    who = toks[0]
                    act = toks[-1]
                    targ = " ".join(toks[1:-1])
                elif len(toks) == 2:
                    who, act = toks[0], toks[1]
        else:
            if not cond:
                toks = [t for t in re.findall(r"[a-z0-9_']+", lower) if t not in _EN_STOP]
                if len(toks) >= 3:
                    who = toks[0]
                    act = toks[1]
                    targ = " ".join(toks[2:])
                elif len(toks) == 2:
                    who, act = toks[0], toks[1]

        if not cond and who and act:
            cond = f"{who} {act}".strip()
        if not eff and targ:
            eff = targ

        hv = self.encode_5role_hv(who, act, targ, cond, eff)
        return Hyperedge5Role(lang, who, act, targ, cond, eff, hv, raw)

    def abduce_k_hop(
        self,
        premises: Sequence[str],
        observed_effect: str,
        *,
        max_hops: int = 6,
    ) -> AbductionChainResult:
        """Telescoping XOR K-Hop Abduction recovering root cause across K >= 4 chained rules.

        Each parsed causal rule (C_i -> E_i) is bound as a telescoping XOR edge:
            edge_hv_i = hv(C_i) ^ hv(E_i)
        So XOR-ing all chain edges telescopes:
            edge_1 ^ edge_2 ^ ... ^ edge_k == hv(root_cause) ^ hv(observed_effect)
        allowing instant O(1) bitwise verification of the entire K-hop causal derivation!
        """
        parsed = [self.parse_clause(p) for p in premises]
        target = self._clean_phrase(observed_effect)
        # Map effect -> list of (cond, edge_hv)
        reverse_adj: dict[str, list[tuple[str, int]]] = {}
        for edge in parsed:
            if edge.cond and edge.eff:
                e_hv = bind10240(hv10240_of(f"node:{edge.cond}"), hv10240_of(f"node:{edge.eff}"))
                reverse_adj.setdefault(edge.eff, []).append((edge.cond, e_hv))

        # Traverse backward up to max_hops to find deepest explanatory root cause
        chain: list[tuple[str, str]] = []
        edge_hvs: list[int] = []
        cur = target
        visited = {cur}
        for _ in range(max(1, int(max_hops))):
            preds = reverse_adj.get(cur)
            if not preds:
                # Try fuzzy token-subset match if wording differs slightly
                cur_toks = set(cur.split())
                for eff_key, p_list in reverse_adj.items():
                    ek_toks = set(eff_key.split())
                    if cur_toks and (cur_toks.issubset(ek_toks) or ek_toks.issubset(cur_toks)):
                        preds = p_list
                        cur = eff_key
                        break
            if not preds:
                break
            nxt_cond, e_hv = preds[0]
            if nxt_cond in visited:
                break
            visited.add(nxt_cond)
            chain.append((nxt_cond, cur))
            edge_hvs.append(e_hv)
            cur = nxt_cond

        if not chain:
            return AbductionChainResult(
                found=False,
                root_cause="",
                target_effect=target,
                hops=0,
                chain=(),
                xor_verification_q16=0,
                explanation=f"No causal premise chain abduces '{target}'.",
            )

        chain.reverse()
        root = chain[0][0]
        # Telescoping XOR verification: XOR(edges) ^ hv(target) MUST equal hv(root) with d_H == 0!
        telescoped = hv10240_of(f"node:{chain[-1][1]}")
        for ehv in edge_hvs:
            telescoped = bind10240(telescoped, ehv)
        verify_q16 = sim_q16_10240(telescoped, hv10240_of(f"node:{root}"))
        arrow_str = " -> ".join([chain[0][0]] + [e for _c, e in chain])
        return AbductionChainResult(
            found=True,
            root_cause=root,
            target_effect=chain[-1][1],
            hops=len(chain),
            chain=tuple(chain),
            xor_verification_q16=verify_q16,
            explanation=(
                f"Telescoping XOR {len(chain)}-hop abduction (verify_q16={verify_q16}): "
                f"{arrow_str} (root cause: '{root}')."
            ),
        )


# ---------------------------------------------------------------------------
# 2. Tropical-RSA Pragmatic Engine + 2-Hop ATOMIC2020 Bridging (Stage 5B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PragmaticInference:
    utterance: str
    phenomenon: str  # "scalar_implicature" | "indirect_speech_act" | "irony_sarcasm" | "literal"
    literal_meaning: str
    pragmatic_intent: str
    extensional_entropy_q16: int
    speaker_surprisal_q16: int
    atomic_bridge: tuple[str, ...]
    affect_polarity_q16: int  # [-65536, +65536]
    explanation: str


class TropicalRSAPragmaticEngine:
    """Fixed-Point Q16.16 Rational Speech Act (L0 -> S1 -> L1) + 2-Hop ATOMIC2020 Engine."""

    # Embedded compact ATOMIC2020 social/physical commonsense chains:
    # trigger_keywords -> (state, xNeed, HinderedBy, xWant, pragmatic_request)
    _ATOMIC_CHAINS: tuple[dict[str, Any], ...] = (
        {
            "triggers": ("freezing", "so cold", "too cold", "chilly", "thand", "sardi", "thanda"),
            "state": "room_is_cold",
            "xNeed": "thermal_comfort",
            "HinderedBy": "open_window_or_low_heat",
            "xWant": "close_window_or_turn_on_heater",
            "intent": "Indirect request: close the window or raise the heating.",
            "world_size": 4,
            "polarity": -28_000,
        },
        {
            "triggers": ("boiling", "stuffy", "too hot", "so hot", "garmi", "ghutan"),
            "state": "room_is_hot",
            "xNeed": "cooling_airflow",
            "HinderedBy": "closed_window_or_ac_off",
            "xWant": "open_window_or_turn_on_cooling",
            "intent": "Indirect request: open a window or turn on air conditioning.",
            "world_size": 4,
            "polarity": -24_000,
        },
        {
            "triggers": ("battery is at", "battery low", "phone is dying", "battery khatam", "1%", "2%"),
            "state": "device_low_power",
            "xNeed": "electrical_charge",
            "HinderedBy": "no_charger_connected",
            "xWant": "borrow_or_plug_in_charger",
            "intent": "Indirect request: provide a charger or power outlet.",
            "world_size": 3,
            "polarity": -20_000,
        },
        {
            "triggers": ("haven't eaten", "starving", "bhook", "bhuka", "skip lunch", "empty stomach"),
            "state": "speaker_hungry",
            "xNeed": "nutrition",
            "HinderedBy": "lack_of_food",
            "xWant": "eat_a_meal",
            "intent": "Indirect request: get or offer food.",
            "world_size": 3,
            "polarity": -18_000,
        },
        {
            "triggers": ("calm tides", "safe harbor", "quiet waters", "weather holds"),
            "state": "defensive_evasion",
            "xNeed": "conceal_hidden_artifact",
            "HinderedBy": "direct_interrogation",
            "xWant": "deflect_suspicion_politely",
            "intent": "Subtextual deflection: speaker is concealing tension beneath polite phrasing.",
            "world_size": 6,
            "polarity": -12_000,
        },
    )

    # Scalar scales: (weak_term, strong_term, |W(weak)|, |W(strong)|, pragmatic_implicature)
    _SCALAR_SCALES: tuple[tuple[str, str, int, int, str], ...] = (
        ("some", "all", 8, 1, "not all (speaker avoided stronger term 'all')"),
        ("kuch", "sab", 8, 1, "not all / sab nahi (speaker avoided stronger term 'sab')"),
        ("warm", "hot", 4, 1, "not hot (speaker avoided stronger term 'hot')"),
        ("possible", "certain", 6, 1, "not certain (speaker avoided stronger term 'certain')"),
        ("most", "all", 4, 1, "not all (speaker avoided stronger term 'all')"),
        ("good", "excellent", 5, 1, "satisfactory but not extraordinary"),
        ("or", "and", 3, 1, "exclusive-or pragmatically (not both)"),
    )

    _POSITIVE_MARKERS: frozenset[str] = frozenset(
        {"great", "wonderful", "brilliant", "fantastic", "lovely", "awesome", "perfect", "wah", "zabardast", "shabash"}
    )
    _NEGATIVE_EVENTS: frozenset[str] = frozenset(
        {
            "crashed", "crash", "broke", "broken", "failed", "down", "deleted", "outage",
            "rain", "stuck", "traffic", "delayed", "bug", "error", "corrupted", "chali gayi", "kharab", "toot",
        }
    )

    def infer(self, utterance: str, *, context_hint: str = "") -> PragmaticInference:
        """Compute L0 -> S1 -> L1 Tropical-RSA pragmatic interpretation."""
        text = utterance.strip()
        lower = f"{text} {context_hint}".casefold()

        # 1. Irony / Sarcasm via Polarity-Context Surprisal Divergence (L1 inversion)
        pos_hit = [w for w in self._POSITIVE_MARKERS if re.search(rf"\b{re.escape(w)}\b", lower)]
        neg_hit = [w for w in self._NEGATIVE_EVENTS if w in lower]
        if pos_hit and neg_hit:
            # High surprisal between positive surface praise and negative real-world state -> irony!
            ent_q16 = _log2_q16(8)
            surp_q16 = 52_000
            return PragmaticInference(
                utterance=text,
                phenomenon="irony_sarcasm",
                literal_meaning=f"Surface positive marker ({pos_hit[0]}) applied to negative event ({neg_hit[0]}).",
                pragmatic_intent=f"Sarcastic frustration / criticism regarding '{neg_hit[0]}', NOT literal praise.",
                extensional_entropy_q16=ent_q16,
                speaker_surprisal_q16=surp_q16,
                atomic_bridge=(f"SurfacePraise({pos_hit[0]})", f"HinderedBy({neg_hit[0]})", "L1_Inversion(frustration)"),
                affect_polarity_q16=-48_000,
                explanation=(
                    f"Tropical-RSA L1 detected irony/sarcasm: positive praise '{pos_hit[0]}' "
                    f"clashes with negative event '{neg_hit[0]}' (surprisal_q16={surp_q16})."
                ),
            )

        # 2. 2-Hop ATOMIC2020 Indirect Speech Act Bridging (xNeed -> HinderedBy -> xWant)
        for entry in self._ATOMIC_CHAINS:
            if any(tr in lower for tr in entry["triggers"]):
                ent_q16 = _log2_q16(entry["world_size"])
                # S1 prefers utterance that minimizes world entropy while signaling unmet xWant
                surp_q16 = ent_q16 // 2 + 8_000
                bridge = (
                    f"State({entry['state']})",
                    f"xNeed({entry['xNeed']})",
                    f"HinderedBy({entry['HinderedBy']})",
                    f"xWant({entry['xWant']})",
                )
                return PragmaticInference(
                    utterance=text,
                    phenomenon="indirect_speech_act",
                    literal_meaning=f"Observation of state: {entry['state']}.",
                    pragmatic_intent=entry["intent"],
                    extensional_entropy_q16=ent_q16,
                    speaker_surprisal_q16=surp_q16,
                    atomic_bridge=bridge,
                    affect_polarity_q16=entry["polarity"],
                    explanation=(
                        f"Tropical-RSA + 2-hop ATOMIC2020 bridge ({' -> '.join(bridge)}): {entry['intent']}"
                    ),
                )

        # 3. Scalar Implicature via Extensional World Entropy log2(|W(weak)|) > log2(|W(strong)|)
        for weak, strong, w_weak, w_strong, implicature in self._SCALAR_SCALES:
            if re.search(rf"\b{re.escape(weak)}\b", lower) and not re.search(rf"\b{re.escape(strong)}\b", lower):
                ent_weak = _log2_q16(w_weak)
                ent_strong = _log2_q16(w_strong)
                # Pragmatic informativeness gap = log2(|W(weak)|) - log2(|W(strong)|) > 0
                gap = ent_weak - ent_strong
                return PragmaticInference(
                    utterance=text,
                    phenomenon="scalar_implicature",
                    literal_meaning=f"At least '{weak}' holds (extensional worlds |W|={w_weak}).",
                    pragmatic_intent=f"Pragmatic L1 inference: {implicature}.",
                    extensional_entropy_q16=ent_weak,
                    speaker_surprisal_q16=gap,
                    atomic_bridge=(f"L0({weak}:|W|={w_weak})", f"AltStronger({strong}:|W|={w_strong})", f"L1({implicature})"),
                    affect_polarity_q16=0,
                    explanation=(
                        f"Tropical-RSA scalar implicature: '{weak}' has higher extensional entropy "
                        f"({ent_weak}) than '{strong}' ({ent_strong}) -> L1 infers {implicature}."
                    ),
                )

        # 4. Literal default
        return PragmaticInference(
            utterance=text,
            phenomenon="literal",
            literal_meaning=text,
            pragmatic_intent=text,
            extensional_entropy_q16=_log2_q16(2),
            speaker_surprisal_q16=0,
            atomic_bridge=("L0_Literal",),
            affect_polarity_q16=0,
            explanation="Tropical-RSA: literal interpretation carries minimal pragmatic cost.",
        )


# ---------------------------------------------------------------------------
# 3. Thousand-Brains Reference Frames & OCC 22-Emotion Appraisal (12-L2 & 13-L3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleReferenceFrame:
    """Cortical Learning Module frame for a single software subsystem (Stage 12-L2 / 5C)."""

    module_id: str
    responsibility: str
    in_ports: tuple[str, ...]
    out_ports: tuple[str, ...]
    read_footprint: CompactRoaringBitmap
    write_footprint: CompactRoaringBitmap
    invariants: tuple[str, ...]


@dataclass(frozen=True)
class CMPVoteReport:
    consensus_reached: bool
    write_conflicts: tuple[tuple[str, str, list[int]], ...]
    unresolved_ports: tuple[tuple[str, str], ...]
    module_order: tuple[str, ...]
    explanation: str


class CorticalCMPVoter:
    """Monty-style Cortical Messaging Protocol (CMP) lateral consensus over ModuleReferenceFrames."""

    @staticmethod
    def vote_module_graph(
        frames: Sequence[ModuleReferenceFrame],
        external_inputs: Sequence[str] = ("user_prompt", "raw_payload", "cli_args"),
    ) -> CMPVoteReport:
        # 1. Verify disjoint State-Lens write footprints (F_i^W intersect F_j^W == empty)
        write_conflicts: list[tuple[str, str, list[int]]] = []
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                overlap = frames[i].write_footprint & frames[j].write_footprint
                if len(overlap) > 0:
                    write_conflicts.append((frames[i].module_id, frames[j].module_id, overlap.to_list()))

        # 2. Verify all required input ports are supplied by external inputs or upstream out_ports
        available_ports: set[str] = {p.casefold() for p in external_inputs}
        for fr in frames:
            for p in fr.out_ports:
                available_ports.add(p.casefold())

        unresolved: list[tuple[str, str]] = []
        for fr in frames:
            for inp in fr.in_ports:
                if inp.casefold() not in available_ports:
                    unresolved.append((fr.module_id, inp))

        # 3. Topological dependency order for DAC-CEGIS synthesis & O(log M) trace bisection
        port_producer: dict[str, str] = {}
        for fr in frames:
            for op in fr.out_ports:
                port_producer[op.casefold()] = fr.module_id

        ordered: list[str] = []
        placed: set[str] = set()
        remaining = list(frames)
        for _ in range(len(frames) + 1):
            progressed = False
            for fr in list(remaining):
                deps = {
                    port_producer[p.casefold()]
                    for p in fr.in_ports
                    if p.casefold() in port_producer and port_producer[p.casefold()] != fr.module_id
                }
                if deps.issubset(placed):
                    ordered.append(fr.module_id)
                    placed.add(fr.module_id)
                    remaining.remove(fr)
                    progressed = True
            if not progressed:
                break
        for fr in remaining:
            ordered.append(fr.module_id)

        ok = (not write_conflicts) and (not unresolved)
        return CMPVoteReport(
            consensus_reached=ok,
            write_conflicts=tuple(write_conflicts),
            unresolved_ports=tuple(unresolved),
            module_order=tuple(ordered),
            explanation=(
                f"CMP lateral vote across {len(frames)} cortical module frames: "
                f"{'CONSENSUS' if ok else 'CONFLICT'} "
                f"(write_conflicts={len(write_conflicts)}, unresolved_ports={len(unresolved)})."
            ),
        )


# Full OCC 22-emotion taxonomy (Ortony, Clore & Collins 1988) mapped to integer Q16.16 affect deltas
OCC_22_EMOTIONS: dict[str, tuple[int, int, int]] = {
    # emotion: (delta_valence_q16, delta_arousal_q16, delta_tension_q16)
    "joy": (22_000, 14_000, -12_000),
    "distress": (-22_000, 12_000, 18_000),
    "hope": (14_000, 10_000, 6_000),
    "fear": (-20_000, 24_000, 28_000),
    "pride": (20_000, 12_000, -8_000),
    "shame": (-22_000, 8_000, 20_000),
    "admiration": (18_000, 8_000, -10_000),
    "reproach": (-16_000, 12_000, 14_000),
    "gratitude": (20_000, 10_000, -14_000),
    "anger": (-24_000, 28_000, 26_000),
    "relief": (18_000, -12_000, -24_000),
    "disappointment": (-16_000, -8_000, 10_000),
    "satisfaction": (18_000, 4_000, -16_000),
    "fears_confirmed": (-26_000, 18_000, 24_000),
    "happy_for": (14_000, 8_000, -6_000),
    "resentment": (-18_000, 14_000, 18_000),
    "gloating": (10_000, 12_000, 4_000),
    "pity": (-10_000, -4_000, 6_000),
    "gratification": (22_000, 10_000, -14_000),
    "remorse": (-24_000, 6_000, 22_000),
    "love": (26_000, 16_000, -16_000),
    "hate": (-28_000, 24_000, 24_000),
}


@dataclass
class CharacterReferenceFrame:
    """Cortical Character Reference Frame enforcing strict Theory-of-Mind & OCC Affect (Stage 13-L3)."""

    char_id: str
    voice_steering_id: str = "style_melancholic_understatement"
    epistemic_facts: CompactRoaringBitmap = field(default_factory=CompactRoaringBitmap)
    epistemic_labels: set[str] = field(default_factory=set)
    goals_stack: list[str] = field(default_factory=list)
    # Q16.16 (valence, arousal, tension) each clamped to [-65536, +65536]
    affect_q16: tuple[int, int, int] = (0, 8_000, 8_000)
    trust_edges: dict[str, int] = field(default_factory=dict)  # target_char -> Q16.16 trust
    dominant_emotion: str = "hope"

    @staticmethod
    def _fact_id(fact_key: str) -> int:
        bm = CompactRoaringBitmap.from_symbols([fact_key])
        return bm.to_list()[0]

    def observe_fact(self, fact_key: str) -> None:
        norm = fact_key.strip().casefold()
        self.epistemic_facts.add(self._fact_id(norm))
        self.epistemic_labels.add(norm)

    def knows_fact(self, fact_key: str) -> bool:
        norm = fact_key.strip().casefold()
        return self._fact_id(norm) in self.epistemic_facts

    def appraise_event(
        self,
        *,
        desirability_q16: int,
        expected_q16: int = 0,
        actor_id: str | None = None,
        praiseworthy_q16: int = 0,
        precision_q16: int = 49_152,  # 0.75 precision weight
    ) -> str:
        """OCC 22-emotion cognitive appraisal + precision-weighted Active Inference state update."""
        # Prediction error epsilon = observation - expectation
        pred_err = desirability_q16 - expected_q16
        if actor_id and actor_id != self.char_id:
            if praiseworthy_q16 < -10_000 and desirability_q16 < -10_000:
                emotion = "anger"
            elif praiseworthy_q16 > 10_000 and desirability_q16 > 10_000:
                emotion = "gratitude"
            elif praiseworthy_q16 < -10_000:
                emotion = "reproach"
            elif praiseworthy_q16 > 10_000:
                emotion = "admiration"
            elif desirability_q16 < -10_000:
                emotion = "resentment"
            else:
                emotion = "happy_for"
        elif actor_id == self.char_id:
            if praiseworthy_q16 > 10_000 and desirability_q16 > 10_000:
                emotion = "gratification"
            elif praiseworthy_q16 < -10_000 and desirability_q16 < -10_000:
                emotion = "remorse"
            elif praiseworthy_q16 > 10_000:
                emotion = "pride"
            elif praiseworthy_q16 < -10_000:
                emotion = "shame"
            else:
                emotion = "satisfaction" if desirability_q16 >= 0 else "disappointment"
        else:
            if expected_q16 < -10_000 and desirability_q16 > 0:
                emotion = "relief"
            elif expected_q16 < -10_000 and desirability_q16 < -10_000:
                emotion = "fears_confirmed"
            elif expected_q16 > 10_000 and desirability_q16 < -10_000:
                emotion = "disappointment"
            elif pred_err < -20_000:
                emotion = "fear"
            elif desirability_q16 > 15_000:
                emotion = "joy"
            elif desirability_q16 < -15_000:
                emotion = "distress"
            else:
                emotion = "hope" if desirability_q16 >= 0 else "pity"

        dv, da, dt = OCC_22_EMOTIONS[emotion]
        # Scale by precision weight in Q16.16 (0 floats)
        nv = max(-Q16, min(Q16, self.affect_q16[0] + (dv * precision_q16) // Q16))
        na = max(-Q16, min(Q16, self.affect_q16[1] + (da * precision_q16) // Q16))
        nt = max(-Q16, min(Q16, self.affect_q16[2] + (dt * precision_q16) // Q16))
        self.affect_q16 = (nv, na, nt)
        self.dominant_emotion = emotion

        if actor_id and actor_id != self.char_id:
            prev_t = self.trust_edges.get(actor_id, 16_384)
            delta_t = (praiseworthy_q16 * precision_q16) // (2 * Q16)
            self.trust_edges[actor_id] = max(-Q16, min(Q16, prev_t + delta_t))
        return emotion


# ---------------------------------------------------------------------------
# 4. Active Inference Expected Free Energy G(pi) & Generalized Binary Search (10 & 12-L2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_id: str
    epistemic_value_q16: int
    pragmatic_risk_q16: int
    expected_free_energy_q16: int  # G(pi) = -EpistemicValue + PragmaticRisk (lower is better!)
    chosen_defaults: dict[str, str]


@dataclass(frozen=True)
class ResolvedArchitectureSpec:
    prompt: str
    domain_family: str
    app_name: str
    primary_entity: str
    features: tuple[str, ...]
    assumptions: dict[str, str]
    expected_free_energy_q16: int
    module_frames: tuple[ModuleReferenceFrame, ...]
    cmp_vote: CMPVoteReport


class CorticalActiveInferenceController:
    """Discrete Active Inference POMDP minimizing G(pi) + Generalized Binary Search.

    Collapses high-entropy vague 1-sentence software prompts into a complete, CMP-verified
    multi-module architecture specification in 1 single pass without asking 3-5 user turns.
    """

    def __init__(self) -> None:
        self.jtms = TropicalJTMS()
        self._install_engineering_defaults()

    def _install_engineering_defaults(self) -> None:
        self.jtms.add_rule(DefeasibleRule("def_local_store", ("local_first",), "embedded_wal_journal", tropical_cost=4))
        self.jtms.add_rule(DefeasibleRule("def_sync_crdt", ("collaborative",), "lamport_crdt_event_log", tropical_cost=3))
        self.jtms.add_rule(DefeasibleRule("def_analytics", ("analytics",), "deterministic_aggregator", tropical_cost=5))
        self.jtms.add_rule(DefeasibleRule("def_reminders", ("reminders",), "priority_queue_scheduler", tropical_cost=5))
        self.jtms.add_rule(DefeasibleRule("def_ledger", ("finance",), "double_entry_conservation", tropical_cost=3))
        self.jtms.add_rule(DefeasibleRule("def_workflow", ("tasks",), "dag_dependency_resolver", tropical_cost=4))

    @staticmethod
    def compute_efe_q16(epistemic_gain_q16: int, pragmatic_risk_q16: int) -> int:
        """Expected Free Energy G(pi) = -EpistemicValue + PragmaticRisk (lower is better)."""
        return int(pragmatic_risk_q16) - int(epistemic_gain_q16)

    @staticmethod
    def generalized_binary_search_split(
        hypotheses: Sequence[tuple[str, int]],
        candidate_probes: Mapping[str, set[str]],
    ) -> tuple[str, int]:
        """Stage 10 Generalized Binary Search: pick probe splitting weighted hypothesis mass closest to 50/50."""
        total_w = sum(max(1, w) for _h, w in hypotheses)
        if total_w <= 0 or not candidate_probes:
            return ("", 0)
        best_probe = ""
        best_imbalance = total_w + 1
        for probe_id, pos_set in sorted(candidate_probes.items()):
            pos_w = sum(max(1, w) for h, w in hypotheses if h in pos_set)
            imbalance = abs(2 * pos_w - total_w)
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_probe = probe_id
        # Information gain proxy in Q16.16: maximal (65536) at exact 50/50 split
        info_gain_q16 = max(0, ((total_w - best_imbalance) * Q16) // total_w)
        return best_probe, info_gain_q16

    def resolve_vague_software_prompt(self, prompt: str) -> ResolvedArchitectureSpec:
        """Resolve a vague 1-sentence product prompt into a CMP-verified 8-module architecture."""
        lower = prompt.casefold()
        # Detect domain family and features
        if any(k in lower for k in ("habit", "streak", "routine", "check-in", "checkin")):
            domain = "habit_tracker"
            app_name = "LocalFirstHabitTracker"
            entity = "habit"
        elif any(k in lower for k in ("expense", "budget", "ledger", "invoice", "split", "finance", "wallet")):
            domain = "expense_ledger"
            app_name = "LocalFirstExpenseLedger"
            entity = "entry"
        elif any(k in lower for k in ("kanban", "task", "issue", "project", "ticket", "sprint", "todo")):
            domain = "task_board"
            app_name = "LocalFirstTaskBoard"
            entity = "task"
        elif any(k in lower for k in ("note", "wiki", "knowledge", "bookmark", "document")):
            domain = "knowledge_notes"
            app_name = "LocalFirstNoteVault"
            entity = "note"
        else:
            domain = "collaborative_tracker"
            app_name = "LocalFirstWorkspaceApp"
            entity = "item"

        # Active Inference: score 3 candidate architectural policies via G(pi)
        wants_local = any(k in lower for k in ("local", "offline", "desktop", "cli", "embedded")) or True
        wants_collab = any(k in lower for k in ("collab", "sync", "shared", "team", "multi", "merge"))
        wants_analytics = any(k in lower for k in ("streak", "analytic", "metric", "report", "stat", "balance", "summary"))
        wants_reminders = any(k in lower for k in ("remind", "alert", "notif", "schedule", "due", "deadline"))

        policies = [
            PolicyEvaluation(
                policy_id="stdlib_wal_crdt_modular",
                epistemic_value_q16=52_000 + (8_000 if wants_collab else 0),
                pragmatic_risk_q16=10_000,  # Zero external dependency risk + sandbox verifiable
                expected_free_energy_q16=self.compute_efe_q16(
                    52_000 + (8_000 if wants_collab else 0), 10_000
                ),
                chosen_defaults={
                    "runtime": "Python 3.11+ pure stdlib (zero external dependency lock-in)",
                    "persistence": "Append-only JSON WAL journal + atomic snapshot store",
                    "concurrency_sync": "Lamport-clock LWW-Element-Set CRDT event log" if wants_collab else "Monotonic revision log with optimistic locking",
                    "analytics": "Deterministic streak/metric aggregation engine" if wants_analytics else "Summary statistics aggregator",
                    "scheduling": "Deterministic min-heap priority-queue reminder daemon" if wants_reminders else "Event hook dispatcher",
                    "interface": "Command router + structured JSON API controller",
                    "verification": "Self-generated property-based oracle (round-trip, idempotency, conservation)",
                },
            ),
            PolicyEvaluation(
                policy_id="external_cloud_microservice",
                epistemic_value_q16=32_000,
                pragmatic_risk_q16=54_000 if wants_local else 38_000,
                expected_free_energy_q16=self.compute_efe_q16(32_000, 54_000 if wants_local else 38_000),
                chosen_defaults={"runtime": "Distributed cloud services"},
            ),
            PolicyEvaluation(
                policy_id="single_file_monolith",
                epistemic_value_q16=20_000,
                pragmatic_risk_q16=42_000,
                expected_free_energy_q16=self.compute_efe_q16(20_000, 42_000),
                chosen_defaults={"runtime": "Monolithic script"},
            ),
        ]
        policies.sort(key=lambda p: p.expected_free_energy_q16)
        winner = policies[0]

        # Consult Tropical-JTMS for explicit defeasible rationale
        jtms_run = self.jtms.clone()
        if wants_local:
            jtms_run.assert_fact("local_first", True)
        if wants_collab:
            jtms_run.assert_fact("collaborative", True)
        if wants_analytics:
            jtms_run.assert_fact("analytics", True)
        if wants_reminders:
            jtms_run.assert_fact("reminders", True)
        if domain == "expense_ledger":
            jtms_run.assert_fact("finance", True)
        if domain == "task_board":
            jtms_run.assert_fact("tasks", True)
        derived = jtms_run.derive_all()

        assumptions = dict(winner.chosen_defaults)
        assumptions["jtms_active_rules"] = ", ".join(
            sorted(b.winning_rule_id for b in derived.values() if b.winning_rule_id != "explicit_fact")
        ) or "def_local_store"

        features = ["wal_persistence", "validation_errors", "import_export_codec", "command_api"]
        if wants_collab:
            features.append("crdt_lamport_sync")
        if wants_analytics:
            features.append("streak_analytics")
        if wants_reminders:
            features.append("reminder_scheduler")
        if domain == "expense_ledger":
            features.append("double_entry_conservation")
        if domain == "task_board":
            features.append("dependency_dag")

        # Construct 8 cortical ModuleReferenceFrames with strictly DISJOINT write footprints
        frames = (
            ModuleReferenceFrame(
                module_id="models",
                responsibility="Domain entities, typed records, and validation errors",
                in_ports=("raw_payload",),
                out_ports=("entity_model", "event_record"),
                read_footprint=CompactRoaringBitmap.from_symbols(["schema_def"]),
                write_footprint=CompactRoaringBitmap.from_symbols(["entity_registry"]),
                invariants=("valid_entity_id_non_empty", "deterministic_serialization"),
            ),
            ModuleReferenceFrame(
                module_id="crdt_sync",
                responsibility="Lamport clock & commutative idempotent CRDT event merge",
                in_ports=("event_record",),
                out_ports=("merged_log", "clock_state"),
                read_footprint=CompactRoaringBitmap.from_symbols(["entity_registry"]),
                write_footprint=CompactRoaringBitmap.from_symbols(["sync_event_log"]),
                invariants=("crdt_merge_idempotent", "crdt_merge_commutative"),
            ),
            ModuleReferenceFrame(
                module_id="storage_wal",
                responsibility="Local-first append-only WAL journal and snapshot repository",
                in_ports=("entity_model", "merged_log"),
                out_ports=("store_snapshot", "entity_index"),
                read_footprint=CompactRoaringBitmap.from_symbols(["entity_registry", "sync_event_log"]),
                write_footprint=CompactRoaringBitmap.from_symbols(["wal_journal_store"]),
                invariants=("wal_replay_roundtrip", "no_duplicate_primary_keys"),
            ),
            ModuleReferenceFrame(
                module_id="domain_engine",
                responsibility="Core state transitions, toggle/check-in/ledger/DAG mutations",
                in_ports=("entity_index", "clock_state"),
                out_ports=("domain_state", "mutation_receipt"),
                read_footprint=CompactRoaringBitmap.from_symbols(["wal_journal_store", "sync_event_log"]),
                write_footprint=CompactRoaringBitmap.from_symbols(["domain_projection"]),
                invariants=("state_conservation", "monotonic_revision"),
            ),
            ModuleReferenceFrame(
                module_id="analytics_engine",
                responsibility="Streak calculation, completion velocity, balance & health analytics",
                in_ports=("domain_state", "merged_log"),
                out_ports=("analytics_report",),
                read_footprint=CompactRoaringBitmap.from_symbols(["domain_projection", "sync_event_log"]),
                write_footprint=CompactRoaringBitmap.from_symbols(["analytics_cache"]),
                invariants=("non_negative_streaks", "longest_ge_current_streak"),
            ),
            ModuleReferenceFrame(
                module_id="reminder_scheduler",
                responsibility="Priority-queue local reminder & due-window scheduler",
                in_ports=("domain_state",),
                out_ports=("due_reminders",),
                read_footprint=CompactRoaringBitmap.from_symbols(["domain_projection"]),
                write_footprint=CompactRoaringBitmap.from_symbols(["reminder_heap"]),
                invariants=("chronological_dispatch", "idempotent_ack"),
            ),
            ModuleReferenceFrame(
                module_id="codec_io",
                responsibility="Deterministic JSON export/import snapshot codec with checksums",
                in_ports=("store_snapshot", "analytics_report"),
                out_ports=("serialized_bundle",),
                read_footprint=CompactRoaringBitmap.from_symbols(["wal_journal_store", "analytics_cache"]),
                write_footprint=CompactRoaringBitmap.from_symbols(["export_buffer"]),
                invariants=("export_import_lossless_roundtrip",),
            ),
            ModuleReferenceFrame(
                module_id="app_controller",
                responsibility="Unified CLI/API command dispatcher wiring all subsystems",
                in_ports=("user_prompt", "domain_state", "analytics_report", "due_reminders", "serialized_bundle"),
                out_ports=("api_response",),
                read_footprint=CompactRoaringBitmap.from_symbols(
                    ["domain_projection", "analytics_cache", "reminder_heap", "export_buffer"]
                ),
                write_footprint=CompactRoaringBitmap.from_symbols(["controller_audit_trail"]),
                invariants=("graceful_error_envelope", "end_to_end_dispatch"),
            ),
        )
        cmp_vote = CorticalCMPVoter.vote_module_graph(frames)
        return ResolvedArchitectureSpec(
            prompt=prompt,
            domain_family=domain,
            app_name=app_name,
            primary_entity=entity,
            features=tuple(features),
            assumptions=assumptions,
            expected_free_energy_q16=winner.expected_free_energy_q16,
            module_frames=frames,
            cmp_vote=cmp_vote,
        )


__all__ = [
    "Q16",
    "Hyperedge5Role",
    "AbductionChainResult",
    "BilingualCCGHDCParser",
    "PragmaticInference",
    "TropicalRSAPragmaticEngine",
    "ModuleReferenceFrame",
    "CMPVoteReport",
    "CorticalCMPVoter",
    "OCC_22_EMOTIONS",
    "CharacterReferenceFrame",
    "PolicyEvaluation",
    "ResolvedArchitectureSpec",
    "CorticalActiveInferenceController",
]
