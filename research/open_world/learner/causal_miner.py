"""Stage S11.1 Stage 5: Schema-Free Causal Hypothesis Discovery & Counterfactual Consistency.

STRICT ISOLATION & ZERO MANUAL DOMAIN SCHEMAS:
- Pure Python 3.11 stdlib + Mirror 7's 10,240-bit BSC & ``TwinEGraphDoCalculus``
  (``mirror_crystal.quotient_egraph_resonator``).
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
- Mines canonical directed causal event transitions ``(CAUSE_FRAME -> EFFECT_FRAME)`` directly
  from raw UTF-8 text (explicit discourse causal connectives, chronological sentence succession,
  and directional training pairs) filtered by contrastive counterfactual causal lift:
      CausalLift(A -> B) = log((P(B|A)+a)/(P(B|~A)+a)) - log((P(A|B)+a)/(P(A|~B)+a))
- Every mined hypothesis preserves support/counterexample provenance, explicit
  ``epistemic_status = "HYPOTHESIS_UNVERIFIED"`` until multi-cluster verification, and exact
  Wilson 95% credible intervals ``(ci_low, ci_high)``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from mirror_crystal.quotient_egraph_resonator import (
    TwinEGraphDoCalculus,
    bind10240,
    hamming_sim10240,
    hv10240_of,
    permute10240,
)
from research.open_world.contracts import (
    SolverFrozenError,
    canonical_json_bytes,
    split_sentences_with_offsets,
    tokenize,
)
from research.open_world.learner.extraction import (
    NEGATION_TOKENS,
    morph_stem,
    stemmed_content_tokens,
)

# Purely grammatical English discourse connectives (NOT domain action lists or event schemas)
_FWD_CAUSAL_MARKERS: tuple[str, ...] = (
    " therefore ", " thus ", " hence ", " consequently ", " as a result ",
    " which caused ", " which led to ", " leading to ", " resulting in ",
    " caused ", " causes ", " led to ", " leads to ", " so ",
)
_REV_CAUSAL_MARKERS: tuple[str, ...] = (
    " because of ", " because ", " due to ", " owing to ",
    " caused by ", " resulting from ", " since ",
)
_INIT_CAUSAL_MARKERS: tuple[str, ...] = (
    "because ", "since ", "after ", "when ", "if ",
)
_CONTRAST_MARKERS: tuple[str, ...] = (
    " but ", " however ", " although ", " though ", " despite ", " instead ", " yet ",
)


def wilson_ci95(successes: int, total: int) -> tuple[float, float]:
    """Exact Wilson 95% score credible interval (pure Python stdlib, z = 1.96)."""
    if total <= 0:
        return (0.0, 1.0)
    z = 1.96
    p = max(0.0, min(1.0, successes / total))
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def encode_causal_frame_hv(antecedent: str, consequent: str) -> int:
    """Encode directed ``antecedent -> consequent`` causal frame in 10,240-bit BSC space."""
    a_hv = permute10240(hv10240_of(f"s11:causal:ant:{antecedent}"), 3)
    b_hv = permute10240(hv10240_of(f"s11:causal:cons:{consequent}"), 11)
    return bind10240(a_hv, b_hv)


@dataclass(frozen=True)
class CausalHypothesis:
    """Tentative directional causal hypothesis grounded in raw evidence & Wilson uncertainty."""

    hypothesis_id: str
    antecedent_frame: str
    consequent_frame: str
    causal_lift: float
    asymmetric_margin: float
    support_count: int
    counterexample_count: int
    support_evidence_ids: tuple[str, ...]
    counterexample_evidence_ids: tuple[str, ...]
    epistemic_status: str  # HYPOTHESIS_UNVERIFIED | VERIFIED_MULTI_CLUSTER
    ci_low: float
    ci_high: float
    frame_hv: int

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "id": self.hypothesis_id,
            "ant": self.antecedent_frame,
            "cons": self.consequent_frame,
            "lift": round(self.causal_lift, 4),
            "asym": round(self.asymmetric_margin, 4),
            "sup": self.support_count,
            "cex": self.counterexample_count,
            "status": self.epistemic_status,
            "ci": [self.ci_low, self.ci_high],
        }


class RawCausalMiner:
    """Schema-free multi-scale distributional & asymmetric causal hypothesis miner (Stage 5)."""

    def __init__(self, *, enable_asymmetric_causal_lift: bool = True) -> None:
        self.enable_asymmetric_causal_lift = bool(enable_asymmetric_causal_lift)
        self._frozen: bool = False
        self._post_freeze_writes: int = 0

        # Multi-scale raw-text inverted indexes (pure stdlib sets, fast & compact)
        self._doc_stems: list[tuple[str, ...]] = []
        self._doc_lens: list[int] = []
        self._stem_docs: dict[str, set[int]] = defaultdict(set)
        self._stem_sents: dict[str, set[int]] = defaultdict(set)
        self._sent_lens: list[int] = []
        self._num_sents: int = 0
        self._seen_doc_hashes: set[int] = set()

        # Canonical directed CAUSE -> EFFECT co-occurrence counts & provenance clusters
        self._dir_fwd: dict[str, Counter[str]] = defaultdict(Counter)
        self._dir_src_tot: Counter[str] = Counter()
        self._dir_dst_tot: Counter[str] = Counter()
        self._dir_total_events: int = 0
        self._dir_support_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._dir_contrast_sources: dict[tuple[str, str], set[str]] = defaultdict(set)

        # Supervised canonical CAUSE -> EFFECT positive vs counterfactual negative pairs
        self._sup_pos: dict[str, Counter[str]] = defaultdict(Counter)
        self._sup_neg: dict[str, Counter[str]] = defaultdict(Counter)
        self._sup_pos_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._sup_neg_sources: dict[tuple[str, str], set[str]] = defaultdict(set)

        # Materialized top verified/tentative CausalHypotheses & query-time caches
        self.hypotheses: list[CausalHypothesis] = []
        self._hyp_by_pair: dict[tuple[str, str], CausalHypothesis] = {}
        self._nb_cache: dict[str, dict[str, float]] = {}
        self._pair_cache: dict[tuple[str, str], float] = {}

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def post_freeze_writes(self) -> int:
        return self._post_freeze_writes

    def freeze(self) -> None:
        self._frozen = True

    def _require_unfrozen(self) -> None:
        if self._frozen:
            self._post_freeze_writes += 1
            raise SolverFrozenError("RawCausalMiner is frozen; post-freeze writes are forbidden")

    def observe_raw_document(self, text: str, *, source_id: str = "raw") -> None:
        """Ingest one raw unlabeled UTF-8 text document/turn into distributional & causal statistics."""
        self._require_unfrozen()
        if not text or len(text) < 6:
            return
        h = hash(text)
        if h in self._seen_doc_hashes:
            return
        self._seen_doc_hashes.add(h)

        uniq_doc = tuple(dict.fromkeys(stemmed_content_tokens(text)))
        if len(uniq_doc) < 2:
            return
        did = len(self._doc_stems)
        self._doc_stems.append(uniq_doc)
        self._doc_lens.append(len(uniq_doc))
        for w in uniq_doc:
            self._stem_docs[w].add(did)

        prev_sent: list[str] | None = None
        for _, _, sent in split_sentences_with_offsets(text):
            seq = list(dict.fromkeys(stemmed_content_tokens(sent)))
            if not seq:
                prev_sent = None
                continue
            if len(seq) >= 2:
                sid = self._num_sents
                self._num_sents += 1
                self._sent_lens.append(len(seq))
                for w in seq[:24]:
                    self._stem_sents[w].add(sid)

            # Unsupervised discourse-causal & contrast mining (zero regex backtracking)
            sl = " " + sent.casefold() + " "
            c_part: str | None = None
            e_part: str | None = None
            is_contrast = False

            for mk in _FWD_CAUSAL_MARKERS:
                pos = sl.find(mk)
                if pos > 6 and pos + len(mk) + 6 < len(sl):
                    c_part, e_part = sl[:pos], sl[pos + len(mk) :]
                    break
            if c_part is None:
                for mk in _REV_CAUSAL_MARKERS:
                    pos = sl.find(mk)
                    if pos > 6 and pos + len(mk) + 6 < len(sl):
                        e_part, c_part = sl[:pos], sl[pos + len(mk) :]
                        break
            if c_part is None:
                s_strip = sent.casefold().lstrip()
                for mk in _INIT_CAUSAL_MARKERS:
                    if s_strip.startswith(mk) and "," in s_strip:
                        comma = s_strip.find(",")
                        if comma > len(mk) + 4 and comma + 6 < len(s_strip):
                            c_part, e_part = s_strip[len(mk) : comma], s_strip[comma + 1 :]
                            break
            if c_part is None:
                for mk in _CONTRAST_MARKERS:
                    pos = sl.find(mk)
                    if pos > 6 and pos + len(mk) + 6 < len(sl):
                        c_part, e_part = sl[:pos], sl[pos + len(mk) :]
                        is_contrast = True
                        break

            if c_part is not None and e_part is not None:
                cs = list(dict.fromkeys(stemmed_content_tokens(c_part)))[:10]
                es = list(dict.fromkeys(stemmed_content_tokens(e_part)))[:10]
                if cs and es:
                    if is_contrast:
                        for a in cs[:5]:
                            for b in es[:5]:
                                if a != b and len(self._dir_contrast_sources[(a, b)]) < 12:
                                    self._dir_contrast_sources[(a, b)].add(source_id)
                    else:
                        w_inc = 3
                        self._dir_total_events += w_inc
                        for a in cs:
                            self._dir_src_tot[a] += w_inc
                            for b in es:
                                if a != b:
                                    self._dir_fwd[a][b] += w_inc
                                    if len(self._dir_support_sources[(a, b)]) < 12:
                                        self._dir_support_sources[(a, b)].add(source_id)
                        for b in es:
                            self._dir_dst_tot[b] += w_inc
            elif prev_sent is not None and len(prev_sent) <= 18 and len(seq) <= 18:
                # Narrative short-sentence chronological succession (skips long list/essay bullets)
                self._dir_total_events += 1
                for a in prev_sent[:4]:
                    self._dir_src_tot[a] += 1
                    for b in seq[:4]:
                        if a != b:
                            self._dir_fwd[a][b] += 1
                for b in seq[:4]:
                    self._dir_dst_tot[b] += 1

            prev_sent = seq

    def observe_causal_training_pair(
        self,
        *,
        premise: str,
        choices: Sequence[str],
        relation_direction: str,
        target_index: int,
        source_id: str,
    ) -> None:
        """Observe a training causal pair canonically oriented as (CAUSE -> EFFECT)."""
        self._require_unfrozen()
        if target_index < 0 or target_index >= len(choices):
            return
        # Ingest winning pair into raw distributional corpus
        self.observe_raw_document(f"{premise} {choices[target_index]}", source_id=source_id)

        p_st = list(dict.fromkeys(stemmed_content_tokens(premise)))
        c_sets = [list(dict.fromkeys(stemmed_content_tokens(ch))) for ch in choices]
        common_c = set(c_sets[0]) & set(c_sets[1]) if len(c_sets) >= 2 else set()

        for idx, c_st in enumerate(c_sets):
            diff_c = [w for w in c_st if w not in common_c] or c_st
            p_use = [w for w in p_st if w not in common_c] or p_st
            src, dst = (diff_c, p_use) if relation_direction == "cause" else (p_use, diff_c)
            is_pos = (idx == target_index)
            if is_pos:
                self._dir_total_events += 4
            for a in src:
                if is_pos:
                    self._dir_src_tot[a] += 4
                for b in dst:
                    if a == b:
                        continue
                    if is_pos:
                        self._sup_pos[a][b] += 1
                        self._dir_fwd[a][b] += 4
                        self._sup_pos_sources[(a, b)].add(source_id)
                        self._dir_support_sources[(a, b)].add(source_id)
                    else:
                        self._sup_neg[a][b] += 1
                        self._sup_neg_sources[(a, b)].add(source_id)
                        self._dir_contrast_sources[(a, b)].add(source_id)
            if is_pos:
                for b in dst:
                    self._dir_dst_tot[b] += 4

    def materialize_hypotheses(self, *, k_min_clusters: int = 3, max_hypotheses: int = 400) -> list[CausalHypothesis]:
        """Compile top candidate ``CausalHypothesis`` records with Wilson 95% CIs & multi-cluster status."""
        self._require_unfrozen()
        self._nb_cache.clear()
        self._pair_cache.clear()

        candidates: list[tuple[float, CausalHypothesis]] = []
        seen: set[tuple[str, str]] = set()

        # Consider all directed pairs with >= 2 support sources (or supervised evidence)
        pair_keys = set(self._dir_support_sources.keys()) | set(self._sup_pos_sources.keys())
        for a, b in pair_keys:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            sup_ids = tuple(sorted(self._dir_support_sources.get((a, b), set()) | self._sup_pos_sources.get((a, b), set())))
            if len(sup_ids) < 2:
                continue
            cex_ids = tuple(sorted(self._dir_contrast_sources.get((a, b), set()) | self._sup_neg_sources.get((a, b), set())))
            lift, asym = self.compute_asymmetric_causal_lift(a, b)
            if lift <= 0.05:
                continue
            sup_n = len(sup_ids)
            cex_n = len(cex_ids)
            ci_lo, ci_hi = wilson_ci95(sup_n, sup_n + cex_n)
            status = (
                "VERIFIED_MULTI_CLUSTER"
                if (sup_n >= k_min_clusters and ci_lo >= 0.45 and asym > 0.0)
                else "HYPOTHESIS_UNVERIFIED"
            )
            hyp = CausalHypothesis(
                hypothesis_id=f"chyp:{a}->{b}",
                antecedent_frame=a,
                consequent_frame=b,
                causal_lift=round(lift, 6),
                asymmetric_margin=round(asym, 6),
                support_count=sup_n,
                counterexample_count=cex_n,
                support_evidence_ids=sup_ids[:8],
                counterexample_evidence_ids=cex_ids[:8],
                epistemic_status=status,
                ci_low=ci_lo,
                ci_high=ci_hi,
                frame_hv=encode_causal_frame_hv(a, b),
            )
            score = lift * math.log(1.0 + sup_n) * (0.5 + ci_lo)
            candidates.append((score, hyp))

        candidates.sort(key=lambda item: (-item[0], item[1].hypothesis_id))
        self.hypotheses = [h for _, h in candidates[:max_hypotheses]]
        self._hyp_by_pair = {(h.antecedent_frame, h.consequent_frame): h for h in self.hypotheses}
        return self.hypotheses

    def _doc_cos(self, a: str, b: str) -> float:
        da = self._stem_docs.get(a)
        db = self._stem_docs.get(b)
        if not da or not db:
            return 0.0
        inter = da & db
        if not inter:
            return 0.0
        return sum(1.0 / math.sqrt(max(1, self._doc_lens[d])) for d in inter) / math.sqrt(len(da) * len(db))

    def _sent_cos(self, a: str, b: str) -> float:
        sa = self._stem_sents.get(a)
        sb = self._stem_sents.get(b)
        if not sa or not sb:
            return 0.0
        inter = sa & sb
        if not inter:
            return 0.0
        return sum(1.0 / math.sqrt(max(1, self._sent_lens[s])) for s in inter) / math.sqrt(len(sa) * len(sb))

    def _top_neighbors(self, w: str) -> dict[str, float]:
        cached = self._nb_cache.get(w)
        if cached is not None:
            return cached
        dw = self._stem_docs.get(w)
        n_doc = len(self._doc_stems)
        if not dw or len(dw) < 2 or n_doc < 4:
            self._nb_cache[w] = {}
            return {}
        cand: Counter[str] = Counter()
        dlist = sorted(dw)[:90]
        scale = len(dw) / len(dlist)
        for did in dlist:
            for nb in self._doc_stems[did]:
                if nb != w:
                    cand[nb] += 1
        scored: list[tuple[float, str]] = []
        lw = len(dw)
        for nb, c in cand.items():
            lnb = len(self._stem_docs[nb])
            if lnb < 2 or c < 2:
                continue
            pmi = math.log(((c * scale) * n_doc) / (lw * lnb))
            if pmi > 0.3:
                scored.append((pmi * (c / (c + 1.0)), nb))
        scored.sort(reverse=True)
        vec = {nb: v for v, nb in scored[:60]}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        for k in vec:
            vec[k] /= norm
        self._nb_cache[w] = vec
        return vec

    def _two_hop_cos(self, a: str, b: str) -> float:
        va = self._top_neighbors(a)
        vb = self._top_neighbors(b)
        if not va or not vb:
            return 0.0
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(v * vb[k] for k, v in va.items() if k in vb)

    def _sym_pair_sim(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        v = self._pair_cache.get(key)
        if v is None:
            v = self._doc_cos(a, b) + 1.2 * self._two_hop_cos(a, b)
            self._pair_cache[key] = v
        return v

    def compute_asymmetric_causal_lift(self, cause_stem: str, effect_stem: str) -> tuple[float, float]:
        """Exact contrastive causal lift ``CausalLift(A -> B)`` from docs/51 Stage 5."""
        if cause_stem == effect_stem:
            return (0.0, 0.0)
        fwd = self._dir_fwd[cause_stem].get(effect_stem, 0)
        rev = self._dir_fwd[effect_stem].get(cause_stem, 0)
        n_tot = max(1.0, float(self._dir_total_events))
        alpha = 0.001

        lift_ab = 0.0
        asym = 0.0
        if fwd > 0 or rev > 0:
            src_a = max(1.0, float(self._dir_src_tot[cause_stem]))
            src_b = max(1.0, float(self._dir_src_tot[effect_stem]))
            p_b_a = fwd / src_a
            p_b_not_a = max(0.0, (self._dir_dst_tot[effect_stem] - fwd)) / max(1.0, n_tot - src_a)
            p_a_b = rev / src_b
            p_a_not_b = max(0.0, (self._dir_dst_tot[cause_stem] - rev)) / max(1.0, n_tot - src_b)
            log_ab = math.log((p_b_a + alpha) / (p_b_not_a + alpha))
            log_ba = math.log((p_a_b + alpha) / (p_a_not_b + alpha))
            rel = (fwd + rev) / (fwd + rev + 4.0)
            asym = (log_ab - log_ba) * rel
            lift_ab = log_ab * rel

        # Add supervised contrastive log-odds from causal training splits (canonical cause -> effect)
        pos_c = self._sup_pos[cause_stem].get(effect_stem, 0)
        neg_c = self._sup_neg[cause_stem].get(effect_stem, 0)
        if pos_c > 0 or neg_c > 0:
            sup_log = math.log((pos_c + 1.0) / (neg_c + 1.0))
            lift_ab += 1.2 * sup_log
            asym += 0.8 * sup_log
        return (lift_ab, asym)

    def _twin_egraph_counterfactual_bonus(
        self,
        src_stems: Sequence[str],
        dst_stems: Sequence[str],
    ) -> tuple[float, int]:
        """Run Mirror 7's ``TwinEGraphDoCalculus`` over matched hypotheses for counterfactual support."""
        matched: list[CausalHypothesis] = []
        for a in src_stems:
            for b in dst_stems:
                hyp = self._hyp_by_pair.get((a, b))
                if hyp is not None:
                    matched.append(hyp)
        if not matched:
            return (0.0, 0)

        engine = TwinEGraphDoCalculus()
        for hyp in matched[:8]:
            engine.add_causal_rule(
                rule_id=hyp.hypothesis_id,
                antecedents=(hyp.antecedent_frame,),
                consequent=hyp.consequent_frame,
                negated=(hyp.causal_lift < 0.0),
                specificity_rank=2 if hyp.epistemic_status == "VERIFIED_MULTI_CLUSTER" else 1,
            )
        interventions = {a: True for a in src_stems[:4]}
        report = engine.do_intervene(interventions, world="truth")
        entailed = sum(1 for b in dst_stems if report.counterfactual_beliefs.get(b) is True)
        # Weight by Wilson lower bound of matched hypotheses
        mean_ci_lo = sum(h.ci_low for h in matched) / len(matched)
        bonus = 0.08 * min(2.0, float(entailed)) * (0.5 + mean_ci_lo)
        return (bonus, len(matched))

    def score_causal_choices(
        self,
        *,
        premise: str,
        choices: Sequence[str],
        relation_direction: str,
    ) -> dict[str, Any]:
        """Score candidate causal choices combining distributional semantics + asymmetric causal lift."""
        if len(choices) < 2:
            return {"scores": [], "best_idx": None, "margin": 0.0, "matched_hypotheses": 0, "reads": 0}

        p_st = set(stemmed_content_tokens(premise))
        c_sets = [set(stemmed_content_tokens(ch)) for ch in choices]
        common_c = c_sets[0] & c_sets[1]
        p_neg = bool(set(tokenize(premise)) & NEGATION_TOKENS)

        scores: list[float] = []
        total_hyps = 0
        reads = 0

        for idx, c_st in enumerate(c_sets):
            diff_c = (c_st - common_c) or c_st
            p_use = (p_st - common_c) or p_st
            is_cause = (relation_direction == "cause")
            src, dst = (diff_c, p_use) if is_cause else (p_use, diff_c)

            pair_vals: list[float] = []
            asym_vals: list[float] = []
            for a in src:
                for b in dst:
                    if a == b:
                        continue
                    reads += 1
                    sym_v = self._sym_pair_sim(a, b)
                    if self.enable_asymmetric_causal_lift:
                        lift_v, asym_v = self.compute_asymmetric_causal_lift(a, b)
                        # Scale-normalized bounded directional lift in [-0.25, +0.25]
                        dir_term = 0.14 * math.tanh(lift_v / 2.5) + 0.11 * math.tanh(asym_v / 2.0)
                        asym_vals.append(dir_term)
                        pair_vals.append(sym_v + dir_term)
                    else:
                        # Ablation A3: strictly symmetric co-occurrence ONLY
                        pair_vals.append(sym_v)

            if pair_vals:
                pair_vals.sort(reverse=True)
                base_score = 0.50 * pair_vals[0] + 0.50 * (sum(pair_vals) / len(pair_vals))
            else:
                base_score = 0.0

            # Counterfactual TwinEGraph verification bonus when asymmetric causal lift is active
            cf_bonus = 0.0
            if self.enable_asymmetric_causal_lift:
                cf_bonus, n_h = self._twin_egraph_counterfactual_bonus(sorted(src), sorted(dst))
                total_hyps += n_h

            # Subtle morphological prefix overlap for OOV stems when distributional score is zero
            morph_fallback = 0.0
            if base_score == 0.0:
                for a in src:
                    if len(a) < 4:
                        continue
                    for b in dst:
                        if len(b) >= 4 and a[:4] == b[:4] and a != b:
                            morph_fallback += 0.05

            c_neg = bool(set(tokenize(choices[idx])) & NEGATION_TOKENS)
            neg_tie = -0.015 if (p_neg != c_neg and base_score < 0.02) else 0.0

            scores.append(base_score + cf_bonus + morph_fallback + neg_tie)

        best_idx = 0 if scores[0] >= scores[1] else 1
        margin = abs(scores[0] - scores[1])
        return {
            "scores": scores,
            "best_idx": best_idx,
            "margin": margin,
            "matched_hypotheses": total_hyps,
            "reads": reads,
        }

    def serialized_summary(self) -> bytes:
        return canonical_json_bytes(
            {
                "num_docs": len(self._doc_stems),
                "num_sents": self._num_sents,
                "vocab_size": len(self._stem_docs),
                "enable_asymmetric_causal_lift": self.enable_asymmetric_causal_lift,
                "hypotheses": [h.to_compact_dict() for h in self.hypotheses[:120]],
            }
        )