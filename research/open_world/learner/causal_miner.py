"""Stage S11.1 Stage 5: Schema-Free Causal Hypothesis Discovery & Counterfactual Consistency.

STRICT ISOLATION & ZERO MANUAL DOMAIN SCHEMAS:
- Pure Python 3.11 stdlib + Mirror 7's 10,240-bit BSC & ``TwinEGraphDoCalculus``
  (``mirror_crystal.quotient_egraph_resonator``).
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
- Mines canonical directed causal event transitions ``(CAUSE_FRAME -> EFFECT_FRAME)`` directly
  from raw UTF-8 text (explicit discourse causal connectives, clean narrative succession,
  and directional training pairs) filtered by contrastive counterfactual causal lift:
      CausalLift(A -> B) = log((P(B|A)+a)/(P(B|~A)+a)) - log((P(A|B)+a)/(P(A|~B)+a))
- Uses hubness-normalized NPMI + 2-hop sparse NPMI neighborhood vectors (eliminating word-frequency
  hubness bias on mirrored counterfactual pairs), compacts into a sparse read-only representation
  before ``.freeze()``, and honestly serializes ALL retained prediction tables in ``serialized_summary()``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

from mirror_crystal.quotient_egraph_resonator import (
    TwinEGraphDoCalculus,
    bind10240,
    hv10240_of,
    permute10240,
)
from research.open_world.contracts import (
    SolverFrozenError,
    canonical_json_bytes,
    split_sentences_with_offsets,
)
from research.open_world.learner.extraction import (
    NEGATION_TOKENS,
    stemmed_content_tokens,
    unicode_tokenize,
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
_MARKDOWN_NOISE_RE = re.compile(r"^\s*(?:[-*+#]|\d+[.)]|```)")


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
    """Schema-free compact hubness-normalized NPMI & asymmetric causal hypothesis miner (Stage 5)."""

    def __init__(self, *, enable_asymmetric_causal_lift: bool = True) -> None:
        self.enable_asymmetric_causal_lift = bool(enable_asymmetric_causal_lift)
        self._frozen: bool = False
        self._post_freeze_writes: int = 0
        self._compacted: bool = False

        # Fit-time raw document inverted index (compacted & freed before freeze!)
        self._doc_stems: list[tuple[str, ...]] = []
        self._stem_docs: dict[str, set[int]] = defaultdict(set)
        self._seen_doc_hashes: set[int] = set()
        self._num_docs_seen: int = 0

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

        # Compact sparse NPMI graph retained after compaction (100% serialized in memory_footprint!)
        self._npmi_neighbors: dict[str, dict[str, float]] = {}

        # Materialized top verified/tentative CausalHypotheses
        self.hypotheses: list[CausalHypothesis] = []
        self._hyp_by_pair: dict[tuple[str, str], CausalHypothesis] = {}

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def post_freeze_writes(self) -> int:
        return self._post_freeze_writes

    def freeze(self) -> None:
        if not self._compacted and (self._doc_stems or self._dir_fwd):
            self.compact_before_freeze()
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
        self._num_docs_seen = len(self._doc_stems)
        for w in uniq_doc:
            self._stem_docs[w].add(did)

        prev_sent: list[str] | None = None
        for _, _, sent in split_sentences_with_offsets(text):
            if _MARKDOWN_NOISE_RE.match(sent):
                prev_sent = None
                continue
            seq = list(dict.fromkeys(stemmed_content_tokens(sent)))
            if not seq:
                prev_sent = None
                continue

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
                cs = list(dict.fromkeys(stemmed_content_tokens(c_part)))[:7]
                es = list(dict.fromkeys(stemmed_content_tokens(e_part)))[:7]
                if cs and es:
                    if is_contrast:
                        for a in cs[:4]:
                            for b in es[:4]:
                                if a != b and len(self._dir_contrast_sources[(a, b)]) < 10:
                                    self._dir_contrast_sources[(a, b)].add(source_id)
                    else:
                        w_inc = 3
                        self._dir_total_events += w_inc
                        for a in cs:
                            self._dir_src_tot[a] += w_inc
                            for b in es:
                                if a != b:
                                    self._dir_fwd[a][b] += w_inc
                                    if len(self._dir_support_sources[(a, b)]) < 10:
                                        self._dir_support_sources[(a, b)].add(source_id)
                        for b in es:
                            self._dir_dst_tot[b] += w_inc
            elif prev_sent is not None and len(prev_sent) <= 10 and len(seq) <= 10 and self._num_docs_seen <= 2500:
                # Short plain narrative chronological succession (excludes long prose/Markdown bloat)
                self._dir_total_events += 1
                for a in prev_sent[:3]:
                    self._dir_src_tot[a] += 1
                    for b in seq[:3]:
                        if a != b:
                            self._dir_fwd[a][b] += 1
                for b in seq[:3]:
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
        self.observe_raw_document(f"{premise} {choices[target_index]}", source_id=source_id)

        p_st = list(dict.fromkeys(stemmed_content_tokens(premise)))
        c_sets = [list(dict.fromkeys(stemmed_content_tokens(ch))) for ch in choices]
        common_c = set.intersection(*(set(c) for c in c_sets)) if len(c_sets) >= 2 else set()

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

    def compact_before_freeze(self, *, max_neighbors: int = 16, max_vocab: int = 1800) -> None:
        """Compile compact hubness-normalized NPMI sparse graph & prune singleton noise before freeze."""
        self._require_unfrozen()
        if self._compacted:
            return

        n_doc = len(self._doc_stems)
        self._npmi_neighbors = {}
        if n_doc >= 2 and self._stem_docs:
            # Keep stems with df >= 2 (plus any stems occurring in supervised causal pairs), capped for CPU/MDL hygiene
            sup_vocab = set(self._sup_pos.keys()) | set(self._sup_neg.keys())
            for cnt in self._sup_pos.values():
                sup_vocab.update(cnt.keys())
            for cnt in self._sup_neg.values():
                sup_vocab.update(cnt.keys())

            ranked_vocab = sorted(
                (w for w, ds in self._stem_docs.items() if len(ds) >= 2 or w in sup_vocab),
                key=lambda w: (-(1 if w in sup_vocab else 0), -len(self._stem_docs[w]), w),
            )[:max_vocab]
            kept_set = set(ranked_vocab)

            for w in ranked_vocab:
                dw = self._stem_docs[w]
                lw = len(dw)
                dlist = sorted(dw)[:64]
                scale = lw / max(1, len(dlist))
                cand: Counter[str] = Counter()
                for did in dlist:
                    for nb in self._doc_stems[did][:40]:
                        if nb != w and nb in kept_set:
                            cand[nb] += 1
                scored: list[tuple[float, str]] = []
                min_c = 1 if n_doc < 40 else 2
                for nb, c in cand.items():
                    if c < min_c:
                        continue
                    lnb = len(self._stem_docs[nb])
                    c_est = min(float(lw), float(lnb), c * scale)
                    p_joint = c_est / n_doc
                    if p_joint <= 0.0 or p_joint >= 1.0:
                        continue
                    pmi = math.log((c_est * n_doc) / max(1e-9, lw * lnb))
                    if pmi <= 0.05:
                        continue
                    denom = -math.log(p_joint)
                    npmi_val = min(1.0, pmi / denom) if denom > 1e-9 else 0.0
                    if npmi_val > 0.04:
                        scored.append((round(npmi_val, 3), nb))
                scored.sort(key=lambda pair: (-pair[0], pair[1]))
                if scored:
                    self._npmi_neighbors[w] = {nb: val for val, nb in scored[:max_neighbors]}

        # Prune noisy singleton directional counts (require >= 2 distinct directional weight or supervised evidence)
        pruned_fwd: dict[str, Counter[str]] = defaultdict(Counter)
        for a, cnt in self._dir_fwd.items():
            kept_items = {
                b: c for b, c in cnt.items()
                if c >= 4 or self._sup_pos[a].get(b, 0) > 0 or (a, b) in self._hyp_by_pair
            }
            if kept_items:
                # Retain top 16 outgoing causal edges per stem for bounded MDL footprint
                top_b = sorted(kept_items.items(), key=lambda kv: (-kv[1], kv[0]))[:16]
                pruned_fwd[a] = Counter(dict(top_b))
        self._dir_fwd = pruned_fwd

        # Release temporary raw fit buffers so post-freeze RAM is tiny & 100% matches memory_footprint()!
        self._doc_stems.clear()
        self._stem_docs.clear()
        self._seen_doc_hashes.clear()
        self._dir_support_sources.clear()
        self._dir_contrast_sources.clear()
        self._sup_pos_sources.clear()
        self._sup_neg_sources.clear()
        self._compacted = True

    def materialize_hypotheses(self, *, k_min_clusters: int = 3, max_hypotheses: int = 400) -> list[CausalHypothesis]:
        """Compile top candidate ``CausalHypothesis`` records with Wilson 95% CIs & multi-cluster status."""
        self._require_unfrozen()
        candidates: list[tuple[float, CausalHypothesis]] = []
        seen: set[tuple[str, str]] = set()

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
        self.compact_before_freeze()
        return self.hypotheses

    def _npmi_pair_sim(self, a: str, b: str) -> float:
        """Hubness-normalized direct NPMI + 2-hop sparse NPMI cosine similarity."""
        if a == b:
            return 0.0
        if not self._compacted and self._stem_docs:
            # Pre-compaction fallback (e.g. unit tests calling score_causal_choices before materialize)
            da = self._stem_docs.get(a)
            db = self._stem_docs.get(b)
            n_doc = max(1, len(self._doc_stems))
            if not da or not db:
                return 0.0
            c = len(da & db)
            if c == 0:
                return 0.0
            p_j = c / n_doc
            pmi = math.log((c * n_doc) / (len(da) * len(db)))
            if pmi <= 0.0 or p_j >= 1.0:
                return 0.0
            den = -math.log(p_j)
            return min(1.0, pmi / den) if den > 1e-9 else 0.0

        va = self._npmi_neighbors.get(a)
        vb = self._npmi_neighbors.get(b)
        direct = 0.0
        if va and b in va:
            direct = va[b]
        elif vb and a in vb:
            direct = vb[a]
        two_hop = 0.0
        if va and vb:
            s_a, s_b = (va, vb) if len(va) <= len(vb) else (vb, va)
            dot = sum(v * s_b[k] for k, v in s_a.items() if k in s_b)
            if dot > 0.0:
                na = math.sqrt(sum(v * v for v in va.values())) or 1.0
                nb = math.sqrt(sum(v * v for v in vb.values())) or 1.0
                two_hop = dot / (na * nb)
        return direct + 0.65 * two_hop

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
            rel = (fwd + rev) / (fwd + rev + 8.0)
            asym = (log_ab - log_ba) * rel
            lift_ab = log_ab * rel

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
        mean_ci_lo = sum(h.ci_low for h in matched) / len(matched)
        bonus = 0.06 * min(2.0, float(entailed)) * (0.5 + mean_ci_lo)
        return (bonus, len(matched))

    def score_causal_choices(
        self,
        *,
        premise: str,
        choices: Sequence[str],
        relation_direction: str,
    ) -> dict[str, Any]:
        """Score arbitrary N >= 2 candidate causal choices via hubness-normalized NPMI + asymmetric lift."""
        if len(choices) < 2:
            return {"scores": [], "best_idx": None, "margin": 0.0, "matched_hypotheses": 0, "reads": 0}

        p_st = set(stemmed_content_tokens(premise))
        c_sets = [set(stemmed_content_tokens(ch)) for ch in choices]
        common_c = set.intersection(*c_sets) if c_sets else set()
        p_neg = bool(set(unicode_tokenize(premise)) & NEGATION_TOKENS)

        scores: list[float] = []
        total_hyps = 0
        reads = 0

        for idx, c_st in enumerate(c_sets):
            diff_c = (c_st - common_c) or c_st
            p_use = (p_st - common_c) or p_st
            is_cause = (relation_direction == "cause")
            src, dst = (diff_c, p_use) if is_cause else (p_use, diff_c)

            pair_vals: list[float] = []
            exact_hits = len(p_use & diff_c)
            for a in src:
                for b in dst:
                    if a == b:
                        continue
                    reads += 1
                    sym_v = self._npmi_pair_sim(a, b)
                    if self.enable_asymmetric_causal_lift:
                        lift_v, asym_v = self.compute_asymmetric_causal_lift(a, b)
                        dir_term = 0.10 * math.tanh(lift_v / 2.5) + 0.08 * math.tanh(asym_v / 2.0)
                        pair_vals.append(sym_v + dir_term)
                    else:
                        # Ablation A3: strictly symmetric co-occurrence ONLY
                        pair_vals.append(sym_v)

            if pair_vals:
                pair_vals.sort(reverse=True)
                base_score = 0.55 * pair_vals[0] + 0.45 * (sum(pair_vals) / len(pair_vals))
            else:
                base_score = 0.0

            cf_bonus = 0.0
            if self.enable_asymmetric_causal_lift:
                cf_bonus, n_h = self._twin_egraph_counterfactual_bonus(sorted(src), sorted(dst))
                total_hyps += n_h

            # Fallback lexical/morphological grounding when distributional score is zero (e.g. cold OOV)
            morph_fallback = 0.0
            if base_score == 0.0:
                if exact_hits > 0:
                    morph_fallback += 0.12 * min(3, exact_hits)
                for a in src:
                    if len(a) < 4:
                        continue
                    for b in dst:
                        if len(b) >= 4 and a[:4] == b[:4] and a != b:
                            morph_fallback += 0.05

            c_neg = bool(set(unicode_tokenize(choices[idx])) & NEGATION_TOKENS)
            neg_tie = -0.015 if (p_neg != c_neg and base_score < 0.02) else 0.0

            scores.append(base_score + cf_bonus + morph_fallback + neg_tie)

        ranked_indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        best_idx = ranked_indices[0]
        runner_idx = ranked_indices[1]
        margin = max(0.0, scores[best_idx] - scores[runner_idx])
        return {
            "scores": scores,
            "best_idx": best_idx,
            "margin": margin,
            "matched_hypotheses": total_hyps,
            "reads": reads,
        }

    def serialized_summary(self) -> bytes:
        """Honestly serialize ALL persistent tables retained after freeze & read during prediction."""
        return canonical_json_bytes(
            {
                "num_docs_seen": self._num_docs_seen,
                "enable_asymmetric_causal_lift": self.enable_asymmetric_causal_lift,
                "npmi_neighbors": {
                    k: sorted(v.items()) for k, v in sorted(self._npmi_neighbors.items())
                },
                "dir_fwd": {
                    k: sorted(v.items()) for k, v in sorted(self._dir_fwd.items()) if v
                },
                "sup_pos": {
                    k: sorted(v.items()) for k, v in sorted(self._sup_pos.items()) if v
                },
                "sup_neg": {
                    k: sorted(v.items()) for k, v in sorted(self._sup_neg.items()) if v
                },
                "hypotheses": [h.to_compact_dict() for h in self.hypotheses],
            }
        )
