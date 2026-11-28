"""Stage S11.1 Unified 9-Stage CPU-Native Raw-Observation Open-World Learner.

Implements the complete 9-stage architecture specified in
``docs/51-open-world-generalization-s11-1-raw-observation-learning-design.md``.

STRICT ISOLATION & CPU-NATIVE CONSTRAINTS:
- Pure Python 3.11 stdlib + Mirror 7 pure-CPU BSC / TropicalJTMS / TwinEGraphDoCalculus ONLY.
- ZERO GPU, ZERO pretrained LLMs, ZERO external AI APIs, ZERO neural training, ZERO manual
  domain action lists or event schemas, ZERO production ``.dcrystal`` vault mutations.
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from research.open_world.contracts import (
    OpenWorldObservation,
    OpenWorldPrediction,
    OpenWorldSolver,
    OpenWorldTrainingExample,
    SolverMemoryFootprint,
    canonical_json_bytes,
    compressed_bit_length,
    content_tokens,
    normalize_text,
    tokenize,
)
from research.open_world.learner.capability_inducer import (
    CapabilityVersionStore,
    OpenWorldCapabilityCard,
    extract_query_signature,
)
from research.open_world.learner.causal_miner import RawCausalMiner
from research.open_world.learner.evidence_ledger import (
    RawEvidenceLedger,
    RawEvidenceUnit,
    extract_evidence_units,
)
from research.open_world.learner.extraction import (
    StructuralPivotTable,
    extract_candidate_spans,
    morph_stem,
    token_grounded_in_set,
    unicode_content_tokens,
    unicode_tokenize,
)
from research.open_world.learner.temporal_graph import BiTemporalGraph
from research.open_world.learner.verifier import (
    _QUESTION_META_WORDS,
    _extract_question_proper_entities,
    compute_local_idf,
    detect_required_answer_type,
    execute_temporal_or_numeric_verifier,
    unit_satisfies_answer_type,
    verify_evidence_grounding,
)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-min(40.0, x))
        return 1.0 / (1.0 + z)
    z = math.exp(max(-40.0, x))
    return z / (1.0 + z)


def _fit_platt_2param(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Deterministic 2-parameter Platt sigmoid calibration minimizing Brier score on validation."""
    if len(xs) < 6:
        base_p = sum(ys) / max(1, len(ys))
        base_p = max(0.08, min(0.92, base_p))
        return (0.0, math.log(base_p / (1.0 - base_p)))
    best_loss = 1e9
    best_ab = (1.5, 0.0)
    for a in (0.0, 0.5, 1.0, 1.8, 2.8, 4.0, 6.0):
        for b_i in range(-16, 17):
            b = b_i * 0.25
            loss = 0.0
            for x, y in zip(xs, ys):
                d = _sigmoid(a * x + b) - y
                loss += d * d
            if loss < best_loss:
                best_loss = loss
                best_ab = (a, b)
    a0, b0 = best_ab
    for da in (-0.4, -0.2, 0.0, 0.2, 0.4):
        for db in (-0.2, -0.1, 0.0, 0.1, 0.2):
            a = max(0.0, a0 + da)
            b = b0 + db
            loss = 0.0
            for x, y in zip(xs, ys):
                d = _sigmoid(a * x + b) - y
                loss += d * d
            if loss < best_loss:
                best_loss = loss
                best_ab = (a, b)
    return best_ab


def _approx_token_f1(pred: str, gold: str) -> float:
    pt = normalize_text(pred).split()
    gt = normalize_text(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    inter = sum((Counter(pt) & Counter(gt)).values())
    if inter == 0:
        return 0.0
    return 2.0 * inter / (len(pt) + len(gt))


@dataclass(frozen=True)
class AblationConfig:
    """Explicit toggle matrix for all 6 preregistered S11.1 controlled ablations (docs/51 §3.1)."""

    enable_evidence_grounding_verifier: bool = True  # A1_no_evidence_grounding_verifier
    enable_bitemporal_supersession: bool = True      # A2_no_bitemporal_supersession
    enable_asymmetric_causal_lift: bool = True       # A3_no_asymmetric_causal_lift
    enable_capability_anti_unification: bool = True  # A4_no_capability_anti_unification
    enable_shard_lsh_indexing: bool = True           # A5_no_shard_lsh_indexing
    enable_abstention_gate: bool = True              # A6_no_abstention_gate

    def ablation_id(self) -> str:
        off: list[str] = []
        if not self.enable_evidence_grounding_verifier:
            off.append("A1_no_evidence_grounding_verifier")
        if not self.enable_bitemporal_supersession:
            off.append("A2_no_bitemporal_supersession")
        if not self.enable_asymmetric_causal_lift:
            off.append("A3_no_asymmetric_causal_lift")
        if not self.enable_capability_anti_unification:
            off.append("A4_no_capability_anti_unification")
        if not self.enable_shard_lsh_indexing:
            off.append("A5_no_shard_lsh_indexing")
        if not self.enable_abstention_gate:
            off.append("A6_no_abstention_gate")
        return "s11_1_full" if not off else "+".join(off)


class RawObservationOpenWorldLearner(OpenWorldSolver):
    """Unified 9-stage pure-CPU raw-observation open-world learner (Stage S11.1)."""

    solver_id: str = "s11_1_raw_observation_learner"
    category: str = "s11_1_raw_observation_learner"
    uses_gpu: bool = False
    uses_pretrained_llm: bool = False
    uses_external_api: bool = False
    uses_neural_training: bool = False
    evaluator_only: bool = False

    def __init__(
        self,
        *,
        seed: int = 101,
        config: AblationConfig | None = None,
        shared_unlabeled_train_texts: Sequence[str] = (),
        k_min_clusters: int = 3,
    ) -> None:
        super().__init__(seed=seed)
        self.config = config or AblationConfig()
        self.solver_id = (
            "s11_1_raw_observation_learner"
            if self.config.ablation_id() == "s11_1_full"
            else f"s11_1_ablation_{self.config.ablation_id()}"
        )
        self._shared_unlabeled_train_texts: tuple[str, ...] = tuple(shared_unlabeled_train_texts)
        self._ledger = RawEvidenceLedger(db_path=":memory:")
        self._pivots = StructuralPivotTable()
        self._causal_miner = RawCausalMiner(
            enable_asymmetric_causal_lift=self.config.enable_asymmetric_causal_lift
        )
        self._cap_store = CapabilityVersionStore(
            enable_anti_unification=self.config.enable_capability_anti_unification,
            use_shard_lsh=self.config.enable_shard_lsh_indexing,
            k_min_clusters=k_min_clusters,
        )
        self._idf_by_bench: dict[str, dict[str, float]] = {}
        self._abstain_threshold: dict[str, float] = {
            "balanced_copa": 0.0,
            "squad_v2": 0.56,
            "longmemeval": 0.42,
        }
        self._platt_answered: dict[str, tuple[float, float]] = {}
        self._platt_abstained: dict[str, tuple[float, float]] = {}
        self._observed_shared_raw: bool = False

    def freeze(self) -> None:
        super().freeze()
        self._ledger.freeze()
        self._causal_miner.freeze()
        self._cap_store.freeze()

    @property
    def total_post_freeze_writes(self) -> int:
        return (
            self._post_freeze_writes
            + self._ledger.post_freeze_writes
            + self._causal_miner.post_freeze_writes
            + self._cap_store.post_freeze_writes
        )

    def observe_unlabeled_raw_texts(self, raw_texts: Iterable[str], *, source_prefix: str = "train_raw") -> int:
        """Ingest strictly public unlabeled training raw UTF-8 texts (before freeze)."""
        self._require_unfrozen()
        count = 0
        batch_texts: list[str] = []
        for idx, txt in enumerate(raw_texts):
            if not txt or len(txt) < 8:
                continue
            self._causal_miner.observe_raw_document(txt, source_id=f"{source_prefix}:{idx}")
            batch_texts.append(txt)
            count += 1
        if batch_texts:
            self._pivots.observe_texts(batch_texts)
        return count

    def fit(self, benchmark_id: str, train_examples: Sequence[OpenWorldTrainingExample]) -> None:
        """Learn entities, structural pivots, causal hypotheses, and SHARD+ITL+RDC capability cards strictly on train."""
        self._require_unfrozen()

        if self._shared_unlabeled_train_texts and not self._observed_shared_raw:
            self.observe_unlabeled_raw_texts(self._shared_unlabeled_train_texts, source_prefix="s11_train_raw")
            self._observed_shared_raw = True

        df: Counter[str] = Counter()
        doc_count = 0
        raw_texts_for_pivots: list[str] = []

        for ex in train_examples:
            obs = ex.observation
            if obs.task_kind == "long_memory_qa":
                # Avoid redundant full-haystack sentence splitting during fit; record first session turns into ledger
                if obs.sessions:
                    sample_obs = OpenWorldObservation(
                        task_id=obs.task_id, benchmark_id=obs.benchmark_id, split=obs.split,
                        task_kind=obs.task_kind, prompt_text=obs.prompt_text, question_text=obs.question_text,
                        sessions=obs.sessions[:1], question_date=obs.question_date,
                    )
                    units = extract_evidence_units(
                        sample_obs, flatten_timestamps=not self.config.enable_bitemporal_supersession
                    )
                    self._ledger.append_batch(units[:6])
            else:
                units = extract_evidence_units(
                    obs, flatten_timestamps=not self.config.enable_bitemporal_supersession
                )
                self._ledger.append_batch(units[:12])

            if obs.task_kind == "causal_choice":
                if ex.target_choice_index is not None:
                    self._causal_miner.observe_causal_training_pair(
                        premise=obs.prompt_text,
                        choices=obs.candidate_choices,
                        relation_direction=obs.relation_direction,
                        target_index=ex.target_choice_index,
                        source_id=obs.task_id,
                    )
                for txt in (obs.prompt_text,) + obs.candidate_choices:
                    raw_texts_for_pivots.append(txt)
                    doc_count += 1
                    for tok in set(unicode_content_tokens(txt)):
                        df[tok] += 1
                        df[morph_stem(tok)] += 1
            elif obs.task_kind == "extractive_qa":
                raw_texts_for_pivots.append(obs.prompt_text)
                for u in units:
                    doc_count += 1
                    seen_st: set[str] = set()
                    for tok in unicode_content_tokens(u.text):
                        seen_st.add(tok)
                        seen_st.add(morph_stem(tok))
                    for t in seen_st:
                        df[t] += 1
            else:
                seen_sess_texts: set[str] = set()
                for sess in obs.sessions:
                    if sess.session_id in seen_sess_texts:
                        continue
                    seen_sess_texts.add(sess.session_id)
                    doc_count += 1
                    s_toks: set[str] = set()
                    for turn in sess.turns:
                        if turn.content:
                            raw_texts_for_pivots.append(turn.content[:400])
                            for tok in unicode_content_tokens(turn.content):
                                s_toks.add(tok)
                                s_toks.add(morph_stem(tok))
                    for t in s_toks:
                        df[t] += 1

        if raw_texts_for_pivots:
            self._pivots.observe_texts(raw_texts_for_pivots[:2500])

        # Keep informative persistent IDF entries (df >= 2 when corpus is large) + true OOV IDF
        min_df = 2 if len(df) > 2500 else 1
        idf_tbl = {
            tok: round(math.log(1.0 + (doc_count - c + 0.5) / (c + 0.5)), 2)
            for tok, c in df.items()
            if c >= min_df
        }
        idf_tbl["__oov_idf__"] = round(min(5.2, math.log(1.0 + (doc_count + 0.5) / 3.0)), 2)
        self._idf_by_bench[benchmark_id] = idf_tbl

        hyps = []
        if any(ex.observation.task_kind == "causal_choice" for ex in train_examples):
            hyps = self._causal_miner.materialize_hypotheses(k_min_clusters=3, max_hypotheses=350)

        self._cap_store.induce_from_training(train_examples)
        if hyps:
            self._cap_store.index_causal_hypotheses(hyps)
    def _idf_for(self, benchmark_id: str) -> dict[str, float]:
        return self._idf_by_bench.get(benchmark_id, {})

    def _score_sentence_unit(
        self,
        question: str,
        q_content: Sequence[str],
        q_bigrams: set[tuple[str, str]],
        q_entities: tuple[str, ...],
        q_nums: set[str],
        unit: RawEvidenceUnit,
        idf: Mapping[str, float],
        *,
        local_idf: Mapping[str, float] | None = None,
        req_type: str = "general",
        speaker_target: str = "document",
        allow_fuzzy: bool = True,
    ) -> float:
        u_raw = set(unicode_tokenize(unit.text))
        if not u_raw:
            return 0.0
        u_stems = {morph_stem(t) for t in u_raw}
        u_low = unit.text.casefold()

        total_w = 0.0
        hit_w = 0.0
        for tok in q_content:
            st = morph_stem(tok)
            oov_w = float(idf.get("__oov_idf__", 3.5)) if idf else 2.2
            c_w = float(idf.get(tok, idf.get(st, oov_w)))
            if local_idf:
                l_w = float(local_idf.get(st, 2.2))
                w = 0.65 * l_w + 0.35 * c_w
            else:
                w = c_w
            total_w += w
            if token_grounded_in_set(tok, u_stems, u_raw, allow_fuzzy=allow_fuzzy):
                hit_w += w
        lex_cov = hit_w / max(1e-6, total_w)

        u_seq = [morph_stem(t) for t in unicode_content_tokens(unit.text)]
        u_bigs = {(u_seq[i], u_seq[i + 1]) for i in range(len(u_seq) - 1)} if len(u_seq) >= 2 else set()
        bg_bonus = 0.16 * (len(q_bigrams & u_bigs) / max(1, len(q_bigrams))) if q_bigrams else 0.0

        ent_bonus = 0.0
        if q_entities:
            ehits = sum(
                1 for e in q_entities
                if e.casefold() in u_low
                or all(token_grounded_in_set(t, u_stems, u_raw, allow_fuzzy=allow_fuzzy) for t in unicode_content_tokens(e)[:2])
            )
            ent_bonus = 0.18 * (ehits / len(q_entities))
        num_bonus = 0.0
        if q_nums:
            nhits = sum(1 for n in q_nums if n in u_raw or n in u_low)
            num_bonus = 0.16 * (nhits / len(q_nums))

        type_bonus = 0.0
        if req_type != "general":
            type_bonus = 0.14 if unit_satisfies_answer_type(unit.text, question, req_type) else -0.10

        role_bonus = 0.0
        if speaker_target == "user":
            role_bonus = 0.10 if unit.speaker_role == "user" else -0.05
        elif speaker_target == "assistant":
            role_bonus = 0.10 if unit.speaker_role == "assistant" else -0.02

        len_pen = 0.03 * max(0.0, (len(u_raw) - 55) / 100.0)
        return max(0.0, lex_cov + bg_bonus + ent_bonus + num_bonus + type_bonus + role_bonus - len_pen)

    def _select_best_span_from_unit(
        self,
        *,
        question: str,
        unit: RawEvidenceUnit,
        proposed_cards: Sequence[tuple[OpenWorldCapabilityCard, int]],
        idf: Mapping[str, float],
        local_idf: Mapping[str, float] | None = None,
    ) -> tuple[str, float, str]:
        spans = extract_candidate_spans(unit, pivots=self._pivots.pivots, compute_hv=False)
        if not spans:
            return (unit.text[:90].strip(), 0.2, "fallback_unit_head")

        qsig = dict(extract_query_signature(question, task_kind="extractive_qa"))
        wh = qsig["wh_class"]
        req_type = detect_required_answer_type(question)
        q_toks = set(unicode_tokenize(question))
        q_stems = {morph_stem(t) for t in unicode_content_tokens(question) if t not in _QUESTION_META_WORDS}
        q_all_stems = {morph_stem(t) for t in unicode_content_tokens(question)}

        anchors: list[tuple[int, float]] = []
        for m in re.finditer(r"[^\W_]+", unit.text):
            w = m.group(0).casefold()
            st = morph_stem(w)
            if st in q_stems:
                aw = float((local_idf or {}).get(st, idf.get(st, 2.0)))
                anchors.append(((m.start() + m.end()) // 2, max(0.8, aw)))
            elif w in q_toks and w not in _QUESTION_META_WORDS and len(w) > 2:
                anchors.append(((m.start() + m.end()) // 2, 1.0))

        shape_weights: Counter[str] = Counter()
        lp_weights: Counter[str] = Counter()
        rp_weights: Counter[str] = Counter()
        len_bounds: list[tuple[int, int]] = []
        anchor_pref: Counter[str] = Counter()
        active_card_id = "none"

        for card, sim in proposed_cards:
            if not self.config.enable_capability_anti_unification:
                if len(q_stems & set(card.concrete_tokens)) < 2:
                    continue
            w_card = max(0.1, (sim / 10240.0 + 0.5)) * (0.4 + card.precision)
            shape_weights[card.expected_span_shape] += w_card
            anchor_pref[card.anchor_pattern] += w_card
            for lp in card.left_pivots:
                lp_weights[lp] += w_card
            for rp in card.right_pivots:
                rp_weights[rp] += w_card
            len_bounds.append((card.preferred_len_min, card.preferred_len_max))
            if active_card_id == "none":
                active_card_id = card.card_id

        wh_shape_prior: dict[str, set[str]] = {
            "temporal_point": {"DATE_TIME"},
            "temporal_duration": {"QUANTITY_NUM", "DATE_TIME"},
            "numeric_quantity": {"QUANTITY_NUM", "PERCENT_FRAC"},
            "agent_entity": {"PROPER_SPAN"},
            "location_entity": {"PROPER_SPAN", "NOUN_CHUNK"},
            "named_entity": {"PROPER_SPAN", "QUOTED_SPAN"},
            "reason_clause": {"CLAUSE_SPAN", "NOUN_CHUNK"},
            "manner_clause": {"NOUN_CHUNK", "CLAUSE_SPAN"},
        }
        prior_shapes = {"PERCENT_FRAC"} if req_type == "percentage" else wh_shape_prior.get(wh, set())

        best_span = spans[0].surface
        best_score = -1e9
        max_sw = max(shape_weights.values()) if shape_weights else 1.0
        total_aw = sum(w for _, w in anchors) or 1.0

        for cand in spans:
            surf = cand.surface
            s_toks = unicode_tokenize(surf)
            if not s_toks:
                continue
            s_stems = [morph_stem(t) for t in unicode_content_tokens(surf)]
            if s_stems:
                novel_frac = sum(1 for st in s_stems if st not in q_all_stems) / len(s_stems)
            else:
                novel_frac = 1.0 if surf.casefold() not in question.casefold() else 0.0
            if novel_frac <= 0.05:
                continue

            card_score = 0.0
            if shape_weights:
                card_score += 0.36 * (shape_weights.get(cand.entity_shape, 0.0) / max_sw)
                if cand.left_pivot in lp_weights:
                    card_score += 0.12
                if cand.right_pivot in rp_weights:
                    card_score += 0.08
                if len_bounds:
                    avg_lo = sum(a for a, _ in len_bounds) / len(len_bounds)
                    avg_hi = sum(b for _, b in len_bounds) / len(len_bounds)
                    if avg_lo <= len(s_toks) <= avg_hi + 1:
                        card_score += 0.10
            if cand.entity_shape in prior_shapes:
                card_score += 0.36 if self.config.enable_capability_anti_unification else 0.10

            if req_type == "percentage":
                card_score += 0.55 if cand.entity_shape == "PERCENT_FRAC" else -0.45
            elif req_type == "temporal":
                card_score += 0.45 if cand.entity_shape == "DATE_TIME" else -0.38
            elif req_type == "numeric":
                is_yr = bool(re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", surf.strip()))
                card_score += 0.40 if (cand.entity_shape in {"QUANTITY_NUM", "PERCENT_FRAC"} and not is_yr) else -0.32
            elif wh == "agent_entity" and cand.entity_shape == "PROPER_SPAN":
                card_score += 0.22
            elif wh == "location_entity" and cand.left_pivot in {"in", "at", "on", "to", "into", "under", "near", "inside", "from", "behind", "above", "below"}:
                card_score += 0.40
            elif wh == "named_entity" and cand.entity_shape in {"PROPER_SPAN", "QUOTED_SPAN"}:
                card_score += 0.24
            if cand.entity_shape == "QUOTED_SPAN" and re.search(r"\b(?:mean|means|called|named|known|term|word|phrase|title|book|song|playlist)\b", question, re.I):
                card_score += 0.35

            center = (cand.char_start + cand.char_end) // 2
            if anchors:
                density = sum(aw / (1.0 + abs(center - ap) / 22.0) for ap, aw in anchors) / total_aw
                min_dist = min(abs(center - ap) for ap, _ in anchors)
                prox_score = 0.55 * density + 0.25 / (1.0 + min_dist / 20.0)
                if anchor_pref:
                    left_cnt = sum(1 for ap, _ in anchors if ap < cand.char_start)
                    right_cnt = sum(1 for ap, _ in anchors if ap > cand.char_end)
                    top_ap = anchor_pref.most_common(1)[0][0]
                    if top_ap == "LEFT_ANCHOR" and left_cnt > right_cnt:
                        prox_score += 0.08
                    elif top_ap == "RIGHT_ANCHOR" and right_cnt > left_cnt:
                        prox_score += 0.08
                    elif top_ap == "BOTH_ANCHORS" and left_cnt > 0 and right_cnt > 0:
                        prox_score += 0.10
            else:
                prox_score = 0.0

            n_w = len(s_toks)
            if wh in {"reason_clause", "manner_clause"}:
                conc = 0.14 if 2 <= n_w <= 8 else 0.0
            else:
                conc = 0.18 if 1 <= n_w <= 3 else (0.08 if n_w <= 5 else -0.10)

            pivot_boost = 0.12 if cand.left_pivot in {"called", "named", "means", "mean", "to", "in", "by", "as", "at", "from", "under", "on"} else 0.0
            total = card_score + prox_score + 0.30 * novel_frac + conc + pivot_boost
            if total > best_score:
                best_score = total
                best_span = surf

        return (best_span, max(0.0, best_score), active_card_id)

    def _evaluate_extractive_qa(self, obs: OpenWorldObservation) -> dict[str, Any]:
        idf = self._idf_for(obs.benchmark_id)
        units = extract_evidence_units(obs, flatten_timestamps=not self.config.enable_bitemporal_supersession)
        if not units:
            return {"raw_score": 0.0, "answer": "", "cited_ids": (), "cited_spans": (), "abstain_reason": "empty_evidence", "reads": 0}

        local_idf = compute_local_idf(units)
        req_type = detect_required_answer_type(obs.question_text)
        q_content = list(dict.fromkeys(t for t in unicode_content_tokens(obs.question_text) if t not in _QUESTION_META_WORDS))
        if not q_content:
            q_content = list(dict.fromkeys(unicode_content_tokens(obs.question_text)))
        q_seq = [morph_stem(t) for t in unicode_content_tokens(obs.question_text)]
        q_bigs = {(q_seq[i], q_seq[i + 1]) for i in range(len(q_seq) - 1)} if len(q_seq) >= 2 else set()
        q_ents = _extract_question_proper_entities(obs.question_text)
        q_nums = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", obs.question_text))

        scored_units: list[tuple[float, RawEvidenceUnit]] = []
        for u in units:
            sc = self._score_sentence_unit(
                obs.question_text, q_content, q_bigs, q_ents, q_nums, u, idf,
                local_idf=local_idf, req_type=req_type, speaker_target="document", allow_fuzzy=True,
            )
            scored_units.append((sc, u))
        scored_units.sort(key=lambda pair: (-pair[0], pair[1].sent_index))

        best_sent_sc, best_unit = scored_units[0]
        proposed_cards = self._cap_store.propose_cards(obs.question_text, task_kind="extractive_qa", top_k=6)
        span, span_sc, card_id = self._select_best_span_from_unit(
            question=obs.question_text,
            unit=best_unit,
            proposed_cards=proposed_cards,
            idf=idf,
            local_idf=local_idf,
        )

        if self.config.enable_evidence_grounding_verifier:
            ver = verify_evidence_grounding(
                question=obs.question_text,
                proposed_answer=span,
                cited_units=(best_unit,),
                all_units=units,
                idf_lookup=idf,
                is_computed_answer=False,
            )
            epistemic_score = 0.15 * min(1.0, best_sent_sc) + 0.65 * ver.support_score + 0.20 * min(1.0, span_sc / 2.2)
            if not ver.answer_type_compatible:
                epistemic_score *= 0.25
            if ver.antonym_contradiction or not ver.polarity_consistent:
                epistemic_score *= 0.25
            if ver.neg_implicative_ungrounded:
                epistemic_score *= 0.45
            if ver.coord_operand_ungrounded:
                epistemic_score *= 0.50
            if ver.bitemporal_contested:
                epistemic_score *= 0.15
            if ver.missing_numbers and ver.number_grounding_cov < 0.5:
                epistemic_score *= 0.30
            if ver.missing_entities and ver.entity_passage_cov < 0.5:
                epistemic_score *= 0.35
            abstain_reason = (
                ver.rejection_reasons[0] if ver.rejection_reasons else "epistemic_support_below_threshold"
            )
            ver_dict = ver.to_dict()
        else:
            epistemic_score = min(1.0, best_sent_sc)
            abstain_reason = "raw_lexical_score_below_threshold"
            ver_dict = {"verifier_disabled_ablation_a1": True}

        hard_reject = bool(
            self.config.enable_evidence_grounding_verifier
            and (
                (not ver.span_grounded)
                or (not ver.answer_type_compatible)
                or ver.antonym_contradiction
                or (not ver.polarity_consistent)
                or ver.bitemporal_contested
                or (bool(ver.missing_numbers) and ver.number_grounding_cov < 0.5)
                or (bool(ver.missing_entities) and ver.entity_passage_cov < 0.5)
            )
        )
        return {
            "raw_score": epistemic_score,
            "hard_reject": hard_reject,
            "sent_score": best_sent_sc,
            "span_score": span_sc,
            "answer": span,
            "cited_ids": (best_unit.citation_id,),
            "cited_spans": (span,),
            "abstain_reason": abstain_reason,
            "card_id": card_id,
            "verification": ver_dict,
            "reads": len(units) + len(proposed_cards),
        }
    def _evaluate_long_memory(self, obs: OpenWorldObservation) -> dict[str, Any]:
        idf = self._idf_for(obs.benchmark_id)
        units = extract_evidence_units(obs, flatten_timestamps=not self.config.enable_bitemporal_supersession)
        if not units:
            return {"raw_score": 0.0, "answer": "", "cited_ids": (), "cited_spans": (), "abstain_reason": "empty_haystack", "reads": 0}

        qsig = dict(extract_query_signature(obs.question_text, task_kind="long_memory_qa"))
        speaker_target = qsig["speaker_target"]
        temporal_mode = "earliest" if qsig["temporal_mode"] == "historical_earliest" else "latest"
        lm_req_type = detect_required_answer_type(obs.question_text)

        q_content = list(dict.fromkeys(t for t in unicode_content_tokens(obs.question_text) if t not in _QUESTION_META_WORDS))
        if not q_content:
            q_content = list(dict.fromkeys(unicode_content_tokens(obs.question_text)))
        q_stems_set = {morph_stem(t) for t in q_content}
        q_seq = [morph_stem(t) for t in unicode_content_tokens(obs.question_text)]
        q_bigs = {(q_seq[i], q_seq[i + 1]) for i in range(len(q_seq) - 1)} if len(q_seq) >= 2 else set()
        q_ents = _extract_question_proper_entities(obs.question_text)
        q_nums = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", obs.question_text))

        scored_units: list[tuple[float, RawEvidenceUnit]] = []
        by_session: dict[str, list[tuple[float, RawEvidenceUnit]]] = defaultdict(list)
        sess_bags: dict[str, set[str]] = defaultdict(set)

        for u in units:
            u_toks = unicode_content_tokens(u.text)
            if not u_toks:
                continue
            u_stems = {morph_stem(t) for t in u_toks}
            sess_bags[u.source_id].update(u_stems)
            if not (u_stems & q_stems_set):
                continue
            sc = self._score_sentence_unit(
                obs.question_text, q_content, q_bigs, q_ents, q_nums, u, idf,
                req_type=lm_req_type, speaker_target=speaker_target, allow_fuzzy=False,
            )
            scored_units.append((sc, u))
            by_session[u.source_id].append((sc, u))

        if not scored_units:
            for u in units[:30]:
                sc = self._score_sentence_unit(
                    obs.question_text, q_content, q_bigs, q_ents, q_nums, u, idf,
                    req_type=lm_req_type, speaker_target=speaker_target, allow_fuzzy=True,
                )
                scored_units.append((sc, u))
                by_session[u.source_id].append((sc, u))

        scored_units.sort(key=lambda pair: (-pair[0], -pair[1].observed_ordinal, pair[1].evidence_id))

        q_w_tot = max(1e-6, sum(float(idf.get(t, 2.0)) for t in q_content))
        sess_ranked: list[tuple[float, int, str, RawEvidenceUnit, list[tuple[float, RawEvidenceUnit]]]] = []
        for sid, s_units in by_session.items():
            s_units.sort(key=lambda p: (-p[0], p[1].turn_index, p[1].sent_index))
            top1_sc, top1_u = s_units[0]
            top2_sc = s_units[1][0] if len(s_units) > 1 else top1_sc * 0.6
            s_bag = sess_bags.get(sid, set())
            bag_cov = sum(float(idf.get(t, 2.0)) for t in q_content if morph_stem(t) in s_bag) / q_w_tot
            agg_sc = 0.58 * top1_sc + 0.22 * top2_sc + 0.20 * bag_cov
            sess_ranked.append((agg_sc, top1_u.observed_ordinal, sid, top1_u, s_units))

        sess_ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        top_sess_sc = sess_ranked[0][0]

        # Build BiTemporalGraph over top candidate sessions (down to 0.50 * top score so paraphrased
        # update sessions like "stored" -> "moved" enter the supersession graph!).
        graph_sessions = [
            item for item in sess_ranked[:6]
            if item[0] >= max(0.18, 0.50 * top_sess_sc)
        ] or [sess_ranked[0]]
        strong_sessions = [
            item for item in sess_ranked[:4]
            if item[0] >= max(0.22, 0.72 * top_sess_sc)
        ] or [sess_ranked[0]]
        top_units_for_graph = [u for _, _, _, _, su_list in graph_sessions[:5] for _, u in su_list[:4]]
        bt_graph = BiTemporalGraph(enable_bitemporal_supersession=self.config.enable_bitemporal_supersession)
        bt_graph.ingest_units(top_units_for_graph)

        sess_by_id = {it[2]: it for it in graph_sessions}
        top_sid = strong_sessions[0][2]
        top_rel_ids = list(bt_graph.by_source_id.get(top_sid, ()))
        resolved_bt = bt_graph.resolve_query_state(top_rel_ids, temporal_mode=temporal_mode)
        bt_winner = resolved_bt.get("winner")
        bt_chain_sids = tuple(resolved_bt.get("chain_sources", ()))

        if (
            self.config.enable_bitemporal_supersession
            and bt_winner is not None
            and len(bt_chain_sids) >= 2
            and bt_winner.source_id in sess_by_id
        ):
            chosen_sess = sess_by_id[bt_winner.source_id]
            cited_sids = tuple(dict.fromkeys((bt_winner.source_id,) + bt_chain_sids))[:2]
        elif self.config.enable_bitemporal_supersession and len(strong_sessions) >= 2:
            chron = sorted(strong_sessions, key=lambda it: (it[1], -it[0], it[2]))
            chosen_sess = chron[0] if temporal_mode == "earliest" else chron[-1]
            cited_sids = tuple(dict.fromkeys(it[2] for it in strong_sessions[:2]))
        else:
            chosen_sess = strong_sessions[0]
            cited_sids = (chosen_sess[2],)

        winning_unit = chosen_sess[3]
        if qsig["modality"] == "PREFERENCE":
            user_cands = [(sc, u) for sc, u in chosen_sess[4] if u.speaker_role == "user"]
            if user_cands and user_cands[0][0] >= 0.70 * chosen_sess[4][0][0]:
                winning_unit = user_cands[0][1]
        elif lm_req_type != "general" and not unit_satisfies_answer_type(winning_unit.text, obs.question_text, lm_req_type):
            # Prefer a high-scoring sentence in the same winning session that actually satisfies the grammatical answer type
            typed_cands = [
                (sc, u) for sc, u in chosen_sess[4]
                if sc >= 0.62 * chosen_sess[4][0][0] and unit_satisfies_answer_type(u.text, obs.question_text, lm_req_type)
            ]
            if typed_cands:
                winning_unit = typed_cands[0][1]

        proposed_cards = self._cap_store.propose_cards(obs.question_text, task_kind="long_memory_qa", top_k=6)

        exec_res = execute_temporal_or_numeric_verifier(
            question=obs.question_text,
            question_date=obs.question_date,
            ranked_units=scored_units[:30],
            idf_lookup=idf,
            enable_bitemporal=self.config.enable_bitemporal_supersession,
        )

        is_computed = False
        op_trace = "extractive_span"
        cited_units_for_ver: tuple[RawEvidenceUnit, ...] = (winning_unit,)

        if exec_res is not None:
            answer = str(exec_res["answer"])
            cited_sids = tuple(exec_res["cited_ids"])
            cited_units_for_ver = tuple(exec_res["cited_units"])
            is_computed = True
            op_trace = str(exec_res["operation"])
            card_id = proposed_cards[0][0].card_id if proposed_cards else "exec_temporal_numeric"
        else:
            if qsig["modality"] == "PREFERENCE" or qsig["wh_class"] == "boolean_verify":
                answer = winning_unit.text[:200].strip()
                card_id = proposed_cards[0][0].card_id if proposed_cards else "preference_clause"
            else:
                answer, _span_sc, card_id = self._select_best_span_from_unit(
                    question=obs.question_text,
                    unit=winning_unit,
                    proposed_cards=proposed_cards,
                    idf=idf,
                )
                if len(answer) < 2:
                    answer = winning_unit.text[:140].strip()

        if self.config.enable_evidence_grounding_verifier:
            # Scope passage verification to the cited session(s) so coincidental words in 10 unrelated
            # distractor sessions CANNOT falsely validate a missing entity or attribute!
            # Include cited sessions + runner-up strong session for multi-entity comparisons (e.g. A or B)
            ver_sid_set = set(cited_sids)
            if re.search(r"\b(?:or|between|compared)\b", obs.question_text, re.I):
                for item in strong_sessions[:2]:
                    ver_sid_set.add(item[2])
            cited_session_units = tuple(u for u in units if u.source_id in ver_sid_set)
            ver = verify_evidence_grounding(
                question=obs.question_text,
                proposed_answer=answer,
                cited_units=cited_units_for_ver,
                all_units=cited_session_units or cited_units_for_ver,
                idf_lookup=idf,
                is_computed_answer=is_computed,
                bitemporal_graph=bt_graph,
                temporal_mode=temporal_mode,
            )
            epistemic_score = 0.32 * min(1.0, top_sess_sc) + 0.68 * ver.support_score
            if is_computed:
                epistemic_score = min(1.0, epistemic_score + 0.06)
            if (not ver.answer_type_compatible) and lm_req_type in ("numeric", "percentage", "temporal"):
                epistemic_score *= 0.32
            if ver.coord_operand_ungrounded:
                epistemic_score *= 0.28
            if ver.neg_implicative_ungrounded:
                epistemic_score *= 0.45
            if ver.antonym_contradiction or (not ver.polarity_consistent):
                epistemic_score *= 0.25
            if ver.bitemporal_contested:
                epistemic_score *= 0.15
            if ver.missing_numbers and ver.number_grounding_cov < 0.5:
                epistemic_score *= 0.30
            if ver.missing_entities and ver.entity_passage_cov < 0.5:
                epistemic_score *= 0.28
            if ver.rare_token_grounded_frac < 0.40:
                epistemic_score *= 0.42
            abstain_reason = (
                ver.rejection_reasons[0] if ver.rejection_reasons else "insufficient_session_evidence_grounding"
            )
            ver_dict = ver.to_dict()
        else:
            epistemic_score = min(1.0, top_sess_sc)
            abstain_reason = "raw_session_score_below_threshold"
            ver_dict = {"verifier_disabled_ablation_a1": True}

        hard_reject = bool(
            self.config.enable_evidence_grounding_verifier
            and (
                (not ver.span_grounded)
                or ver.bitemporal_contested
                or (lm_req_type in ("numeric", "percentage") and (not is_computed) and (not ver.answer_type_compatible))
                or ver.antonym_contradiction
                or (not ver.polarity_consistent)
                or (bool(ver.missing_numbers) and ver.number_grounding_cov < 0.5)
                or (bool(ver.missing_entities) and ver.entity_passage_cov < 0.5)
            )
        )
        return {
            "raw_score": epistemic_score,
            "hard_reject": hard_reject,
            "top_sess_score": top_sess_sc,
            "answer": answer,
            "cited_ids": cited_sids,
            "cited_spans": (answer[:160],) if answer else (),
            "abstain_reason": abstain_reason,
            "card_id": card_id,
            "operation": op_trace,
            "bitemporal_stats": bt_graph.stats(),
            "verification": ver_dict,
            "reads": len(obs.sessions) + len(proposed_cards),
        }

    def _evaluate_causal_choice(self, obs: OpenWorldObservation) -> dict[str, Any]:
        proposed_cards = self._cap_store.propose_cards(obs.question_text, task_kind="causal_choice", top_k=4)
        res = self._causal_miner.score_causal_choices(
            premise=obs.prompt_text,
            choices=obs.candidate_choices,
            relation_direction=obs.relation_direction,
        )
        res["reads"] = int(res.get("reads", 0)) + len(proposed_cards)
        res["card_id"] = proposed_cards[0][0].card_id if proposed_cards else "causal_direct"
        return res

    def calibrate(
        self, benchmark_id: str, val_examples: Sequence[OpenWorldTrainingExample]
    ) -> None:
        """Calibrate abstention threshold and Platt confidence strictly on validation split."""
        self._require_unfrozen()
        if not val_examples:
            return
        kind = val_examples[0].observation.task_kind

        if kind == "causal_choice":
            scored = [(ex, self._evaluate_causal_choice(ex.observation)) for ex in val_examples]
            self._abstain_threshold[benchmark_id] = 0.0
            xs = [float(info["margin"]) for _, info in scored]
            ys = [1.0 if info["best_idx"] == ex.target_choice_index else 0.0 for ex, info in scored]
            self._platt_answered[benchmark_id] = _fit_platt_2param(xs, ys)
            self._platt_abstained[benchmark_id] = (0.0, 0.0)
            return

        eval_fn = self._evaluate_extractive_qa if kind == "extractive_qa" else self._evaluate_long_memory
        scored = [(ex, eval_fn(ex.observation)) for ex in val_examples]
        raw_scores = sorted(set(round(float(info["raw_score"]), 4) for _, info in scored))
        if not raw_scores:
            return

        n_ans = sum(1 for ex, _ in scored if not ex.is_unanswerable)
        n_unans = sum(1 for ex, _ in scored if ex.is_unanswerable)

        if self.config.enable_abstention_gate and n_ans > 0 and n_unans > 0:
            grid = raw_scores[:: max(1, len(raw_scores) // 60)]
            scored_grid: list[tuple[float, float]] = []
            for thr in grid:
                ans_kept = sum(
                    1 for ex, info in scored
                    if (not ex.is_unanswerable) and (not bool(info.get("hard_reject", False))) and info["raw_score"] >= thr
                )
                unans_abs = sum(
                    1 for ex, info in scored
                    if ex.is_unanswerable and (bool(info.get("hard_reject", False)) or info["raw_score"] < thr)
                )
                false_abs = 1.0 - (ans_kept / n_ans)
                unans_rec = unans_abs / n_unans
                ans_f1_sum = 0.0
                ev_hit_sum = 0.0
                for ex, info in scored:
                    if (not ex.is_unanswerable) and (not bool(info.get("hard_reject", False))) and info["raw_score"] >= thr:
                        if ex.target_answer:
                            ans_f1_sum += _approx_token_f1(str(info["answer"]), ex.target_answer)
                        if ex.supporting_evidence_ids and (set(info.get("cited_ids", ())) & set(ex.supporting_evidence_ids)):
                            ev_hit_sum += 1.0
                mean_ans_f1 = ans_f1_sum / n_ans
                mean_ev_hit = ev_hit_sum / n_ans
                fa_cap = 0.16 if kind == "long_memory_qa" else 0.20
                penalty = 4.0 * (false_abs - fa_cap) if false_abs > fa_cap else 0.0
                # Class-balanced recall weight when unanswerables are sparse (e.g. LongMemEval _abs)
                u_w = 0.78 if (n_unans / max(1, n_ans)) < 0.20 else 0.50
                a_w = 0.55 if (n_unans / max(1, n_ans)) < 0.20 else 0.50
                obj = u_w * unans_rec + a_w * mean_ans_f1 + 0.12 * mean_ev_hit - penalty
                scored_grid.append((obj, thr))
            best_obj = max(o for o, _ in scored_grid)
            near_best = [t for o, t in scored_grid if o >= best_obj - 0.015]
            self._abstain_threshold[benchmark_id] = near_best[len(near_best) // 2]
        elif not self.config.enable_abstention_gate:
            self._abstain_threshold[benchmark_id] = -1.0
        else:
            self._abstain_threshold[benchmark_id] = max(0.25, raw_scores[0] * 0.5)

        thr = self._abstain_threshold[benchmark_id]
        ans_xs: list[float] = []
        ans_ys: list[float] = []
        abs_xs: list[float] = []
        abs_ys: list[float] = []

        for ex, info in scored:
            sc = float(info["raw_score"])
            abstained = self.config.enable_abstention_gate and (bool(info.get("hard_reject", False)) or sc < thr)
            if abstained:
                abs_xs.append(thr - sc)
                abs_ys.append(1.0 if ex.is_unanswerable else 0.0)
            else:
                ans_xs.append(sc - thr)
                if ex.is_unanswerable:
                    ans_ys.append(0.0)
                elif ex.target_answer:
                    f1 = _approx_token_f1(str(info["answer"]), ex.target_answer)
                    ans_ys.append(1.0 if f1 >= 0.5 else 0.0)
                else:
                    ans_ys.append(0.0)

        self._platt_answered[benchmark_id] = _fit_platt_2param(ans_xs, ans_ys)
        self._platt_abstained[benchmark_id] = _fit_platt_2param(abs_xs, abs_ys)

    def _calibrated_conf(self, benchmark_id: str, x: float, *, abstained: bool) -> float:
        table = self._platt_abstained if abstained else self._platt_answered
        a, b = table.get(benchmark_id, (2.0, 0.0))
        return max(0.02, min(0.98, _sigmoid(a * x + b)))

    def predict(self, observation: OpenWorldObservation) -> OpenWorldPrediction:
        """Predict or abstain strictly from public ``observation`` (0 post-freeze memory writes!)."""
        bid = observation.benchmark_id
        thr = self._abstain_threshold.get(bid, 0.45)

        if observation.task_kind == "causal_choice":
            info = self._evaluate_causal_choice(observation)
            if info["best_idx"] is None:
                return OpenWorldPrediction(
                    task_id=observation.task_id, benchmark_id=bid, abstained=True,
                    abstain_reason="missing_candidate_choices", confidence=0.5, uncertainty=0.5,
                    memory_reads=int(info["reads"]), memory_writes=0,
                )
            margin = float(info["margin"])
            idx = int(info["best_idx"])
            conf = self._calibrated_conf(bid, margin, abstained=False)
            return OpenWorldPrediction(
                task_id=observation.task_id, benchmark_id=bid,
                predicted_answer=observation.candidate_choices[idx],
                predicted_choice_index=idx, abstained=False,
                confidence=conf, uncertainty=1.0 - conf,
                cited_evidence_ids=(f"{observation.task_id}:premise", f"{observation.task_id}:choice_{idx}"),
                cited_evidence_spans=(observation.candidate_choices[idx],),
                memory_reads=int(info["reads"]), memory_writes=0,
                metadata={
                    "ablation": self.config.ablation_id(),
                    "margin": round(margin, 5),
                    "matched_hypotheses": info.get("matched_hypotheses", 0),
                    "card_id": info.get("card_id", ""),
                },
            )

        info = (
            self._evaluate_extractive_qa(observation)
            if observation.task_kind == "extractive_qa"
            else self._evaluate_long_memory(observation)
        )
        raw_score = float(info["raw_score"])
        cited_ids = tuple(info.get("cited_ids", ()))
        ans = str(info.get("answer", "")).strip()

        should_abstain = (
            self.config.enable_abstention_gate
            and (not cited_ids or not ans or bool(info.get("hard_reject", False)) or raw_score < thr)
        )
        if should_abstain:
            conf = self._calibrated_conf(bid, thr - raw_score, abstained=True)
            return OpenWorldPrediction(
                task_id=observation.task_id, benchmark_id=bid, predicted_answer=None,
                abstained=True, abstain_reason=str(info.get("abstain_reason", "calibrated_abstention_gate")),
                confidence=conf, uncertainty=1.0 - conf, cited_evidence_ids=(), cited_evidence_spans=(),
                memory_reads=int(info.get("reads", 0)), memory_writes=0,
                metadata={
                    "ablation": self.config.ablation_id(),
                    "epistemic_score": round(raw_score, 4),
                    "threshold": round(thr, 4),
                    "verification": info.get("verification", {}),
                },
            )

        conf = self._calibrated_conf(bid, raw_score - thr, abstained=False)
        return OpenWorldPrediction(
            task_id=observation.task_id, benchmark_id=bid, predicted_answer=ans,
            abstained=False, confidence=conf, uncertainty=1.0 - conf,
            cited_evidence_ids=cited_ids, cited_evidence_spans=tuple(info.get("cited_spans", ())),
            memory_reads=int(info.get("reads", 0)), memory_writes=0,
            metadata={
                "ablation": self.config.ablation_id(),
                "epistemic_score": round(raw_score, 4),
                "threshold": round(thr, 4),
                "card_id": info.get("card_id", ""),
                "operation": info.get("operation", "extractive_span"),
                "verification": info.get("verification", {}),
            },
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "ablation": self.config.ablation_id(),
            "ledger_integrity": self._ledger.verify_integrity(),
            "capability_mdl": dict(self._cap_store.mdl_report),
            "shard_index_telemetry": self._cap_store.shard_index.telemetry(),
            "causal_hypotheses_count": len(self._causal_miner.hypotheses),
            "causal_hypotheses_verified": sum(
                1 for h in self._causal_miner.hypotheses if h.epistemic_status == "VERIFIED_MULTI_CLUSTER"
            ),
            "post_freeze_writes": self.total_post_freeze_writes,
        }

    def memory_footprint(self) -> SolverMemoryFootprint:
        """Honestly serialize ALL persistent tables retained after freeze and read by ``predict()``."""
        payload = {
            "solver_id": self.solver_id,
            "ablation": self.config.ablation_id(),
            "ledger": self._ledger.summary_bytes().decode("utf-8"),
            "capabilities": self._cap_store.serialized_bytes().decode("utf-8"),
            "causal": self._causal_miner.serialized_summary().decode("utf-8"),
            "idf_by_bench": {
                b: sorted(tbl.items()) for b, tbl in sorted(self._idf_by_bench.items())
            },
            "induced_pivots": sorted(self._pivots.pivots),
            "thresholds": self._abstain_threshold,
            "platt_answered": self._platt_answered,
            "platt_abstained": self._platt_abstained,
        }
        raw = canonical_json_bytes(payload)
        entries = (
            len(self._cap_store.verified_cards)
            + len(self._causal_miner.hypotheses)
            + len(self._causal_miner._npmi_neighbors)
            + sum(len(v) for v in self._idf_by_bench.values())
            + len(self._ledger)
        )
        return SolverMemoryFootprint(
            serialized_bytes=len(raw),
            compressed_mdl_bits=compressed_bit_length(raw),
            active_entry_count=entries,
            details={
                "verified_capability_cards": len(self._cap_store.verified_cards),
                "shadow_capability_cards": len(self._cap_store.shadow_cards),
                "quarantined_capability_cards": len(self._cap_store.quarantined_cards),
                "causal_hypotheses": len(self._causal_miner.hypotheses),
                "ledger_evidence_records": len(self._ledger),
                "shard_index": self._cap_store.shard_index.telemetry(),
            },
        )