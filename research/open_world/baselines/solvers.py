"""Four strictly isolated CPU-native Stage S11.0 baseline solvers.

STRICT ISOLATION:
- NEVER imports ``research.open_world.evaluator``, ``hidden_eval``, ``oracle``, or
  ``datasets.loader``.
- NEVER writes to or modifies any production ``.dcrystal`` vault.
- Pure Python 3.11 stdlib + read-only Mirror 7 native CPU organs only.
- ZERO GPU, ZERO pretrained LLMs, ZERO external AI APIs, ZERO neural training.

Baselines implemented:
1. ``ExistingMirrorSystemBaseline`` — read-only invocation of Mirror 7's shipped CPU organs
   (``BilingualCCGHDCParser``, ``LTAGKneserNeyRealizer.pdcst``, isolated per-episode
   ``KanervaSDMInt8``, ``ContextReasoner``).
2. ``RetrievalOnlyBaseline`` — pure BM25/IDF lexical retrieval over sentences, turns, and
   training demonstrations with validation-calibrated abstention thresholds.
3. ``ExistingRuleBlueprintBaseline`` — read-only lookup over Mirror 7's existing staged
   blueprints (997), authored domain seed rules (110), and source-supported rules (26),
   never promoting unverified entries and abstaining when uncovered.
4. ``SimpleSymbolicBaseline`` — pure CPU domain-agnostic symbolic extractor (directed causal
   token-pair PMI from train pairs; WH-type + entity/number/date + negation rules; temporal
   date ordering + recency precedence + entity-grounding abstention for multi-session memory).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

from mirror_crystal.blueprint_runtime import BlueprintCompositionEngine
from mirror_crystal.cortical_active_inference import BilingualCCGHDCParser
from mirror_crystal.hippocampal_sdm_memory import KanervaSDMInt8
from mirror_crystal.native_intelligence.context_reasoner import ContextReasoner
from mirror_crystal.ternary_lut_realizer import LTAGKneserNeyRealizer

from research.open_world.contracts import (
    ConversationSession,
    EvidenceSentence,
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

REPO_ROOT = Path(__file__).resolve().parents[3]

_NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "no", "never", "none", "neither", "nor", "without", "cannot", "can't", "didn't", "doesn't", "don't", "isn't", "wasn't", "weren't", "haven't", "hasn't", "won't", "wouldn't"}
)
_UPDATE_CUES: frozenset[str] = frozenset(
    {"actually", "now", "currently", "updated", "update", "changed", "switched", "moved", "replaced", "instead", "anymore", "latest", "recent", "recently", "new", "newly", "bought", "painted"}
)
_TEMPORAL_ORDER_CUES: frozenset[str] = frozenset(
    {"first", "earlier", "before", "after", "later", "last", "recently", "order", "long", "days", "weeks", "months", "ago"}
)
_DATE_YMD_RE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?(?:\s*(?:%|percent|million|billion|thousand|km|miles|years|days|hours))?\b", re.I)
_CAP_SPAN_RE = re.compile(r"\b[A-Z][a-zA-Z0-9'-]+(?:\s+(?:of|the|and|de|van|von|al)?\s*[A-Z][a-zA-Z0-9'-]+)*\b")


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(50.0, x))
        return 1.0 / (1.0 + z)
    z = math.exp(max(-50.0, x))
    return z / (1.0 + z)


def _parse_date_ordinal(date_str: str) -> int:
    m = _DATE_YMD_RE.search(date_str or "")
    if not m:
        return 0
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return y * 10000 + mo * 100 + d


def _flatten_session_text(session: ConversationSession) -> str:
    return " ".join(t.content for t in session.turns if t.content).strip()


def _extract_non_question_span(sentence: str, question: str, max_words: int = 12) -> str:
    """Extract a concise candidate answer phrase from a matched sentence excluding question words."""
    q_toks = set(content_tokens(question))
    words = sentence.strip().rstrip(".?!").split()
    if not words:
        return ""
    # Find longest contiguous run of tokens containing non-question content words.
    best_span: list[str] = []
    cur: list[str] = []
    for w in words:
        w_clean = re.sub(r"[^a-zA-Z0-9'-]", "", w).casefold()
        if w_clean and w_clean not in q_toks:
            cur.append(w)
            if len(cur) > len(best_span):
                best_span = list(cur)
        else:
            cur = []
    chosen = best_span[:max_words] if best_span else words[:max_words]
    return " ".join(chosen).strip(" ,;:-")


def _bm25_score(
    query_toks: Sequence[str],
    doc_toks: Sequence[str],
    idf: dict[str, float],
    avg_len: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if not query_toks or not doc_toks:
        return 0.0
    tf = Counter(doc_toks)
    dlen = len(doc_toks)
    denom_norm = k1 * (1.0 - b + b * (dlen / max(1.0, avg_len)))
    score = 0.0
    for qt in set(query_toks):
        f = tf.get(qt, 0)
        if f <= 0:
            continue
        w = idf.get(qt, 1.0)
        score += w * ((f * (k1 + 1.0)) / (f + denom_norm))
    return score


# ---------------------------------------------------------------------------
# Baseline 1: Existing Mirror System (Read-Only Existing CPU Organs)
# ---------------------------------------------------------------------------


class ExistingMirrorSystemBaseline(OpenWorldSolver):
    """Evaluates Mirror 7's shipped CPU-native organs in strict read-only mode (zero vault writes)."""

    solver_id: str = "existing_mirror_system"
    category: str = "existing_mirror"

    def __init__(self, seed: int = 101) -> None:
        super().__init__(seed=seed)
        self._parser = BilingualCCGHDCParser()
        self._realizer = LTAGKneserNeyRealizer()
        self._context_reasoner = ContextReasoner()
        self._seen_benchmarks: list[str] = []

    def fit(self, benchmark_id: str, train_examples: Sequence[OpenWorldTrainingExample]) -> None:
        self._require_unfrozen()
        # Shipped Mirror organs are evaluated zero-shot without modifying vaults or priors.
        if benchmark_id not in self._seen_benchmarks:
            self._seen_benchmarks.append(benchmark_id)

    def predict(self, observation: OpenWorldObservation) -> OpenWorldPrediction:
        if observation.task_kind == "causal_choice":
            return self._predict_causal_choice(observation)
        if observation.task_kind == "extractive_qa":
            return self._predict_extractive_qa(observation)
        return self._predict_long_memory(observation)

    def _predict_causal_choice(self, obs: OpenWorldObservation) -> OpenWorldPrediction:
        if len(obs.candidate_choices) < 2:
            return OpenWorldPrediction(
                task_id=obs.task_id,
                benchmark_id=obs.benchmark_id,
                abstained=True,
                abstain_reason="missing_candidate_choices",
                confidence=0.0,
                uncertainty=1.0,
            )
        p_parsed = self._parser.parse_clause(obs.prompt_text)
        pdcst = self._realizer.pdcst
        conn = "because" if obs.relation_direction == "cause" else "so"

        scores: list[float] = []
        for ch in obs.candidate_choices:
            combined = f"{obs.prompt_text} {conn} {ch}"
            q16_score = float(pdcst.score_sentence_q16(combined))
            c_parsed = self._parser.parse_clause(ch)
            # Small bonus if Mirror's CCG parser extracts matching actor/target roles.
            role_bonus = 0.0
            if p_parsed.who and c_parsed.who and p_parsed.who == c_parsed.who:
                role_bonus += 0.25
            scores.append(q16_score + role_bonus)

        if abs(scores[0] - scores[1]) < 1e-9:
            return OpenWorldPrediction(
                task_id=obs.task_id,
                benchmark_id=obs.benchmark_id,
                abstained=True,
                abstain_reason="pdcst_exact_tie",
                confidence=0.5,
                uncertainty=0.5,
                cited_evidence_ids=(f"{obs.task_id}:premise",),
                memory_reads=2,
            )

        best_idx = 0 if scores[0] > scores[1] else 1
        margin = abs(scores[0] - scores[1]) / max(1.0, abs(scores[0]) + abs(scores[1]))
        conf = max(0.5, min(0.85, 0.5 + margin))
        return OpenWorldPrediction(
            task_id=obs.task_id,
            benchmark_id=obs.benchmark_id,
            predicted_answer=obs.candidate_choices[best_idx],
            predicted_choice_index=best_idx,
            abstained=False,
            confidence=conf,
            uncertainty=1.0 - conf,
            cited_evidence_ids=(f"{obs.task_id}:premise", f"{obs.task_id}:choice_{best_idx}"),
            cited_evidence_spans=(obs.candidate_choices[best_idx],),
            memory_reads=2,
            metadata={"organ": "BilingualCCGHDCParser+LTAGKneserNeyRealizer.pdcst"},
        )

    def _predict_extractive_qa(self, obs: OpenWorldObservation) -> OpenWorldPrediction:
        # 1. Invoke Mirror's existing ContextReasoner on raw passage + question.
        combined_prompt = f"{obs.prompt_text}\n{obs.question_text}"
        try:
            native_ans = self._context_reasoner.solve(combined_prompt)
        except Exception:
            native_ans = None

        if native_ans is not None and getattr(native_ans, "answer", ""):
            ans_str = str(native_ans.answer).strip()
            if ans_str.casefold().startswith("unknown"):
                return OpenWorldPrediction(
                    task_id=obs.task_id,
                    benchmark_id=obs.benchmark_id,
                    abstained=True,
                    abstain_reason="context_reasoner_unknown",
                    confidence=0.65,
                    uncertainty=0.35,
                    memory_reads=len(obs.evidence_units),
                    metadata={"organ": "ContextReasoner"},
                )
            cited = (obs.evidence_units[0].evidence_id,) if obs.evidence_units else ()
            return OpenWorldPrediction(
                task_id=obs.task_id,
                benchmark_id=obs.benchmark_id,
                predicted_answer=ans_str,
                abstained=False,
                confidence=float(getattr(native_ans, "confidence", 0.7)),
                uncertainty=1.0 - float(getattr(native_ans, "confidence", 0.7)),
                cited_evidence_ids=cited,
                cited_evidence_spans=(ans_str,),
                memory_reads=len(obs.evidence_units),
                metadata={"organ": "ContextReasoner"},
            )

        # 2. Check Mirror's BilingualCCGHDCParser exact role matching against passage clauses.
        q_parsed = self._parser.parse_clause(obs.question_text)
        best_ev: EvidenceSentence | None = None
        best_role_hits = 0
        for ev in obs.evidence_units:
            ep = self._parser.parse_clause(ev.text)
            hits = 0
            if q_parsed.who and ep.who and q_parsed.who == ep.who:
                hits += 1
            if q_parsed.act and ep.act and q_parsed.act == ep.act:
                hits += 1
            if q_parsed.targ and ep.targ and q_parsed.targ == ep.targ:
                hits += 1
            if hits > best_role_hits:
                best_role_hits = hits
                best_ev = ev

        if best_ev is not None and best_role_hits >= 2:
            ans_span = _extract_non_question_span(best_ev.text, obs.question_text, max_words=10)
            return OpenWorldPrediction(
                task_id=obs.task_id,
                benchmark_id=obs.benchmark_id,
                predicted_answer=ans_span,
                abstained=False,
                confidence=0.55,
                uncertainty=0.45,
                cited_evidence_ids=(best_ev.evidence_id,),
                cited_evidence_spans=(ans_span,),
                memory_reads=len(obs.evidence_units),
                metadata={"organ": "BilingualCCGHDCParser", "role_hits": best_role_hits},
            )

        # Honest abstention when shipped Mirror organs cannot ground open-world prose.
        return OpenWorldPrediction(
            task_id=obs.task_id,
            benchmark_id=obs.benchmark_id,
            abstained=True,
            abstain_reason="existing_mirror_organs_no_grounded_parse",
            confidence=0.5,
            uncertainty=0.5,
            memory_reads=len(obs.evidence_units),
            metadata={"organ": "ContextReasoner+BilingualCCGHDCParser", "s0_s10_itl_applicable": False},
        )

    def _predict_long_memory(self, obs: OpenWorldObservation) -> OpenWorldPrediction:
        if not obs.sessions:
            return OpenWorldPrediction(
                task_id=obs.task_id,
                benchmark_id=obs.benchmark_id,
                abstained=True,
                abstain_reason="empty_session_haystack",
                confidence=0.9,
                uncertainty=0.1,
            )

        # Ephemeral per-episode KanervaSDMInt8 (matching scripts/run_five_organ_external_eval.py Organ 3).
        # Note: FlyHash.text_to_input_vec computes 3 * len(bytes) blake2b hashes in pure Python;
        # we bound key_text to a deterministic 1024-char head + 1024-char tail window (2048 chars max)
        # while retaining full session turns for turn-level extraction, and record this bottleneck explicitly.
        sdm = KanervaSDMInt8(
            num_hard_locations=256,
            sdr_dim=256,
            top_k_sdr=16,
            value_dim=64,
            active_locations_m=8,
            seed=f"s11-mirror-sdm-{self.seed}-{obs.task_id}",
        )
        sess_map: dict[str, ConversationSession] = {}
        for sess in obs.sessions:
            text = _flatten_session_text(sess)
            if text:
                sess_map[sess.session_id] = sess
                key_window = text if len(text) <= 2048 else (text[:1024] + " " + text[-1024:])
                sdm.write(
                    episode_id=sess.session_id,
                    key_text=key_window,
                    payload={"session_id": sess.session_id},
                )

        recall = sdm.read(obs.question_text)
        if not recall.found or recall.episode is None:
            return OpenWorldPrediction(
                task_id=obs.task_id,
                benchmark_id=obs.benchmark_id,
                abstained=True,
                abstain_reason="kanerva_sdm_recall_miss",
                confidence=0.6,
                uncertainty=0.4,
                memory_reads=len(sess_map),
                metadata={"organ": "KanervaSDMInt8"},
            )

        recalled_sid = str(recall.episode.episode_id)
        recalled_sess = sess_map.get(recalled_sid)
        q_toks = set(content_tokens(obs.question_text))
        best_turn_text = ""
        best_overlap = -1
        if recalled_sess is not None:
            for turn in recalled_sess.turns:
                t_toks = set(content_tokens(turn.content))
                ov = len(q_toks & t_toks)
                if ov > best_overlap:
                    best_overlap = ov
                    best_turn_text = turn.content

        ans = _extract_non_question_span(best_turn_text, obs.question_text, max_words=18) if best_turn_text else ""
        sim_score = float(getattr(recall, "confidence", getattr(recall, "similarity", 0.5)))
        conf = max(0.1, min(0.85, sim_score))
        return OpenWorldPrediction(
            task_id=obs.task_id,
            benchmark_id=obs.benchmark_id,
            predicted_answer=ans if ans else best_turn_text[:160],
            abstained=False,
            confidence=conf,
            uncertainty=1.0 - conf,
            cited_evidence_ids=(recalled_sid,),
            cited_evidence_spans=(ans[:160],) if ans else (),
            memory_reads=len(sess_map) + 1,
            metadata={"organ": "KanervaSDMInt8", "recalled_session_id": recalled_sid},
        )

    def memory_footprint(self) -> SolverMemoryFootprint:
        # Read-only shipped priors only; zero learned state written during fit.
        meta = canonical_json_bytes({"organs": ["BilingualCCGHDCParser", "LTAGKneserNeyRealizer", "ContextReasoner", "KanervaSDMInt8"]})
        return SolverMemoryFootprint(
            serialized_bytes=len(meta),
            compressed_mdl_bits=compressed_bit_length(meta),
            active_entry_count=4,
            details={"mode": "read_only_shipped_organs_zero_vault_writes"},
        )


# ---------------------------------------------------------------------------
# Baseline 2: Retrieval-Only Baseline (BM25 / IDF Lexical Retrieval)
# ---------------------------------------------------------------------------


class RetrievalOnlyBaseline(OpenWorldSolver):
    """Pure CPU-native BM25/IDF retrieval baseline with validation-calibrated abstention."""

    solver_id: str = "retrieval_only_bm25"
    category: str = "retrieval_only"

    def __init__(self, seed: int = 101) -> None:
        super().__init__(seed=seed)
        self._idf_by_bench: dict[str, dict[str, float]] = {}
        self._avg_len_by_bench: dict[str, float] = {}
        self._copa_train_demos: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = []
        self._abstain_threshold: dict[str, float] = {
            "balanced_copa": 0.01,
            "squad_v2": 2.0,
            "longmemeval": 2.0,
        }

    def fit(self, benchmark_id: str, train_examples: Sequence[OpenWorldTrainingExample]) -> None:
        self._require_unfrozen()
        df: Counter[str] = Counter()
        doc_count = 0
        total_len = 0

        if benchmark_id in {"balanced_copa", "smoke_causal"}:
            self._copa_train_demos = []

        for ex in train_examples:
            obs = ex.observation
            docs: list[tuple[str, ...]] = []
            if obs.task_kind == "causal_choice":
                p_toks = content_tokens(obs.prompt_text)
                docs.append(p_toks)
                for ch in obs.candidate_choices:
                    docs.append(content_tokens(ch))
                if ex.target_choice_index is not None and 0 <= ex.target_choice_index < len(obs.candidate_choices):
                    self._copa_train_demos.append(
                        (p_toks, obs.relation_direction, content_tokens(obs.candidate_choices[ex.target_choice_index]))
                    )
            elif obs.task_kind == "extractive_qa":
                for ev in obs.evidence_units:
                    docs.append(content_tokens(ev.text))
            else:
                for sess in obs.sessions:
                    docs.append(content_tokens(_flatten_session_text(sess)))
            for d in docs:
                doc_count += 1
                total_len += len(d)
                for tok in set(d):
                    df[tok] += 1

        idf: dict[str, float] = {}
        for tok, c in df.items():
            idf[tok] = math.log(1.0 + (doc_count - c + 0.5) / (c + 0.5))
        self._idf_by_bench[benchmark_id] = idf
        self._avg_len_by_bench[benchmark_id] = total_len / max(1, doc_count)

    def _raw_score_and_candidate(self, obs: OpenWorldObservation) -> dict[str, Any]:
        idf = self._idf_by_bench.get(obs.benchmark_id, {})
        avg_len = self._avg_len_by_bench.get(obs.benchmark_id, 20.0)

        if obs.task_kind == "causal_choice":
            p_toks = content_tokens(obs.prompt_text)
            c_scores: list[float] = []
            reads = 0
            for ch in obs.candidate_choices:
                c_toks = content_tokens(ch)
                # Direct premise-choice BM25 overlap + nearest training demonstration support.
                direct = _bm25_score(p_toks, c_toks, idf, avg_len=max(4.0, avg_len))
                nn_bonus = 0.0
                p_set = set(p_toks)
                c_set = set(c_toks)
                for tr_p, tr_dir, tr_c in self._copa_train_demos:
                    reads += 1
                    if tr_dir != obs.relation_direction:
                        continue
                    p_ov = len(p_set & set(tr_p))
                    c_ov = len(c_set & set(tr_c))
                    if p_ov > 0 and c_ov > 0:
                        nn_bonus = max(nn_bonus, (p_ov * c_ov) / max(1.0, len(p_set) + len(c_set)))
                c_scores.append(direct + 2.0 * nn_bonus)
            if len(c_scores) < 2:
                return {"score": 0.0, "best_idx": None, "reads": reads}
            best_idx = 0 if c_scores[0] >= c_scores[1] else 1
            margin = abs(c_scores[0] - c_scores[1])
            return {"score": margin, "best_idx": best_idx, "reads": reads, "scores": c_scores}

        if obs.task_kind == "extractive_qa":
            q_toks = content_tokens(obs.question_text)
            if not q_toks or not obs.evidence_units:
                return {"score": 0.0, "best_ev": None, "reads": len(obs.evidence_units)}
            best_score = -1.0
            best_ev: EvidenceSentence | None = None
            q_max = sum(idf.get(t, 1.0) for t in set(q_toks)) or 1.0
            for ev in obs.evidence_units:
                s = _bm25_score(q_toks, content_tokens(ev.text), idf, avg_len=max(8.0, avg_len)) / q_max
                if s > best_score:
                    best_score = s
                    best_ev = ev
            return {"score": max(0.0, best_score), "best_ev": best_ev, "reads": len(obs.evidence_units)}

        # long_memory_qa
        q_toks = content_tokens(obs.question_text)
        if not q_toks or not obs.sessions:
            return {"score": 0.0, "best_sess": None, "best_turn": None, "reads": len(obs.sessions)}
        q_max = sum(idf.get(t, 1.0) for t in set(q_toks)) or 1.0
        scored_sessions: list[tuple[float, ConversationSession]] = []
        for sess in obs.sessions:
            stoks = content_tokens(_flatten_session_text(sess))
            s = _bm25_score(q_toks, stoks, idf, avg_len=max(20.0, avg_len)) / q_max
            scored_sessions.append((s, sess))
        scored_sessions.sort(key=lambda x: (-x[0], x[1].session_id))
        top_score, top_sess = scored_sessions[0]
        # Pick best turn inside top session
        best_turn_text = ""
        best_turn_score = -1.0
        for turn in top_sess.turns:
            ts = _bm25_score(q_toks, content_tokens(turn.content), idf, avg_len=15.0)
            if ts > best_turn_score:
                best_turn_score = ts
                best_turn_text = turn.content
        cited = [top_sess.session_id]
        if len(scored_sessions) > 1 and scored_sessions[1][0] >= 0.85 * top_score and top_score > 0:
            cited.append(scored_sessions[1][1].session_id)
        return {
            "score": max(0.0, top_score),
            "best_sess": top_sess,
            "best_turn_text": best_turn_text,
            "cited": tuple(cited),
            "reads": len(obs.sessions),
        }

    def calibrate(
        self, benchmark_id: str, val_examples: Sequence[OpenWorldTrainingExample]
    ) -> None:
        self._require_unfrozen()
        if not val_examples:
            return
        kind = val_examples[0].observation.task_kind
        scored = [(ex, self._raw_score_and_candidate(ex.observation)) for ex in val_examples]

        if kind == "causal_choice":
            # Abstain only on exact/near ties; pick best threshold over tiny margins.
            candidates = [0.0, 1e-6, 0.05, 0.2, 0.5]
            best_thr = 1e-6
            best_val = -1.0
            for thr in candidates:
                hits = 0
                for ex, info in scored:
                    if info["score"] < thr:
                        continue
                    if info["best_idx"] == ex.target_choice_index:
                        hits += 1
                acc = hits / len(scored)
                if acc > best_val:
                    best_val = acc
                    best_thr = thr
            self._abstain_threshold[benchmark_id] = best_thr
            return

        # For QA / LongMemEval with unanswerable items: select threshold maximizing ROC Youden's J
        # = P(answered | answerable) + P(abstained | unanswerable) - 1 on validation, avoiding
        # degenerate always-abstain collapse while calibrating purely from validation labels.
        raw_scores = sorted(set(round(info["score"], 4) for _, info in scored))
        if not raw_scores:
            return
        grid = raw_scores[:: max(1, len(raw_scores) // 40)]
        n_ans = sum(1 for ex, _ in scored if not ex.is_unanswerable)
        n_unans = sum(1 for ex, _ in scored if ex.is_unanswerable)
        if n_ans == 0 or n_unans == 0:
            self._abstain_threshold[benchmark_id] = raw_scores[0] * 0.5
            return
        best_thr = raw_scores[len(raw_scores) // 2]
        best_j = -2.0
        for thr in grid:
            tpr_ans = sum(
                1 for ex, info in scored if (not ex.is_unanswerable) and info["score"] >= thr
            ) / n_ans
            tnr_unans = sum(
                1 for ex, info in scored if ex.is_unanswerable and info["score"] < thr
            ) / n_unans
            j_stat = tpr_ans + tnr_unans - 1.0
            if j_stat > best_j + 1e-9:
                best_j = j_stat
                best_thr = thr
        self._abstain_threshold[benchmark_id] = best_thr

    def predict(self, observation: OpenWorldObservation) -> OpenWorldPrediction:
        info = self._raw_score_and_candidate(observation)
        thr = self._abstain_threshold.get(observation.benchmark_id, 0.35)
        score = float(info["score"])

        if observation.task_kind == "causal_choice":
            if info["best_idx"] is None or score < thr:
                return OpenWorldPrediction(
                    task_id=observation.task_id,
                    benchmark_id=observation.benchmark_id,
                    abstained=True,
                    abstain_reason="retrieval_tie_or_zero_overlap",
                    confidence=0.5,
                    uncertainty=0.5,
                    cited_evidence_ids=(f"{observation.task_id}:premise",),
                    memory_reads=int(info["reads"]),
                )
            idx = int(info["best_idx"])
            conf = max(0.5, min(0.92, _sigmoid(score)))
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                predicted_answer=observation.candidate_choices[idx],
                predicted_choice_index=idx,
                abstained=False,
                confidence=conf,
                uncertainty=1.0 - conf,
                cited_evidence_ids=(f"{observation.task_id}:premise", f"{observation.task_id}:choice_{idx}"),
                cited_evidence_spans=(observation.candidate_choices[idx],),
                memory_reads=int(info["reads"]),
            )

        if observation.task_kind == "extractive_qa":
            best_ev: EvidenceSentence | None = info.get("best_ev")
            if best_ev is None or score < thr:
                conf = max(0.5, min(0.9, _sigmoid((thr - score) * 3.0)))
                return OpenWorldPrediction(
                    task_id=observation.task_id,
                    benchmark_id=observation.benchmark_id,
                    abstained=True,
                    abstain_reason="bm25_below_validation_threshold",
                    confidence=conf,
                    uncertainty=1.0 - conf,
                    memory_reads=int(info["reads"]),
                )
            ans = _extract_non_question_span(best_ev.text, observation.question_text, max_words=10)
            conf = max(0.2, min(0.9, _sigmoid((score - thr) * 2.5)))
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                predicted_answer=ans,
                abstained=False,
                confidence=conf,
                uncertainty=1.0 - conf,
                cited_evidence_ids=(best_ev.evidence_id,),
                cited_evidence_spans=(ans,),
                memory_reads=int(info["reads"]),
            )

        # long_memory_qa
        best_sess: ConversationSession | None = info.get("best_sess")
        if best_sess is None or score < thr:
            conf = max(0.5, min(0.9, _sigmoid((thr - score) * 3.0)))
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                abstained=True,
                abstain_reason="bm25_session_below_validation_threshold",
                confidence=conf,
                uncertainty=1.0 - conf,
                memory_reads=int(info["reads"]),
            )
        turn_text = str(info.get("best_turn_text", ""))
        ans = _extract_non_question_span(turn_text, observation.question_text, max_words=18)
        conf = max(0.2, min(0.9, _sigmoid((score - thr) * 2.5)))
        return OpenWorldPrediction(
            task_id=observation.task_id,
            benchmark_id=observation.benchmark_id,
            predicted_answer=ans if ans else turn_text[:160],
            abstained=False,
            confidence=conf,
            uncertainty=1.0 - conf,
            cited_evidence_ids=tuple(info.get("cited", ())),
            cited_evidence_spans=(ans[:160],) if ans else (),
            memory_reads=int(info["reads"]),
        )

    def memory_footprint(self) -> SolverMemoryFootprint:
        payload = {
            "idf": {b: len(v) for b, v in sorted(self._idf_by_bench.items())},
            "idf_tables": {b: sorted(v.items()) for b, v in sorted(self._idf_by_bench.items())},
            "copa_demos": len(self._copa_train_demos),
            "thresholds": self._abstain_threshold,
        }
        raw = canonical_json_bytes(payload)
        entry_count = sum(len(v) for v in self._idf_by_bench.values()) + len(self._copa_train_demos)
        return SolverMemoryFootprint(
            serialized_bytes=len(raw),
            compressed_mdl_bits=compressed_bit_length(raw),
            active_entry_count=entry_count,
        )


# ---------------------------------------------------------------------------
# Baseline 3: Existing Rule / Blueprint Baseline (Read-Only Staged Packs)
# ---------------------------------------------------------------------------


class ExistingRuleBlueprintBaseline(OpenWorldSolver):
    """Evaluates Mirror 7's existing staged rules & blueprints strictly read-only (zero promotion, zero vault writes)."""

    solver_id: str = "existing_rule_blueprint"
    category: str = "existing_rule_blueprint"

    def __init__(self, seed: int = 101, repo_root: Path = REPO_ROOT) -> None:
        super().__init__(seed=seed)
        self._repo_root = Path(repo_root)
        self._composer = BlueprintCompositionEngine()
        self._entries: list[dict[str, Any]] = []
        self._inv_index: dict[str, list[int]] = defaultdict(list)
        self._pack_counts: dict[str, int] = {}
        self._load_readonly_packs()

    def _load_readonly_packs(self) -> None:
        pack_specs = [
            (
                "staged_blueprints_997",
                self._repo_root / "rules_and_blueprints" / "blueprints" / "staging" / "mirror_997_unambiguous_prompt_candidates.jsonl",
            ),
            (
                "authored_domain_seed_110",
                self._repo_root / "research" / "rule_packs" / "mirror_domain_seed_v1.jsonl",
            ),
            (
                "general_reasoning_source_supported_26",
                self._repo_root / "rules_and_blueprints" / "general_reasoning" / "rules" / "general_reasoning_source_supported_26.jsonl",
            ),
        ]
        for pack_name, path in pack_specs:
            count = 0
            if not path.is_file():
                self._pack_counts[pack_name] = 0
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    row = json.loads(s)
                    prompt = str(
                        row.get("prompt")
                        or row.get("question")
                        or row.get("trigger")
                        or row.get("title")
                        or ""
                    ).strip()
                    response = str(
                        row.get("response")
                        or row.get("answer")
                        or row.get("statement")
                        or ""
                    ).strip()
                    if not prompt or not response:
                        continue
                    p_toks = frozenset(content_tokens(prompt))
                    if not p_toks:
                        continue
                    idx = len(self._entries)
                    self._entries.append(
                        {
                            "rule_id": str(row.get("id") or row.get("blueprint_id") or f"{pack_name}_{count}"),
                            "pack": pack_name,
                            "evidence_status": str(row.get("evidence_status") or (row.get("evidence") or {}).get("status") or "unverified"),
                            "prompt_tokens": p_toks,
                            "response_tokens": frozenset(content_tokens(response[:600])),
                            "response_head": response[:240],
                        }
                    )
                    for tok in p_toks:
                        self._inv_index[tok].append(idx)
                    count += 1
            self._pack_counts[pack_name] = count

    def fit(self, benchmark_id: str, train_examples: Sequence[OpenWorldTrainingExample]) -> None:
        self._require_unfrozen()
        # Existing rule/blueprint baseline evaluates existing packs without promoting new unverified rules.

    def _match_existing_rule(self, query_text: str) -> tuple[dict[str, Any] | None, float, int]:
        q_toks = frozenset(content_tokens(query_text))
        if not q_toks:
            return None, 0.0, 0
        cand_counts: Counter[int] = Counter()
        reads = 0
        for tok in q_toks:
            postings = self._inv_index.get(tok)
            if postings:
                reads += len(postings)
                for idx in postings:
                    cand_counts[idx] += 1
        best_entry: dict[str, Any] | None = None
        best_jacc = 0.0
        for idx, inter in cand_counts.most_common(12):
            entry = self._entries[idx]
            union = len(q_toks | entry["prompt_tokens"])
            jacc = inter / max(1, union)
            if jacc > best_jacc:
                best_jacc = jacc
                best_entry = entry
        return best_entry, best_jacc, reads

    def predict(self, observation: OpenWorldObservation) -> OpenWorldPrediction:
        entry, jacc, reads = self._match_existing_rule(observation.question_text)
        # Strict match gate: only apply a staged rule/blueprint if prompt Jaccard >= 0.35.
        if entry is None or jacc < 0.35:
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                abstained=True,
                abstain_reason="no_matching_existing_rule_or_blueprint",
                confidence=0.65,
                uncertainty=0.35,
                memory_reads=reads,
                metadata={
                    "best_rule_jaccard": round(jacc, 4),
                    "packs_checked": self._pack_counts,
                    "unverified_promoted": False,
                },
            )

        if observation.task_kind == "causal_choice" and len(observation.candidate_choices) >= 2:
            r_toks = entry["response_tokens"]
            s0 = len(r_toks & set(content_tokens(observation.candidate_choices[0])))
            s1 = len(r_toks & set(content_tokens(observation.candidate_choices[1])))
            if s0 == s1:
                return OpenWorldPrediction(
                    task_id=observation.task_id,
                    benchmark_id=observation.benchmark_id,
                    abstained=True,
                    abstain_reason="matched_rule_ambiguous_across_choices",
                    confidence=0.5,
                    uncertainty=0.5,
                    memory_reads=reads,
                )
            idx = 0 if s0 > s1 else 1
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                predicted_answer=observation.candidate_choices[idx],
                predicted_choice_index=idx,
                abstained=False,
                confidence=min(0.75, jacc),
                uncertainty=1.0 - min(0.75, jacc),
                cited_evidence_ids=(f"{observation.task_id}:premise", f"{observation.task_id}:choice_{idx}"),
                memory_reads=reads,
                metadata={"matched_rule_id": entry["rule_id"], "evidence_status": entry["evidence_status"]},
            )

        # For QA / memory tasks, rule packs do not ground dynamic unseen document/session evidence.
        return OpenWorldPrediction(
            task_id=observation.task_id,
            benchmark_id=observation.benchmark_id,
            predicted_answer=entry["response_head"],
            abstained=False,
            confidence=min(0.6, jacc),
            uncertainty=1.0 - min(0.6, jacc),
            cited_evidence_ids=(),
            memory_reads=reads,
            metadata={"matched_rule_id": entry["rule_id"], "evidence_status": entry["evidence_status"]},
        )

    def memory_footprint(self) -> SolverMemoryFootprint:
        summary = canonical_json_bytes(
            {
                "pack_counts": self._pack_counts,
                "vocab_size": len(self._inv_index),
                "rule_ids": [e["rule_id"] for e in self._entries],
            }
        )
        return SolverMemoryFootprint(
            serialized_bytes=len(summary),
            compressed_mdl_bits=compressed_bit_length(summary),
            active_entry_count=len(self._entries),
            external_readonly_pack_entries=len(self._entries),
            details={"packs": self._pack_counts, "readonly_unverified_never_promoted": True},
        )


# ---------------------------------------------------------------------------
# Baseline 4: Simple Symbolic Baseline (Domain-Agnostic CPU Symbolic Extractor)
# ---------------------------------------------------------------------------


class SimpleSymbolicBaseline(OpenWorldSolver):
    """Domain-agnostic CPU symbolic baseline: directed causal PMI, WH/entity/negation QA, and temporal recency memory."""

    solver_id: str = "simple_symbolic_extractor"
    category: str = "simple_symbolic"

    def __init__(self, seed: int = 101) -> None:
        super().__init__(seed=seed)
        self._idf_by_bench: dict[str, dict[str, float]] = {}
        # Directed token-pair causal log-odds learned strictly from Balanced COPA train split
        self._dir_pair_pos: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        self._dir_pair_neg: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        self._thresholds: dict[str, float] = {
            "balanced_copa": 0.05,
            "squad_v2": 0.45,
            "longmemeval": 0.35,
        }

    def fit(self, benchmark_id: str, train_examples: Sequence[OpenWorldTrainingExample]) -> None:
        self._require_unfrozen()
        df: Counter[str] = Counter()
        doc_count = 0

        if benchmark_id in {"balanced_copa", "smoke_causal"}:
            self._dir_pair_pos.clear()
            self._dir_pair_neg.clear()

        for ex in train_examples:
            obs = ex.observation
            if obs.task_kind == "causal_choice":
                p_toks = list(dict.fromkeys(content_tokens(obs.prompt_text)))
                for tok in p_toks:
                    df[tok] += 1
                doc_count += 1
                for idx, ch in enumerate(obs.candidate_choices):
                    c_toks = list(dict.fromkeys(content_tokens(ch)))
                    for tok in c_toks:
                        df[tok] += 1
                    doc_count += 1
                    if ex.target_choice_index is None:
                        continue
                    target_counter = (
                        self._dir_pair_pos[obs.relation_direction]
                        if idx == ex.target_choice_index
                        else self._dir_pair_neg[obs.relation_direction]
                    )
                    for pt in p_toks:
                        for ct in c_toks:
                            target_counter[(pt, ct)] += 1
            elif obs.task_kind == "extractive_qa":
                for ev in obs.evidence_units:
                    doc_count += 1
                    for tok in set(content_tokens(ev.text)):
                        df[tok] += 1
            else:
                for sess in obs.sessions:
                    doc_count += 1
                    for tok in set(content_tokens(_flatten_session_text(sess))):
                        df[tok] += 1

        self._idf_by_bench[benchmark_id] = {
            tok: math.log(1.0 + (doc_count - c + 0.5) / (c + 0.5))
            for tok, c in df.items()
        }

    @staticmethod
    def _detect_wh_type(question: str) -> str:
        ql = question.strip().casefold()
        if re.search(r"\b(?:when|what year|which year|what date|in what century)\b", ql):
            return "temporal"
        if re.search(r"\b(?:how many|how much|what percentage|what number|how old|how long)\b", ql):
            return "numeric"
        if re.search(r"\b(?:who|whose|whom)\b", ql):
            return "person_entity"
        if re.search(r"\b(?:where|which city|what city|which country|what country)\b", ql):
            return "location_entity"
        return "general"

    def _score_causal(self, obs: OpenWorldObservation) -> dict[str, Any]:
        idf = self._idf_by_bench.get(obs.benchmark_id, {})
        p_toks = list(dict.fromkeys(content_tokens(obs.prompt_text)))
        p_set = set(p_toks)
        p_neg = bool(set(tokenize(obs.prompt_text)) & _NEGATION_WORDS)
        pos_tbl = self._dir_pair_pos.get(obs.relation_direction, Counter())
        neg_tbl = self._dir_pair_neg.get(obs.relation_direction, Counter())

        scores: list[float] = []
        reads = 0
        for ch in obs.candidate_choices:
            c_toks = list(dict.fromkeys(content_tokens(ch)))
            c_set = set(c_toks)
            # 1. IDF-weighted shared entity/lexical overlap
            shared = sum(idf.get(t, 1.0) for t in (p_set & c_set))
            # 2. Directed causal PMI log-odds strictly from training pairs
            pmi = 0.0
            for pt in p_toks:
                for ct in c_toks:
                    reads += 1
                    pos_c = pos_tbl.get((pt, ct), 0)
                    neg_c = neg_tbl.get((pt, ct), 0)
                    if pos_c > 0 or neg_c > 0:
                        pmi += math.log((pos_c + 1.0) / (neg_c + 1.0))
            # 3. Stem/prefix morphological overlap (len >= 4)
            morph = 0.0
            for pt in p_toks:
                if len(pt) < 4:
                    continue
                for ct in c_toks:
                    if len(ct) >= 4 and pt[:4] == ct[:4] and pt != ct:
                        morph += 0.35
            c_neg = bool(set(tokenize(ch)) & _NEGATION_WORDS)
            neg_penalty = -0.1 if (p_neg != c_neg and shared == 0.0 and pmi == 0.0) else 0.0
            scores.append(shared + 1.5 * pmi + morph + neg_penalty)

        if len(scores) < 2:
            return {"score": 0.0, "best_idx": None, "reads": reads}
        best_idx = 0 if scores[0] >= scores[1] else 1
        return {"score": abs(scores[0] - scores[1]), "best_idx": best_idx, "reads": reads}

    def _score_extractive_qa(self, obs: OpenWorldObservation) -> dict[str, Any]:
        idf = self._idf_by_bench.get(obs.benchmark_id, {})
        q_toks = set(content_tokens(obs.question_text))
        if not q_toks or not obs.evidence_units:
            return {"score": 0.0, "best_ev": None, "span": "", "reads": len(obs.evidence_units)}

        wh_type = self._detect_wh_type(obs.question_text)
        q_caps = {m.group(0) for m in _CAP_SPAN_RE.finditer(obs.question_text[1:])}
        q_nums = set(_NUM_RE.findall(obs.question_text))
        q_neg = bool(set(tokenize(obs.question_text)) & _NEGATION_WORDS)
        q_weight = sum(idf.get(t, 1.0) for t in q_toks) or 1.0

        best_score = -1.0
        best_ev: EvidenceSentence | None = None
        best_span = ""

        for ev in obs.evidence_units:
            s_toks = set(content_tokens(ev.text))
            lex_cov = sum(idf.get(t, 1.0) for t in (q_toks & s_toks)) / q_weight

            # Entity / number grounding checks catch SQuAD v2 entity-swap unanswerables!
            s_caps = {m.group(0) for m in _CAP_SPAN_RE.finditer(ev.text)}
            s_nums = set(_NUM_RE.findall(ev.text))
            s_neg = bool(set(tokenize(ev.text)) & _NEGATION_WORDS)

            ent_penalty = 0.0
            if q_caps:
                cap_hits = sum(1 for c in q_caps if c.casefold() in ev.text.casefold())
                if cap_hits == 0:
                    ent_penalty -= 0.25
                else:
                    ent_penalty += 0.12 * (cap_hits / len(q_caps))
            if q_nums:
                num_hits = sum(1 for n in q_nums if n in ev.text)
                if num_hits == 0:
                    ent_penalty -= 0.25
            if q_neg != s_neg:
                ent_penalty -= 0.15

            # WH-type candidate extraction & compatibility bonus
            wh_bonus = 0.0
            cand_span = ""
            if wh_type == "temporal":
                years = [y for y in _YEAR_RE.findall(ev.text) if y not in q_nums]
                if years:
                    cand_span = years[0]
                    wh_bonus = 0.20
                else:
                    wh_bonus = -0.20
            elif wh_type == "numeric":
                nums = [n for n in _NUM_RE.findall(ev.text) if n not in q_nums]
                if nums:
                    cand_span = nums[0]
                    wh_bonus = 0.20
                else:
                    wh_bonus = -0.20
            elif wh_type in {"person_entity", "location_entity"}:
                novel_caps = [c for c in s_caps if c.casefold() not in obs.question_text.casefold()]
                if novel_caps:
                    cand_span = novel_caps[0]
                    wh_bonus = 0.15
                else:
                    wh_bonus = -0.12

            if not cand_span:
                cand_span = _extract_non_question_span(ev.text, obs.question_text, max_words=9)

            total_score = lex_cov + ent_penalty + wh_bonus
            if total_score > best_score:
                best_score = total_score
                best_ev = ev
                best_span = cand_span

        return {
            "score": max(0.0, best_score),
            "best_ev": best_ev,
            "span": best_span,
            "reads": len(obs.evidence_units),
        }

    def _score_long_memory(self, obs: OpenWorldObservation) -> dict[str, Any]:
        idf = self._idf_by_bench.get(obs.benchmark_id, {})
        q_toks = set(content_tokens(obs.question_text))
        if not q_toks or not obs.sessions:
            return {"score": 0.0, "cited": (), "span": "", "reads": len(obs.sessions)}

        q_raw_toks = set(tokenize(obs.question_text))
        wants_recency = bool(q_raw_toks & _UPDATE_CUES)
        wants_temporal = bool(q_raw_toks & _TEMPORAL_ORDER_CUES)
        q_weight = sum(idf.get(t, 1.0) for t in q_toks) or 1.0

        # Order sessions chronologically by parsed date ordinal.
        ordered = sorted(
            obs.sessions,
            key=lambda s: (_parse_date_ordinal(s.session_date), s.session_id),
        )
        n_sess = len(ordered)
        scored: list[tuple[float, float, int, ConversationSession, str]] = []

        for rank_idx, sess in enumerate(ordered):
            recency_frac = (rank_idx + 1) / max(1, n_sess)
            best_turn_cov = 0.0
            best_turn_text = ""
            sess_toks: set[str] = set()
            has_update_cue = False

            for turn in sess.turns:
                t_toks = set(content_tokens(turn.content))
                sess_toks.update(t_toks)
                if set(tokenize(turn.content)) & _UPDATE_CUES:
                    has_update_cue = True
                cov = sum(idf.get(t, 1.0) for t in (q_toks & t_toks)) / q_weight
                if cov > best_turn_cov:
                    best_turn_cov = cov
                    best_turn_text = turn.content

            sess_cov = sum(idf.get(t, 1.0) for t in (q_toks & sess_toks)) / q_weight
            base_cov = 0.55 * sess_cov + 0.45 * best_turn_cov
            bonus = 0.0
            if base_cov > 0.15:
                if wants_recency or has_update_cue:
                    bonus += 0.18 * recency_frac
                if wants_temporal:
                    bonus += 0.05 * recency_frac
            scored.append((base_cov + bonus, base_cov, rank_idx, sess, best_turn_text))

        scored.sort(key=lambda item: (-item[0], -item[2]))
        top_total, top_cov, _, top_sess, top_turn = scored[0]
        cited = [top_sess.session_id]
        if len(scored) > 1 and scored[1][1] >= 0.70 * top_cov and top_cov > 0.15:
            cited.append(scored[1][3].session_id)

        # For explicit "which came first" temporal questions with 2 matched sessions, emit earliest session snippet.
        span_source = top_turn
        if wants_temporal and "first" in q_raw_toks and len(scored) > 1 and scored[1][1] >= 0.65 * top_cov:
            earliest = min(scored[:2], key=lambda item: item[2])
            span_source = earliest[4]

        ans = _extract_non_question_span(span_source, obs.question_text, max_words=16)
        return {
            "score": max(0.0, top_cov),
            "cited": tuple(cited),
            "span": ans if ans else span_source[:160],
            "reads": len(obs.sessions),
        }

    def calibrate(
        self, benchmark_id: str, val_examples: Sequence[OpenWorldTrainingExample]
    ) -> None:
        self._require_unfrozen()
        if not val_examples:
            return
        kind = val_examples[0].observation.task_kind
        if kind == "causal_choice":
            scored = [(ex, self._score_causal(ex.observation)) for ex in val_examples]
            best_thr, best_acc = 0.01, -1.0
            for thr in (0.0, 0.01, 0.05, 0.15, 0.35):
                hits = sum(
                    1
                    for ex, info in scored
                    if info["score"] >= thr and info["best_idx"] == ex.target_choice_index
                )
                acc = hits / len(scored)
                if acc > best_acc:
                    best_acc, best_thr = acc, thr
            self._thresholds[benchmark_id] = best_thr
            return

        scorer = self._score_extractive_qa if kind == "extractive_qa" else self._score_long_memory
        scored = [(ex, scorer(ex.observation)) for ex in val_examples]
        raw_scores = sorted(set(round(info["score"], 4) for _, info in scored))
        if not raw_scores:
            return
        grid = raw_scores[:: max(1, len(raw_scores) // 40)]
        n_ans = sum(1 for ex, _ in scored if not ex.is_unanswerable)
        n_unans = sum(1 for ex, _ in scored if ex.is_unanswerable)
        if n_ans == 0 or n_unans == 0:
            self._thresholds[benchmark_id] = raw_scores[0] * 0.5
            return
        best_thr = raw_scores[len(raw_scores) // 2]
        best_j = -2.0
        for thr in grid:
            tpr_ans = sum(
                1 for ex, info in scored if (not ex.is_unanswerable) and info["score"] >= thr
            ) / n_ans
            tnr_unans = sum(
                1 for ex, info in scored if ex.is_unanswerable and info["score"] < thr
            ) / n_unans
            j_stat = tpr_ans + tnr_unans - 1.0
            if j_stat > best_j + 1e-9:
                best_j = j_stat
                best_thr = thr
        self._thresholds[benchmark_id] = best_thr

    def predict(self, observation: OpenWorldObservation) -> OpenWorldPrediction:
        thr = self._thresholds.get(observation.benchmark_id, 0.35)

        if observation.task_kind == "causal_choice":
            info = self._score_causal(observation)
            score = float(info["score"])
            if info["best_idx"] is None or score < thr:
                return OpenWorldPrediction(
                    task_id=observation.task_id,
                    benchmark_id=observation.benchmark_id,
                    abstained=True,
                    abstain_reason="symbolic_causal_margin_below_threshold",
                    confidence=0.5,
                    uncertainty=0.5,
                    cited_evidence_ids=(f"{observation.task_id}:premise",),
                    memory_reads=int(info["reads"]),
                )
            idx = int(info["best_idx"])
            conf = max(0.5, min(0.92, _sigmoid(score * 0.8)))
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                predicted_answer=observation.candidate_choices[idx],
                predicted_choice_index=idx,
                abstained=False,
                confidence=conf,
                uncertainty=1.0 - conf,
                cited_evidence_ids=(f"{observation.task_id}:premise", f"{observation.task_id}:choice_{idx}"),
                cited_evidence_spans=(observation.candidate_choices[idx],),
                memory_reads=int(info["reads"]),
            )

        if observation.task_kind == "extractive_qa":
            info = self._score_extractive_qa(observation)
            score = float(info["score"])
            best_ev: EvidenceSentence | None = info.get("best_ev")
            if best_ev is None or score < thr:
                conf = max(0.5, min(0.92, _sigmoid((thr - score) * 3.0)))
                return OpenWorldPrediction(
                    task_id=observation.task_id,
                    benchmark_id=observation.benchmark_id,
                    abstained=True,
                    abstain_reason="symbolic_entity_or_wh_constraint_unmet",
                    confidence=conf,
                    uncertainty=1.0 - conf,
                    memory_reads=int(info["reads"]),
                )
            span = str(info.get("span", ""))
            conf = max(0.25, min(0.92, _sigmoid((score - thr) * 3.0)))
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                predicted_answer=span,
                abstained=False,
                confidence=conf,
                uncertainty=1.0 - conf,
                cited_evidence_ids=(best_ev.evidence_id,),
                cited_evidence_spans=(span,),
                memory_reads=int(info["reads"]),
            )

        info = self._score_long_memory(observation)
        score = float(info["score"])
        cited = tuple(info.get("cited", ()))
        if not cited or score < thr:
            conf = max(0.5, min(0.92, _sigmoid((thr - score) * 3.0)))
            return OpenWorldPrediction(
                task_id=observation.task_id,
                benchmark_id=observation.benchmark_id,
                abstained=True,
                abstain_reason="insufficient_entity_grounding_in_sessions",
                confidence=conf,
                uncertainty=1.0 - conf,
                memory_reads=int(info["reads"]),
            )
        span = str(info.get("span", ""))
        conf = max(0.25, min(0.92, _sigmoid((score - thr) * 3.0)))
        return OpenWorldPrediction(
            task_id=observation.task_id,
            benchmark_id=observation.benchmark_id,
            predicted_answer=span,
            abstained=False,
            confidence=conf,
            uncertainty=1.0 - conf,
            cited_evidence_ids=cited,
            cited_evidence_spans=(span[:160],) if span else (),
            memory_reads=int(info["reads"]),
        )

    def memory_footprint(self) -> SolverMemoryFootprint:
        payload = {
            "idf_tables": {b: sorted(v.items()) for b, v in sorted(self._idf_by_bench.items())},
            "dir_pair_pos": {d: sorted((f"{a}|{b}", c) for (a, b), c in cnt.items()) for d, cnt in sorted(self._dir_pair_pos.items())},
            "dir_pair_neg": {d: sorted((f"{a}|{b}", c) for (a, b), c in cnt.items()) for d, cnt in sorted(self._dir_pair_neg.items())},
            "thresholds": self._thresholds,
        }
        raw = canonical_json_bytes(payload)
        entries = (
            sum(len(v) for v in self._idf_by_bench.values())
            + sum(len(c) for c in self._dir_pair_pos.values())
            + sum(len(c) for c in self._dir_pair_neg.values())
        )
        return SolverMemoryFootprint(
            serialized_bytes=len(raw),
            compressed_mdl_bits=compressed_bit_length(raw),
            active_entry_count=entries,
        )
