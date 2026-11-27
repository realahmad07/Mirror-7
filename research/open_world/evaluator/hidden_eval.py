"""EVALUATOR-ONLY sealed benchmark construction keeping answers, labels, and gold structures outside the learner.

ISOLATION CONTRACT:
- Every item parsed from raw external datasets is immediately split into:
  1. Public ``OpenWorldObservation`` (ZERO gold answers, labels, ``is_impossible``, ``has_answer``,
     ``answer_session_ids``, or ``mirrored`` flags).
  2. Hidden ``HiddenGoldRecord`` stored strictly inside ``SealedBenchmarkBundle.hidden_gold``.
- Training/validation supervision (``OpenWorldTrainingExample``) is built ONLY for ``train`` and
  ``validation`` splits — NEVER for ``hidden_eval``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any

from research.open_world.contracts import (
    ConversationSession,
    ConversationTurn,
    EvidenceSentence,
    OpenWorldObservation,
    OpenWorldTrainingExample,
    canonical_json_bytes,
    sha256_hex,
    split_sentences_with_offsets,
)
from research.open_world.datasets.loader import (
    DEFAULT_CACHE_DIR,
    load_balanced_copa_raw,
    load_longmemeval_raw,
    load_squad_v2_raw,
)
from research.open_world.datasets.splitter import (
    SourceDisjointSplitResult,
    build_connected_components,
    partition_source_clusters,
)


@dataclass(frozen=True)
class HiddenGoldRecord:
    """EVALUATOR-ONLY ground truth kept strictly outside learners and baselines."""

    task_id: str
    benchmark_id: str
    split: str
    task_kind: str
    source_cluster: str
    category: str
    gold_choice_index: int | None = None
    gold_answers: tuple[str, ...] = ()
    is_unanswerable: bool = False
    gold_evidence_ids: tuple[str, ...] = ()
    gold_secondary_evidence_ids: tuple[str, ...] = ()  # e.g. turn-level has_answer IDs
    pair_cluster_id: str = ""
    is_mirrored: bool = False


@dataclass(frozen=True)
class SealedBenchmarkBundle:
    """Complete sealed benchmark with public splits separated from evaluator-only hidden gold."""

    benchmark_id: str
    task_kind: str
    source_metadata: dict[str, Any]
    split_result: SourceDisjointSplitResult
    train_examples: tuple[OpenWorldTrainingExample, ...]
    val_examples: tuple[OpenWorldTrainingExample, ...]
    val_observations: tuple[OpenWorldObservation, ...]
    hidden_observations: tuple[OpenWorldObservation, ...]
    hidden_gold: dict[str, HiddenGoldRecord]  # EVALUATOR-ONLY (covers val + hidden_eval + train)
    manifest_digest: dict[str, Any]


def _stable_hash(seed_str: str) -> str:
    return hashlib.sha256(seed_str.encode("utf-8")).hexdigest()


def _bundle_manifest_digest(
    benchmark_id: str,
    split_res: SourceDisjointSplitResult,
    train_examples: tuple[OpenWorldTrainingExample, ...],
    val_observations: tuple[OpenWorldObservation, ...],
    hidden_observations: tuple[OpenWorldObservation, ...],
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "train_count": len(train_examples),
        "val_count": len(val_observations),
        "hidden_eval_count": len(hidden_observations),
        "train_ids_sha256": sha256_hex(canonical_json_bytes([e.observation.task_id for e in train_examples])),
        "val_ids_sha256": sha256_hex(canonical_json_bytes([o.task_id for o in val_observations])),
        "hidden_ids_sha256": sha256_hex(canonical_json_bytes([o.task_id for o in hidden_observations])),
        "train_clusters_sha256": sha256_hex(canonical_json_bytes(list(split_res.train_clusters))),
        "val_clusters_sha256": sha256_hex(canonical_json_bytes(list(split_res.val_clusters))),
        "hidden_clusters_sha256": sha256_hex(canonical_json_bytes(list(split_res.hidden_clusters))),
        "disjointness_verified": split_res.disjointness_verified,
        "disjointness_report": split_res.disjointness_report,
    }


# ---------------------------------------------------------------------------
# 1. Balanced COPA (Real-World Causal Commonsense, CC-BY-4.0)
# ---------------------------------------------------------------------------


def build_balanced_copa_bundle(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    allow_download: bool = True,
) -> SealedBenchmarkBundle:
    """Build source-disjoint Balanced COPA bundle (cluster-disjoint train/val + official test hidden)."""
    train_rows, test_rows, src_meta = load_balanced_copa_raw(
        cache_dir=cache_dir, allow_download=allow_download
    )

    item_to_resources: dict[str, list[str]] = {}
    row_by_id: dict[str, tuple[str, dict[str, str]]] = {}

    for row in train_rows:
        tid = f"bcopa_tr_{row['id']}"
        c_pair = "|".join(sorted((row["choice1"].strip().casefold(), row["choice2"].strip().casefold())))
        row_by_id[tid] = ("train_pool", row)
        item_to_resources[tid] = [f"pair:{c_pair}", f"prem:{row['premise'].strip().casefold()}"]

    for row in test_rows:
        tid = f"bcopa_te_{row['id']}"
        c_pair = "|".join(sorted((row["choice1"].strip().casefold(), row["choice2"].strip().casefold())))
        row_by_id[tid] = ("official_test", row)
        item_to_resources[tid] = [f"pair:{c_pair}", f"prem:{row['premise'].strip().casefold()}"]

    item_to_cluster = build_connected_components(item_to_resources)
    fixed_hidden_clusters = {
        item_to_cluster[tid]
        for tid, (origin, _) in row_by_id.items()
        if origin == "official_test"
    }
    cluster_strata: dict[str, str] = {}
    for tid, (_, row) in row_by_id.items():
        cid = item_to_cluster[tid]
        cluster_strata.setdefault(cid, str(row["question"]).strip().casefold())

    split_res = partition_source_clusters(
        "balanced_copa",
        item_to_cluster,
        train_ratio=0.70,
        val_ratio=0.30,
        salt="s11.0-bcopa-v1",
        cluster_strata=cluster_strata,
        fixed_hidden_clusters=fixed_hidden_clusters,
        item_to_resources=item_to_resources,
    )

    split_lookup: dict[str, str] = {}
    for tid in split_res.train_item_ids:
        split_lookup[tid] = "train"
    for tid in split_res.val_item_ids:
        split_lookup[tid] = "validation"
    for tid in split_res.hidden_item_ids:
        split_lookup[tid] = "hidden_eval"

    train_examples: list[OpenWorldTrainingExample] = []
    val_examples: list[OpenWorldTrainingExample] = []
    val_obs: list[OpenWorldObservation] = []
    hidden_obs: list[OpenWorldObservation] = []
    hidden_gold: dict[str, HiddenGoldRecord] = {}

    for tid in sorted(row_by_id.keys()):
        _, row = row_by_id[tid]
        split = split_lookup[tid]
        premise = str(row["premise"]).strip()
        c0 = str(row["choice1"]).strip()
        c1 = str(row["choice2"]).strip()
        qdir = str(row["question"]).strip().casefold()  # "cause" or "effect"
        label = int(row["label"])
        mirrored = str(row.get("mirrored", "False")).strip().casefold() == "true"

        ev_units = (
            EvidenceSentence(evidence_id=f"{tid}:premise", text=premise, char_start=0, char_end=len(premise)),
            EvidenceSentence(evidence_id=f"{tid}:choice_0", text=c0, char_start=0, char_end=len(c0)),
            EvidenceSentence(evidence_id=f"{tid}:choice_1", text=c1, char_start=0, char_end=len(c1)),
        )
        question_prompt = (
            f"What was the CAUSE of this event? {premise}"
            if qdir == "cause"
            else f"What happened as a RESULT (EFFECT) of this event? {premise}"
        )
        obs = OpenWorldObservation(
            task_id=tid,
            benchmark_id="balanced_copa",
            split=split,
            task_kind="causal_choice",
            prompt_text=premise,
            question_text=question_prompt,
            candidate_choices=(c0, c1),
            evidence_units=ev_units,
            relation_direction=qdir,
            public_metadata=(("relation_direction", qdir),),
        )
        gold_ev = (f"{tid}:premise", f"{tid}:choice_{label}")
        gold = HiddenGoldRecord(
            task_id=tid,
            benchmark_id="balanced_copa",
            split=split,
            task_kind="causal_choice",
            source_cluster=item_to_cluster[tid],
            category=qdir,
            gold_choice_index=label,
            gold_answers=(c0 if label == 0 else c1,),
            is_unanswerable=False,
            gold_evidence_ids=gold_ev,
            pair_cluster_id=item_to_cluster[tid],
            is_mirrored=mirrored,
        )
        hidden_gold[tid] = gold

        if split == "train":
            train_examples.append(
                OpenWorldTrainingExample(
                    observation=obs,
                    target_choice_index=label,
                    target_answer=c0 if label == 0 else c1,
                    is_unanswerable=False,
                    supporting_evidence_ids=gold_ev,
                )
            )
        elif split == "validation":
            val_obs.append(obs)
            val_examples.append(
                OpenWorldTrainingExample(
                    observation=obs,
                    target_choice_index=label,
                    target_answer=c0 if label == 0 else c1,
                    is_unanswerable=False,
                    supporting_evidence_ids=gold_ev,
                )
            )
        else:
            hidden_obs.append(obs)

    tr_t = tuple(train_examples)
    va_e = tuple(val_examples)
    va_o = tuple(val_obs)
    hi_o = tuple(hidden_obs)
    return SealedBenchmarkBundle(
        benchmark_id="balanced_copa",
        task_kind="causal_choice",
        source_metadata=src_meta,
        split_result=split_res,
        train_examples=tr_t,
        val_examples=va_e,
        val_observations=va_o,
        hidden_observations=hi_o,
        hidden_gold=hidden_gold,
        manifest_digest=_bundle_manifest_digest("balanced_copa", split_res, tr_t, va_o, hi_o),
    )


# ---------------------------------------------------------------------------
# 2. SQuAD v2 (Real-World Evidence-Grounded QA + Unanswerable Abstention, CC-BY-SA-4.0)
# ---------------------------------------------------------------------------


def build_squad_v2_bundle(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    allow_download: bool = True,
    per_article_answerable_cap: int = 15,
    per_article_unanswerable_cap: int = 15,
) -> SealedBenchmarkBundle:
    """Build 100% Wikipedia-article-disjoint SQuAD 2.0 bundle with balanced answerable/unanswerable items."""
    articles, src_meta = load_squad_v2_raw(cache_dir=cache_dir, allow_download=allow_download)

    selected_items: list[dict[str, Any]] = []
    item_to_cluster: dict[str, str] = {}
    item_to_resources: dict[str, list[str]] = {}

    for art in sorted(articles, key=lambda a: str(a.get("title", ""))):
        title = str(art["title"])
        ans_pool: list[dict[str, Any]] = []
        unans_pool: list[dict[str, Any]] = []
        for p_idx, para in enumerate(art.get("paragraphs") or []):
            context = str(para.get("context", ""))
            for qa in para.get("qas") or []:
                rec = {
                    "task_id": f"sq2_{qa['id']}",
                    "title": title,
                    "para_idx": p_idx,
                    "context": context,
                    "question": str(qa["question"]).strip(),
                    "is_impossible": bool(qa.get("is_impossible", False)),
                    "answers": list(qa.get("answers") or []),
                }
                if rec["is_impossible"] or not rec["answers"]:
                    unans_pool.append(rec)
                else:
                    ans_pool.append(rec)

        ans_pool.sort(key=lambda r: _stable_hash(f"s11.0-sq2-ans:{r['task_id']}"))
        unans_pool.sort(key=lambda r: _stable_hash(f"s11.0-sq2-unans:{r['task_id']}"))
        chosen = ans_pool[:per_article_answerable_cap] + unans_pool[:per_article_unanswerable_cap]
        for rec in chosen:
            selected_items.append(rec)
            cluster = f"wiki_article:{title}"
            item_to_cluster[rec["task_id"]] = cluster
            item_to_resources[rec["task_id"]] = [cluster]

    split_res = partition_source_clusters(
        "squad_v2",
        item_to_cluster,
        train_ratio=0.40,
        val_ratio=0.20,
        salt="s11.0-squadv2-v1",
        item_to_resources=item_to_resources,
    )
    split_lookup: dict[str, str] = {}
    for tid in split_res.train_item_ids:
        split_lookup[tid] = "train"
    for tid in split_res.val_item_ids:
        split_lookup[tid] = "validation"
    for tid in split_res.hidden_item_ids:
        split_lookup[tid] = "hidden_eval"

    train_examples: list[OpenWorldTrainingExample] = []
    val_examples: list[OpenWorldTrainingExample] = []
    val_obs: list[OpenWorldObservation] = []
    hidden_obs: list[OpenWorldObservation] = []
    hidden_gold: dict[str, HiddenGoldRecord] = {}

    for rec in sorted(selected_items, key=lambda r: r["task_id"]):
        tid = rec["task_id"]
        split = split_lookup[tid]
        context = rec["context"]
        sent_spans = split_sentences_with_offsets(context)
        ev_units = tuple(
            EvidenceSentence(
                evidence_id=f"{tid}:s{idx}",
                text=stext,
                char_start=cstart,
                char_end=cend,
            )
            for idx, (cstart, cend, stext) in enumerate(sent_spans)
        )
        obs = OpenWorldObservation(
            task_id=tid,
            benchmark_id="squad_v2",
            split=split,
            task_kind="extractive_qa",
            prompt_text=context,
            question_text=rec["question"],
            evidence_units=ev_units,
            public_metadata=(("sentence_count", str(len(ev_units))),),
        )

        is_unans = bool(rec["is_impossible"] or not rec["answers"])
        gold_answers: list[str] = []
        gold_ev_ids: set[str] = set()
        if not is_unans:
            for ans in rec["answers"]:
                atext = str(ans.get("text", "")).strip()
                if atext:
                    gold_answers.append(atext)
                astart = int(ans.get("answer_start", -1))
                aend = astart + len(atext)
                for ev in ev_units:
                    if astart >= 0 and not (aend <= ev.char_start or astart >= ev.char_end):
                        gold_ev_ids.add(ev.evidence_id)
            if not gold_ev_ids and ev_units and gold_answers:
                # Fallback substring match if offset falls on leading whitespace.
                first_low = gold_answers[0].casefold()
                for ev in ev_units:
                    if first_low in ev.text.casefold():
                        gold_ev_ids.add(ev.evidence_id)
                        break

        dedup_answers = tuple(dict.fromkeys(gold_answers))
        gold_ev_tuple = tuple(sorted(gold_ev_ids))
        gold = HiddenGoldRecord(
            task_id=tid,
            benchmark_id="squad_v2",
            split=split,
            task_kind="extractive_qa",
            source_cluster=item_to_cluster[tid],
            category="unanswerable" if is_unans else "answerable",
            gold_answers=dedup_answers,
            is_unanswerable=is_unans,
            gold_evidence_ids=gold_ev_tuple,
        )
        hidden_gold[tid] = gold

        if split == "train":
            train_examples.append(
                OpenWorldTrainingExample(
                    observation=obs,
                    target_answer=dedup_answers[0] if dedup_answers else None,
                    is_unanswerable=is_unans,
                    supporting_evidence_ids=gold_ev_tuple,
                )
            )
        elif split == "validation":
            val_obs.append(obs)
            val_examples.append(
                OpenWorldTrainingExample(
                    observation=obs,
                    target_answer=dedup_answers[0] if dedup_answers else None,
                    is_unanswerable=is_unans,
                    supporting_evidence_ids=gold_ev_tuple,
                )
            )
        else:
            hidden_obs.append(obs)

    tr_t = tuple(train_examples)
    va_e = tuple(val_examples)
    va_o = tuple(val_obs)
    hi_o = tuple(hidden_obs)
    src_meta = dict(src_meta)
    src_meta["sampling_rule"] = (
        f"balanced deterministic subsample up to {per_article_answerable_cap} answerable + "
        f"{per_article_unanswerable_cap} unanswerable per Wikipedia article across all 35 articles"
    )
    return SealedBenchmarkBundle(
        benchmark_id="squad_v2",
        task_kind="extractive_qa",
        source_metadata=src_meta,
        split_result=split_res,
        train_examples=tr_t,
        val_examples=va_e,
        val_observations=va_o,
        hidden_observations=hi_o,
        hidden_gold=hidden_gold,
        manifest_digest=_bundle_manifest_digest("squad_v2", split_res, tr_t, va_o, hi_o),
    )


# ---------------------------------------------------------------------------
# 3. LongMemEval (Real-World Long-Term Multi-Session Memory, MIT)
# ---------------------------------------------------------------------------


def _build_public_session(session_id: str, session_date: str, raw_turns: list[Any]) -> ConversationSession:
    """Convert raw LongMemEval session turns into strictly public ConversationTurn objects.

    CRITICAL: Deliberately strips ``has_answer`` and any other hidden annotation fields.
    """
    turns: list[ConversationTurn] = []
    for idx, turn in enumerate(raw_turns):
        if isinstance(turn, dict):
            role = str(turn.get("role", "user")).strip()
            content = str(turn.get("content", "")).strip()
        else:
            role = "user"
            content = str(turn).strip()
        turns.append(
            ConversationTurn(
                turn_id=f"{session_id}::t{idx}",
                session_id=session_id,
                turn_index=idx,
                role=role,
                content=content,
            )
        )
    return ConversationSession(
        session_id=session_id,
        session_date=str(session_date or "").strip(),
        turns=tuple(turns),
    )


def build_longmemeval_bundle(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    allow_download: bool = True,
    distractor_sessions_per_episode: int = 10,
) -> SealedBenchmarkBundle:
    """Build session-disjoint LongMemEval bundle with split-internal chronological distractor haystacks."""
    episodes, src_meta = load_longmemeval_raw(cache_dir=cache_dir, allow_download=allow_download)

    ep_by_id: dict[str, dict[str, Any]] = {}
    item_to_resources: dict[str, list[str]] = {}
    for ep in episodes:
        qid = str(ep["question_id"])
        ep_by_id[qid] = ep
        sids = [str(s) for s in (ep.get("haystack_session_ids") or [])]
        item_to_resources[qid] = sids if sids else [f"nosess:{qid}"]

    # Connected-component clustering guarantees the 8 shared session IDs NEVER cross splits.
    item_to_cluster = build_connected_components(item_to_resources)
    cluster_strata: dict[str, str] = {}
    for qid, ep in sorted(ep_by_id.items()):
        cid = item_to_cluster[qid]
        is_abs = qid.endswith("_abs")
        strat = f"{ep.get('question_type', 'unknown')}|{'abs' if is_abs else 'ans'}"
        cluster_strata.setdefault(cid, strat)

    split_res = partition_source_clusters(
        "longmemeval",
        item_to_cluster,
        train_ratio=0.40,
        val_ratio=0.20,
        salt="s11.0-longmemeval-v1",
        cluster_strata=cluster_strata,
        item_to_resources=item_to_resources,
    )
    split_lookup: dict[str, str] = {}
    for qid in split_res.train_item_ids:
        split_lookup[qid] = "train"
    for qid in split_res.val_item_ids:
        split_lookup[qid] = "validation"
    for qid in split_res.hidden_item_ids:
        split_lookup[qid] = "hidden_eval"

    # Build split-internal session pools (STRICTLY isolated per split: zero cross-split distractors!).
    split_session_pools: dict[str, list[tuple[str, str, ConversationSession]]] = {
        "train": [],
        "validation": [],
        "hidden_eval": [],
    }
    seen_sid_per_split: dict[str, set[str]] = {"train": set(), "validation": set(), "hidden_eval": set()}

    for qid in sorted(ep_by_id.keys()):
        ep = ep_by_id[qid]
        split = split_lookup[qid]
        cid = item_to_cluster[qid]
        raw_sids = ep.get("haystack_session_ids") or []
        raw_dates = ep.get("haystack_dates") or []
        raw_sessions = ep.get("haystack_sessions") or []
        for idx, (sid, raw_turns) in enumerate(zip(raw_sids, raw_sessions)):
            sid_s = str(sid)
            if sid_s in seen_sid_per_split[split]:
                continue
            seen_sid_per_split[split].add(sid_s)
            sdate = str(raw_dates[idx]) if idx < len(raw_dates) else ""
            pub_sess = _build_public_session(sid_s, sdate, list(raw_turns or []))
            split_session_pools[split].append((cid, sid_s, pub_sess))

    # Hard check: zero session ID overlap across split pools.
    assert seen_sid_per_split["train"].isdisjoint(seen_sid_per_split["validation"])
    assert seen_sid_per_split["train"].isdisjoint(seen_sid_per_split["hidden_eval"])
    assert seen_sid_per_split["validation"].isdisjoint(seen_sid_per_split["hidden_eval"])

    train_examples: list[OpenWorldTrainingExample] = []
    val_examples: list[OpenWorldTrainingExample] = []
    val_obs: list[OpenWorldObservation] = []
    hidden_obs: list[OpenWorldObservation] = []
    hidden_gold: dict[str, HiddenGoldRecord] = {}

    for qid in sorted(ep_by_id.keys()):
        ep = ep_by_id[qid]
        split = split_lookup[qid]
        cid = item_to_cluster[qid]
        is_abs = qid.endswith("_abs")
        qtype = str(ep.get("question_type", "unknown"))
        qdate = str(ep.get("question_date", "")).strip()
        question = str(ep.get("question", "")).strip()
        answer = str(ep.get("answer", "")).strip()

        raw_sids = [str(s) for s in (ep.get("haystack_session_ids") or [])]
        raw_dates = [str(d) for d in (ep.get("haystack_dates") or [])]
        raw_sessions = list(ep.get("haystack_sessions") or [])

        native_sessions: list[ConversationSession] = []
        gold_turn_ids: list[str] = []
        native_sid_set: set[str] = set()
        for idx, (sid_s, raw_turns) in enumerate(zip(raw_sids, raw_sessions)):
            sdate = raw_dates[idx] if idx < len(raw_dates) else ""
            native_sessions.append(_build_public_session(sid_s, sdate, list(raw_turns or [])))
            native_sid_set.add(sid_s)
            if not is_abs:
                for t_idx, turn in enumerate(raw_turns or []):
                    if isinstance(turn, dict) and turn.get("has_answer"):
                        gold_turn_ids.append(f"{sid_s}::t{t_idx}")

        # Deterministically sample split-internal distractor sessions from OTHER components in same split.
        candidate_distractors = [
            sess
            for other_cid, other_sid, sess in split_session_pools[split]
            if other_cid != cid and other_sid not in native_sid_set
        ]
        rng = random.Random(f"s11.0-lme-distractors:{split}:{qid}")
        rng.shuffle(candidate_distractors)
        chosen_distractors = candidate_distractors[:distractor_sessions_per_episode]

        # Combine and order chronologically by session_date (ties broken deterministically by session_id).
        combined_sessions = sorted(
            native_sessions + chosen_distractors,
            key=lambda s: (s.session_date, s.session_id),
        )

        obs = OpenWorldObservation(
            task_id=qid,
            benchmark_id="longmemeval",
            split=split,
            task_kind="long_memory_qa",
            prompt_text=f"Question date: {qdate}. Sessions available: {len(combined_sessions)}.",
            question_text=question,
            sessions=tuple(combined_sessions),
            question_date=qdate,
            public_metadata=(
                ("question_type", qtype),
                ("session_count", str(len(combined_sessions))),
            ),
        )

        # For genuinely unanswerable *_abs questions, gold evidence is empty (answer is absent!).
        gold_session_ids = (
            ()
            if is_abs
            else tuple(sorted(str(s) for s in (ep.get("answer_session_ids") or [])))
        )
        gold = HiddenGoldRecord(
            task_id=qid,
            benchmark_id="longmemeval",
            split=split,
            task_kind="long_memory_qa",
            source_cluster=cid,
            category=qtype,
            gold_answers=() if is_abs else (answer,),
            is_unanswerable=is_abs,
            gold_evidence_ids=gold_session_ids,
            gold_secondary_evidence_ids=tuple(sorted(gold_turn_ids)),
        )
        hidden_gold[qid] = gold

        if split == "train":
            train_examples.append(
                OpenWorldTrainingExample(
                    observation=obs,
                    target_answer=None if is_abs else answer,
                    is_unanswerable=is_abs,
                    supporting_evidence_ids=gold_session_ids,
                )
            )
        elif split == "validation":
            val_obs.append(obs)
            val_examples.append(
                OpenWorldTrainingExample(
                    observation=obs,
                    target_answer=None if is_abs else answer,
                    is_unanswerable=is_abs,
                    supporting_evidence_ids=gold_session_ids,
                )
            )
        else:
            hidden_obs.append(obs)

    tr_t = tuple(train_examples)
    va_e = tuple(val_examples)
    va_o = tuple(val_obs)
    hi_o = tuple(hidden_obs)
    src_meta = dict(src_meta)
    src_meta["haystack_construction"] = (
        f"split-internal chronological distractor augmentation (+{distractor_sessions_per_episode} "
        "distractor sessions strictly from unrelated connected components inside the same split; "
        "0 cross-split session overlap)"
    )
    return SealedBenchmarkBundle(
        benchmark_id="longmemeval",
        task_kind="long_memory_qa",
        source_metadata=src_meta,
        split_result=split_res,
        train_examples=tr_t,
        val_examples=va_e,
        val_observations=va_o,
        hidden_observations=hi_o,
        hidden_gold=hidden_gold,
        manifest_digest=_bundle_manifest_digest("longmemeval", split_res, tr_t, va_o, hi_o),
    )


# ---------------------------------------------------------------------------
# 4. Offline Clean Synthetic Smoke Fixture (for zero-network unit/CI tests)
# ---------------------------------------------------------------------------


def build_offline_smoke_bundles() -> dict[str, SealedBenchmarkBundle]:
    """Build tiny license-free synthetic in-memory fixtures mirroring all 3 task shapes for offline tests."""
    # 1. Tiny causal choice fixture
    copa_rows = [
        ("smoke_c1", "cl_a", "train", "cause", "The glass fell onto the tile floor.", "The glass shattered.", "The floor was painted.", 0),
        ("smoke_c2", "cl_b", "train", "effect", "Heavy rain flooded the street.", "Cars stopped moving.", "The sun melted snow.", 0),
        ("smoke_c3", "cl_c", "train", "cause", "The bread burned in the oven.", "The timer failed to ring.", "The bakery painted walls.", 0),
        ("smoke_c4", "cl_d", "validation", "effect", "The power cable snapped during the storm.", "The lights went out.", "The garden bloomed.", 0),
        ("smoke_c5", "cl_e", "hidden_eval", "cause", "The ice cube melted on the counter.", "The room was warm.", "The freezer broke a hinge.", 0),
        ("smoke_c6", "cl_f", "hidden_eval", "effect", "She forgot her umbrella in the rain.", "Her coat got soaked.", "She solved an equation.", 0),
    ]
    c_tr: list[OpenWorldTrainingExample] = []
    c_ve: list[OpenWorldTrainingExample] = []
    c_vo: list[OpenWorldObservation] = []
    c_ho: list[OpenWorldObservation] = []
    c_hg: dict[str, HiddenGoldRecord] = {}
    c_i2c: dict[str, str] = {}
    for tid, cl, sp, qd, prem, o0, o1, lbl in copa_rows:
        c_i2c[tid] = cl
        evs = (
            EvidenceSentence(f"{tid}:premise", prem, 0, len(prem)),
            EvidenceSentence(f"{tid}:choice_0", o0, 0, len(o0)),
            EvidenceSentence(f"{tid}:choice_1", o1, 0, len(o1)),
        )
        obs = OpenWorldObservation(
            task_id=tid,
            benchmark_id="smoke_causal",
            split=sp,
            task_kind="causal_choice",
            prompt_text=prem,
            question_text=f"What was the {qd.upper()}? {prem}",
            candidate_choices=(o0, o1),
            evidence_units=evs,
            relation_direction=qd,
            public_metadata=(("relation_direction", qd),),
        )
        gev = (f"{tid}:premise", f"{tid}:choice_{lbl}")
        c_hg[tid] = HiddenGoldRecord(
            task_id=tid,
            benchmark_id="smoke_causal",
            split=sp,
            task_kind="causal_choice",
            source_cluster=cl,
            category=qd,
            gold_choice_index=lbl,
            gold_answers=(o0 if lbl == 0 else o1,),
            gold_evidence_ids=gev,
            pair_cluster_id=cl,
        )
        ex = OpenWorldTrainingExample(obs, target_choice_index=lbl, target_answer=o0 if lbl == 0 else o1, supporting_evidence_ids=gev)
        if sp == "train":
            c_tr.append(ex)
        elif sp == "validation":
            c_ve.append(ex)
            c_vo.append(obs)
        else:
            c_ho.append(obs)
    c_sr = SourceDisjointSplitResult(
        benchmark_id="smoke_causal",
        train_item_ids=tuple(e.observation.task_id for e in c_tr),
        val_item_ids=tuple(o.task_id for o in c_vo),
        hidden_item_ids=tuple(o.task_id for o in c_ho),
        train_clusters=("cl_a", "cl_b", "cl_c"),
        val_clusters=("cl_d",),
        hidden_clusters=("cl_e", "cl_f"),
        disjointness_verified=True,
        disjointness_report={"zero_leakage_verified": True},
    )

    # 2. Tiny extractive QA + unanswerable fixture
    qa_specs = [
        ("smoke_q1", "art_1", "train", "Cobalt River flows north across Veridia. Its bridge opened in 1984.", "When did the Cobalt River bridge open?", "1984", False, 1),
        ("smoke_q2", "art_1", "train", "Cobalt River flows north across Veridia. Its bridge opened in 1984.", "Who painted the Cobalt River bridge?", None, True, -1),
        ("smoke_q3", "art_2", "validation", "Dr. Lina Patel founded the Solis Observatory in 2012. It tracks comets.", "Who founded the Solis Observatory?", "Dr. Lina Patel", False, 0),
        ("smoke_q4", "art_2", "validation", "Dr. Lina Patel founded the Solis Observatory in 2012. It tracks comets.", "How many submarines does Solis Observatory operate?", None, True, -1),
        ("smoke_q5", "art_3", "hidden_eval", "The Meridian Archive stores 420 maps. Curator Omar Reed cataloged them in 2019.", "How many maps does the Meridian Archive store?", "420 maps", False, 0),
        ("smoke_q6", "art_3", "hidden_eval", "The Meridian Archive stores 420 maps. Curator Omar Reed cataloged them in 2019.", "What is the melting point of titanium on Jupiter?", None, True, -1),
    ]
    q_tr: list[OpenWorldTrainingExample] = []
    q_ve: list[OpenWorldTrainingExample] = []
    q_vo: list[OpenWorldObservation] = []
    q_ho: list[OpenWorldObservation] = []
    q_hg: dict[str, HiddenGoldRecord] = {}
    for tid, art, sp, ctx, qtxt, ans, unans, sidx in qa_specs:
        spans = split_sentences_with_offsets(ctx)
        evs = tuple(EvidenceSentence(f"{tid}:s{i}", st, cs, ce) for i, (cs, ce, st) in enumerate(spans))
        obs = OpenWorldObservation(
            task_id=tid,
            benchmark_id="smoke_qa",
            split=sp,
            task_kind="extractive_qa",
            prompt_text=ctx,
            question_text=qtxt,
            evidence_units=evs,
        )
        gev = () if unans else (f"{tid}:s{sidx}",)
        q_hg[tid] = HiddenGoldRecord(
            task_id=tid,
            benchmark_id="smoke_qa",
            split=sp,
            task_kind="extractive_qa",
            source_cluster=art,
            category="unanswerable" if unans else "answerable",
            gold_answers=() if unans else (str(ans),),
            is_unanswerable=unans,
            gold_evidence_ids=gev,
        )
        ex = OpenWorldTrainingExample(obs, target_answer=ans, is_unanswerable=unans, supporting_evidence_ids=gev)
        if sp == "train":
            q_tr.append(ex)
        elif sp == "validation":
            q_ve.append(ex)
            q_vo.append(obs)
        else:
            q_ho.append(obs)
    q_sr = SourceDisjointSplitResult(
        benchmark_id="smoke_qa",
        train_item_ids=tuple(e.observation.task_id for e in q_tr),
        val_item_ids=tuple(o.task_id for o in q_vo),
        hidden_item_ids=tuple(o.task_id for o in q_ho),
        train_clusters=("art_1",),
        val_clusters=("art_2",),
        hidden_clusters=("art_3",),
        disjointness_verified=True,
        disjointness_report={"zero_leakage_verified": True},
    )

    # 3. Tiny long-memory fixture (knowledge-update + unanswerable _abs)
    def _sess(sid: str, dt: str, text: str) -> ConversationSession:
        return ConversationSession(
            session_id=sid,
            session_date=dt,
            turns=(ConversationTurn(f"{sid}::t0", sid, 0, "user", text),),
        )

    m_specs = [
        (
            "smoke_m1", "mc_1", "train", "single-session-user", False, "2024/05/10",
            "What musical instrument do I play?", "cello", ("s1",),
            (_sess("s0", "2024/05/01", "I cooked pasta tonight."), _sess("s1", "2024/05/04", "I practice cello every Sunday.")),
        ),
        (
            "smoke_m2", "mc_2", "validation", "knowledge-update", False, "2024/06/10",
            "Which city do I live in now?", "Lisbon", ("s3",),
            (_sess("s2", "2024/05/02", "I used to live in Madrid."), _sess("s3", "2024/06/02", "Actually I moved to Lisbon last week.")),
        ),
        (
            "smoke_m3", "mc_3", "hidden_eval", "knowledge-update", False, "2024/07/10",
            "What color is my bicycle now?", "green", ("s5",),
            (_sess("s4", "2024/06/01", "My bicycle was blue."), _sess("s5", "2024/07/01", "I painted my bicycle green yesterday."), _sess("s6", "2024/06/15", "I read a book about astronomy.")),
        ),
        (
            "smoke_m4_abs", "mc_4", "hidden_eval", "single-session-user", True, "2024/07/12",
            "What is the name of my pet iguana?", "", (),
            (_sess("s7", "2024/07/02", "I went hiking in the mountains."), _sess("s8", "2024/07/05", "I bought new running shoes.")),
        ),
    ]
    m_tr: list[OpenWorldTrainingExample] = []
    m_ve: list[OpenWorldTrainingExample] = []
    m_vo: list[OpenWorldObservation] = []
    m_ho: list[OpenWorldObservation] = []
    m_hg: dict[str, HiddenGoldRecord] = {}
    for qid, cl, sp, cat, unans, qdt, qtxt, ans, g_sids, sess_t in m_specs:
        obs = OpenWorldObservation(
            task_id=qid,
            benchmark_id="smoke_memory",
            split=sp,
            task_kind="long_memory_qa",
            prompt_text=f"Question date: {qdt}",
            question_text=qtxt,
            sessions=sess_t,
            question_date=qdt,
            public_metadata=(("question_type", cat),),
        )
        m_hg[qid] = HiddenGoldRecord(
            task_id=qid,
            benchmark_id="smoke_memory",
            split=sp,
            task_kind="long_memory_qa",
            source_cluster=cl,
            category=cat,
            gold_answers=() if unans else (ans,),
            is_unanswerable=unans,
            gold_evidence_ids=g_sids,
            gold_secondary_evidence_ids=tuple(f"{s}::t0" for s in g_sids),
        )
        ex = OpenWorldTrainingExample(obs, target_answer=None if unans else ans, is_unanswerable=unans, supporting_evidence_ids=g_sids)
        if sp == "train":
            m_tr.append(ex)
        elif sp == "validation":
            m_ve.append(ex)
            m_vo.append(obs)
        else:
            m_ho.append(obs)
    m_sr = SourceDisjointSplitResult(
        benchmark_id="smoke_memory",
        train_item_ids=tuple(e.observation.task_id for e in m_tr),
        val_item_ids=tuple(o.task_id for o in m_vo),
        hidden_item_ids=tuple(o.task_id for o in m_ho),
        train_clusters=("mc_1",),
        val_clusters=("mc_2",),
        hidden_clusters=("mc_3", "mc_4"),
        disjointness_verified=True,
        disjointness_report={"zero_leakage_verified": True},
    )

    return {
        "smoke_causal": SealedBenchmarkBundle(
            "smoke_causal", "causal_choice", {"license": "Apache-2.0 (synthetic smoke)"}, c_sr,
            tuple(c_tr), tuple(c_ve), tuple(c_vo), tuple(c_ho), c_hg,
            _bundle_manifest_digest("smoke_causal", c_sr, tuple(c_tr), tuple(c_vo), tuple(c_ho)),
        ),
        "smoke_qa": SealedBenchmarkBundle(
            "smoke_qa", "extractive_qa", {"license": "Apache-2.0 (synthetic smoke)"}, q_sr,
            tuple(q_tr), tuple(q_ve), tuple(q_vo), tuple(q_ho), q_hg,
            _bundle_manifest_digest("smoke_qa", q_sr, tuple(q_tr), tuple(q_vo), tuple(q_ho)),
        ),
        "smoke_memory": SealedBenchmarkBundle(
            "smoke_memory", "long_memory_qa", {"license": "Apache-2.0 (synthetic smoke)"}, m_sr,
            tuple(m_tr), tuple(m_ve), tuple(m_vo), tuple(m_ho), m_hg,
            _bundle_manifest_digest("smoke_memory", m_sr, tuple(m_tr), tuple(m_vo), tuple(m_ho)),
        ),
    }
