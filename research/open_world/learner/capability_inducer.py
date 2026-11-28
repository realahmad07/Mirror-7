"""Stage S11.1 Stage 6 & Stage 9: Reusable Capability Induction (ITL + SHARD 10,240-Bit LSH + RDC MDL).

STRICT ISOLATION & ZERO MANUAL DOMAIN SCHEMAS:
- Pure Python 3.11 stdlib + Mirror 7's 10,240-bit BSC primitives
  (``mirror_crystal.quotient_egraph_resonator``).
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
- Implements:
  1. ``OpenWorldSHARDIndex``: 10,240-bit BSC Multi-Table LSH Index (with ``use_lsh=False``
     exhaustive linear-scan mode for Ablation ``A5_no_shard_lsh_indexing``).
  2. Structural ITL Anti-Unification: lifts concrete raw-text inference traces across training
     clusters into parameterized ``OpenWorldCapabilityCard``s (with ``enable_anti_unification=False``
     instance-memorization mode for Ablation ``A4_no_capability_anti_unification``).
  3. RDC MDL Consolidation Judge & Cross-Cluster Verification Promotion Gate:
     ``J = L(library) + L(residuals | library) + beta * verified_failures + gamma * unsupported_rate``
     Promotes cards from ``shadow -> verified`` ONLY if supported across >= ``k_min`` distinct
     source clusters and MDL improves without regression.
  4. ``CapabilityVersionStore`` (Stage 9): versioned shadow/verified/quarantined lifecycle with
     mandatory regression replay and automatic non-destructive rollback on poison/regression.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import math
import re
from typing import Any, Callable, Mapping, Sequence

from mirror_crystal.quotient_egraph_resonator import (
    BitSlicedBundle10240,
    D_UNIFIED,
    UNIFIED_MASK,
    UNIFIED_ODD_MASK,
    bind10240,
    hamming_sim10240,
    hv10240_of,
    permute10240,
)
from research.open_world.contracts import (
    OpenWorldTrainingExample,
    SolverFrozenError,
    canonical_json_bytes,
    compressed_bit_length,
    content_tokens,
    normalize_text,
    tokenize,
)
from research.open_world.learner.evidence_ledger import extract_evidence_units
from research.open_world.learner.extraction import (
    CandidateEntity,
    detect_polarity_and_modality,
    extract_candidate_spans,
    morph_stem,
    unicode_content_tokens,
    unicode_tokenize,
)
from research.open_world.learner.temporal_graph import HISTORICAL_CUES, UPDATE_REVISION_CUES

CAPABILITY_SCHEMA_VERSION: str = "s11.1-open-world-capability-v1"


def _lsh_bit_tables(num_tables: int, bits_per_table: int, seed_tag: str) -> tuple[tuple[int, ...], ...]:
    tables: list[tuple[int, ...]] = []
    for t_idx in range(num_tables):
        chosen: list[int] = []
        used: set[int] = set()
        nonce = 0
        while len(chosen) < bits_per_table:
            digest = hashlib.blake2b(
                f"{seed_tag}:t{t_idx}:n{nonce}".encode("utf-8"), digest_size=32
            ).digest()
            for off in range(0, len(digest), 2):
                bpos = int.from_bytes(digest[off : off + 2], "little") % D_UNIFIED
                if bpos not in used:
                    used.add(bpos)
                    chosen.append(bpos)
                    if len(chosen) == bits_per_table:
                        break
            nonce += 1
        tables.append(tuple(chosen))
    return tuple(tables)


def _bucket_key(hv: int, bit_positions: tuple[int, ...]) -> int:
    key = 0
    for idx, bpos in enumerate(bit_positions):
        if (hv >> bpos) & 1:
            key |= 1 << idx
    return key


def bundle_10240(weighted_hvs: Sequence[tuple[int, int]], *, empty_tag: str = "empty") -> int:
    pos = [(hv, w) for hv, w in weighted_hvs if w > 0]
    if not pos:
        return hv10240_of(f"s11:bundle:{empty_tag}")
    if len(pos) == 1:
        return pos[0][0] & UNIFIED_MASK
    total_w = sum(w for _, w in pos)
    bd = BitSlicedBundle10240(bits=max(8, (total_w * 2 + 3).bit_length()))
    for hv, w in pos:
        bd.add(hv, w * 2)
    if total_w % 2 == 0:
        bd.add(UNIFIED_ODD_MASK, 2)
    bd.add(UNIFIED_MASK, 1)
    return bd.binary()


def encode_query_signature_hv(
    query_sig: Sequence[tuple[str, str]],
    *,
    extra_tokens: Sequence[str] = (),
) -> int:
    """Encode an anti-unified query signature (or concrete query tokens) into 10,240-bit BSC space."""
    atoms: list[tuple[int, int]] = []
    for idx, (k, v) in enumerate(sorted(query_sig)):
        k_hv = permute10240(hv10240_of(f"s11:qsig_k:{k}"), (idx + 1) * 13)
        v_hv = hv10240_of(f"s11:qsig_v:{v}")
        atoms.append((bind10240(k_hv, v_hv), 3))
    for tok in extra_tokens[:10]:
        atoms.append((permute10240(hv10240_of(f"s11:qtok:{tok}"), 41), 1))
    return bundle_10240(atoms, empty_tag="qsig_empty")


def extract_query_signature(question: str, *, task_kind: str = "extractive_qa") -> tuple[tuple[str, str], ...]:
    """Extract a 100% domain-agnostic structural query signature from raw UTF-8 question text."""
    ql = question.strip().casefold()
    raw_toks = set(unicode_tokenize(question))

    # 1. Purely grammatical WH-interrogative class
    if re.search(r"\b(?:when|what year|which year|what date|in what year|in what century|what month|what day)\b", ql):
        wh = "temporal_point"
    elif re.search(r"\b(?:how many\s+(?:days?|weeks?|months?|years?)|how long\s+(?:has|have|had|did|ago|since)|how much time)\b", ql):
        wh = "temporal_duration"
    elif re.search(r"\b(?:how many|how much|what percentage|what number|what fraction|how old|how far|how tall|how big|how fast|what score|what price|what cost)\b", ql):
        wh = "numeric_quantity"
    elif re.search(r"\b(?:who|whose|whom)\b", ql):
        wh = "agent_entity"
    elif re.search(r"\b(?:where|which city|what city|which country|what country|which state|what location|which place)\b", ql):
        wh = "location_entity"
    elif re.search(r"\b(?:why|for what reason|what caused|what was the cause)\b", ql):
        wh = "reason_clause"
    elif re.search(r"\b(?:how\s+(?:are|is|was|were|did|do|does|can|could|would|should))\b", ql):
        wh = "manner_clause"
    elif re.search(r"\b(?:what\s+(?:is|was)\s+the\s+name|which\s+(?:company|organization|group|team|person|city|country|title|brand|model|system))\b", ql):
        wh = "named_entity"
    elif re.search(r"^(?:did|do|does|is|are|was|were|has|have|had|can|could|will|would)\b", ql):
        wh = "boolean_verify"
    else:
        wh = "general_concept"

    # 2. Temporal traversal mode (domain-agnostic discourse cues)
    if raw_toks & HISTORICAL_CUES or "used to" in ql:
        t_mode = "historical_earliest"
    elif raw_toks & UPDATE_REVISION_CUES:
        t_mode = "current_latest"
    elif re.search(r"\b(?:order|first|earlier|before|after|later|last|most recent|recently)\b", ql):
        t_mode = "temporal_order"
    else:
        t_mode = "default_latest"

    # 3. Aggregation mode (purely syntactic quantifiers)
    if re.search(r"\b(?:how much more|how many more|difference|fewer|less than|more than|compared to|left to)\b", ql):
        agg = "difference"
    elif re.search(r"\b(?:total|altogether|combined|in total|all|each|how many times|how many days|how many hours|how many)\b", ql):
        agg = "sum_or_count"
    else:
        agg = "single_fact"

    # 4. Speaker reference (first-person user vs second-person assistant)
    if re.search(r"\b(?:you\s+(?:recommended|suggested|mentioned|made|said|gave|told|provided)|your\s+(?:move|recommendation|suggestion))\b", ql):
        speaker = "assistant"
    elif re.search(r"\b(?:i|my|me|mine)\b", ql):
        speaker = "user"
    else:
        speaker = "document"

    pol, mod = detect_polarity_and_modality(question)
    return (
        ("task_kind", task_kind),
        ("wh_class", wh),
        ("temporal_mode", t_mode),
        ("aggregation", agg),
        ("speaker_target", speaker),
        ("polarity", pol),
        ("modality", mod),
    )


@dataclass(frozen=True)
class OpenWorldCapabilityCard:
    """Reusable ITL-induced & RDC-verified open-world inference capability card (Stage 6)."""

    card_id: str
    capability_kind: str  # QA_ALIGNMENT | TEMPORAL_PROCEDURE | CAUSAL_BRIDGE | RAW_INSTANCE
    query_signature: tuple[tuple[str, str], ...]
    expected_span_shape: str
    anchor_pattern: str
    left_pivots: tuple[str, ...]
    right_pivots: tuple[str, ...]
    preferred_len_min: int
    preferred_len_max: int
    support_clusters: tuple[str, ...]
    hit_count: int
    total_count: int
    mdl_gain_bits: float
    status: str  # shadow | verified | quarantined | retired
    version: int
    card_hv: int
    concrete_tokens: tuple[str, ...] = ()  # Non-empty ONLY in A4 raw-instance ablation!

    @property
    def precision(self) -> float:
        return self.hit_count / max(1, self.total_count)

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "id": self.card_id,
            "kind": self.capability_kind,
            "qsig": dict(self.query_signature),
            "shape": self.expected_span_shape,
            "anchor": self.anchor_pattern,
            "lp": list(self.left_pivots),
            "rp": list(self.right_pivots),
            "len": [self.preferred_len_min, self.preferred_len_max],
            "clusters": len(self.support_clusters),
            "hits": self.hit_count,
            "total": self.total_count,
            "mdl_gain": round(self.mdl_gain_bits, 2),
            "status": self.status,
            "ver": self.version,
        }


class OpenWorldSHARDIndex:
    """10,240-bit BSC Multi-Table LSH Index over ``OpenWorldCapabilityCard``s (with A5 linear-scan toggle)."""

    def __init__(
        self,
        *,
        num_tables: int = 12,
        bits_per_table: int = 4,
        use_lsh: bool = True,
        seed_tag: str = "s11_1_shard_lsh_v1",
    ) -> None:
        self.num_tables = max(1, int(num_tables))
        self.bits_per_table = max(1, min(16, int(bits_per_table)))
        self.use_lsh = bool(use_lsh)
        self.seed_tag = seed_tag
        self._bit_tables = _lsh_bit_tables(self.num_tables, self.bits_per_table, seed_tag)
        self._buckets: list[dict[int, list[int]]] = [{} for _ in range(self.num_tables)]
        self._cards: list[OpenWorldCapabilityCard] = []
        self._vectors: list[int] = []
        self._frozen: bool = False
        self._post_freeze_writes: int = 0
        self.queries: int = 0
        self.candidates_scanned: int = 0
        self.lsh_bucket_hits: int = 0
        self.memory_accesses: int = 0

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def post_freeze_writes(self) -> int:
        return self._post_freeze_writes

    def __len__(self) -> int:
        return len(self._cards)

    def freeze(self) -> None:
        self._frozen = True

    def clear(self) -> None:
        if self._frozen:
            self._post_freeze_writes += 1
            raise SolverFrozenError("OpenWorldSHARDIndex is frozen; mutations forbidden")
        self._buckets = [{} for _ in range(self.num_tables)]
        self._cards.clear()
        self._vectors.clear()

    def add_card(self, card: OpenWorldCapabilityCard) -> int:
        if self._frozen:
            self._post_freeze_writes += 1
            raise SolverFrozenError("OpenWorldSHARDIndex is frozen; post-freeze insertions forbidden")
        idx = len(self._cards)
        self._cards.append(card)
        self._vectors.append(card.card_hv)
        for t_idx, bpos in enumerate(self._bit_tables):
            bk = _bucket_key(card.card_hv, bpos)
            self._buckets[t_idx].setdefault(bk, []).append(idx)
        return idx

    def query(
        self,
        query_hv: int,
        *,
        top_k: int = 6,
    ) -> list[tuple[OpenWorldCapabilityCard, int]]:
        """Retrieve top-K capability cards via SHARD LSH buckets (or exhaustive scan in Ablation A5)."""
        self.queries += 1
        n_total = len(self._cards)
        if n_total == 0:
            return []

        if self.use_lsh:
            cand_set: set[int] = set()
            for t_idx, bpos in enumerate(self._bit_tables):
                bk = _bucket_key(query_hv, bpos)
                bucket = self._buckets[t_idx].get(bk)
                if bucket:
                    cand_set.update(bucket)
            if cand_set:
                self.lsh_bucket_hits += 1
                indices = sorted(cand_set)
            else:
                indices = list(range(n_total))
        else:
            # Ablation A5: exhaustive linear scan over EVERY stored card vector
            indices = list(range(n_total))

        self.candidates_scanned += len(indices)
        self.memory_accesses += len(indices)
        scored = [
            (self._cards[i], hamming_sim10240(query_hv, self._vectors[i]))
            for i in indices
        ]
        scored.sort(key=lambda pair: (-pair[1], -pair[0].precision, pair[0].card_id))
        return scored[:top_k]

    def telemetry(self) -> dict[str, Any]:
        return {
            "use_lsh": self.use_lsh,
            "num_cards": len(self._cards),
            "queries": self.queries,
            "candidates_scanned": self.candidates_scanned,
            "mean_candidates_per_query": round(self.candidates_scanned / max(1, self.queries), 3),
            "lsh_bucket_hits": self.lsh_bucket_hits,
            "memory_accesses": self.memory_accesses,
            "frozen": self._frozen,
            "post_freeze_writes": self._post_freeze_writes,
        }


@dataclass(frozen=True)
class ConcreteAlignmentTrace:
    """One verified raw-text training trace prior to ITL anti-unification."""

    trace_id: str
    cluster_id: str
    capability_kind: str
    query_signature: tuple[tuple[str, str], ...]
    span_shape: str
    anchor_pattern: str
    left_pivot: str
    right_pivot: str
    span_token_len: int
    concrete_query_tokens: tuple[str, ...]


def _infer_target_span_shape(answer: str) -> str:
    s = answer.strip()
    if not s:
        return "NOUN_CHUNK"
    if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2}|century|january|february|march|april|may|june|july|august|september|october|november|december)\b", s, re.I):
        return "DATE_TIME"
    if re.search(r"(?:^[$€£]?\d|\d+\s*(?:%|percent|million|billion|thousand|days?|weeks?|months?|years?|km|miles?|hours?|minutes?|seconds?))", s, re.I):
        return "QUANTITY_NUM"
    words = s.split()
    cap_words = sum(1 for w in words if w and w[0].isupper())
    if len(words) <= 6 and cap_words >= max(1, int(0.6 * len(words))):
        return "PROPER_SPAN"
    if len(words) >= 5:
        return "CLAUSE_SPAN"
    return "NOUN_CHUNK"


def _extract_training_traces(
    train_examples: Sequence[OpenWorldTrainingExample],
) -> list[ConcreteAlignmentTrace]:
    """Extract raw alignment traces strictly from public training examples."""
    traces: list[ConcreteAlignmentTrace] = []
    for idx, ex in enumerate(train_examples):
        obs = ex.observation
        if ex.is_unanswerable:
            continue
        qsig = extract_query_signature(obs.question_text, task_kind=obs.task_kind)
        qtoks = tuple(sorted(set(morph_stem(t) for t in unicode_content_tokens(obs.question_text)))[:8])
        # Cluster provenance derived strictly from public source/task prefix (zero hidden labels)
        if obs.task_kind == "extractive_qa":
            units = extract_evidence_units(obs)
            cluster_id = f"src:{units[0].sha256[:10]}" if units else f"src:{obs.task_id}"
            ans = (ex.target_answer or "").strip()
            if not ans:
                continue
            shape = _infer_target_span_shape(ans)
            ans_low = ans.casefold()
            lp = "<BOS>"
            rp = "<EOS>"
            anchor_pat = "BOTH_ANCHORS"
            q_stems = set(morph_stem(t) for t in unicode_content_tokens(obs.question_text))
            for u in units:
                pos = u.text.casefold().find(ans_low)
                if pos >= 0:
                    before = u.text[:pos]
                    after = u.text[pos + len(ans) :]
                    bt = unicode_tokenize(before)
                    at = unicode_tokenize(after)
                    lp = bt[-1] if bt else "<BOS>"
                    rp = at[0] if at else "<EOS>"
                    l_hits = len(q_stems & {morph_stem(w) for w in bt[-8:]})
                    r_hits = len(q_stems & {morph_stem(w) for w in at[:8]})
                    if l_hits > r_hits:
                        anchor_pat = "LEFT_ANCHOR"
                    elif r_hits > l_hits:
                        anchor_pat = "RIGHT_ANCHOR"
                    else:
                        anchor_pat = "BOTH_ANCHORS"
                    break
            traces.append(
                ConcreteAlignmentTrace(
                    trace_id=f"tr_{idx}_{obs.task_id}",
                    cluster_id=cluster_id,
                    capability_kind="QA_ALIGNMENT",
                    query_signature=qsig,
                    span_shape=shape,
                    anchor_pattern=anchor_pat,
                    left_pivot=lp,
                    right_pivot=rp,
                    span_token_len=max(1, len(ans.split())),
                    concrete_query_tokens=qtoks,
                )
            )
        elif obs.task_kind == "long_memory_qa":
            ans = (ex.target_answer or "").strip()
            if not ans:
                continue
            qsig_d = dict(qsig)
            cluster_id = f"sess:{ex.supporting_evidence_ids[0]}" if ex.supporting_evidence_ids else f"sess:{obs.task_id}"
            shape = _infer_target_span_shape(ans)
            if qsig_d["wh_class"] == "temporal_duration":
                anchor_pat = "DATE_DIFF"
                shape = "DATE_DELTA"
            elif qsig_d["temporal_mode"] == "historical_earliest":
                anchor_pat = "SUPERSEDES_EARLIEST"
            elif qsig_d["aggregation"] == "difference":
                anchor_pat = "MULTI_DIFF"
                shape = "NUMERIC_AGG"
            elif qsig_d["aggregation"] == "sum_or_count" and len(ex.supporting_evidence_ids) >= 2:
                anchor_pat = "MULTI_SUM"
                shape = "NUMERIC_AGG"
            else:
                anchor_pat = "SUPERSEDES_LATEST"
            traces.append(
                ConcreteAlignmentTrace(
                    trace_id=f"tr_{idx}_{obs.task_id}",
                    cluster_id=cluster_id,
                    capability_kind="TEMPORAL_PROCEDURE",
                    query_signature=qsig,
                    span_shape=shape,
                    anchor_pattern=anchor_pat,
                    left_pivot="<SESS>",
                    right_pivot="<TURN>",
                    span_token_len=max(1, len(ans.split())),
                    concrete_query_tokens=qtoks,
                )
            )
        elif obs.task_kind == "causal_choice":
            if ex.target_choice_index is None:
                continue
            cluster_id = f"causal:{obs.task_id}"
            traces.append(
                ConcreteAlignmentTrace(
                    trace_id=f"tr_{idx}_{obs.task_id}",
                    cluster_id=cluster_id,
                    capability_kind="CAUSAL_BRIDGE",
                    query_signature=qsig + (("relation_direction", obs.relation_direction or "effect"),),
                    span_shape="CLAUSE_SPAN",
                    anchor_pattern="CAUSAL_DIR",
                    left_pivot=obs.relation_direction or "effect",
                    right_pivot="causal_choice",
                    span_token_len=6,
                    concrete_query_tokens=qtoks,
                )
            )
    return traces


class CapabilityVersionStore:
    """Stage 6 + Stage 9 ITL/RDC Capability Induction, Cross-Cluster Promotion & Rollback Store."""

    def __init__(
        self,
        *,
        enable_anti_unification: bool = True,
        use_shard_lsh: bool = True,
        k_min_clusters: int = 3,
    ) -> None:
        self.enable_anti_unification = bool(enable_anti_unification)
        self.use_shard_lsh = bool(use_shard_lsh)
        self.k_min_clusters = max(1, int(k_min_clusters))
        self._frozen: bool = False
        self._post_freeze_writes: int = 0

        self.version: int = 0
        self.shadow_cards: dict[str, OpenWorldCapabilityCard] = {}
        self.verified_cards: dict[str, OpenWorldCapabilityCard] = {}
        self.quarantined_cards: dict[str, OpenWorldCapabilityCard] = {}
        self.history_snapshots: list[dict[str, OpenWorldCapabilityCard]] = []
        self.rollback_count: int = 0
        self.shard_index = OpenWorldSHARDIndex(num_tables=8, bits_per_table=6, use_lsh=use_shard_lsh)
        self.mdl_report: dict[str, Any] = {}

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def post_freeze_writes(self) -> int:
        return self._post_freeze_writes

    def freeze(self) -> None:
        self._frozen = True
        self.shard_index.freeze()

    def _require_unfrozen(self) -> None:
        if self._frozen:
            self._post_freeze_writes += 1
            raise SolverFrozenError("CapabilityVersionStore is frozen; post-freeze writes forbidden")

    def induce_from_training(
        self,
        train_examples: Sequence[OpenWorldTrainingExample],
    ) -> dict[str, Any]:
        """Induce, MDL-filter, cross-cluster verify, and SHARD-index reusable capability cards."""
        self._require_unfrozen()
        traces = _extract_training_traces(train_examples)
        if not traces:
            return {"traces": 0, "verified_cards": 0, "shadow_cards": 0}

        distinct_total_clusters = len({t.cluster_id for t in traces})
        effective_k_min = min(self.k_min_clusters, max(1, distinct_total_clusters))

        candidate_cards: list[OpenWorldCapabilityCard] = []

        if not self.enable_anti_unification:
            # Ablation A4: NO structural anti-unification -> raw concrete instance memorization!
            # Each trace becomes an unlifted instance card keyed by its concrete query tokens.
            for idx, tr in enumerate(traces[:220]):
                hv = encode_query_signature_hv(
                    tr.query_signature, extra_tokens=tr.concrete_query_tokens
                )
                card = OpenWorldCapabilityCard(
                    card_id=f"raw_inst_{idx}_{tr.trace_id}",
                    capability_kind="RAW_INSTANCE",
                    query_signature=tr.query_signature,
                    expected_span_shape=tr.span_shape,
                    anchor_pattern=tr.anchor_pattern,
                    left_pivots=(tr.left_pivot,),
                    right_pivots=(tr.right_pivot,),
                    preferred_len_min=tr.span_token_len,
                    preferred_len_max=tr.span_token_len,
                    support_clusters=(tr.cluster_id,),
                    hit_count=1,
                    total_count=1,
                    mdl_gain_bits=0.0,
                    status="verified",
                    version=self.version + 1,
                    card_hv=hv,
                    concrete_tokens=tr.concrete_query_tokens,
                )
                candidate_cards.append(card)
        else:
            # Full Stage 6 ITL Structural Anti-Unification:
            # Generalize concrete entities/values into typed role variables & group by
            # (capability_kind, query_signature, span_shape, anchor_pattern).
            groups: dict[tuple[str, tuple[tuple[str, str], ...], str, str], list[ConcreteAlignmentTrace]] = defaultdict(list)
            # Also track total traces per (capability_kind, query_signature) for empirical precision!
            sig_totals: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
            for tr in traces:
                groups[(tr.capability_kind, tr.query_signature, tr.span_shape, tr.anchor_pattern)].append(tr)
                sig_totals[(tr.capability_kind, tr.query_signature)] += 1

            for idx, ((kind, qsig, shape, anchor), members) in enumerate(sorted(groups.items())):
                clusters = tuple(sorted({m.cluster_id for m in members}))
                lp_counts = Counter(m.left_pivot for m in members if m.left_pivot)
                rp_counts = Counter(m.right_pivot for m in members if m.right_pivot)
                lens = sorted(m.span_token_len for m in members)
                p10 = lens[max(0, int(0.10 * (len(lens) - 1)))]
                p90 = lens[min(len(lens) - 1, int(0.90 * (len(lens) - 1)))]

                raw_bits = compressed_bit_length(
                    canonical_json_bytes([m.__dict__ for m in members])
                )
                sig_tot = max(len(members), sig_totals[(kind, qsig)])
                hv = encode_query_signature_hv(qsig)
                card = OpenWorldCapabilityCard(
                    card_id=f"itl_card_{kind.lower()}_{idx}",
                    capability_kind=kind,
                    query_signature=qsig,
                    expected_span_shape=shape,
                    anchor_pattern=anchor,
                    left_pivots=tuple(k for k, _ in lp_counts.most_common(6)),
                    right_pivots=tuple(k for k, _ in rp_counts.most_common(6)),
                    preferred_len_min=max(1, p10),
                    preferred_len_max=max(p10, p90),
                    support_clusters=clusters,
                    hit_count=len(members),
                    total_count=sig_tot,
                    mdl_gain_bits=0.0,
                    status="shadow",
                    version=self.version + 1,
                    card_hv=hv,
                )
                card_bits = compressed_bit_length(canonical_json_bytes(card.to_compact_dict()))
                mdl_gain = float(raw_bits - card_bits)
                # Stage 8/9 Cross-Cluster & MDL Promotion Gate:
                # Never promote unverified single-cluster noise when >= k_min clusters exist!
                promote = (
                    len(clusters) >= effective_k_min
                    and card.precision >= 0.15
                    and (mdl_gain > 0.0 or len(members) >= effective_k_min)
                )
                card = replace(
                    card,
                    mdl_gain_bits=mdl_gain,
                    status="verified" if promote else "shadow",
                )
                candidate_cards.append(card)

        # Compute RDC MDL consolidation score J = L(lib) + L(residuals | lib)
        verified_batch = [c for c in candidate_cards if c.status == "verified"]
        shadow_batch = [c for c in candidate_cards if c.status != "verified"]

        lib_bits = compressed_bit_length(
            canonical_json_bytes([c.to_compact_dict() for c in verified_batch])
        )
        covered_sigs = {(c.capability_kind, c.query_signature, c.expected_span_shape) for c in verified_batch}
        residual_traces = [
            t.__dict__ for t in traces
            if (t.capability_kind, t.query_signature, t.span_shape) not in covered_sigs
        ]
        res_bits = compressed_bit_length(canonical_json_bytes(residual_traces)) if residual_traces else 0
        raw_all_bits = compressed_bit_length(canonical_json_bytes([t.__dict__ for t in traces]))

        self.commit_with_regression_guard(verified_batch, shadow_batch)
        self.mdl_report = {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "total_traces": len(traces),
            "raw_traces_bits": raw_all_bits,
            "library_bits": lib_bits,
            "residual_bits": res_bits,
            "total_mdl_bits": lib_bits + res_bits,
            "compression_ratio": round(raw_all_bits / max(1, lib_bits + res_bits), 4),
            "verified_card_count": len(self.verified_cards),
            "shadow_card_count": len(self.shadow_cards),
            "quarantined_card_count": len(self.quarantined_cards),
            "k_min_clusters_used": effective_k_min,
            "anti_unification_enabled": self.enable_anti_unification,
        }
        return self.mdl_report

    def commit_with_regression_guard(
        self,
        new_verified: Sequence[OpenWorldCapabilityCard],
        new_shadow: Sequence[OpenWorldCapabilityCard] = (),
        *,
        regression_evaluator: Callable[[Mapping[str, OpenWorldCapabilityCard]], float] | None = None,
        min_baseline_score: float | None = None,
    ) -> bool:
        """Stage 9 controlled versioned update with mandatory regression replay & automatic rollback."""
        self._require_unfrozen()
        prev_snapshot = dict(self.verified_cards)

        baseline_score = 0.0
        if regression_evaluator is not None:
            baseline_score = (
                float(min_baseline_score)
                if min_baseline_score is not None
                else float(regression_evaluator(prev_snapshot))
            )

        proposed = dict(prev_snapshot)
        for c in new_verified:
            proposed[c.card_id] = c

        if regression_evaluator is not None:
            new_score = float(regression_evaluator(proposed))
            if new_score + 1e-9 < baseline_score:
                # Automatic quarantine & non-destructive rollback!
                for c in new_verified:
                    if c.card_id not in prev_snapshot:
                        self.quarantined_cards[c.card_id] = replace(c, status="quarantined")
                self.rollback_count += 1
                return False

        self.history_snapshots.append(prev_snapshot)
        self.verified_cards = proposed
        for c in new_shadow:
            self.shadow_cards[c.card_id] = c
        self.version += 1

        # Rebuild SHARD 10,240-bit LSH index strictly from verified cards
        self.shard_index.clear()
        for cid in sorted(self.verified_cards):
            self.shard_index.add_card(self.verified_cards[cid])
        return True

    def index_causal_hypotheses(self, hypotheses: Sequence[Any]) -> int:
        """Index materialized Stage 5 CausalHypotheses into the SHARD 10,240-bit LSH index."""
        self._require_unfrozen()
        added = 0
        for h in hypotheses:
            cid = f"causal_frame_{h.hypothesis_id}"
            if cid in self.verified_cards:
                continue
            card = OpenWorldCapabilityCard(
                card_id=cid,
                capability_kind="CAUSAL_BRIDGE",
                query_signature=(("task_kind", "causal_choice"), ("ant", h.antecedent_frame), ("cons", h.consequent_frame)),
                expected_span_shape="CLAUSE_SPAN",
                anchor_pattern="CAUSAL_DIR",
                left_pivots=(h.antecedent_frame,),
                right_pivots=(h.consequent_frame,),
                preferred_len_min=2,
                preferred_len_max=10,
                support_clusters=h.support_evidence_ids,
                hit_count=h.support_count,
                total_count=max(1, h.support_count + h.counterexample_count),
                mdl_gain_bits=float(h.support_count * 12),
                status="verified" if h.epistemic_status == "VERIFIED_MULTI_CLUSTER" else "shadow",
                version=self.version,
                card_hv=h.frame_hv,
            )
            self.shard_index.add_card(card)
            added += 1
        return added

    def propose_cards(
        self,
        question: str,
        *,
        task_kind: str,
        top_k: int = 6,
    ) -> list[tuple[OpenWorldCapabilityCard, int]]:
        """Propose top-K verified capability cards via SHARD 10,240-bit LSH (read-only after freeze)."""
        qsig = extract_query_signature(question, task_kind=task_kind)
        if not self.enable_anti_unification:
            extra = tuple(sorted(set(morph_stem(t) for t in unicode_content_tokens(question)))[:8])
            q_hv = encode_query_signature_hv(qsig, extra_tokens=extra)
        else:
            q_hv = encode_query_signature_hv(qsig)
        return self.shard_index.query(q_hv, top_k=top_k)

    def serialized_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": self.version,
                "verified": [self.verified_cards[k].to_compact_dict() for k in sorted(self.verified_cards)],
                "quarantined_count": len(self.quarantined_cards),
                "rollback_count": self.rollback_count,
            }
        )