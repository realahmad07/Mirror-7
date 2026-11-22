"""S0 baseline solvers for the hidden relational simulator benchmark.

STRICT ISOLATION: This module MUST NEVER import ``hidden_actions``. All baselines
observe ONLY public ``TrainingEpisode`` demonstrations during ``fit()`` and public
``EvaluationTaskPublic`` inputs during ``predict()``.

Baselines implemented for S0:
1. ``NoOpIdentityBaseline`` — sanity floor predicting unchanged state.
2. ``EpisodeLookupBaseline`` — exact (state, action) memory store with honest abstention.
3. ``StructuralKNNBaseline`` — nearest-neighbor retrieval over state/action features.
4. ``SHARDHDCRetrievalBaseline`` — 10,240-bit BSC HDC retrieval reusing Mirror 7's
   ``mirror_crystal.quotient_egraph_resonator`` primitives (SHARD without ITL).
5. ``CurrentMirrorSolverBaseline`` — invokes Mirror 7's existing native reasoning organs
   (``ContextReasoner`` / ``native_chain`` + ``BilingualCCGHDCParser``) with zero vault writes.
6. ``SmallCPUNeuralReferenceBaseline`` — comparison-only tiny pure-Python MLP trained on
   the same 6-demo/action CPU budget (never a candidate Mirror learner).
"""
from __future__ import annotations

import hashlib
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Sequence

from mirror_crystal.cortical_active_inference import BilingualCCGHDCParser
from mirror_crystal.quotient_egraph_resonator import (
    D_UNIFIED,
    BitSlicedBundle10240,
    bind10240,
    hamming_sim10240,
    hv10240_of,
)
from mirror_crystal import native_intelligence as _mirror_ni
from mirror_eval.generalization.suts import Response as _MirrorResponse, _source_for as _mirror_source_for

from research.cpu_learning.evaluation.metrics import BitBudgetAudit, TaskPrediction
from research.cpu_learning.simulator.state_graph import (
    ActionInvocation,
    EvaluationTaskPublic,
    SparseDelta,
    StateGraph,
    TrainingEpisode,
    canonical_json_bytes,
    compressed_bit_length,
)


class BaseS0Solver(ABC):
    """Common interface for all S0 baselines and future S1-S10 candidate learners."""

    name: str = "base"
    category: str = "baseline"
    comparison_only: bool = False
    evaluator_only: bool = False

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._frozen = False

    def freeze(self) -> None:
        """Freeze solver parameters/memory before temporal holdout evaluation."""
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def _check_not_frozen(self) -> None:
        if self._frozen:
            raise RuntimeError(f"Solver {self.name!r} is frozen and cannot fit new episodes")

    @abstractmethod
    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        """Observe public training demonstrations only."""

    @abstractmethod
    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        """Predict final state (or abstain) given public task only."""

    def recovered_operator_schemas(self) -> dict[str, dict[str, Any]]:
        """Abstract operator schemas discovered by the solver (empty for non-ITL baselines)."""
        return {}

    @abstractmethod
    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        """Report explicit storage & MDL bit footprint."""


class NoOpIdentityBaseline(BaseS0Solver):
    """Sanity floor: always predicts zero state changes (never abstains)."""

    name = "noop_identity_floor"
    category = "sanity_floor"

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        self._check_not_frozen()

    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        steps = tuple(task.initial_state.copy() for _ in task.actions)
        return TaskPrediction(
            predicted_final_state=task.initial_state.copy(),
            abstained=False,
            retrieval_hits=0,
            memory_accesses=0,
            step_predictions=steps,
        )

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        return BitBudgetAudit(
            raw_training_data_bits=raw_training_bits,
            compressed_training_data_bits=compressed_training_bits,
            episode_store_bits=0,
            active_bundle_bits=0,
            operator_library_bits=0,
            compressed_model_mdl_bits=0,
            compression_ratio_vs_raw=0.0,
        )


class EpisodeLookupBaseline(BaseS0Solver):
    """Exact episode memory lookup. Abstains on any unseen (state, action) key."""

    name = "episode_lookup"
    category = "memory_baseline"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._store: dict[str, SparseDelta] = {}
        self._raw_store_bytes: bytes = b""

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        self._check_not_frozen()
        self._store = {ep.state_action_key(): ep.sparse_delta for ep in episodes}
        self._raw_store_bytes = canonical_json_bytes(
            {k: d.to_dict() for k, d in sorted(self._store.items())}
        )

    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        cur = task.initial_state.copy()
        step_preds: list[StateGraph | None] = []
        retrieval_hits = 0
        mem_accesses = 0

        for idx in range(len(task.actions)):
            key = task.step_state_action_key(cur, idx)
            mem_accesses += 1
            delta = self._store.get(key)
            if delta is None:
                return TaskPrediction(
                    predicted_final_state=None,
                    abstained=True,
                    retrieval_hits=retrieval_hits,
                    memory_accesses=mem_accesses,
                    step_predictions=tuple(step_preds),
                    metadata={"abstain_step": idx, "reason": "exact_key_miss"},
                )
            retrieval_hits += 1
            cur = delta.apply_to(cur)
            step_preds.append(cur)

        return TaskPrediction(
            predicted_final_state=cur,
            abstained=False,
            retrieval_hits=retrieval_hits,
            memory_accesses=mem_accesses,
            step_predictions=tuple(step_preds),
        )

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        if not self._store:
            ep_bits = 0
            mdl_bits = 0
        else:
            ep_bits = len(self._raw_store_bytes) * 8
            mdl_bits = compressed_bit_length(self._raw_store_bytes)
        return BitBudgetAudit(
            raw_training_data_bits=raw_training_bits,
            compressed_training_data_bits=compressed_training_bits,
            episode_store_bits=ep_bits,
            active_bundle_bits=0,
            operator_library_bits=0,
            compressed_model_mdl_bits=mdl_bits,
            compression_ratio_vs_raw=mdl_bits / max(1, raw_training_bits),
        )


def _state_action_feature_set(state: StateGraph, action: ActionInvocation) -> frozenset[str]:
    feats: set[str] = {f"act:{action.action_id}"}
    ent_map = state.entity_map()
    for role_k, eid in action.roles:
        ent = ent_map.get(eid)
        if ent is not None:
            feats.add(f"role_type:{role_k}={ent.entity_type}")
            for ak, av in ent.attributes:
                feats.add(f"role_attr:{role_k}.{ak}={av}")
        feats.add(f"role_ent:{role_k}={eid}")
        for _s, pred, obj in state.facts_for(eid):
            feats.add(f"role_pred:{role_k}.{pred}={obj if obj in ('true', 'false') else 'ENT'}")
    for s, p, o in state.facts:
        feats.add(f"fact:{s}|{p}|{o}")
    return frozenset(feats)


class StructuralKNNBaseline(BaseS0Solver):
    """1-NN retrieval over state-action feature Jaccard similarity; applies raw stored delta."""

    name = "structural_knn"
    category = "memory_baseline"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._entries: list[tuple[str, str, frozenset[str], SparseDelta]] = []
        self._raw_store_bytes: bytes = b""

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        self._check_not_frozen()
        self._entries = [
            (
                ep.state_action_key(),
                ep.action.action_id,
                _state_action_feature_set(ep.before_state, ep.action),
                ep.sparse_delta,
            )
            for ep in episodes
        ]
        self._raw_store_bytes = canonical_json_bytes(
            [
                {"key": k, "action": aid, "features": sorted(feats), "delta": d.to_dict()}
                for k, aid, feats, d in self._entries
            ]
        )

    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        cur = task.initial_state.copy()
        step_preds: list[StateGraph | None] = []
        retrieval_hits = 0
        mem_accesses = 0

        for idx, act in enumerate(task.actions):
            exact_key = task.step_state_action_key(cur, idx)
            q_feats = _state_action_feature_set(cur, act)
            best_sim = -1.0
            best_delta: SparseDelta | None = None
            best_exact = False
            for k, aid, feats, delta in self._entries:
                if aid != act.action_id:
                    continue
                mem_accesses += 1
                if k == exact_key:
                    best_sim = 2.0
                    best_delta = delta
                    best_exact = True
                    break
                inter = len(q_feats & feats)
                union = len(q_feats | feats)
                sim = inter / float(union) if union > 0 else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_delta = delta
            if best_delta is None:
                return TaskPrediction(
                    predicted_final_state=None,
                    abstained=True,
                    retrieval_hits=retrieval_hits,
                    memory_accesses=mem_accesses,
                    step_predictions=tuple(step_preds),
                )
            if best_exact:
                retrieval_hits += 1
            cur = best_delta.apply_to(cur)
            step_preds.append(cur)

        return TaskPrediction(
            predicted_final_state=cur,
            abstained=False,
            retrieval_hits=retrieval_hits,
            memory_accesses=mem_accesses,
            step_predictions=tuple(step_preds),
        )

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        if not self._entries:
            ep_bits = 0
            mdl_bits = 0
        else:
            ep_bits = len(self._raw_store_bytes) * 8
            mdl_bits = compressed_bit_length(self._raw_store_bytes)
        return BitBudgetAudit(
            raw_training_data_bits=raw_training_bits,
            compressed_training_data_bits=compressed_training_bits,
            episode_store_bits=ep_bits,
            active_bundle_bits=0,
            operator_library_bits=0,
            compressed_model_mdl_bits=mdl_bits,
            compression_ratio_vs_raw=mdl_bits / max(1, raw_training_bits),
        )


def _encode_state_action_hv10240(state: StateGraph, action: ActionInvocation) -> int:
    """Encode a relational (state, action) pair into a 10,240-bit Mirror BSC hypervector."""
    bd = BitSlicedBundle10240(bits=10)
    bd.add(bind10240(hv10240_of("role:action_id"), hv10240_of(f"val:{action.action_id}")), 4)
    ent_map = state.entity_map()
    for role_k, eid in action.roles:
        role_hv = hv10240_of(f"role:{role_k}")
        bd.add(bind10240(role_hv, hv10240_of(f"ent:{eid}")), 2)
        ent = ent_map.get(eid)
        if ent is not None:
            bd.add(bind10240(hv10240_of(f"role_type:{role_k}"), hv10240_of(f"type:{ent.entity_type}")), 2)
            for ak, av in ent.attributes:
                bd.add(
                    bind10240(hv10240_of(f"role_attr:{role_k}.{ak}"), hv10240_of(f"aval:{av}")),
                    1,
                )
        for _s, pred, obj in state.facts_for(eid):
            val_tag = obj if obj in ("true", "false") else f"ent:{obj}"
            bd.add(bind10240(hv10240_of(f"role_fact:{role_k}.{pred}"), hv10240_of(val_tag)), 2)
    return bd.binary()


class SHARDHDCRetrievalBaseline(BaseS0Solver):
    """SHARD associative retrieval WITHOUT ITL operator induction.

    Reuses Mirror 7's 10,240-bit BSC algebra (``mirror_crystal.quotient_egraph_resonator``)
    to retrieve the nearest training episode by POPCNT Hamming similarity and apply
    its stored delta directly. Answers: 'How far does HDC similarity alone go?'
    """

    name = "shard_hdc_retrieval_no_itl"
    category = "shard_baseline"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._entries: list[tuple[str, int, SparseDelta]] = []
        self._delta_bytes: bytes = b""

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        self._check_not_frozen()
        self._entries = [
            (
                ep.state_action_key(),
                _encode_state_action_hv10240(ep.before_state, ep.action),
                ep.sparse_delta,
            )
            for ep in episodes
        ]
        self._delta_bytes = canonical_json_bytes([d.to_dict() for _k, _hv, d in self._entries])

    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        cur = task.initial_state.copy()
        step_preds: list[StateGraph | None] = []
        retrieval_hits = 0
        mem_accesses = 0

        for idx, act in enumerate(task.actions):
            exact_key = task.step_state_action_key(cur, idx)
            q_hv = _encode_state_action_hv10240(cur, act)
            best_sim = -D_UNIFIED - 1
            best_delta: SparseDelta | None = None
            best_key = ""
            for k, ep_hv, delta in self._entries:
                mem_accesses += 1
                sim = hamming_sim10240(q_hv, ep_hv)
                if sim > best_sim:
                    best_sim = sim
                    best_delta = delta
                    best_key = k
            if best_delta is None:
                return TaskPrediction(
                    predicted_final_state=None,
                    abstained=True,
                    retrieval_hits=retrieval_hits,
                    memory_accesses=mem_accesses,
                    step_predictions=tuple(step_preds),
                )
            if best_key == exact_key:
                retrieval_hits += 1
            cur = best_delta.apply_to(cur)
            step_preds.append(cur)

        return TaskPrediction(
            predicted_final_state=cur,
            abstained=False,
            retrieval_hits=retrieval_hits,
            memory_accesses=mem_accesses,
            step_predictions=tuple(step_preds),
        )

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        if not self._entries:
            bundle_bits = 0
            total_raw = 0
            mdl_bits = 0
        else:
            bundle_bits = len(self._entries) * D_UNIFIED
            delta_bits = len(self._delta_bytes) * 8
            total_raw = bundle_bits + delta_bits
            # Compress delta payload + raw random-like HVs (incompressible bits = bundle_bits)
            mdl_bits = bundle_bits + compressed_bit_length(self._delta_bytes)
        return BitBudgetAudit(
            raw_training_data_bits=raw_training_bits,
            compressed_training_data_bits=compressed_training_bits,
            episode_store_bits=total_raw,
            active_bundle_bits=bundle_bits,
            operator_library_bits=0,
            compressed_model_mdl_bits=mdl_bits,
            compression_ratio_vs_raw=mdl_bits / max(1, raw_training_bits),
        )


def _format_task_for_mirror(task: EvaluationTaskPublic) -> str:
    """Render a relational task as plain natural language for Current Mirror evaluation."""
    clauses: list[str] = []
    for subj, pred, obj in sorted(task.initial_state.facts):
        if pred == "located_on":
            clauses.append(f"{subj} is located on {obj}.")
        elif pred == "owned_by":
            clauses.append(f"{obj} owns {subj}.")
        elif pred == "is_open":
            clauses.append(f"{subj} is {'open' if obj == 'true' else 'closed'}.")
        elif pred == "is_locked":
            clauses.append(f"{subj} is {'locked' if obj == 'true' else 'unlocked'}.")
        elif pred == "accessible":
            clauses.append(f"{subj} is {'accessible' if obj == 'true' else 'inaccessible'}.")
    for act in task.actions:
        clauses.append(f"{act.instruction_text()}.")
    last_roles = task.actions[-1].role_dict() if task.actions else {}
    focus = last_roles.get("object") or last_roles.get("target") or "the world"
    clauses.append(f"What is the final state of {focus}?")
    return " ".join(clauses)


class CurrentMirrorSolverBaseline(BaseS0Solver):
    """Evaluates Mirror 7's existing native reasoning organs in strictly read-only mode.

    Mirrors ``native_chain(include_context_reasoner=True)`` plus ``BilingualCCGHDCParser``,
    deliberately excluding ``PersonalFactMemory.try_store`` and isolating session IDs per
    task so zero test prompts can ever be ingested into Mirror memory or leak across tasks
    (including after ``.freeze()``).
    """

    name = "current_mirror_solver"
    category = "existing_mirror"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._parser = BilingualCCGHDCParser()
        self._chat = _mirror_ni.ConversationOrgan()
        self._logic = _mirror_ni.LogicOrgan()
        self._words = _mirror_ni.WordProblemSolver()
        self._curated = _mirror_ni.CuratedKnowledgeOrgan()
        self._kg = _mirror_ni.KnowledgeGraphEngine()
        self._multi_hop = _mirror_ni.MultiHopChainingEngine(kg=self._kg)
        reasoner_cls = getattr(_mirror_ni, "ContextReasoner", None)
        self._reasoner = reasoner_cls() if reasoner_cls is not None else None

    def _query_read_only_organs(self, prompt: str, session_id: str) -> _MirrorResponse:
        calls = [
            lambda: self._chat.respond(prompt, session_id=session_id),
        ]
        if self._reasoner is not None:
            calls.append(lambda: self._reasoner.respond(prompt))
        calls.extend(
            [
                lambda: self._multi_hop.resolve_chain(prompt),
                lambda: self._logic.respond(prompt),
                lambda: self._words.solve(prompt),
                lambda: self._kg.lookup_from_prompt(prompt) or self._curated.lookup(prompt),
            ]
        )
        for call in calls:
            try:
                hit = call()
            except Exception:
                hit = None
            if hit is not None:
                return _MirrorResponse(text=hit.answer, source=_mirror_source_for(hit.kind))
        return _MirrorResponse(text="", source="abstain")

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        self._check_not_frozen()
        # Existing Mirror native organs are non-learning for relational state-graph operators;
        # we intentionally do NOT ingest training or test episodes into Mirror vaults/memory.

    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        parsed_steps = 0
        for act in task.actions:
            edge = self._parser.parse_clause(act.instruction_text())
            rdict = act.role_dict()
            # BilingualCCGHDCParser only exposes (who, act, targ) slots; it can only
            # exactly bind 2-role actions (open/lock) when who==actor, act==action_id, targ==target.
            if (
                len(rdict) == 2
                and edge.act == act.action_id
                and edge.who == rdict.get("actor", "").casefold()
                and edge.targ == rdict.get("target", "").casefold()
            ):
                parsed_steps += 1

        prompt = _format_task_for_mirror(task)
        resp = self._query_read_only_organs(prompt, session_id=f"s0_ro:{self.seed}:{task.task_id}")
        meta = {
            "parser_nonzero_steps": parsed_steps,
            "total_steps": len(task.actions),
            "mirror_source": resp.source,
            "mirror_text_preview": resp.text[:120],
        }
        if resp.abstained:
            return TaskPrediction(
                predicted_final_state=None,
                abstained=True,
                retrieval_hits=0,
                memory_accesses=0,
                metadata=meta,
            )
        # Non-empty text response from an unrelated organ cannot construct a verified StateGraph
        return TaskPrediction(
            predicted_final_state=None,
            abstained=False,
            retrieval_hits=0,
            memory_accesses=0,
            metadata=meta,
        )

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        return BitBudgetAudit(
            raw_training_data_bits=raw_training_bits,
            compressed_training_data_bits=compressed_training_bits,
            episode_store_bits=0,
            active_bundle_bits=0,
            operator_library_bits=0,
            compressed_model_mdl_bits=0,
            compression_ratio_vs_raw=0.0,
        )


def _hash_feature_vector(state: StateGraph, action: ActionInvocation, dim: int = 48) -> list[float]:
    vec = [0.0] * dim
    feats = _state_action_feature_set(state, action)
    for f in feats:
        h = hashlib.blake2b(f.encode("utf-8"), digest_size=4).digest()
        val = int.from_bytes(h, "little")
        idx = val % dim
        sign = 1.0 if ((val >> 16) & 1) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]
    return vec


class SmallCPUNeuralReferenceBaseline(BaseS0Solver):
    """Comparison-only pure-Python tiny 1-hidden-layer MLP on CPU.

    Strictly for research comparison under the same 6-demo/action data and CPU budget
    (PDF pp. 10, 15, 16). NEVER used by candidate Mirror learners.
    """

    name = "small_cpu_neural_reference"
    category = "comparison_only_neural"
    comparison_only = True

    def __init__(
        self,
        seed: int = 0,
        input_dim: int = 32,
        hidden_dim: int = 12,
        epochs: int = 25,
        lr: float = 0.15,
    ) -> None:
        super().__init__(seed=seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self._templates: list[SparseDelta] = []
        self._w1: list[list[float]] = []
        self._b1: list[float] = []
        self._w2: list[list[float]] = []
        self._b2: list[float] = []

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        self._check_not_frozen()
        rng = random.Random(self.seed * 131 + 7)
        delta_to_idx: dict[str, int] = {}
        self._templates = []
        dataset: list[tuple[list[float], int]] = []
        for ep in episodes:
            key = canonical_json_bytes(ep.sparse_delta.to_dict()).decode("utf-8")
            if key not in delta_to_idx:
                delta_to_idx[key] = len(self._templates)
                self._templates.append(ep.sparse_delta)
            x = _hash_feature_vector(ep.before_state, ep.action, self.input_dim)
            dataset.append((x, delta_to_idx[key]))

        n_cls = max(1, len(self._templates))
        scale1 = 1.0 / math.sqrt(self.input_dim)
        scale2 = 1.0 / math.sqrt(self.hidden_dim)
        self._w1 = [[rng.uniform(-scale1, scale1) for _ in range(self.input_dim)] for _ in range(self.hidden_dim)]
        self._b1 = [0.0] * self.hidden_dim
        self._w2 = [[rng.uniform(-scale2, scale2) for _ in range(self.hidden_dim)] for _ in range(n_cls)]
        self._b2 = [0.0] * n_cls

        for _epoch in range(self.epochs):
            rng.shuffle(dataset)
            for x, y in dataset:
                # Forward: tanh hidden -> softmax logits
                h = [
                    math.tanh(sum(w * xi for w, xi in zip(row, x)) + b)
                    for row, b in zip(self._w1, self._b1)
                ]
                logits = [
                    sum(w * hi for w, hi in zip(row, h)) + b
                    for row, b in zip(self._w2, self._b2)
                ]
                m = max(logits)
                exps = [math.exp(v - m) for v in logits]
                z = sum(exps) or 1.0
                probs = [e / z for e in exps]

                # Backward (cross-entropy)
                d_logits = [probs[c] - (1.0 if c == y else 0.0) for c in range(n_cls)]
                d_h = [0.0] * self.hidden_dim
                for c in range(n_cls):
                    dl = d_logits[c]
                    row2 = self._w2[c]
                    for j in range(self.hidden_dim):
                        d_h[j] += dl * row2[j]
                        row2[j] -= self.lr * dl * h[j]
                    self._b2[c] -= self.lr * dl
                for j in range(self.hidden_dim):
                    dh = d_h[j] * (1.0 - h[j] * h[j])
                    row1 = self._w1[j]
                    for i in range(self.input_dim):
                        row1[i] -= self.lr * dh * x[i]
                    self._b1[j] -= self.lr * dh

    def _predict_template(self, state: StateGraph, action: ActionInvocation) -> SparseDelta:
        x = _hash_feature_vector(state, action, self.input_dim)
        h = [
            math.tanh(sum(w * xi for w, xi in zip(row, x)) + b)
            for row, b in zip(self._w1, self._b1)
        ]
        logits = [
            sum(w * hi for w, hi in zip(row, h)) + b
            for row, b in zip(self._w2, self._b2)
        ]
        best_c = max(range(len(logits)), key=lambda c: logits[c])
        return self._templates[best_c]

    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        if not self._templates:
            return TaskPrediction(predicted_final_state=None, abstained=True)
        cur = task.initial_state.copy()
        step_preds: list[StateGraph | None] = []
        for act in task.actions:
            delta = self._predict_template(cur, act)
            cur = delta.apply_to(cur)
            step_preds.append(cur)
        return TaskPrediction(
            predicted_final_state=cur,
            abstained=False,
            retrieval_hits=0,
            memory_accesses=0,
            step_predictions=tuple(step_preds),
        )

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        if not self._templates:
            return BitBudgetAudit(
                raw_training_data_bits=raw_training_bits,
                compressed_training_data_bits=compressed_training_bits,
                episode_store_bits=0,
                active_bundle_bits=0,
                operator_library_bits=0,
                compressed_model_mdl_bits=0,
                compression_ratio_vs_raw=0.0,
            )
        n_cls = len(self._templates)
        n_params = self.hidden_dim * (self.input_dim + 1) + n_cls * (self.hidden_dim + 1)
        # 32-bit float parameter accounting + stored delta templates
        param_bits = n_params * 32
        tmpl_bytes = canonical_json_bytes([d.to_dict() for d in self._templates])
        tmpl_bits = len(tmpl_bytes) * 8
        mdl_bits = param_bits + compressed_bit_length(tmpl_bytes)
        return BitBudgetAudit(
            raw_training_data_bits=raw_training_bits,
            compressed_training_data_bits=compressed_training_bits,
            episode_store_bits=tmpl_bits,
            active_bundle_bits=param_bits,
            operator_library_bits=0,
            compressed_model_mdl_bits=mdl_bits,
            compression_ratio_vs_raw=mdl_bits / max(1, raw_training_bits),
        )
