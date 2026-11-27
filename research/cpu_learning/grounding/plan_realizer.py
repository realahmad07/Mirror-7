"""Stage S7 Plan-Conditioned Truth-Locked Natural Language Realizer & Separated Dual Metrics.

STRICT SEPARATION OF SEMANTIC TRUTH VS SURFACE FLUENCY (PDF pp. 12, 14):
- *"Do not call a correct graph a fluent response."*
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- ``GroundedPlanRealizer`` renders natural-language explanations strictly from verified
  ``GroundedExecutionResult`` traces using inverted learned ``ConstructionCard`` templates and
  ``ByteMarkovGrammar`` continuation scoring (``0%`` entity/outcome hallucination).
- Evaluates every realization along TWO strictly independent axes:
  1. **Semantic Truthfulness:** ``semantic_correctness`` and ``hallucination_rate``.
  2. **Surface Readability:** ``surface_fluency_score`` and ``reference_perplexity_bpb``.
- Includes two diagnostic ablations proving metric independence:
  - ``telegraphic_tuple_dump``: ``100%`` semantic truth / ``0%`` hallucination, low fluency.
  - ``ungated_fluent_assumer``: high surface fluency, fails truth / hallucinates on blocked & negated steps.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.grounding.construction_learner import (
    ClauseExecutionTrace,
    ConstructionCard,
    GroundedExecutionResult,
    RawByteConstructionLearner,
)
from research.cpu_learning.simulator.state_graph import ObservationMask, StateGraph

PLAN_REALIZER_SCHEMA_VERSION: str = "s7-plan-realizer-v1"


class PlanRealizerFrozenError(RuntimeError):
    """Raised when attempting to fit prose statistics on a frozen ``GroundedPlanRealizer``."""


class _VariableOrderContinuationScorer:
    """Self-contained variable-order (orders 2 & 4) UTF-8 byte continuation table (0 float weights)."""

    def __init__(self, max_contexts: int = 8192) -> None:
        self.max_contexts = int(max_contexts)
        self._order4: dict[bytes, dict[int, int]] = {}
        self._order2: dict[bytes, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.total_bytes_observed: int = 0

    def observe_bytes(self, text: str, max_sample_bytes: int = 512) -> None:
        raw = text.encode("utf-8", errors="ignore")[:max_sample_bytes]
        n = len(raw)
        if n < 3:
            return
        self.total_bytes_observed += n
        for i in range(2, n):
            b_next = raw[i]
            self._order2[raw[i - 2 : i]][b_next] += 1
            if i >= 4:
                ctx4 = raw[i - 4 : i]
                tbl = self._order4.get(ctx4)
                if tbl is None:
                    if len(self._order4) < self.max_contexts:
                        self._order4[ctx4] = {b_next: 1}
                else:
                    tbl[b_next] = tbl.get(b_next, 0) + 1

    def byte_fluency_score(self, text: str) -> float:
        raw = text.encode("utf-8", errors="ignore")[:256]
        if len(raw) < 4 or self.total_bytes_observed == 0:
            return 0.5
        hits = 0
        checks = 0
        for i in range(4, len(raw)):
            checks += 1
            b_next = raw[i]
            ctx4 = raw[i - 4 : i]
            if ctx4 in self._order4 and b_next in self._order4[ctx4]:
                hits += 1
            elif b_next in self._order2.get(raw[i - 2 : i], {}):
                hits += 1
        return hits / max(1, checks)

_ENTITY_ID_REGEX = re.compile(r"\b[^\W\d_]+_[0-9]+\b", flags=re.UNICODE)
_MACHINE_SYNTAX_CHARS: frozenset[str] = frozenset({"=", "|", "[", "]", "{", "}", "<", ">", "$"})

_DISCOURSE_OPENERS: tuple[str, ...] = ("First,", "Next,", "Then,", "Afterward,", "Subsequently,")

# Positive vs non-execution lexical markers used strictly by the evaluator to verify outcome polarity
_APPLIED_MARKERS: tuple[str, ...] = ("succeeded", "completed", "executed", "updated", "applied")
_BLOCKED_MARKERS: tuple[str, ...] = ("blocked", "could not proceed", "failed", "prevented")
_NEGATED_MARKERS: tuple[str, ...] = ("negation", "intentionally skipped", "refrained", "not performed")
_COND_SKIP_MARKERS: tuple[str, ...] = ("condition did not hold", "condition was unmet", "skipped because")
_ABSTAIN_MARKERS: tuple[str, ...] = ("abstained", "ungrounded", "ambiguous", "unobserved")


@dataclass(frozen=True)
class RealizationEvaluation:
    """Strictly separated semantic truthfulness vs surface readability metrics."""

    method: str
    realized_text: str
    semantic_correctness: float
    hallucination_rate: float
    surface_fluency_score: float
    reference_perplexity_bpb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "realized_text": self.realized_text,
            "semantic_correctness": round(self.semantic_correctness, 6),
            "hallucination_rate": round(self.hallucination_rate, 6),
            "surface_fluency_score": round(self.surface_fluency_score, 6),
            "reference_perplexity_bpb": round(self.reference_perplexity_bpb, 4),
        }


class ReferenceProseByteModel:
    """Lightweight interpolated order-3 byte cross-entropy scorer fit on natural prose."""

    def __init__(self, discount: float = 0.75) -> None:
        self.discount = float(discount)
        self._c0: dict[int, int] = defaultdict(int)
        self._total0: int = 0
        self._c1: dict[bytes, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._tot1: dict[bytes, int] = defaultdict(int)
        self._c2: dict[bytes, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._tot2: dict[bytes, int] = defaultdict(int)

    def fit_texts(self, texts: Sequence[str]) -> None:
        for txt in texts:
            raw = txt.encode("utf-8")
            for i, b in enumerate(raw):
                self._c0[b] += 1
                self._total0 += 1
                if i >= 1:
                    h1 = raw[i - 1 : i]
                    self._c1[h1][b] += 1
                    self._tot1[h1] += 1
                if i >= 2:
                    h2 = raw[i - 2 : i]
                    self._c2[h2][b] += 1
                    self._tot2[h2] += 1

    def byte_prob(self, ctx2: bytes, b: int) -> float:
        # Uniform base floor
        p = 1.0 / 256.0
        # Unigram
        if self._total0 > 0:
            u_cnt = len(self._c0)
            c = self._c0.get(b, 0)
            lam0 = (self.discount * u_cnt) / float(self._total0)
            p = max(0.0, c - self.discount) / float(self._total0) + lam0 * p
        # Bigram
        if len(ctx2) >= 1:
            h1 = ctx2[-1:]
            t1 = self._tot1.get(h1, 0)
            if t1 > 0:
                d1 = self._c1[h1]
                u1 = len(d1)
                c1 = d1.get(b, 0)
                lam1 = (self.discount * u1) / float(t1)
                p = max(0.0, c1 - self.discount) / float(t1) + lam1 * p
        # Trigram
        if len(ctx2) >= 2:
            h2 = ctx2[-2:]
            t2 = self._tot2.get(h2, 0)
            if t2 > 0:
                d2 = self._c2[h2]
                u2 = len(d2)
                c2 = d2.get(b, 0)
                lam2 = (self.discount * u2) / float(t2)
                p = max(0.0, c2 - self.discount) / float(t2) + lam2 * p
        return max(p, 1e-9)

    def cross_entropy_bpb(self, text: str) -> float:
        raw = text.encode("utf-8")
        if not raw:
            return 8.0
        bits = 0.0
        for i, b in enumerate(raw):
            ctx = raw[max(0, i - 2) : i]
            bits -= math.log2(self.byte_prob(ctx, b))
        return bits / float(len(raw))


def _render_clause_via_construction(
    card: ConstructionCard | None,
    action_id: str | None,
    roles: Mapping[str, str],
) -> str:
    """Invert a learned ``ConstructionCard`` pattern into a surface phrase over ``roles``."""
    if card is not None:
        smap = card.slot_map()
        words: list[str] = []
        for tok in card.pattern_tokens:
            if tok.startswith("$SLOT:"):
                s_idx = int(tok.split(":", 1)[1])
                rname = smap.get(s_idx, "")
                words.append(roles.get(rname, tok))
            else:
                words.append(tok)
        return " ".join(words)
    if action_id is None:
        return "the requested step"
    ent_list = [v for _k, v in sorted(roles.items())]
    return f"{action_id} involving " + " and ".join(ent_list)


class GroundedPlanRealizer:
    """Truth-locked plan-to-text realizer backed by learned constructions & variable-order continuation scoring."""

    def __init__(self, construction_learner: RawByteConstructionLearner) -> None:
        self.learner = construction_learner
        self.fluency_grammar = _VariableOrderContinuationScorer(max_contexts=8192)
        self.ref_prose_model = ReferenceProseByteModel()
        self._templates_by_action: dict[str, list[ConstructionCard]] = {}
        self._frozen: bool = False
        self._refresh_templates()

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._refresh_templates()
        self._frozen = True

    def _refresh_templates(self) -> None:
        by_act: dict[str, list[ConstructionCard]] = {}
        for cx in self.learner.promoted_constructions:
            by_act.setdefault(cx.action_id, []).append(cx)
        for aid in by_act:
            by_act[aid].sort(key=lambda c: (-c.support_count, len(c.pattern_tokens), c.construction_id))
        self._templates_by_action = by_act

    def fit_prose_corpus(self, prose_samples: Sequence[str]) -> None:
        """Fit variable-order continuation scorer and ``ReferenceProseByteModel`` on natural prose."""
        if self._frozen:
            raise PlanRealizerFrozenError(
                "GroundedPlanRealizer is frozen; cannot call fit_prose_corpus()"
            )
        self._refresh_templates()
        for sample in prose_samples:
            self.fluency_grammar.observe_bytes(sample, max_sample_bytes=512)
        self.ref_prose_model.fit_texts(prose_samples)

    def _best_clause_phrase(self, action_id: str | None, roles: Mapping[str, str]) -> str:
        if action_id is None:
            return "the requested step"
        cands = self._templates_by_action.get(action_id, [])
        if not cands:
            return _render_clause_via_construction(None, action_id, roles)
        # Rank candidate learned construction realizations by ByteMarkovGrammar fluency
        scored = [
            (
                self.fluency_grammar.byte_fluency_score(
                    _render_clause_via_construction(cx, action_id, roles)
                ),
                -idx,
                _render_clause_via_construction(cx, action_id, roles),
            )
            for idx, cx in enumerate(cands[:4])
        ]
        scored.sort(reverse=True)
        return scored[0][2]

    def realize(self, exec_result: GroundedExecutionResult) -> str:
        """Generate a truth-locked natural-language report from ``exec_result``."""
        self._refresh_templates()
        if exec_result.abstained:
            if exec_result.is_ambiguous:
                return (
                    "The system abstained because the instruction remained structurally ambiguous "
                    "across multiple candidate role bindings, leaving the world unchanged."
                )
            return (
                "The system abstained because the instruction contained ungrounded or unobserved "
                "conditions, leaving every entity unchanged."
            )

        traces = exec_result.clause_traces
        if not traces:
            return "No executable clauses were present, and the world remained unchanged."

        sentences: list[str] = []
        n = len(traces)
        for idx, tr in enumerate(traces):
            phrase = self._best_clause_phrase(tr.action_id, tr.roles)
            if n == 1:
                lead = "In this step,"
            elif idx == n - 1:
                lead = "Finally,"
            else:
                lead = _DISCOURSE_OPENERS[idx % len(_DISCOURSE_OPENERS)]

            if tr.outcome == "applied":
                n_edits = len(tr.added_facts) + len(tr.deleted_facts)
                sentences.append(
                    f"{lead} the action to {phrase} succeeded and updated {n_edits} verified world relations."
                )
            elif tr.outcome == "blocked":
                sentences.append(
                    f"{lead} the attempt to {phrase} was blocked because a required guard failed, leaving the world unchanged."
                )
            elif tr.outcome == "negated_noop":
                sentences.append(
                    f"{lead} the instruction to {phrase} was intentionally skipped due to negation, preserving all existing relations."
                )
            elif tr.outcome == "condition_false_noop":
                sentences.append(
                    f"{lead} because the condition did not hold, the step to {phrase} was skipped without altering any facts."
                )
            else:
                sentences.append(
                    f"{lead} the system abstained on {phrase}, leaving all relations intact."
                )

        return " ".join(sentences)

    def realize_telegraphic_tuple_dump(self, exec_result: GroundedExecutionResult) -> str:
        """Diagnostic baseline 1: 100% semantically accurate graph/tuple dump, low surface fluency."""
        if exec_result.abstained:
            return f"OUT=ABSTAINED|AMBIG={int(exec_result.is_ambiguous)};"
        parts: list[str] = []
        for tr in exec_result.clause_traces:
            role_kv = ",".join(f"{k}={v}" for k, v in sorted(tr.roles.items()))
            out_tag = {
                "applied": "APPLIED",
                "blocked": "BLOCKED",
                "negated_noop": "NEGATION_SKIPPED",
                "condition_false_noop": "CONDITION_DID_NOT_HOLD_SKIPPED",
            }.get(tr.outcome, "ABSTAINED")
            parts.append(f"[{tr.clause_index}|ACT={tr.action_id}|ROLES={{{role_kv}}}|OUT={out_tag}]")
        return " -> ".join(parts)

    def realize_ungated_fluent_assumer(
        self,
        raw_instruction_bytes: bytes,
        initial_state: StateGraph,
    ) -> str:
        """Diagnostic baseline 2: fluent prose template that assumes every requested step succeeded
        (high surface fluency, but hallucinates false transitions on blocked/negated/abstained inputs)."""
        ents = [eid for eid, _ in initial_state.entities[:2]]
        e_str = " and ".join(ents) if ents else "the target entities"
        text_in = raw_instruction_bytes.decode("utf-8", errors="ignore").strip().rstrip(".!")
        return (
            f"First, the requested operation ({text_in}) succeeded smoothly and updated "
            f"the world relations for {e_str} without any issues."
        )

    def evaluate_realization(
        self,
        method: str,
        realized_text: str,
        exec_result: GroundedExecutionResult,
        initial_state: StateGraph,
        *,
        mask: ObservationMask | None = None,
    ) -> RealizationEvaluation:
        """Compute strictly separated Semantic Truthfulness and Surface Readability metrics."""
        obs = mask or ObservationMask(fully_observed=True)
        lower_txt = realized_text.casefold()
        world_eids = {
            eid.casefold()
            for eid, _ in initial_state.entities
            if obs.is_entity_observed(eid)
        }
        if exec_result.final_state is not None:
            world_eids |= {
                eid.casefold()
                for eid, _ in exec_result.final_state.entities
                if obs.is_entity_observed(eid)
            }

        mentioned_eids = {m.casefold() for m in _ENTITY_ID_REGEX.findall(realized_text)}
        ungrounded_eids = mentioned_eids - world_eids

        # --- Axis 1: Semantic Correctness & Hallucination Rate ---
        if exec_result.abstained:
            has_abstain = any(mk in lower_txt for mk in _ABSTAIN_MARKERS) or "out=abstained" in lower_txt
            claims_applied = any(mk in lower_txt for mk in _APPLIED_MARKERS) and not has_abstain
            sem_score = 1.0 if (has_abstain and not claims_applied and not ungrounded_eids) else 0.0
            halluc_events = len(ungrounded_eids) + (1 if claims_applied else 0)
            denom = max(1, len(mentioned_eids) + 1)
            halluc_rate = min(1.0, halluc_events / float(denom))
        else:
            traces = exec_result.clause_traces
            if not traces:
                sem_score = 1.0 if not ungrounded_eids else 0.0
                halluc_rate = 1.0 if ungrounded_eids else 0.0
            else:
                if " -> " in realized_text:
                    raw_segs = [s.casefold() for s in realized_text.split(" -> ") if s.strip()]
                else:
                    raw_segs = [
                        s.casefold()
                        for s in re.split(r"(?<=[.!])\s+", realized_text.strip())
                        if s.strip()
                    ]
                step_hits = 0
                false_claims = 0
                for idx_t, tr in enumerate(traces):
                    seg_txt = raw_segs[idx_t] if len(raw_segs) == len(traces) else lower_txt
                    roles_present = all(eid.casefold() in seg_txt for eid in tr.roles.values())
                    if tr.outcome == "applied":
                        pol_ok = any(mk in seg_txt for mk in _APPLIED_MARKERS) or "out=applied" in seg_txt
                    elif tr.outcome == "blocked":
                        pol_ok = any(mk in seg_txt for mk in _BLOCKED_MARKERS) or "out=blocked" in seg_txt
                        if not pol_ok and any(mk in seg_txt for mk in _APPLIED_MARKERS):
                            false_claims += 1
                    elif tr.outcome == "negated_noop":
                        pol_ok = any(mk in seg_txt for mk in _NEGATED_MARKERS) or "negation_skipped" in seg_txt
                        if not pol_ok and any(mk in seg_txt for mk in _APPLIED_MARKERS):
                            false_claims += 1
                    elif tr.outcome == "condition_false_noop":
                        pol_ok = (
                            any(mk in seg_txt for mk in _COND_SKIP_MARKERS)
                            or "condition_did_not_hold" in seg_txt
                        )
                        if not pol_ok and any(mk in seg_txt for mk in _APPLIED_MARKERS):
                            false_claims += 1
                    else:
                        pol_ok = any(mk in seg_txt for mk in _ABSTAIN_MARKERS)
                    if roles_present and pol_ok:
                        step_hits += 1
                sem_score = (step_hits / float(len(traces))) if not ungrounded_eids else 0.0
                halluc_events = len(ungrounded_eids) + false_claims
                denom = max(1, len(traces) + len(ungrounded_eids))
                halluc_rate = min(1.0, halluc_events / float(denom))

        # --- Axis 2: Surface Readability (100% independent of world truth!) ---
        stripped = realized_text.strip()
        if not stripped:
            fluency = 0.0
        else:
            starts_upper = 1.0 if stripped[0].isupper() else 0.0
            ends_punct = 1.0 if stripped[-1] in {".", "!"} else 0.0
            bad_char_cnt = sum(1 for ch in stripped if ch in _MACHINE_SYNTAX_CHARS)
            clean_syntax = max(0.0, 1.0 - (bad_char_cnt / float(max(8, len(stripped) // 3))))
            words = stripped.split()
            if len(words) >= 4:
                trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
                non_rep = len(set(trigrams)) / float(len(trigrams))
            else:
                non_rep = 0.5
            markov_fl = self.fluency_grammar.byte_fluency_score(stripped)
            fluency = (
                0.20 * starts_upper
                + 0.20 * ends_punct
                + 0.30 * clean_syntax
                + 0.15 * non_rep
                + 0.15 * markov_fl
            )

        ref_bpb = self.ref_prose_model.cross_entropy_bpb(stripped)
        return RealizationEvaluation(
            method=method,
            realized_text=realized_text,
            semantic_correctness=sem_score,
            hallucination_rate=halluc_rate,
            surface_fluency_score=fluency,
            reference_perplexity_bpb=ref_bpb,
        )
