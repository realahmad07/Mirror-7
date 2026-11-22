"""Separate Parser-Only Scoring Harness for Stage S1 (PDF pp. 10, 14, 17).

Evaluates raw UTF-8 instruction -> typed candidate role-frame alignment accuracy
independently of operator induction. Compares Mirror 7's existing ``BilingualCCGHDCParser``
against S1's learner-facing ``TypedRoleFrameParser`` (with and without L2 state typing)
across all 5 fixed seeds, S0 unseen splits, unseen paraphrases, type-disambiguated
constructions, partial observation masks, genuinely ambiguous inputs, and unparseable OOD inputs.

NOTE: Never imports ``hidden_actions`` or ``oracle``.
"""
from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from research.cpu_learning.frames.frame_parser import (
    BaseFrameParser,
    BilingualCCGFrameAdapter,
    TypedRoleFrameParser,
)
from research.cpu_learning.frames.state_frame import CandidateActionFrame, FrameParseResult, TypedStateFrame
from research.cpu_learning.simulator.generators import (
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    GENERATOR_VERSION,
    HELDOUT_POOL,
    TEMPORAL_GENERATOR_VERSION,
    TEMPORAL_POOL,
    UNSEEN_EVAL_SPLITS,
    _assemble_instruction,
    _make_base_world,
    generate_heldout_suite,
    generate_training_episodes,
)
from research.cpu_learning.simulator.state_graph import (
    Entity,
    ObservationMask,
    StateGraph,
    canonical_json_bytes,
    sha256_hex,
)

S1_PARSER_GENERATOR_VERSION: str = "s1-parser-bench-v1.0"

S1_PARSER_SPLIT_ORDER: tuple[str, ...] = (
    "s0_train_instructions",
    "s0_unseen_entities_and_compositions",
    "s0_temporal_frozen",
    "s1_unseen_paraphrases",
    "s1_state_typed_disambiguation",
    "s1_partial_mask_occluded",
    "s1_genuinely_ambiguous",
    "s1_unparseable_ood",
)

UNAMBIGUOUS_EVAL_SPLITS: tuple[str, ...] = (
    "s0_unseen_entities_and_compositions",
    "s0_temporal_frozen",
    "s1_unseen_paraphrases",
    "s1_state_typed_disambiguation",
    "s1_partial_mask_occluded",
)


@dataclass(frozen=True)
class ParserEvalCase:
    """Single standalone parser-only evaluation item (public inputs + parser gold labels)."""

    case_id: str
    split_id: str
    generator_id: str
    seed: int
    raw_instruction_bytes: bytes
    before_frame: TypedStateFrame
    gold_action_id: str
    gold_roles: tuple[tuple[str, str], ...]
    gold_byte_spans: tuple[tuple[str, int, int], ...]
    expected_ambiguous: bool = False
    expected_unparseable: bool = False
    metadata: tuple[tuple[str, Any], ...] = ()

    def gold_role_dict(self) -> dict[str, str]:
        return dict(self.gold_roles)

    def gold_span_dict(self) -> dict[str, tuple[int, int]]:
        return {k: (s, e) for k, s, e in self.gold_byte_spans}

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "split_id": self.split_id,
            "generator_id": self.generator_id,
            "seed": self.seed,
            "raw_instruction_sha256": sha256_hex(self.raw_instruction_bytes),
            "raw_instruction_byte_length": len(self.raw_instruction_bytes),
            "before_frame_fp": self.before_frame.fingerprint(),
            "gold_action_id": self.gold_action_id,
            "gold_roles": dict(self.gold_roles),
            "gold_byte_spans": {k: [s, e] for k, s, e in self.gold_byte_spans},
            "expected_ambiguous": self.expected_ambiguous,
            "expected_unparseable": self.expected_unparseable,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParserCaseScore:
    """Per-case parser-only scoring record."""

    case_id: str
    split_id: str
    abstained: bool
    is_ambiguous: bool
    candidate_count: int
    action_match: bool
    exact_role_match: bool
    unambiguous_exact_role_match: bool
    exact_byte_span_match: bool
    all_spans_valid_utf8: bool
    top_k_gold_recall: bool
    ambiguity_flag_correct: bool
    ood_abstention_correct: bool
    role_tp: int
    role_fp: int
    role_fn: int


def _frame_matches_gold_roles(frame: CandidateActionFrame, case: ParserEvalCase) -> bool:
    return frame.action_id == case.gold_action_id and frame.role_dict() == case.gold_role_dict()


def _frame_matches_gold_spans(frame: CandidateActionFrame, case: ParserEvalCase) -> bool:
    return (
        _frame_matches_gold_roles(frame, case)
        and frame.span_dict() == case.gold_span_dict()
    )


def score_parser_case(case: ParserEvalCase, result: FrameParseResult) -> ParserCaseScore:
    """Score one parser output independently of operator induction."""
    all_spans_valid = all(
        cf.validate_spans_against(case.raw_instruction_bytes) for cf in result.candidate_frames
    )

    if case.expected_unparseable:
        ok_abstain = bool(result.abstained and len(result.candidate_frames) == 0)
        return ParserCaseScore(
            case_id=case.case_id,
            split_id=case.split_id,
            abstained=result.abstained,
            is_ambiguous=result.is_ambiguous,
            candidate_count=len(result.candidate_frames),
            action_match=False,
            exact_role_match=False,
            unambiguous_exact_role_match=False,
            exact_byte_span_match=False,
            all_spans_valid_utf8=all_spans_valid,
            top_k_gold_recall=False,
            ambiguity_flag_correct=(not result.is_ambiguous),
            ood_abstention_correct=ok_abstain,
            role_tp=0,
            role_fp=sum(len(cf.roles) for cf in result.candidate_frames[:1]),
            role_fn=0,
        )

    top = result.top_frame
    gold_role_set = set(case.gold_roles)
    if top is None or result.abstained:
        return ParserCaseScore(
            case_id=case.case_id,
            split_id=case.split_id,
            abstained=True,
            is_ambiguous=result.is_ambiguous,
            candidate_count=len(result.candidate_frames),
            action_match=False,
            exact_role_match=False,
            unambiguous_exact_role_match=False,
            exact_byte_span_match=False,
            all_spans_valid_utf8=all_spans_valid,
            top_k_gold_recall=False,
            ambiguity_flag_correct=(result.is_ambiguous == case.expected_ambiguous),
            ood_abstention_correct=False,
            role_tp=0,
            role_fp=0,
            role_fn=len(gold_role_set),
        )

    pred_role_set = set(top.roles)
    tp = len(pred_role_set & gold_role_set)
    fp = len(pred_role_set - gold_role_set)
    fn = len(gold_role_set - pred_role_set)

    action_ok = top.action_id == case.gold_action_id
    exact_role_ok = _frame_matches_gold_roles(top, case)
    unamb_exact_ok = bool(exact_role_ok and (not result.is_ambiguous) and len(result.candidate_frames) == 1)
    exact_span_ok = _frame_matches_gold_spans(top, case)
    top_k_ok = any(_frame_matches_gold_roles(cf, case) for cf in result.candidate_frames)
    amb_flag_ok = (result.is_ambiguous == case.expected_ambiguous) and (
        (not case.expected_ambiguous) or len(result.candidate_frames) >= 2
    )

    return ParserCaseScore(
        case_id=case.case_id,
        split_id=case.split_id,
        abstained=False,
        is_ambiguous=result.is_ambiguous,
        candidate_count=len(result.candidate_frames),
        action_match=action_ok,
        exact_role_match=exact_role_ok,
        unambiguous_exact_role_match=unamb_exact_ok,
        exact_byte_span_match=exact_span_ok,
        all_spans_valid_utf8=all_spans_valid,
        top_k_gold_recall=top_k_ok,
        ambiguity_flag_correct=amb_flag_ok,
        ood_abstention_correct=False,
        role_tp=tp,
        role_fp=fp,
        role_fn=fn,
    )


def aggregate_parser_scores(scores: Sequence[ParserCaseScore]) -> dict[str, Any]:
    """Aggregate per-case parser metrics into split summary statistics."""
    n = len(scores)
    if n == 0:
        return {
            "total_cases": 0,
            "action_id_accuracy": 0.0,
            "exact_role_alignment_accuracy": 0.0,
            "unambiguous_exact_role_accuracy": 0.0,
            "exact_byte_span_accuracy": 0.0,
            "valid_utf8_span_rate": 0.0,
            "top_k_candidate_recall": 0.0,
            "role_slot_precision": 0.0,
            "role_slot_recall": 0.0,
            "role_slot_f1": 0.0,
            "ambiguity_rate": 0.0,
            "ambiguity_flag_accuracy": 0.0,
            "abstention_rate": 0.0,
            "ood_abstention_accuracy": 0.0,
            "mean_candidate_count": 0.0,
        }

    tp = sum(s.role_tp for s in scores)
    fp = sum(s.role_fp for s in scores)
    fn = sum(s.role_fn for s in scores)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "total_cases": n,
        "action_id_accuracy": round(sum(1 for s in scores if s.action_match) / n, 6),
        "exact_role_alignment_accuracy": round(sum(1 for s in scores if s.exact_role_match) / n, 6),
        "unambiguous_exact_role_accuracy": round(
            sum(1 for s in scores if s.unambiguous_exact_role_match) / n, 6
        ),
        "exact_byte_span_accuracy": round(sum(1 for s in scores if s.exact_byte_span_match) / n, 6),
        "valid_utf8_span_rate": round(sum(1 for s in scores if s.all_spans_valid_utf8) / n, 6),
        "top_k_candidate_recall": round(sum(1 for s in scores if s.top_k_gold_recall) / n, 6),
        "role_slot_precision": round(prec, 6),
        "role_slot_recall": round(rec, 6),
        "role_slot_f1": round(f1, 6),
        "ambiguity_rate": round(sum(1 for s in scores if s.is_ambiguous) / n, 6),
        "ambiguity_flag_accuracy": round(sum(1 for s in scores if s.ambiguity_flag_correct) / n, 6),
        "abstention_rate": round(sum(1 for s in scores if s.abstained) / n, 6),
        "ood_abstention_accuracy": round(sum(1 for s in scores if s.ood_abstention_correct) / n, 6),
        "mean_candidate_count": round(sum(s.candidate_count for s in scores) / n, 4),
    }

def _make_utf8_world(seed_rng: random.Random) -> tuple[StateGraph, dict[str, str]]:
    """Create a world containing multi-byte UTF-8 entity IDs to stress byte-vs-char offsets."""
    sfx = seed_rng.randint(1, 9)
    a0 = f"élise_{sfx}"
    a1 = f"josé_{sfx}"
    o0 = f"café_{sfx}"
    s0 = f"étagère_{sfx}"
    s1 = f"naïve_altar_{sfx}"
    c0 = f"coffret_é_{sfx}"
    entities = {
        a0: Entity(a0, "agent", (("role_badge", "amber"), ("status", "active"))),
        a1: Entity(a1, "agent", (("role_badge", "teal"), ("status", "active"))),
        o0: Entity(o0, "object", (("color", "crimson"), ("portable", True))),
        s0: Entity(s0, "surface", (("color", "silver"),)),
        s1: Entity(s1, "surface", (("color", "obsidian"),)),
        c0: Entity(c0, "container", (("color", "violet"), ("status", "closed"))),
    }
    facts = {
        (o0, "located_on", s0),
        (o0, "owned_by", a0),
        (s0, "accessible", "true"),
        (s1, "accessible", "true"),
        (c0, "accessible", "true"),
        (c0, "is_open", "false"),
        (c0, "is_locked", "false"),
    }
    return (
        StateGraph.create(entities, facts),
        {"a0": a0, "a1": a1, "o0": o0, "s0": s0, "s1": s1, "c0": c0},
    )


def _gen_s1_unseen_paraphrases(seed: int) -> list[ParserEvalCase]:
    """28 unseen paraphrases/seed across HELDOUT_POOL, TEMPORAL_POOL, and multi-byte UTF-8."""
    rng = random.Random(seed * 71 + 1101)
    cases: list[ParserEvalCase] = []
    full_mask = ObservationMask(fully_observed=True)

    templates: list[tuple[str, str]] = [
        ("move", "ask_relocate_into"),
        ("move", "please_place_reversed_pp"),
        ("move", "fronted_pp_carry"),
        ("move", "fronted_src_have_reposition"),
        ("move", "passive_moved_fwd"),
        ("move", "passive_relocated_rev"),
        ("move", "shift_off_of_onto"),
        ("give", "instruct_deliver_to"),
        ("give", "please_pass_over_to"),
        ("give", "passive_given_by"),
        ("give", "tell_offer_to"),
        ("give", "kindly_present_to"),
        ("open", "kindly_let_unseal"),
        ("open", "tell_uncover_the"),
        ("open", "passive_opened_by"),
        ("open", "please_make_open"),
        ("lock", "tell_secure_the"),
        ("lock", "passive_latched_by"),
        ("lock", "please_have_fasten"),
        ("lock", "ask_bolt_the"),
        ("move", "uppercase_actor_casefold"),
        ("give", "uppercase_entity_casefold"),
        ("open", "trailing_punct_open"),
        ("lock", "trailing_punct_lock"),
    ]

    for idx, (action_id, style) in enumerate(templates):
        pool = TEMPORAL_POOL if (idx % 3 == 2) else HELDOUT_POOL
        world, ids = _make_base_world(pool, rng)
        frame = TypedStateFrame.from_state_graph(world, full_mask)
        a0, a1 = ids["agents"][0], ids["agents"][1]
        o0 = ids["objects"][0]
        s0, s1 = ids["surfaces"][0], ids["surfaces"][1]
        c0 = ids["containers"][0]

        if style == "ask_relocate_into":
            segs = ((None, "Please ask "), ("actor", a0), (None, " to "), ("action", "relocate"), (None, " the "), ("object", o0), (None, " from "), ("source", s0), (None, " into "), ("destination", c0), (None, "."))
            roles = {"actor": a0, "object": o0, "source": s0, "destination": c0}
        elif style == "please_place_reversed_pp":
            segs = ((None, "Please have "), ("actor", a0), (None, " "), ("action", "place"), (None, " "), ("object", o0), (None, " onto "), ("destination", s1), (None, " out of "), ("source", s0), (None, "."))
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
        elif style == "fronted_pp_carry":
            segs = ((None, "From "), ("source", s0), (None, " to "), ("destination", s1), (None, ", "), ("actor", a0), (None, " "), ("action", "carry"), (None, " "), ("object", o0))
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
        elif style == "fronted_src_have_reposition":
            segs = ((None, "Out of "), ("source", s0), (None, ", have "), ("actor", a0), (None, " "), ("action", "reposition"), (None, " the "), ("object", o0), (None, " onto "), ("destination", s1))
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
        elif style == "passive_moved_fwd":
            segs = ((None, "The "), ("object", o0), (None, " is "), ("action", "moved"), (None, " from "), ("source", s0), (None, " to "), ("destination", s1), (None, " by "), ("actor", a0), (None, "."))
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
        elif style == "passive_relocated_rev":
            segs = (("object", o0), (None, " was "), ("action", "relocated"), (None, " into "), ("destination", c0), (None, " from "), ("source", s0), (None, " by "), ("actor", a0))
            roles = {"actor": a0, "object": o0, "source": s0, "destination": c0}
        elif style == "shift_off_of_onto":
            segs = (("actor", a0), (None, " "), ("action", "shifts"), (None, " "), ("object", o0), (None, " off of "), ("source", s0), (None, " onto "), ("destination", s1))
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
        elif style == "instruct_deliver_to":
            segs = ((None, "Instruct "), ("actor", a0), (None, " to "), ("action", "deliver"), (None, " the "), ("object", o0), (None, " to "), ("recipient", a1), (None, "."))
            roles = {"actor": a0, "object": o0, "recipient": a1}
        elif style == "please_pass_over_to":
            segs = ((None, "Please "), ("actor", a0), (None, " "), ("action", "pass"), (None, " "), ("object", o0), (None, " over to "), ("recipient", a1))
            roles = {"actor": a0, "object": o0, "recipient": a1}
        elif style == "passive_given_by":
            segs = ((None, "The "), ("object", o0), (None, " is "), ("action", "given"), (None, " to "), ("recipient", a1), (None, " by "), ("actor", a0), (None, "."))
            roles = {"actor": a0, "object": o0, "recipient": a1}
        elif style == "tell_offer_to":
            segs = ((None, "Tell "), ("actor", a0), (None, " to "), ("action", "offer"), (None, " "), ("object", o0), (None, " to "), ("recipient", a1))
            roles = {"actor": a0, "object": o0, "recipient": a1}
        elif style == "kindly_present_to":
            segs = ((None, "Kindly let "), ("actor", a0), (None, " "), ("action", "present"), (None, " "), ("object", o0), (None, " to "), ("recipient", a1))
            roles = {"actor": a0, "object": o0, "recipient": a1}
        elif style == "kindly_let_unseal":
            segs = ((None, "Kindly let "), ("actor", a0), (None, " "), ("action", "unseal"), (None, " the "), ("target", c0), (None, "."))
            roles = {"actor": a0, "target": c0}
        elif style == "tell_uncover_the":
            segs = ((None, "Tell "), ("actor", a0), (None, " to "), ("action", "uncover"), (None, " the "), ("target", c0))
            roles = {"actor": a0, "target": c0}
        elif style == "passive_opened_by":
            segs = ((None, "The "), ("target", c0), (None, " is "), ("action", "opened"), (None, " by "), ("actor", a0), (None, "."))
            roles = {"actor": a0, "target": c0}
        elif style == "please_make_open":
            segs = ((None, "Please make "), ("actor", a0), (None, " "), ("action", "open"), (None, " "), ("target", c0))
            roles = {"actor": a0, "target": c0}
        elif style == "tell_secure_the":
            segs = ((None, "Tell "), ("actor", a0), (None, " to "), ("action", "secure"), (None, " the "), ("target", c0), (None, "."))
            roles = {"actor": a0, "target": c0}
        elif style == "passive_latched_by":
            segs = ((None, "The "), ("target", c0), (None, " was "), ("action", "latched"), (None, " by "), ("actor", a0), (None, "."))
            roles = {"actor": a0, "target": c0}
        elif style == "please_have_fasten":
            segs = ((None, "Please have "), ("actor", a0), (None, " "), ("action", "fasten"), (None, " the "), ("target", c0))
            roles = {"actor": a0, "target": c0}
        elif style == "ask_bolt_the":
            segs = ((None, "Ask "), ("actor", a0), (None, " to "), ("action", "bolt"), (None, " "), ("target", c0))
            roles = {"actor": a0, "target": c0}
        elif style == "uppercase_actor_casefold":
            segs = (("actor", a0.upper()), (None, " "), ("action", "move"), (None, " "), ("object", o0.capitalize()), (None, " from "), ("source", s0), (None, " to "), ("destination", s1))
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
        elif style == "uppercase_entity_casefold":
            segs = ((None, "Have "), ("actor", a0.capitalize()), (None, " "), ("action", "give"), (None, " "), ("object", o0.upper()), (None, " to "), ("recipient", a1.capitalize()))
            roles = {"actor": a0, "object": o0, "recipient": a1}
        elif style == "trailing_punct_open":
            segs = (("actor", a0), (None, " "), ("action", "opens"), (None, " the "), ("target", c0), (None, "!"))
            roles = {"actor": a0, "target": c0}
        else:
            segs = (("actor", a0), (None, " "), ("action", "locks"), (None, " the "), ("target", c0), (None, "!"))
            roles = {"actor": a0, "target": c0}

        raw, spans = _assemble_instruction(segs)
        cases.append(
            ParserEvalCase(
                case_id=f"s1_para_s{seed}_{idx:02d}_{style}",
                split_id="s1_unseen_paraphrases",
                generator_id="gen_s1_unseen_paraphrases_v1",
                seed=seed,
                raw_instruction_bytes=raw,
                before_frame=frame,
                gold_action_id=action_id,
                gold_roles=tuple(sorted(roles.items())),
                gold_byte_spans=tuple(sorted(spans)),
                expected_ambiguous=False,
                expected_unparseable=False,
                metadata=(("style", style), ("pool", pool.pool_id)),
            )
        )

    # 4 multi-byte UTF-8 instruction cases/seed (testing exact byte vs char span offsets)
    utf8_world, uids = _make_utf8_world(rng)
    utf8_frame = TypedStateFrame.from_state_graph(utf8_world, full_mask)
    utf8_specs = [
        (
            "move",
            "utf8_multibyte_move",
            ((None, "Please — have "), ("actor", uids["a0"]), (None, " "), ("action", "move"), (None, " "), ("object", uids["o0"]), (None, " from "), ("source", uids["s0"]), (None, " to "), ("destination", uids["s1"])),
            {"actor": uids["a0"], "object": uids["o0"], "source": uids["s0"], "destination": uids["s1"]},
        ),
        (
            "give",
            "utf8_multibyte_give",
            (("actor", uids["a0"]), (None, " "), ("action", "give"), (None, " "), ("object", uids["o0"]), (None, " to "), ("recipient", uids["a1"]), (None, " —")),
            {"actor": uids["a0"], "object": uids["o0"], "recipient": uids["a1"]},
        ),
        (
            "open",
            "utf8_multibyte_open",
            (("actor", uids["a0"]), (None, " "), ("action", "open"), (None, " the "), ("target", uids["c0"])),
            {"actor": uids["a0"], "target": uids["c0"]},
        ),
        (
            "lock",
            "utf8_multibyte_lock",
            ((None, "The "), ("target", uids["c0"]), (None, " is "), ("action", "locked"), (None, " by "), ("actor", uids["a1"])),
            {"actor": uids["a1"], "target": uids["c0"]},
        ),
    ]
    for uidx, (aid, style, segs, roles) in enumerate(utf8_specs):
        raw, spans = _assemble_instruction(segs)
        cases.append(
            ParserEvalCase(
                case_id=f"s1_para_s{seed}_{24 + uidx:02d}_{style}",
                split_id="s1_unseen_paraphrases",
                generator_id="gen_s1_unseen_paraphrases_v1",
                seed=seed,
                raw_instruction_bytes=raw,
                before_frame=utf8_frame,
                gold_action_id=aid,
                gold_roles=tuple(sorted(roles.items())),
                gold_byte_spans=tuple(sorted(spans)),
                expected_ambiguous=False,
                expected_unparseable=False,
                metadata=(("style", style), ("multibyte_utf8", True)),
            )
        )
    return cases

def _gen_s1_state_typed_disambiguation(seed: int) -> list[ParserEvalCase]:
    """16 cases/seed where L2 TypedStateFrame entity types & facts resolve syntactic ambiguity."""
    rng = random.Random(seed * 73 + 2202)
    cases: list[ParserEvalCase] = []
    full_mask = ObservationMask(fully_observed=True)

    for idx in range(16):
        pool = TEMPORAL_POOL if (idx % 2 == 1) else HELDOUT_POOL
        world, ids = _make_base_world(pool, rng)
        frame = TypedStateFrame.from_state_graph(world, full_mask)
        a0, a1 = ids["agents"][0], ids["agents"][1]
        o0 = ids["objects"][0]
        s0, s1 = ids["surfaces"][0], ids["surfaces"][1]

        kind = idx % 4
        if kind == 0:
            # Ditransitive: <actor> hands <recipient:agent> <object:object>
            segs = (("actor", a0), (None, " "), ("action", "hands"), (None, " "), ("recipient", a1), (None, " "), ("object", o0))
            aid = "give"
            roles = {"actor": a0, "object": o0, "recipient": a1}
            tag = "ditransitive_io_do"
        elif kind == 1:
            # Reversed double-object: <actor> gives <object:object> <recipient:agent>
            segs = (("actor", a0), (None, " "), ("action", "gives"), (None, " "), ("object", o0), (None, " "), ("recipient", a1))
            aid = "give"
            roles = {"actor": a0, "object": o0, "recipient": a1}
            tag = "ditransitive_do_io"
        elif kind == 2:
            # Polysemous transfer to agent -> give
            segs = (("actor", a0), (None, " "), ("action", "transfers"), (None, " "), ("object", o0), (None, " to "), ("recipient", a1))
            aid = "give"
            roles = {"actor": a0, "object": o0, "recipient": a1}
            tag = "polysemous_transfer_agent"
        else:
            # Polysemous transfer to surface with observed located_on -> move (source inferred from state)
            segs = (("actor", a0), (None, " "), ("action", "transfers"), (None, " "), ("object", o0), (None, " to "), ("destination", s1))
            aid = "move"
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
            tag = "polysemous_transfer_surface"

        raw, spans = _assemble_instruction(segs)
        cases.append(
            ParserEvalCase(
                case_id=f"s1_typed_s{seed}_{idx:02d}_{tag}",
                split_id="s1_state_typed_disambiguation",
                generator_id="gen_s1_state_typed_disambiguation_v1",
                seed=seed,
                raw_instruction_bytes=raw,
                before_frame=frame,
                gold_action_id=aid,
                gold_roles=tuple(sorted(roles.items())),
                gold_byte_spans=tuple(sorted(spans)),
                expected_ambiguous=False,
                expected_unparseable=False,
                metadata=(("disambiguation_kind", tag),),
            )
        )
    return cases


def _gen_s1_partial_mask_occluded(seed: int) -> list[ParserEvalCase]:
    """16 cases/seed under partial ObservationMasks hiding distractors and/or entity attributes."""
    rng = random.Random(seed * 79 + 3303)
    cases: list[ParserEvalCase] = []
    for idx in range(16):
        world, ids = _make_base_world(HELDOUT_POOL, rng)
        a0, a1 = ids["agents"][0], ids["agents"][1]
        o0 = ids["objects"][0]
        s0, s1 = ids["surfaces"][0], ids["surfaces"][1]
        c0 = ids["containers"][0]

        kind = idx % 4
        if kind == 0:
            # Partial mask hides surfaces & distractors; observes only actor + object + located_on
            mask = ObservationMask(
                observed_entities=frozenset({a0, o0}),
                observed_predicates=frozenset({"located_on"}),
                fully_observed=False,
            )
            segs = (("actor", a0), (None, " "), ("action", "move"), (None, " "), ("object", o0), (None, " from "), ("source", s0), (None, " to "), ("destination", s1))
            aid = "move"
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
        elif kind == 1:
            # Predicate-only mask: all entity attributes occluded!
            mask = ObservationMask(
                observed_entities=frozenset(),
                observed_predicates=frozenset({"owned_by"}),
                fully_observed=False,
            )
            segs = ((None, "Have "), ("actor", a0), (None, " "), ("action", "give"), (None, " "), ("object", o0), (None, " over to "), ("recipient", a1))
            aid = "give"
            roles = {"actor": a0, "object": o0, "recipient": a1}
        elif kind == 2:
            # Blind mask: completely occluded world state (tests syntax resilience without state)
            mask = ObservationMask(
                observed_entities=frozenset(),
                observed_predicates=frozenset(),
                fully_observed=False,
            )
            segs = ((None, "Please "), ("actor", a0), (None, " "), ("action", "open"), (None, " the "), ("target", c0))
            aid = "open"
            roles = {"actor": a0, "target": c0}
        else:
            # Single-entity mask observing only container
            mask = ObservationMask(
                observed_entities=frozenset({c0}),
                observed_predicates=frozenset({"is_locked", "is_open"}),
                fully_observed=False,
            )
            segs = ((None, "The "), ("target", c0), (None, " is "), ("action", "locked"), (None, " by "), ("actor", a0))
            aid = "lock"
            roles = {"actor": a0, "target": c0}

        frame = TypedStateFrame.from_state_graph(world, mask)
        raw, spans = _assemble_instruction(segs)
        cases.append(
            ParserEvalCase(
                case_id=f"s1_occl_s{seed}_{idx:02d}",
                split_id="s1_partial_mask_occluded",
                generator_id="gen_s1_partial_mask_occluded_v1",
                seed=seed,
                raw_instruction_bytes=raw,
                before_frame=frame,
                gold_action_id=aid,
                gold_roles=tuple(sorted(roles.items())),
                gold_byte_spans=tuple(sorted(spans)),
                expected_ambiguous=False,
                expected_unparseable=False,
                metadata=(("mask_kind", kind), ("fully_observed", False)),
            )
        )
    return cases


def _gen_s1_genuinely_ambiguous(seed: int) -> list[ParserEvalCase]:
    """16 genuinely ambiguous cases/seed where honest parsers MUST retain >= 2 candidate frames."""
    rng = random.Random(seed * 83 + 4404)
    cases: list[ParserEvalCase] = []
    full_mask = ObservationMask(fully_observed=True)

    for idx in range(16):
        world, ids = _make_base_world(HELDOUT_POOL, rng)
        a0, a1 = ids["agents"][0], ids["agents"][1]
        o0 = ids["objects"][0]
        s0, s1 = ids["surfaces"][0], ids["surfaces"][1]
        c0 = ids["containers"][0]

        kind = idx % 4
        if kind == 0:
            # Missing directional prepositions: "<actor> move <obj> <loc1> <loc2>"
            frame = TypedStateFrame.from_state_graph(world, full_mask)
            segs = (("actor", a0), (None, " "), ("action", "move"), (None, " "), ("object", o0), (None, " "), ("source", s0), (None, " "), ("destination", s1))
            aid = "move"
            roles = {"actor": a0, "object": o0, "source": s0, "destination": s1}
            tag = "unmarked_move_src_dst"
        elif kind == 1:
            # Polysemous "transfer" with occluded target type under partial mask
            mask = ObservationMask(
                observed_entities=frozenset({a0, o0}),
                observed_predicates=frozenset({"owned_by"}),
                fully_observed=False,
            )
            frame = TypedStateFrame.from_state_graph(world, mask)
            segs = (("actor", a0), (None, " "), ("action", "transfer"), (None, " "), ("object", o0), (None, " to "), ("recipient", a1))
            aid = "give"
            roles = {"actor": a0, "object": o0, "recipient": a1}
            tag = "polysemous_transfer_occluded_target"
        elif kind == 2:
            # Double-object "hand" with both args occluded under partial mask
            mask = ObservationMask(
                observed_entities=frozenset({a0}),
                observed_predicates=frozenset(),
                fully_observed=False,
            )
            frame = TypedStateFrame.from_state_graph(world, mask)
            segs = (("actor", a0), (None, " "), ("action", "hand"), (None, " "), ("recipient", a1), (None, " "), ("object", o0))
            aid = "give"
            roles = {"actor": a0, "object": o0, "recipient": a1}
            tag = "ditransitive_occluded_args"
        else:
            # Imperative omitted actor in a 3-agent world
            frame = TypedStateFrame.from_state_graph(world, full_mask)
            verb = "open" if (idx % 8 < 4) else "lock"
            segs = (("action", verb), (None, " the "), ("target", c0))
            aid = verb
            roles = {"actor": a0, "target": c0}
            tag = f"imperative_omitted_actor_{verb}"

        raw, spans = _assemble_instruction(segs)
        cases.append(
            ParserEvalCase(
                case_id=f"s1_amb_s{seed}_{idx:02d}_{tag}",
                split_id="s1_genuinely_ambiguous",
                generator_id="gen_s1_genuinely_ambiguous_v1",
                seed=seed,
                raw_instruction_bytes=raw,
                before_frame=frame,
                gold_action_id=aid,
                gold_roles=tuple(sorted(roles.items())),
                gold_byte_spans=tuple(sorted(spans)),
                expected_ambiguous=True,
                expected_unparseable=False,
                metadata=(("ambiguity_kind", tag),),
            )
        )
    return cases


def _gen_s1_unparseable_ood(seed: int) -> list[ParserEvalCase]:
    """10 unparseable / non-action / invalid-UTF-8 OOD inputs per seed requiring 100% abstention."""
    rng = random.Random(seed * 89 + 5505)
    world, ids = _make_base_world(HELDOUT_POOL, rng)
    frame = TypedStateFrame.from_state_graph(world, ObservationMask(fully_observed=True))
    a0, o0, s0, c0 = ids["agents"][0], ids["objects"][0], ids["surfaces"][0], ids["containers"][0]

    raw_inputs: list[tuple[str, bytes]] = [
        ("empty_bytes", b""),
        ("whitespace_only", b"   \t\n  "),
        ("invalid_utf8_bytes", b"\xff\xfe\xfd\x80"),
        ("wh_question", f"What is the final state of {o0}?".encode("utf-8")),
        ("declarative_located", f"{o0} is located on {s0}.".encode("utf-8")),
        ("stative_locked", f"{c0} is locked.".encode("utf-8")),
        ("stative_open", f"{c0} is open.".encode("utf-8")),
        ("incomplete_verb", f"{a0} move".encode("utf-8")),
        ("dangling_prep", f"{a0} move {o0} from".encode("utf-8")),
        ("gibberish_tokens", b"zxqjv blorp 12345"),
    ]
    cases: list[ParserEvalCase] = []
    for idx, (tag, raw) in enumerate(raw_inputs):
        cases.append(
            ParserEvalCase(
                case_id=f"s1_ood_s{seed}_{idx:02d}_{tag}",
                split_id="s1_unparseable_ood",
                generator_id="gen_s1_unparseable_ood_v1",
                seed=seed,
                raw_instruction_bytes=raw,
                before_frame=frame,
                gold_action_id="",
                gold_roles=(),
                gold_byte_spans=(),
                expected_ambiguous=False,
                expected_unparseable=True,
                metadata=(("ood_kind", tag),),
            )
        )
    return cases

def generate_s1_parser_suite(
    seed: int,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> dict[str, list[ParserEvalCase]]:
    """Build all 8 deterministic parser-only splits for one seed (zero hidden-rule usage)."""
    train_eps = generate_training_episodes(seed, demos_per_action=demos_per_action)
    s0_suite = generate_heldout_suite(seed, train_eps)

    train_cases: list[ParserEvalCase] = []
    for ep in train_eps:
        frame = TypedStateFrame.from_state_graph(ep.before_state, ep.observation_mask)
        train_cases.append(
            ParserEvalCase(
                case_id=f"s0_train_{ep.episode_id}",
                split_id="s0_train_instructions",
                generator_id=ep.generator_id,
                seed=seed,
                raw_instruction_bytes=ep.action.raw_instruction_bytes,
                before_frame=frame,
                gold_action_id=ep.action.action_id,
                gold_roles=ep.action.roles,
                gold_byte_spans=ep.action.byte_spans,
                expected_ambiguous=False,
                expected_unparseable=False,
                metadata=(("source_split", "train"),),
            )
        )

    s0_unseen_cases: list[ParserEvalCase] = []
    for split_name in UNSEEN_EVAL_SPLITS:
        if split_name == "heldout_temporal_frozen":
            continue
        for case in s0_suite[split_name]:
            frame = TypedStateFrame.from_state_graph(
                case.public.initial_state, case.public.observation_mask
            )
            for step_i, act in enumerate(case.public.actions):
                s0_unseen_cases.append(
                    ParserEvalCase(
                        case_id=f"{case.public.task_id}_step{step_i}",
                        split_id="s0_unseen_entities_and_compositions",
                        generator_id=case.public.generator_id,
                        seed=seed,
                        raw_instruction_bytes=act.raw_instruction_bytes,
                        before_frame=frame,
                        gold_action_id=act.action_id,
                        gold_roles=act.roles,
                        gold_byte_spans=act.byte_spans,
                        expected_ambiguous=False,
                        expected_unparseable=False,
                        metadata=(("source_split", split_name), ("step_index", step_i)),
                    )
                )

    s0_temporal_cases: list[ParserEvalCase] = []
    for case in s0_suite["heldout_temporal_frozen"]:
        frame = TypedStateFrame.from_state_graph(
            case.public.initial_state, case.public.observation_mask
        )
        for step_i, act in enumerate(case.public.actions):
            s0_temporal_cases.append(
                ParserEvalCase(
                    case_id=f"{case.public.task_id}_step{step_i}",
                    split_id="s0_temporal_frozen",
                    generator_id=case.public.generator_id,
                    seed=seed,
                    raw_instruction_bytes=act.raw_instruction_bytes,
                    before_frame=frame,
                    gold_action_id=act.action_id,
                    gold_roles=act.roles,
                    gold_byte_spans=act.byte_spans,
                    expected_ambiguous=False,
                    expected_unparseable=False,
                    metadata=(("source_split", "heldout_temporal_frozen"), ("step_index", step_i)),
                )
            )

    return {
        "s0_train_instructions": train_cases,
        "s0_unseen_entities_and_compositions": s0_unseen_cases,
        "s0_temporal_frozen": s0_temporal_cases,
        "s1_unseen_paraphrases": _gen_s1_unseen_paraphrases(seed),
        "s1_state_typed_disambiguation": _gen_s1_state_typed_disambiguation(seed),
        "s1_partial_mask_occluded": _gen_s1_partial_mask_occluded(seed),
        "s1_genuinely_ambiguous": _gen_s1_genuinely_ambiguous(seed),
        "s1_unparseable_ood": _gen_s1_unparseable_ood(seed),
    }


def build_s1_parser_manifest(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> dict[str, Any]:
    """Build deterministic cryptographic manifest for all 5-seed S1 parser-only splits."""
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        suite = generate_s1_parser_suite(seed, demos_per_action=demos_per_action)
        split_info: dict[str, Any] = {}
        total_cases = 0
        for split_name in S1_PARSER_SPLIT_ORDER:
            cases = suite[split_name]
            total_cases += len(cases)
            digest_bytes = canonical_json_bytes([c.to_manifest_dict() for c in cases])
            split_info[split_name] = {
                "case_count": len(cases),
                "expected_ambiguous_count": sum(1 for c in cases if c.expected_ambiguous),
                "expected_unparseable_count": sum(1 for c in cases if c.expected_unparseable),
                "sha256": sha256_hex(digest_bytes),
            }
        per_seed[str(seed)] = {
            "seed": seed,
            "total_cases": total_cases,
            "splits": split_info,
        }
    return {
        "schema_version": "s1-parser-manifest-v1",
        "s1_parser_generator_version": S1_PARSER_GENERATOR_VERSION,
        "s0_generator_version": GENERATOR_VERSION,
        "s0_temporal_generator_version": TEMPORAL_GENERATOR_VERSION,
        "seeds": list(seeds),
        "demos_per_action": demos_per_action,
        "split_order": list(S1_PARSER_SPLIT_ORDER),
        "per_seed": per_seed,
    }


def build_parser_factories() -> list[Callable[[], BaseFrameParser]]:
    """Return all parsers evaluated in the Stage S1 parser-only ablation matrix."""
    return [
        lambda: BilingualCCGFrameAdapter(),
        lambda: TypedRoleFrameParser(use_state_types=False),
        lambda: TypedRoleFrameParser(use_state_types=True),
    ]


def _profile_parser_peak_heap_bytes(
    factory: Callable[[], BaseFrameParser],
    suite: Mapping[str, Sequence[ParserEvalCase]],
) -> int:
    """Measure peak Python heap in an isolated pass so tracemalloc never distorts CPU timing."""
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    try:
        parser = factory()
        for split_name in S1_PARSER_SPLIT_ORDER:
            for case in suite[split_name][:5]:
                parser.parse(case.raw_instruction_bytes, case.before_frame)
        _cur, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        if not was_tracing:
            tracemalloc.stop()


def evaluate_parser_on_seed(
    factory: Callable[[], BaseFrameParser],
    seed: int,
    *,
    suite: Mapping[str, Sequence[ParserEvalCase]] | None = None,
) -> dict[str, Any]:
    """Evaluate one parser across all 8 S1 parser splits on a single seed."""
    active_suite = suite if suite is not None else generate_s1_parser_suite(seed)
    parser = factory()

    per_split: dict[str, Any] = {}
    unambiguous_scores: list[ParserCaseScore] = []
    failed_samples: list[dict[str, Any]] = []
    total_cases = 0
    total_parse_s = 0.0

    for split_name in S1_PARSER_SPLIT_ORDER:
        cases = active_suite[split_name]
        t0 = time.perf_counter()
        results = [parser.parse(c.raw_instruction_bytes, c.before_frame) for c in cases]
        elapsed = time.perf_counter() - t0
        total_parse_s += elapsed
        total_cases += len(cases)

        scores = [score_parser_case(c, r) for c, r in zip(cases, results)]
        agg = aggregate_parser_scores(scores)
        agg["elapsed_seconds"] = round(elapsed, 6)
        per_split[split_name] = agg

        if split_name in UNAMBIGUOUS_EVAL_SPLITS:
            unambiguous_scores.extend(scores)

        for case, res, sc in zip(cases, results, scores):
            failed = (
                (case.expected_unparseable and not sc.ood_abstention_correct)
                or (case.expected_ambiguous and not (sc.ambiguity_flag_correct and sc.top_k_gold_recall))
                or ((not case.expected_unparseable) and (not case.expected_ambiguous) and not sc.exact_role_match)
            )
            if failed and len(failed_samples) < 8:
                try:
                    inst_preview = case.raw_instruction_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    inst_preview = repr(case.raw_instruction_bytes)
                failed_samples.append(
                    {
                        "case_id": case.case_id,
                        "split_id": split_name,
                        "instruction": inst_preview,
                        "gold_action_id": case.gold_action_id,
                        "gold_roles": case.gold_role_dict(),
                        "expected_ambiguous": case.expected_ambiguous,
                        "expected_unparseable": case.expected_unparseable,
                        "pred_abstained": res.abstained,
                        "pred_is_ambiguous": res.is_ambiguous,
                        "pred_candidate_count": len(res.candidate_frames),
                        "pred_top_action": res.top_frame.action_id if res.top_frame else None,
                        "pred_top_roles": res.top_frame.role_dict() if res.top_frame else None,
                    }
                )

    peak_bytes = _profile_parser_peak_heap_bytes(factory, active_suite)
    unseen_unamb_agg = aggregate_parser_scores(unambiguous_scores)
    pps = total_cases / total_parse_s if total_parse_s > 0 else 0.0

    return {
        "parser": parser.name,
        "parser_version": parser.version,
        "seed": seed,
        "total_cases": total_cases,
        "parse_seconds": round(total_parse_s, 6),
        "parses_per_sec_single_core": round(pps, 2),
        "tracemalloc_peak_kb": round(peak_bytes / 1024.0, 2),
        "unseen_unambiguous_overall": unseen_unamb_agg,
        "per_split": per_split,
        "failed_examples_sample": failed_samples,
    }
