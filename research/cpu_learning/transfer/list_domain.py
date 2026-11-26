"""Stage S6 Transfer Domain 2: Typed List Operators & Multi-Step Compositions.

SCIENTIFIC PRINCIPLE (PDF pp. 12, 13):
- Induces executable ``FunctionalOperatorCard``s from ``<= 6`` short (``len <= 5``)
  before/action/after list demonstrations using the exact same ``TransferProgramInducer``
  (ZERO hardcoded operator definitions in the learner).
- Evaluates on unseen long sequences (``len 8..20``), unseen large & negative integers
  (``[-500, 500]``), empty/singleton edge cases, and held-out Depth-2 and Depth-3
  multi-step list pipelines while auditing zero collateral register corruption.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Sequence

from research.cpu_learning.transfer.transfer_inducer import (
    FunctionalOperatorCard,
    SlotTransitionEpisode,
    TransferProgramInducer,
)

LIST_DOMAIN_SCHEMA_VERSION: str = "s6-list-domain-v1"
LIST_DEMOS_PER_OP: int = 6

# Evaluator-side ground truth used strictly to generate small training demos & grade unseen tests
_ATOMIC_LIST_SPECS: dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]] = {
    "reverse": lambda xs: tuple(reversed(xs)),
    "filter_even": lambda xs: tuple(x for x in xs if x % 2 == 0),
    "filter_positive": lambda xs: tuple(x for x in xs if x > 0),
    "map_double": lambda xs: tuple(x * 2 for x in xs),
    "map_increment": lambda xs: tuple(x + 1 for x in xs),
    "sort_asc": lambda xs: tuple(sorted(xs)),
}

DEPTH2_PIPELINES: tuple[tuple[str, str], ...] = (
    ("filter_positive", "map_double"),
    ("filter_even", "reverse"),
    ("map_increment", "sort_asc"),
    ("reverse", "filter_positive"),
)

DEPTH3_PIPELINES: tuple[tuple[str, str, str], ...] = (
    ("filter_positive", "map_double", "reverse"),
    ("filter_even", "sort_asc", "map_increment"),
    ("reverse", "filter_positive", "sort_asc"),
    ("map_increment", "filter_even", "reverse"),
)


def _generate_short_training_lists(seed: int) -> list[tuple[int, ...]]:
    """Generate 6 diverse short training lists (len <= 5, values in [-9, 9]) ensuring
    non-degenerate distinctions (e.g., unsorted, mixed parity, negative/zero/positive)."""
    rng = random.Random(seed * 401 + 19)
    base = [
        (3, -2, 0, 5),
        (-4, 7, 2, -1, 6),
        (8, 1, -3),
        (0, 4, -5, 2),
    ]
    for _ in range(2):
        k = rng.randint(3, 5)
        cand = tuple(rng.randint(-9, 9) for _ in range(k))
        if list(cand) == sorted(cand) or list(cand) == list(reversed(cand)):
            cand = (5, -1, 4, 0)
        base.append(cand)
    return base[:LIST_DEMOS_PER_OP]


def build_list_training_episodes(seed: int) -> list[SlotTransitionEpisode]:
    """Construct ``6`` short (``len <= 5``) before/action/after episodes per atomic list op."""
    short_lists = _generate_short_training_lists(seed)
    episodes: list[SlotTransitionEpisode] = []
    for op_name, spec_fn in sorted(_ATOMIC_LIST_SPECS.items()):
        for idx, xs in enumerate(short_lists):
            ys = spec_fn(xs)
            episodes.append(
                SlotTransitionEpisode(
                    episode_id=f"list_tr_{op_name}_s{seed}_{idx:02d}",
                    action_id=op_name,
                    before_slots=(
                        ("items", xs),
                        ("tag_reg", f"ctx_{seed}_{idx}"),
                        ("step_reg", idx),
                    ),
                    after_slots=(
                        ("items", ys),
                        ("tag_reg", f"ctx_{seed}_{idx}"),
                        ("step_reg", idx),
                    ),
                    input_slot_order=("items",),
                )
            )
    return episodes


def _execute_list_pipeline(
    cards: dict[str, FunctionalOperatorCard],
    pipeline: Sequence[str],
    initial_items: tuple[int, ...],
    *,
    tag_sentinel: str,
    step_sentinel: int,
) -> tuple[tuple[int, ...], int]:
    """Execute a 1..k step sequence of induced ``FunctionalOperatorCard``s on ``initial_items``."""
    state = {
        "items": initial_items,
        "tag_reg": tag_sentinel,
        "step_reg": step_sentinel,
    }
    collateral = 0
    for op_name in pipeline:
        card = cards[op_name]
        state = card.apply_slots(state, use_sandbox=True)
        if state.get("tag_reg") != tag_sentinel or state.get("step_reg") != step_sentinel:
            collateral += 1
    return (tuple(state["items"]), collateral)


def _gold_list_pipeline(
    pipeline: Sequence[str],
    initial_items: tuple[int, ...],
) -> tuple[int, ...]:
    cur = initial_items
    for op_name in pipeline:
        cur = _ATOMIC_LIST_SPECS[op_name](cur)
    return cur


def run_list_domain_suite(seed: int) -> dict[str, Any]:
    """Train ``TransferProgramInducer`` on ``len <= 5`` list demos and evaluate unseen
    ``len 8..20`` lists, ``[-500, 500]`` values, edge cases, and depth-2/3 compositions."""
    train_eps = build_list_training_episodes(seed)
    # Verify TypedStateFrame conversion on list episodes
    b_f, a_f = train_eps[0].to_typed_frames()
    frame_ok = len(b_f.entities) == 3 and len(a_f.entities) == 3

    inducer = TransferProgramInducer(use_shard_guidance=True, max_candidates=160)
    inducer.fit_episodes(train_eps)
    inducer.freeze()
    cards = inducer.cards

    rng = random.Random(seed * 733 + 91)
    total_collateral = 0

    # 1. Single-step unseen length (8..20), unseen [-500, 500] range, + empty/singleton edge cases
    per_op_acc: dict[str, float] = {}
    atomic_hits = 0
    atomic_total = 0

    for op_name in sorted(_ATOMIC_LIST_SPECS.keys()):
        test_inputs: list[tuple[int, ...]] = [
            (),
            (0,),
            (-317,),
            (412, -412, 0, 412, -19, 88, -1, 250),
        ]
        for _ in range(20):
            length = rng.randint(8, 20)
            test_inputs.append(tuple(rng.randint(-500, 500) for _ in range(length)))

        hits = 0
        for idx, xs in enumerate(test_inputs):
            pred, coll = _execute_list_pipeline(
                cards,
                (op_name,),
                xs,
                tag_sentinel=f"ho_{op_name}_{idx}",
                step_sentinel=idx + 100,
            )
            total_collateral += coll
            if pred == _gold_list_pipeline((op_name,), xs):
                hits += 1
        per_op_acc[op_name] = round(hits / float(len(test_inputs)), 6)
        atomic_hits += hits
        atomic_total += len(test_inputs)

    # 2. Held-out Depth-2 Multi-Step Compositions (len 8..20, values in [-500, 500])
    d2_hits = 0
    d2_total = 0
    for p_idx, pipe in enumerate(DEPTH2_PIPELINES):
        for t_idx in range(8):
            xs = () if t_idx == 0 else tuple(rng.randint(-500, 500) for _ in range(rng.randint(8, 20)))
            pred, coll = _execute_list_pipeline(
                cards,
                pipe,
                xs,
                tag_sentinel=f"d2_{p_idx}_{t_idx}",
                step_sentinel=t_idx,
            )
            total_collateral += coll
            if pred == _gold_list_pipeline(pipe, xs):
                d2_hits += 1
            d2_total += 1

    # 3. Held-out Depth-3 Multi-Step Compositions (len 8..20, values in [-500, 500])
    d3_hits = 0
    d3_total = 0
    for p_idx, pipe in enumerate(DEPTH3_PIPELINES):
        for t_idx in range(8):
            xs = () if t_idx == 0 else tuple(rng.randint(-500, 500) for _ in range(rng.randint(8, 20)))
            pred, coll = _execute_list_pipeline(
                cards,
                pipe,
                xs,
                tag_sentinel=f"d3_{p_idx}_{t_idx}",
                step_sentinel=t_idx,
            )
            total_collateral += coll
            if pred == _gold_list_pipeline(pipe, xs):
                d3_hits += 1
            d3_total += 1

    overall_hits = atomic_hits + d2_hits + d3_hits
    overall_total = atomic_total + d2_total + d3_total

    return {
        "schema_version": LIST_DOMAIN_SCHEMA_VERSION,
        "seed": seed,
        "train_demos_per_op": LIST_DEMOS_PER_OP,
        "train_max_length": 5,
        "typed_frame_bridge_verified": frame_ok,
        "atomic_unseen_len8_20_accuracy": round(atomic_hits / float(atomic_total), 6),
        "per_operator_accuracy": per_op_acc,
        "depth2_composition_accuracy": round(d2_hits / float(d2_total), 6),
        "depth3_composition_accuracy": round(d3_hits / float(d3_total), 6),
        "overall_accuracy": round(overall_hits / float(overall_total), 6),
        "collateral_corruptions": total_collateral,
        "bit_budget": inducer.bit_budget_summary(),
        "executable_cards": {k: c.to_executable_dict() for k, c in sorted(cards.items())},
    }
