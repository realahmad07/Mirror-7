"""Stage S6 Transfer Domain 1: Modular Addition Canary & Multi-Digit Column Arithmetic.

SCIENTIFIC PRINCIPLE (PDF pp. 3, 12, 13):
- Uses the SAME before/action/after programmatic learner (``TransferProgramInducer``) with
  ZERO manually inserted arithmetic rules.
- Models single-column digit addition as ``SlotTransitionEpisode``s (bridged to ``StateGraph``
  and ``TypedStateFrame``) over ``(d_a, d_b, carry_in, radix) -> (digit_out, carry_out)`` while
  preserving unrelated background registers.
- Trains on ``<= 16`` single-column demonstrations (ZERO multi-digit training examples) and
  extrapolates right-to-left across unseen 3-, 4-, 6-, and 8-digit integers, carry cascades
  (e.g. ``99999999 + 1``), and unseen novel radices (``base=7`` and ``base=16``).
"""
from __future__ import annotations

import random
from typing import Any

from research.cpu_learning.transfer.transfer_inducer import (
    FunctionalOperatorCard,
    SlotTransitionEpisode,
    TransferProgramInducer,
)

MATH_DOMAIN_SCHEMA_VERSION: str = "s6-math-domain-v1"
COLUMN_TRAIN_SAMPLE_SIZE: int = 16


def _int_to_lsd_digits(n: int, radix: int, min_len: int = 1) -> list[int]:
    if n < 0:
        raise ValueError("Expected non-negative integer")
    digits: list[int] = []
    cur = int(n)
    while cur > 0:
        digits.append(cur % radix)
        cur //= radix
    while len(digits) < min_len:
        digits.append(0)
    return digits


def _lsd_digits_to_int(digits: list[int], radix: int) -> int:
    val = 0
    mult = 1
    for d in digits:
        val += int(d) * mult
        mult *= radix
    return val


def execute_multidigit_addition(
    column_card: FunctionalOperatorCard,
    num_a: int,
    num_b: int,
    radix: int,
) -> tuple[int, int]:
    """Iteratively apply an induced single-column ``FunctionalOperatorCard`` right-to-left.

    Returns ``(predicted_sum, collateral_corruptions)`` where ``collateral_corruptions``
    counts any unintended mutation to preserved background register slots.
    """
    width = max(
        len(_int_to_lsd_digits(num_a, radix)),
        len(_int_to_lsd_digits(num_b, radix)),
    )
    da_list = _int_to_lsd_digits(num_a, radix, min_len=width)
    db_list = _int_to_lsd_digits(num_b, radix, min_len=width)

    carry = 0
    out_digits: list[int] = []
    collateral = 0

    for col_idx in range(width):
        sentinel_tag = f"col_guard_{col_idx}"
        before = {
            "d_a": da_list[col_idx],
            "d_b": db_list[col_idx],
            "carry_in": carry,
            "radix": radix,
            "digit_out": -1,
            "carry_out": -1,
            "aux_tag": sentinel_tag,
        }
        after = column_card.apply_slots(before, use_sandbox=True)
        if after.get("aux_tag") != sentinel_tag or after.get("radix") != radix:
            collateral += 1
        d_out = int(after["digit_out"])
        carry = int(after["carry_out"])
        out_digits.append(d_out)

    if carry > 0:
        out_digits.append(carry)

    return (_lsd_digits_to_int(out_digits, radix), collateral)


def evaluate_modular_addition_canary(seed: int) -> dict[str, Any]:
    """Evaluate S0's exact modular addition canary ``(a + b) % 10`` via ``TransferProgramInducer``.

    Uses the exact same seed shuffle & 60/40 split protocol as ``shard_canary.py`` where
    pure HDC similarity scored ``4.00%`` and SHARD bit-perceptron scored ``7.00%``, PLUS
    evaluates a sparse 16-example regime (84 unseen test pairs).
    """
    all_pairs = [(a, b) for a in range(10) for b in range(10)]
    rng = random.Random(seed * 1009 + 7)
    pairs = list(all_pairs)
    rng.shuffle(pairs)

    train_60 = pairs[:60]
    test_40 = pairs[60:]

    # Build before/action/after episodes (NO radix=10 slot provided!)
    episodes_60 = [
        SlotTransitionEpisode(
            episode_id=f"mod10_s{seed}_{i:02d}",
            action_id="digit_pair_mod_op",
            before_slots=(("d_a", a), ("d_b", b), ("out_digit", -1), ("ctx_reg", seed % 11)),
            after_slots=(("d_a", a), ("d_b", b), ("out_digit", (a + b) % 10), ("ctx_reg", seed % 11)),
            input_slot_order=("d_a", "d_b"),
        )
        for i, (a, b) in enumerate(train_60)
    ]
    # Verify StateGraph & TypedStateFrame bridge on first episode
    b_frame, a_frame = episodes_60[0].to_typed_frames()
    frames_valid = len(b_frame.entities) == 4 and len(a_frame.entities) == 4

    inducer_60 = TransferProgramInducer(use_shard_guidance=True, max_candidates=120)
    inducer_60.fit_episodes(episodes_60)
    inducer_60.freeze()
    card_60 = inducer_60.get_card("digit_pair_mod_op")
    assert card_60 is not None

    hits_40 = 0
    collateral_40 = 0
    for a, b in test_40:
        sentinel = f"canary_{a}_{b}"
        after = card_60.apply_slots(
            {"d_a": a, "d_b": b, "out_digit": -1, "ctx_reg": sentinel},
            use_sandbox=True,
        )
        if after.get("ctx_reg") != sentinel:
            collateral_40 += 1
        if int(after["out_digit"]) == (a + b) % 10:
            hits_40 += 1

    # Also test sparse 16-demo regime (ensuring output 9 is observed at least once so span=10 is identifiable)
    sparse_16 = list(train_60[:15])
    if not any((a + b) % 10 == 9 for a, b in sparse_16):
        for cand_p in train_60[15:]:
            if (cand_p[0] + cand_p[1]) % 10 == 9:
                sparse_16.append(cand_p)
                break
    else:
        sparse_16.append(train_60[15])
    sparse_set = set(sparse_16)
    unseen_84 = [p for p in all_pairs if p not in sparse_set]

    episodes_16 = [
        SlotTransitionEpisode(
            episode_id=f"mod10_sparse_s{seed}_{i:02d}",
            action_id="digit_pair_mod_op",
            before_slots=(("d_a", a), ("d_b", b), ("out_digit", -1)),
            after_slots=(("d_a", a), ("d_b", b), ("out_digit", (a + b) % 10)),
            input_slot_order=("d_a", "d_b"),
        )
        for i, (a, b) in enumerate(sparse_16)
    ]
    inducer_16 = TransferProgramInducer(use_shard_guidance=True, max_candidates=120)
    inducer_16.fit_episodes(episodes_16)
    inducer_16.freeze()
    card_16 = inducer_16.get_card("digit_pair_mod_op")
    assert card_16 is not None
    hits_84 = sum(
        1
        for a, b in unseen_84
        if int(card_16.apply_slots({"d_a": a, "d_b": b, "out_digit": -1})["out_digit"]) == (a + b) % 10
    )

    return {
        "exact_canary_60_40_accuracy": round(hits_40 / float(len(test_40)), 6),
        "sparse_16_84_accuracy": round(hits_84 / float(len(unseen_84)), 6),
        "collateral_corruptions": collateral_40,
        "typed_frame_bridge_verified": frames_valid,
        "induced_python": dict(card_60.python_sources),
        "card_compressed_bits": card_60.compressed_bits(),
        "executable_card": card_60.to_executable_dict(),
    }


def _sample_column_training_episodes(
    seed: int,
    radix: int,
    sample_size: int = COLUMN_TRAIN_SAMPLE_SIZE,
) -> list[SlotTransitionEpisode]:
    """Sample ``sample_size <= 16`` single-column addition demos (0 multi-digit examples)."""
    rng = random.Random(seed * 313 + radix * 97)
    no_wrap_c0: list[tuple[int, int, int]] = []
    wrap_c0: list[tuple[int, int, int]] = []
    no_wrap_c1: list[tuple[int, int, int]] = []
    tip_c1: list[tuple[int, int, int]] = []
    wrap_c1: list[tuple[int, int, int]] = []

    for da in range(radix):
        for db in range(radix):
            for cin in (0, 1):
                tot = da + db + cin
                if cin == 0 and tot < radix:
                    no_wrap_c0.append((da, db, cin))
                elif cin == 0 and tot >= radix:
                    wrap_c0.append((da, db, cin))
                elif cin == 1 and tot < radix:
                    no_wrap_c1.append((da, db, cin))
                elif cin == 1 and da + db == radix - 1:
                    tip_c1.append((da, db, cin))
                else:
                    wrap_c1.append((da, db, cin))

    q = max(2, sample_size // 4)
    n_tip = min(2, len(tip_c1))
    n_rest_c1 = max(1, sample_size - 3 * q - n_tip)
    chosen = (
        rng.sample(no_wrap_c0, min(q, len(no_wrap_c0)))
        + rng.sample(wrap_c0, min(q, len(wrap_c0)))
        + rng.sample(no_wrap_c1, min(q, len(no_wrap_c1)))
        + rng.sample(tip_c1, n_tip)
        + rng.sample(wrap_c1, min(n_rest_c1, len(wrap_c1)))
    )
    rng.shuffle(chosen)

    episodes: list[SlotTransitionEpisode] = []
    for idx, (da, db, cin) in enumerate(chosen):
        tot = da + db + cin
        episodes.append(
            SlotTransitionEpisode(
                episode_id=f"col_b{radix}_s{seed}_{idx:02d}",
                action_id=f"column_step_b{radix}",
                before_slots=(
                    ("d_a", da),
                    ("d_b", db),
                    ("carry_in", cin),
                    ("radix", radix),
                    ("digit_out", -1),
                    ("carry_out", -1),
                    ("aux_tag", f"train_{idx}"),
                ),
                after_slots=(
                    ("d_a", da),
                    ("d_b", db),
                    ("carry_in", cin),
                    ("radix", radix),
                    ("digit_out", tot % radix),
                    ("carry_out", tot // radix),
                    ("aux_tag", f"train_{idx}"),
                ),
                input_slot_order=("d_a", "d_b", "carry_in", "radix"),
            )
        )
    return episodes


def evaluate_multidigit_and_radix_transfer(seed: int) -> dict[str, Any]:
    """Train on 16 single-column triples and test unseen 3,4,6,8-digit + novel base-7 & base-16."""
    # 1. Base-10 single-column induction (16 single-column demos, 0 multi-digit demos)
    b10_eps = _sample_column_training_episodes(seed, radix=10, sample_size=COLUMN_TRAIN_SAMPLE_SIZE)
    inducer_b10 = TransferProgramInducer(use_shard_guidance=True, max_candidates=320)
    inducer_b10.fit_episodes(b10_eps)
    inducer_b10.freeze()
    card_b10 = inducer_b10.get_card("column_step_b10")
    assert card_b10 is not None

    rng = random.Random(seed * 617 + 29)
    per_length: dict[str, dict[str, Any]] = {}
    total_hits = 0
    total_tasks = 0
    total_collateral = 0

    for n_digits in (3, 4, 6, 8):
        lo = 10 ** (n_digits - 1)
        hi = (10 ** n_digits) - 1
        hits = 0
        n_cases = 16
        for _ in range(n_cases):
            na = rng.randint(lo, hi)
            nb = rng.randint(lo, hi)
            pred, coll = execute_multidigit_addition(card_b10, na, nb, radix=10)
            total_collateral += coll
            if pred == na + nb:
                hits += 1
        per_length[f"digits_{n_digits}"] = {
            "exact_accuracy": round(hits / float(n_cases), 6),
            "n_tasks": n_cases,
        }
        total_hits += hits
        total_tasks += n_cases

    # Deterministic carry-cascade stress cases (including 99999999 + 1)
    cascade_pairs = [
        (999, 1),
        (999, 999),
        (9999, 1),
        (9999, 9999),
        (999999, 1),
        (999999, 888888),
        (99999999, 1),
        (99999999, 99999999),
        (50505050, 49494950),
        (90909099, 9090901),
        (19999999, 80000001),
        (76543219, 23456781),
    ]
    c_hits = 0
    for na, nb in cascade_pairs:
        pred, coll = execute_multidigit_addition(card_b10, na, nb, radix=10)
        total_collateral += coll
        if pred == na + nb:
            c_hits += 1
    per_length["carry_cascade_stress"] = {
        "exact_accuracy": round(c_hits / float(len(cascade_pairs)), 6),
        "n_tasks": len(cascade_pairs),
    }
    total_hits += c_hits
    total_tasks += len(cascade_pairs)

    # 2. Novel Radix Transfer: Base-7 (16 single-column demos -> 24 unseen 5-digit base-7 sums)
    b7_eps = _sample_column_training_episodes(seed, radix=7, sample_size=COLUMN_TRAIN_SAMPLE_SIZE)
    inducer_b7 = TransferProgramInducer(use_shard_guidance=True, max_candidates=320)
    inducer_b7.fit_episodes(b7_eps)
    inducer_b7.freeze()
    card_b7 = inducer_b7.get_card("column_step_b7")
    assert card_b7 is not None

    lo7, hi7 = 7 ** 4, (7 ** 5) - 1
    b7_pairs = [(hi7, 1), (hi7, hi7)] + [
        (rng.randint(lo7, hi7), rng.randint(lo7, hi7)) for _ in range(22)
    ]
    b7_hits = 0
    for na, nb in b7_pairs:
        pred, coll = execute_multidigit_addition(card_b7, na, nb, radix=7)
        total_collateral += coll
        if pred == na + nb:
            b7_hits += 1

    # 3. Novel Radix Transfer: Base-16 (16 single-column demos -> 24 unseen 5-digit hex sums)
    b16_eps = _sample_column_training_episodes(seed, radix=16, sample_size=COLUMN_TRAIN_SAMPLE_SIZE)
    inducer_b16 = TransferProgramInducer(use_shard_guidance=True, max_candidates=320)
    inducer_b16.fit_episodes(b16_eps)
    inducer_b16.freeze()
    card_b16 = inducer_b16.get_card("column_step_b16")
    assert card_b16 is not None

    lo16, hi16 = 16 ** 4, (16 ** 5) - 1
    b16_pairs = [(hi16, 1), (hi16, hi16)] + [
        (rng.randint(lo16, hi16), rng.randint(lo16, hi16)) for _ in range(22)
    ]
    b16_hits = 0
    for na, nb in b16_pairs:
        pred, coll = execute_multidigit_addition(card_b16, na, nb, radix=16)
        total_collateral += coll
        if pred == na + nb:
            b16_hits += 1

    return {
        "train_single_column_demos": len(b10_eps),
        "train_multidigit_demos": 0,
        "multidigit_overall_accuracy": round(total_hits / float(total_tasks), 6),
        "multidigit_per_split": per_length,
        "novel_base7_5digit_accuracy": round(b7_hits / float(len(b7_pairs)), 6),
        "novel_base16_5digit_accuracy": round(b16_hits / float(len(b16_pairs)), 6),
        "collateral_corruptions": total_collateral,
        "induced_b10_python": dict(card_b10.python_sources),
        "b10_card_compressed_bits": card_b10.compressed_bits(),
        "b7_card_compressed_bits": card_b7.compressed_bits(),
        "b16_card_compressed_bits": card_b16.compressed_bits(),
        "executable_cards": {
            "column_step_b10": card_b10.to_executable_dict(),
            "column_step_b7": card_b7.to_executable_dict(),
            "column_step_b16": card_b16.to_executable_dict(),
        },
    }


def build_math_training_payload(seed: int) -> dict[str, Any]:
    """Return canonical serializable math-domain training episodes for exact bit-budget accounting."""
    all_pairs = [(a, b) for a in range(10) for b in range(10)]
    rng = random.Random(seed * 1009 + 7)
    pairs = list(all_pairs)
    rng.shuffle(pairs)
    train_60 = pairs[:60]
    canary_eps = [
        SlotTransitionEpisode(
            episode_id=f"mod10_s{seed}_{i:02d}",
            action_id="digit_pair_mod_op",
            before_slots=(("d_a", a), ("d_b", b), ("out_digit", -1), ("ctx_reg", seed % 11)),
            after_slots=(("d_a", a), ("d_b", b), ("out_digit", (a + b) % 10), ("ctx_reg", seed % 11)),
            input_slot_order=("d_a", "d_b"),
        ).to_dict()
        for i, (a, b) in enumerate(train_60)
    ]
    return {
        "canary_60": canary_eps,
        "column_b10": [e.to_dict() for e in _sample_column_training_episodes(seed, 10, COLUMN_TRAIN_SAMPLE_SIZE)],
        "column_b7": [e.to_dict() for e in _sample_column_training_episodes(seed, 7, COLUMN_TRAIN_SAMPLE_SIZE)],
        "column_b16": [e.to_dict() for e in _sample_column_training_episodes(seed, 16, COLUMN_TRAIN_SAMPLE_SIZE)],
    }


def run_math_domain_suite(seed: int) -> dict[str, Any]:
    """Run complete Stage S6 math transfer evaluation for ``seed``."""
    canary = evaluate_modular_addition_canary(seed)
    multidigit = evaluate_multidigit_and_radix_transfer(seed)
    return {
        "schema_version": MATH_DOMAIN_SCHEMA_VERSION,
        "seed": seed,
        "modular_addition_canary": canary,
        "multidigit_and_radix": multidigit,
    }
