"""Independent 5-seed CPU reproduction of the SHARD HDC bit-learning arithmetic canary.

Reproduces the three digit-pair tasks referenced in the research handoff (PDF pp. 3, 9):
1. ``comparison``: ``a > b`` on unseen digit pairs (a, b) in {0..9}^2.
2. ``carry``: ``(a + b) >= 10`` on unseen digit pairs.
3. ``modular_addition``: ``(a + b) % 10`` on unseen digit pairs (the key S6 canary).

Reuses Mirror 7's 4,096-bit HDC primitives (``mirror_crystal.hdc_associative_engine``)
with graded thermometer-correlated level hypervectors and a bit-exact 2's-complement
vertical bit-sliced signed accumulator for local CPU bit-perceptron readout updates
(0 floats, 0 backprop, explicit accounting of learned readout bits).
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Sequence

from mirror_crystal.hdc_associative_engine import (
    D,
    MASK,
    FastBitSlicedBundle,
    hamming_sim,
    hv_of,
)
from research.cpu_learning.evaluation.metrics import summarize_across_seeds
from research.cpu_learning.simulator.generators import DEFAULT_SEEDS

CANARY_TASKS: dict[str, tuple[int, Callable[[int, int], int]]] = {
    "comparison": (2, lambda a, b: 1 if a > b else 0),
    "carry": (2, lambda a, b: 1 if (a + b) >= 10 else 0),
    "modular_addition": (10, lambda a, b: (a + b) % 10),
}


def _permute(hv: int, shift: int) -> int:
    s = shift % D
    if s == 0:
        return hv & MASK
    return ((hv << s) | (hv >> (D - s))) & MASK


class SignedBitSlicedAccumulator:
    """Bit-exact K-bit 2's-complement signed accumulator across all D=4,096 bits in parallel.

    Performs local bipolar updates (+1 where v_i=1, -1 where v_i=0) in K bitwise passes
    with zero floating-point operations. Sign-bit plane ``planes[K-1]`` directly yields
    the binarized readout hypervector via one bitwise NOT.
    """

    def __init__(self, bits: int = 12) -> None:
        self.K = max(4, int(bits))
        self.planes: list[int] = [0] * self.K
        self._updates: int = 0

    def _ensure_headroom(self) -> None:
        # Dynamic 2's-complement sign extension prevents silent sign-flip wraparound
        # when cumulative updates approach 2^(K-1) - 1.
        if self._updates + 1 >= (1 << (self.K - 1)):
            self.planes.append(self.planes[self.K - 1])
            self.K += 1

    def add_bipolar(self, v: int) -> None:
        """Add +1 where bit i of v is 1, and -1 where bit i of v is 0 (2*v - 1)."""
        self._ensure_headroom()
        self._updates += 1
        v = v & MASK
        planes = self.planes
        K = self.K
        # Step 1: subtract 1 from every lane (2's-complement borrow ripple)
        borrow = MASK
        for k in range(K):
            pk = planes[k]
            new_borrow = ((~pk) & MASK) & borrow
            planes[k] = pk ^ borrow
            borrow = new_borrow
            if not borrow:
                break
        # Step 2: add +2 to lanes where bit i of v is 1 (carry ripple starting at plane 1)
        carry = v
        for k in range(1, K):
            pk = planes[k]
            new_carry = pk & carry
            planes[k] = pk ^ carry
            carry = new_carry
            if not carry:
                break

    def sub_bipolar(self, v: int) -> None:
        """Subtract bipolar(v) == add bipolar(~v)."""
        self.add_bipolar((~v) & MASK)

    def sign_hv(self) -> int:
        """Binarized D-bit readout hypervector: bit i is 1 iff accumulator[i] >= 0."""
        return (~self.planes[self.K - 1]) & MASK


def _build_graded_digit_hvs(seed: int) -> list[int]:
    """Build 10 graded level-correlated 4,096-bit HVs so nearby digits share bits."""
    rng = random.Random(seed * 7919 + 42)
    base = hv_of(f"shard_canary_base:{seed}".encode("utf-8"))
    bit_order = list(range(D))
    rng.shuffle(bit_order)
    step = D // 10
    levels = [base]
    cur = base
    for k in range(1, 10):
        flip_mask = 0
        for b in bit_order[(k - 1) * step : k * step]:
            flip_mask |= 1 << b
        cur = (cur ^ flip_mask) & MASK
        levels.append(cur)
    return levels


def _encode_pair_hv(
    a: int,
    b: int,
    levels: Sequence[int],
    role_a: int,
    role_b: int,
    role_ab: int,
) -> int:
    bd = FastBitSlicedBundle(bits=8)
    va = role_a ^ levels[a]
    vb = role_b ^ levels[b]
    vab = role_ab ^ (levels[a] ^ _permute(levels[b], 17))
    bd.add(va, 2)
    bd.add(vb, 2)
    bd.add(vab, 1)
    return bd.binary()


def _eval_pure_hdc_similarity(
    train_hvs: Sequence[tuple[int, int]],
    test_hvs: Sequence[tuple[int, int]],
    num_classes: int,
) -> float:
    bundles = [FastBitSlicedBundle(bits=10) for _ in range(num_classes)]
    for hv, label in train_hvs:
        bundles[label].add(hv, 1)
    protos = [bd.binary() for bd in bundles]
    correct = 0
    for hv, label in test_hvs:
        best_c = max(range(num_classes), key=lambda c: hamming_sim(hv, protos[c]))
        if best_c == label:
            correct += 1
    return correct / float(max(1, len(test_hvs)))


def _eval_bit_perceptron_readout(
    train_hvs: Sequence[tuple[int, int]],
    test_hvs: Sequence[tuple[int, int]],
    num_classes: int,
    *,
    epochs: int = 25,
    seed: int = 0,
) -> tuple[float, int]:
    """Local CPU bit-update perceptron readout over 4,096-bit HVs (0 floats, no backprop).

    Returns ``(unseen_accuracy, learned_readout_bits)``.
    """
    accs = [SignedBitSlicedAccumulator(bits=12) for _ in range(num_classes)]
    rng = random.Random(seed * 997 + 13)
    order = list(range(len(train_hvs)))

    # Warm-start from class prototypes
    for hv, y in train_hvs:
        accs[y].add_bipolar(hv)
    readouts = [acc.sign_hv() for acc in accs]

    for _epoch in range(epochs):
        rng.shuffle(order)
        mistakes = 0
        for idx in order:
            hv, y = train_hvs[idx]
            pred = max(range(num_classes), key=lambda c: hamming_sim(hv, readouts[c]))
            if pred != y:
                mistakes += 1
                accs[y].add_bipolar(hv)
                accs[pred].sub_bipolar(hv)
                readouts[y] = accs[y].sign_hv()
                readouts[pred] = accs[pred].sign_hv()
        if mistakes == 0:
            break

    correct = 0
    for hv, y in test_hvs:
        pred = max(range(num_classes), key=lambda c: hamming_sim(hv, readouts[c]))
        if pred == y:
            correct += 1
    learned_bits = num_classes * D
    return (correct / float(max(1, len(test_hvs))), learned_bits)


def run_shard_arithmetic_canary(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    train_fraction: float = 0.60,
    epochs: int = 25,
) -> dict[str, Any]:
    """Run the 5-seed CPU SHARD arithmetic canary and return raw + aggregated logs."""
    t0 = time.perf_counter()
    all_pairs = [(a, b) for a in range(10) for b in range(10)]
    per_seed: list[dict[str, Any]] = []

    sim_by_task: dict[str, list[float]] = {t: [] for t in CANARY_TASKS}
    perc_by_task: dict[str, list[float]] = {t: [] for t in CANARY_TASKS}

    for seed in seeds:
        rng = random.Random(seed * 1009 + 7)
        pairs = list(all_pairs)
        rng.shuffle(pairs)
        n_train = int(round(len(pairs) * train_fraction))
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:]

        levels = _build_graded_digit_hvs(seed)
        role_a = hv_of(f"shard_role:a:{seed}".encode("utf-8"))
        role_b = hv_of(f"shard_role:b:{seed}".encode("utf-8"))
        role_ab = hv_of(f"shard_role:ab:{seed}".encode("utf-8"))

        pair_hv = {
            (a, b): _encode_pair_hv(a, b, levels, role_a, role_b, role_ab)
            for a, b in all_pairs
        }

        seed_rec: dict[str, Any] = {
            "seed": seed,
            "train_pairs": len(train_pairs),
            "unseen_test_pairs": len(test_pairs),
            "tasks": {},
        }
        for task_name, (n_cls, fn) in CANARY_TASKS.items():
            tr = [(pair_hv[p], fn(p[0], p[1])) for p in train_pairs]
            te = [(pair_hv[p], fn(p[0], p[1])) for p in test_pairs]
            sim_acc = _eval_pure_hdc_similarity(tr, te, n_cls)
            perc_acc, learned_bits = _eval_bit_perceptron_readout(
                tr, te, n_cls, epochs=epochs, seed=seed
            )
            sim_by_task[task_name].append(sim_acc)
            perc_by_task[task_name].append(perc_acc)
            seed_rec["tasks"][task_name] = {
                "num_classes": n_cls,
                "chance_baseline": round(1.0 / n_cls, 6),
                "hdc_pure_similarity_acc": round(sim_acc, 6),
                "shard_bit_perceptron_acc": round(perc_acc, 6),
                "learned_readout_bits": learned_bits,
            }
        per_seed.append(seed_rec)

    elapsed_s = time.perf_counter() - t0
    summary_by_task: dict[str, Any] = {}
    for task_name, (n_cls, _fn) in CANARY_TASKS.items():
        summary_by_task[task_name] = {
            "num_classes": n_cls,
            "chance_baseline": round(1.0 / n_cls, 6),
            "hdc_pure_similarity": summarize_across_seeds(
                sim_by_task[task_name], bounds=(0.0, 1.0)
            ),
            "shard_bit_perceptron": summarize_across_seeds(
                perc_by_task[task_name], bounds=(0.0, 1.0)
            ),
            "learned_readout_bits": n_cls * D,
        }

    return {
        "canary_name": "shard_hdc_arithmetic_canary_v1",
        "hdc_dimension_bits": D,
        "seeds": list(seeds),
        "train_fraction": train_fraction,
        "epochs": epochs,
        "elapsed_seconds": round(elapsed_s, 4),
        "summary_by_task": summary_by_task,
        "per_seed": per_seed,
    }
