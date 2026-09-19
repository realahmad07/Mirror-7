"""
Mirror 7 — Adaptive Predictive Memory — Test Suite
====================================================

Tests are intentionally NOT designed to exercise specific implementation
details.  They test the *behavioural contract* of a prediction engine:

  1. Basic repeating pattern         — can it learn A B A B?
  2. Longer cycle                    — can it learn A B C D A B C D?
  3. Context-dependent branching     — can it distinguish X→Y from Z→W?
  4. Transfer / unseen pattern       — does it degrade gracefully?
  5. Random / adversarial input      — does it express uncertainty?
  6. Ambiguous context               — does confidence drop when context
                                       legitimately maps to multiple outcomes?
  7. Memory-constrained operation    — does it survive with tiny memory?
  8. Non-stationary adaptation       — can it track a distribution shift?
  9. Multi-step forecast             — does confidence decay with horizon?
 10. Serialisation round-trip        — does to_dict() preserve state info?

Run directly:
    python -m mirror7_algorithm.test_mirror7_algorithm
Or:
    python test_mirror7_algorithm.py
"""

from __future__ import annotations

import sys
import os
import math
import random
import time

# Allow running both as `python test_mirror7_algorithm.py` from the package
# directory and as `python -m mirror7_algorithm.test_mirror7_algorithm`.
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from mirror7_algorithm.mirror7_algorithm import (
    AdaptivePredictiveMemory,
    PredictionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pass_count = 0
_fail_count = 0


def _check(condition: bool, label: str, detail: str = "") -> None:
    """Assert-like helper that prints results instead of raising."""
    global _pass_count, _fail_count
    status = "PASS" if condition else "FAIL"
    if not condition:
        _fail_count += 1
    else:
        _pass_count += 1
    msg = f"  [{status}] {label}"
    if detail and not condition:
        msg += f"  — {detail}"
    print(msg)


def _separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Test 1 — Basic repeating pattern (A B A B ...)
# ---------------------------------------------------------------------------

def test_basic_repeat() -> None:
    _separator("Test 1: Basic Repeating Pattern (A B A B ...)")
    apm = AdaptivePredictiveMemory(max_memory=500, max_context=4)

    # Train on 20 repetitions of AB
    seq = list("AB") * 20
    results = apm.observe_sequence(seq)

    # After training, predict next 2 steps
    preds = apm.predict(n_steps=2)

    _check(
        preds[0].best_prediction == "A",
        "After ...AB, predicts A",
        f"got {preds[0].best_prediction!r}",
    )
    _check(
        preds[0].confidence > 0.3,
        "Reasonable confidence on learned pattern",
        f"confidence={preds[0].confidence:.3f}",
    )

    # Surprise should decrease over time
    early_surprise = sum(r.surprise for r in results[:6]) / 6
    late_surprise = sum(r.surprise for r in results[-6:]) / 6
    _check(
        late_surprise < early_surprise,
        "Surprise decreases as pattern is learned",
        f"early={early_surprise:.2f}, late={late_surprise:.2f}",
    )


# ---------------------------------------------------------------------------
# Test 2 — Longer cycle (A B C D A B C D ...)
# ---------------------------------------------------------------------------

def test_longer_cycle() -> None:
    _separator("Test 2: Longer Cycle (A B C D ...)")
    apm = AdaptivePredictiveMemory(max_memory=1000, max_context=8)

    seq = list("ABCD") * 30
    apm.observe_sequence(seq)

    preds = apm.predict(n_steps=4)
    predicted = [p.best_prediction for p in preds]
    _check(
        predicted == list("ABCD"),
        "Predicts full 4-symbol cycle",
        f"got {predicted}",
    )

    # Confidence should be highest for the immediate next step
    _check(
        preds[0].confidence >= preds[-1].confidence,
        "Nearer predictions have higher confidence",
        f"step0={preds[0].confidence:.3f}, step3={preds[-1].confidence:.3f}",
    )


# ---------------------------------------------------------------------------
# Test 3 — Context-dependent branching
# ---------------------------------------------------------------------------

def test_context_dependent() -> None:
    _separator("Test 3: Context-Dependent Branching")
    apm = AdaptivePredictiveMemory(max_memory=1000, max_context=4)

    # Pattern: X is always followed by Y; Z is always followed by W
    pattern = list("XY") * 20 + list("ZW") * 20
    # Interleave to make context matter
    interleaved = []
    for i in range(20):
        interleaved.extend(["X", "Y", "Z", "W"])
    apm.observe_sequence(interleaved)

    # After seeing X, should predict Y
    apm.reset_context()
    apm.observe("X")
    preds = apm.predict(n_steps=1)
    _check(
        preds[0].best_prediction == "Y",
        "After X, predicts Y",
        f"got {preds[0].best_prediction!r}",
    )

    # After seeing Z, should predict W
    apm.reset_context()
    apm.observe("Z")
    preds = apm.predict(n_steps=1)
    _check(
        preds[0].best_prediction == "W",
        "After Z, predicts W",
        f"got {preds[0].best_prediction!r}",
    )


# ---------------------------------------------------------------------------
# Test 4 — Transfer to unseen / related pattern
# ---------------------------------------------------------------------------

def test_unseen_pattern() -> None:
    _separator("Test 4: Transfer to Unseen Pattern")
    apm = AdaptivePredictiveMemory(max_memory=1000, max_context=8)

    # Train on ABCABC...
    train = list("ABC") * 30
    apm.observe_sequence(train)

    # Now switch to ABXABX... (shares 'AB' prefix but diverges at C→X)
    apm.reset_context()
    test_seq = list("ABX") * 5
    results = apm.observe_sequence(test_seq)

    # First encounter of X after AB should be surprising
    # X appears at indices 2, 5, 8, 11, 14
    first_x_surprise = results[2].surprise  # first X after "AB"
    last_x_surprise = results[-1].surprise  # last X after "AB"

    _check(
        first_x_surprise is not None and first_x_surprise > 1.0,
        "First novel symbol X is surprising",
        f"surprise={first_x_surprise}",
    )
    _check(
        last_x_surprise < first_x_surprise,
        "Surprise for X decreases as pattern is learned",
        f"first={first_x_surprise:.2f}, last={last_x_surprise:.2f}",
    )


# ---------------------------------------------------------------------------
# Test 5 — Random / adversarial input
# ---------------------------------------------------------------------------

def test_random_adversarial() -> None:
    _separator("Test 5: Random / Adversarial Input")
    rng = random.Random(42)
    apm = AdaptivePredictiveMemory(max_memory=500, max_context=4)

    # Feed 200 uniformly random symbols from a 10-symbol alphabet
    alphabet = list("0123456789")
    seq = [rng.choice(alphabet) for _ in range(200)]
    results = apm.observe_sequence(seq)

    # Predictions should be uncertain — confidence should stay low
    late_results = results[-50:]
    mean_conf = sum(r.confidence for r in late_results) / len(late_results)
    mean_surp = sum(r.surprise for r in late_results) / len(late_results)

    _check(
        mean_conf < 0.6,
        "Low confidence on random data",
        f"mean_confidence={mean_conf:.3f}",
    )
    _check(
        mean_surp > 1.0,
        "Sustained high surprise on random data",
        f"mean_surprise={mean_surp:.2f}",
    )

    # It should NOT be confident about any particular prediction
    preds = apm.predict(n_steps=1)
    _check(
        preds[0].is_uncertain or preds[0].best_probability < 0.7,
        "Not over-confident on random data",
        f"best_prob={preds[0].best_probability:.3f}, conf={preds[0].confidence:.3f}",
    )


# ---------------------------------------------------------------------------
# Test 6 — Ambiguous context
# ---------------------------------------------------------------------------

def test_ambiguous_context() -> None:
    _separator("Test 6: Ambiguous Context")
    apm = AdaptivePredictiveMemory(max_memory=500, max_context=4)

    # Context "A" is followed by "B" half the time and "C" half the time
    seq = []
    for _ in range(50):
        seq.extend(["A", "B"])
        seq.extend(["A", "C"])
    apm.observe_sequence(seq)

    # After seeing A, prediction should show roughly 50/50
    apm.reset_context()
    apm.observe("A")
    preds = apm.predict(n_steps=1)
    dist = preds[0].distribution

    b_prob = dist.get("B", 0)
    c_prob = dist.get("C", 0)

    _check(
        abs(b_prob - c_prob) < 0.25,
        "Ambiguous context yields roughly balanced distribution",
        f"P(B)={b_prob:.3f}, P(C)={c_prob:.3f}",
    )
    _check(
        preds[0].entropy > 0.5,
        "High entropy for ambiguous prediction",
        f"entropy={preds[0].entropy:.3f} bits",
    )
    _check(
        preds[0].confidence < 0.7,
        "Reduced confidence for ambiguous context",
        f"confidence={preds[0].confidence:.3f}",
    )


# ---------------------------------------------------------------------------
# Test 7 — Memory-constrained operation
# ---------------------------------------------------------------------------

def test_memory_limit() -> None:
    _separator("Test 7: Memory-Constrained Operation")
    apm = AdaptivePredictiveMemory(max_memory=20, max_context=4)

    # Feed a moderately complex sequence that would normally create many patterns
    seq = list("ABCDEFGH") * 50
    results = apm.observe_sequence(seq)

    stats = apm.get_stats()
    _check(
        stats.n_patterns <= 20,
        f"Memory stays within limit ({stats.n_patterns} <= 20)",
    )
    _check(
        stats.patterns_evicted > 0,
        f"Eviction occurred ({stats.patterns_evicted} patterns evicted)",
    )

    # Even with limited memory, the most important patterns should survive
    # and basic prediction should still work
    preds = apm.predict(n_steps=1)
    _check(
        preds[0].best_prediction is not None,
        "Can still make predictions under memory pressure",
        f"predicted {preds[0].best_prediction!r}",
    )


# ---------------------------------------------------------------------------
# Test 8 — Non-stationary adaptation
# ---------------------------------------------------------------------------

def test_nonstationary() -> None:
    _separator("Test 8: Non-Stationary Adaptation")
    apm = AdaptivePredictiveMemory(
        max_memory=1000, max_context=4, count_decay=0.01
    )

    # Phase 1: Pattern is ABABAB...
    phase1 = list("AB") * 50
    apm.observe_sequence(phase1)

    pred1 = apm.predict(n_steps=1)
    _check(
        pred1[0].best_prediction == "A",
        "Phase 1: predicts A after B",
        f"got {pred1[0].best_prediction!r}",
    )

    # Phase 2: Pattern switches to CDCDCD...
    phase2 = list("CD") * 80
    apm.observe_sequence(phase2)

    pred2 = apm.predict(n_steps=2)
    _check(
        pred2[0].best_prediction == "C",
        "Phase 2: adapts to predict C after D",
        f"got {pred2[0].best_prediction!r}",
    )

    # Check that surprise dropped after adaptation
    recent_surp = apm.get_recent_surprise(window=20)
    _check(
        recent_surp < 5.0,
        "Surprise settles after adapting to new pattern",
        f"recent_surprise={recent_surp:.2f}",
    )


# ---------------------------------------------------------------------------
# Test 9 — Multi-step forecast with confidence decay
# ---------------------------------------------------------------------------

def test_multistep_forecast() -> None:
    _separator("Test 9: Multi-Step Forecast")
    apm = AdaptivePredictiveMemory(max_memory=1000, max_context=8)

    seq = list("ABCDE") * 30
    apm.observe_sequence(seq)

    preds = apm.predict(n_steps=10)

    # Confidence should decay with horizon
    if len(preds) >= 5:
        _check(
            preds[0].confidence > preds[4].confidence,
            "Confidence decays over prediction horizon",
            f"step0={preds[0].confidence:.3f}, step4={preds[4].confidence:.3f}",
        )

    # First few predictions should follow the cycle
    first_few = [p.best_prediction for p in preds[:5]]
    expected_start = list("ABCDE")
    _check(
        first_few == expected_start,
        f"First 5 predicted symbols match cycle",
        f"got {first_few}",
    )


# ---------------------------------------------------------------------------
# Test 10 — Serialisation round-trip
# ---------------------------------------------------------------------------

def test_serialisation() -> None:
    _separator("Test 10: Serialisation Round-Trip")
    apm = AdaptivePredictiveMemory(max_memory=500, max_context=4)
    seq = list("HELLO") * 10
    apm.observe_sequence(seq)

    state = apm.to_dict()

    _check(
        isinstance(state, dict),
        "to_dict() returns a dict",
    )
    _check(
        "patterns" in state and "config" in state,
        "State contains 'patterns' and 'config' keys",
    )
    _check(
        state["step"] == len(seq),
        f"Step count matches ({state['step']} == {len(seq)})",
    )

    # Patterns should be non-empty
    _check(
        len(state["patterns"]) > 0,
        f"Serialised state contains {len(state['patterns'])} patterns",
    )


# ---------------------------------------------------------------------------
# Test 11 — Edge cases and robustness
# ---------------------------------------------------------------------------

def test_edge_cases() -> None:
    _separator("Test 11: Edge Cases & Robustness")

    # Single observation
    apm = AdaptivePredictiveMemory(max_memory=100, max_context=4)
    r = apm.observe("X")
    _check(
        r.surprise is not None,
        "Single observation returns a valid result",
    )
    _check(
        r.confidence == 0.0,
        "No confidence before any context is built",
        f"confidence={r.confidence}",
    )

    # Predict with no context
    apm2 = AdaptivePredictiveMemory()
    preds = apm2.predict(n_steps=3)
    _check(
        len(preds) == 1 and preds[0].best_prediction is None,
        "Predict on empty model returns no prediction",
    )

    # Numeric observations
    apm3 = AdaptivePredictiveMemory(max_memory=200, max_context=4)
    nums = [1, 2, 3, 1, 2, 3, 1, 2, 3] * 10
    apm3.observe_sequence(nums)
    p = apm3.predict(n_steps=1)
    _check(
        p[0].best_prediction == 1,
        "Works with integer observations",
        f"got {p[0].best_prediction!r}",
    )

    # Tuple observations
    apm4 = AdaptivePredictiveMemory(max_memory=200, max_context=4)
    tuples = [(0, 0), (0, 1), (1, 0), (1, 1)] * 15
    apm4.observe_sequence(tuples)
    p4 = apm4.predict(n_steps=1)
    _check(
        p4[0].best_prediction is not None,
        "Works with tuple observations",
        f"predicted {p4[0].best_prediction!r}",
    )

    # Invalid parameters
    try:
        AdaptivePredictiveMemory(max_memory=0)
        _check(False, "Should reject max_memory=0")
    except ValueError:
        _check(True, "Rejects max_memory=0")

    try:
        AdaptivePredictiveMemory(count_decay=1.5)
        _check(False, "Should reject count_decay=1.5")
    except ValueError:
        _check(True, "Rejects count_decay=1.5")


# ---------------------------------------------------------------------------
# Test 12 — Performance / resource sanity
# ---------------------------------------------------------------------------

def test_performance() -> None:
    _separator("Test 12: Performance / Resource Sanity")
    apm = AdaptivePredictiveMemory(max_memory=5000, max_context=8)

    rng = random.Random(123)
    alphabet = list("ABCDEFGHIJKLMNOP")  # 16 symbols

    # Generate 5000 observations with some structure
    seq = []
    for _ in range(1000):
        # 80% structured, 20% noise
        if rng.random() < 0.8:
            seq.extend(rng.choice(["ABC", "DEF", "GHI", "JKL"]))
        else:
            seq.append(rng.choice(alphabet))

    t0 = time.perf_counter()
    apm.observe_sequence(seq)
    elapsed = time.perf_counter() - t0

    _check(
        elapsed < 10.0,
        f"5000+ observations processed in {elapsed:.2f}s (< 10s)",
    )

    stats = apm.get_stats()
    _check(
        stats.n_patterns <= 5000,
        f"Memory bounded ({stats.n_patterns} patterns)",
    )

    # Predict should be fast
    t1 = time.perf_counter()
    preds = apm.predict(n_steps=20)
    pred_elapsed = time.perf_counter() - t1
    _check(
        pred_elapsed < 1.0,
        f"20-step prediction in {pred_elapsed*1000:.1f}ms (< 1s)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global _pass_count, _fail_count
    _pass_count = 0
    _fail_count = 0

    print("\n" + "=" * 60)
    print("  Mirror 7 — Adaptive Predictive Memory — Test Suite")
    print("=" * 60)

    test_basic_repeat()
    test_longer_cycle()
    test_context_dependent()
    test_unseen_pattern()
    test_random_adversarial()
    test_ambiguous_context()
    test_memory_limit()
    test_nonstationary()
    test_multistep_forecast()
    test_serialisation()
    test_edge_cases()
    test_performance()

    print(f"\n{'='*60}")
    print(f"  Results: {_pass_count} passed, {_fail_count} failed")
    print(f"{'='*60}\n")

    return 1 if _fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
