# Phase 65 — Streaming Delayed/Stochastic Hidden-State Learning

Phase 65 removes the controlled-reset assumption from Phase 64 and moves the learner into a continuous observation/action stream.

## Core mechanism

```text
continuous stream
      ↓
partial / noisy observations
      ↓
delayed-action evidence
      ↓
stochastic effect statistics
      ↓
rolling regime evidence
      ↓
sustained discrepancy
      ↓
new / revised unlabeled hypothesis
      ↓
autonomous pulse + gap experiment
      ↓
goal-directed model reuse
```

The implementation does not require `env.reset()` and does not receive regime labels or a predefined regime catalogue.

## Implemented mechanisms

- streaming action-effect learning with configurable delay horizon;
- running mean/variance and confidence bookkeeping for stochastic effects;
- partial observations using `None` for unavailable fields;
- delayed attribution with no-op baseline isolation so delayed effects are not counted as ordinary drift;
- rolling-window regime evidence so old regimes remain distinguishable after dynamics change;
- sustained-mismatch confirmation before creating a new hypothesis;
- autonomous experiment sequences composed from focal actions and observation gaps;
- experiment scoring from disagreement/uncertainty rather than regime labels;
- hard experiment-budget enforcement;
- fail-closed prediction/action selection when evidence is insufficient;
- continuous `run_stream()` integration with no reset call.

## Acceptance boundary

```text
Progressive: 3 effect families × 3 seeds
Held-out:    1 / 1
Controls:    4 / 4
Integration: continuous no-reset stream
Pytest:      9 passed
```

The suite includes delayed stochastic effects, held-out regime invention, hidden internal regime change, disagreement-driven experiment selection, sensor dropout/noise, same-regime cluster stability, hard budget enforcement, malformed-input rejection, and a continuous no-reset execution test.

## Important boundary

Phase 65 is still a bounded research mechanism. It does **not** demonstrate unrestricted belief-state inference, open-world scientific discovery, arbitrary multimodal grounding, or AGI.

The test environments use a bounded numeric observation space and explicit legal actions. The advance over Phase 64 is the removal of controlled resets plus online delayed/stochastic evidence and hidden regime changes.

## Gate

```bash
python -m phase65_streaming_hidden_state.run_phase65_gate
python -m pytest phase65_streaming_hidden_state/test_phase65.py -q
```
