# Phase 66 — Richer Hidden-State Inference

Phase 66 extends Phase 65 without removing its fail-closed principles.

## Research boundary

- multidimensional stochastic observations/effects;
- overlapping delayed action effects;
- explicit transition variance rather than mean-only evidence;
- longer autonomous experiment sequences with observation gaps;
- online unlabeled hidden-state hypothesis revision;
- portable hypothesis snapshots for cross-environment reuse;
- longer-lag prediction;
- partial-observation safety.

## Gate

```bash
python -m phase66_richer_hidden_state.run_phase66_gate
python -m pytest phase66_richer_hidden_state/test_phase66.py -q
```

Acceptance is intentionally bounded. Passing this phase does not establish unrestricted belief-state inference, general world modeling, autonomous science, or AGI.
