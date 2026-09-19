# Mirror 7 — Phase 63: Nonstationary World + Autonomous Experiment Design

Phase 63 removes the assumption that the environment's dynamics stay fixed.

Mirror 7 now maintains multiple regime hypotheses, detects repeated
prediction mismatch as dynamics drift, and actively chooses experiments whose
possible outcomes disagree across hypotheses.

```text
learn regime hypotheses
        ↓
predict
        ↓
observe mismatch
        ↓
mark stale / replan
        ↓
design experiment
        ↓
maximize predicted outcome disagreement
        ↓
identify active regime
        ↓
continue goal-directed control
```

## Acceptance

- 3 progressive change levels × 3 seeds;
- held-out regime identification;
- autonomous experiment selection;
- repeated-mismatch drift detection;
- false-drift negative control;
- post-experiment regime identification;
- bounded experiment budget;
- Phase 62 handoff contract;
- malformed-input rejection;
- 9/9 executable regression.

Run:

```bash
python -m phase63_nonstationary_experiments.run_phase63_gate
python -m pytest phase63_nonstationary_experiments/test_phase63.py -q
```

Expected:

```text
PHASE 63 ACCEPTANCE GATE: PASS
progressive: 3 change levels × 3 seeds
held-out: 2/2
adversarial: 3/3
integration: Phase 62 handoff contract
regression: 9/9
```

## Boundary

Phase 63 establishes bounded nonstationary dynamics handling and autonomous
experiment selection from a finite regime hypothesis bank. It does not claim
general nonstationary-world understanding, arbitrary causal discovery under
drift, or AGI.
