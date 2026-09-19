# Mirror 7 — Phase 59: Predictive Concept Learning

Phase 59 learns context-conditioned transitions over the Phase 58 hierarchy.

```text
hierarchical concepts
        ↓
context history
        ↓
longest supported context
        ↓
prediction + confidence
        ↓
abstain / back off when uncertain
```

The model explicitly tracks competing successors and does not force a
prediction when evidence is insufficient or tied.

## Acceptance

- 3 progressive families × 3 seeds
- held-out recombination
- context-specific disambiguation
- ambiguity abstention
- unseen-context backoff
- deterministic signature
- malformed-input rejection

Run:

```bash
python -m phase59_predictive_concepts.run_phase59_gate
python -m pytest phase59_predictive_concepts/test_phase59.py -q
```

Expected: **8/8**.

Boundary: bounded symbolic predictive learning, not open-world forecasting or AGI.
