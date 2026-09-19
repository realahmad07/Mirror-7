# Mirror 7 — Phase 60: Long-Horizon World Modeling

Phase 60 adds a bounded predictive world model with:

- exact state/action transition memory;
- factorized per-feature action rules;
- repeated numeric delta rules for unseen states;
- explicit conflict/uncertainty handling;
- discrepancy detection and online updates;
- fail-closed long-horizon rollout.

```text
state + action
      ↓
exact memory
      ↓
factorized rule
      ↓
delta generalization
      ↓
next-state prediction
      ↓
actual observation
      ↓
discrepancy / update
```

## Acceptance

- 3 horizons × 3 seeds
- unseen-state generalization
- long-horizon recombination
- discrepancy + online correction
- conflict abstention
- unknown-action fail-closed behavior
- Phase 58 + Phase 59 integration
- malformed-input rejection

Run:

```bash
python -m phase60_long_horizon_world_model.run_phase60_gate
python -m pytest phase60_long_horizon_world_model/test_phase60.py -q
```

Expected: **8/8**.

Boundary: bounded structured world modeling and rollout, not unrestricted real-world world modeling.
