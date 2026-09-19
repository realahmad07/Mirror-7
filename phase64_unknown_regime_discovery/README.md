# Mirror 7 — Phase 64: Unknown-Regime Discovery + Autonomous Experiment Sequences

Phase 64 removes the finite, developer-supplied regime bank from Phase 63.

The system starts with **no regime labels and no predefined regime set**. It
runs controlled multi-action experiments, compresses the observed transition
effects into unlabeled hypothesis profiles, clusters repeated observations into
regimes, and chooses later experiments from predicted hypothesis disagreement.

```text
no regime labels
       ↓
controlled experiment
       ↓
transition-effect profile
       ↓
hypothesis creation / merge
       ↓
candidate regime set
       ↓
multi-step experiment design
       ↓
predicted disagreement
       ↓
new evidence
       ↓
regime identification
       ↓
goal-directed planning
```

## What changed from Phase 63

Phase 63 selected among a finite regime hypothesis bank supplied by the
developer.

Phase 64 instead:

- creates hypotheses from raw experiment outcomes;
- keeps regime identities as internal IDs rather than semantic labels;
- repeats controlled action sequences to build support;
- uses intermediate-prefix disagreement, not just terminal disagreement;
- favors experiments that cover the available action vocabulary;
- reuses an identified regime for goal planning;
- fails closed when no supported regime/action model is available.

## Acceptance

- 3 progressively harder unknown-regime families × 3 seeds;
- held-out regime discovered without supplied regime labels;
- autonomous experiment sequence selected by hypothesis disagreement;
- repeated same-regime control without uncontrolled cluster proliferation;
- irrelevant numeric state variation does not split a regime;
- controlled regime switch followed by re-identification;
- hard experiment-budget bound;
- Phase 63-compatible state/goal contract;
- malformed-input rejection;
- 9/9 executable regression.

Run:

```bash
python -m phase64_unknown_regime_discovery.run_phase64_gate
python -m pytest phase64_unknown_regime_discovery/test_phase64.py -q
```

Expected:

```text
PHASE 64 ACCEPTANCE GATE: PASS
progressive: 3 regime families × 3 seeds
held-out: 2/2
adversarial: 3/3
integration: Phase 63 state/goal contract
regression: 9/9
```

## Boundary

Phase 64 demonstrates bounded **unknown-regime discovery from controlled
experiments** over deterministic structured state/action systems.

It is not unrestricted open-world hypothesis generation, stochastic scientific
discovery, arbitrary causal inference, or AGI. The experiments assume controlled
resets and comparable state/action spaces so that observed transition effects
can be meaningfully compared.
