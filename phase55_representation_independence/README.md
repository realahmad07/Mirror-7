# Mirror 7 — Phase 55: Representation Independence

Phase 55 asks a specific upstream question:

> Can Mirror 7 recover the same underlying structure when the surface representation changes?

## Scope

The tested learner receives only raw bytes. It does not receive:

- a task-family label;
- semantic node names;
- the expected graph family;
- a canonical node numbering;
- hidden ground-truth edges.

The discoverer accepts three generic syntax families:

1. binary adjacency matrices;
2. delimited edge lists;
3. delimited neighbor maps.

All three can encode the same unlabeled graph with different node IDs, ordering, and serialization.

The discovered artifact is an **unlabeled canonical graph fingerprint**.

## Acceptance gate

- 3 progressively harder graph families;
- sizes 4, 5, and 6;
- 3 random seeds per size/family;
- 27/27 cross-representation comparisons;
- 2 held-out graph families;
- 6/6 adversarial negative controls;
- 20-seed determinism/invariance regression;
- exact canonicalization for the bounded <=8-node test domain.

Run:

```bash
python -m phase55_representation_independence.run_phase55_gate
python -m pytest phase55_representation_independence/test_phase55.py -q
```

Expected:

```text
PROGRESSIVE: 27/27 representation comparisons pass
HELD-OUT: 2/2 unseen graph families pass
ADVERSARIAL: 6/6 negative controls pass
REGRESSION: parser + canonicalization + invariance + determinism pass
PHASE 55 ACCEPTANCE GATE: PASS
```

## Boundary

This phase demonstrates representation independence for bounded relational structures. It does **not** yet prove raw visual/audio grounding, open-world concept acquisition, large-scale canonicalization, or AGI.
