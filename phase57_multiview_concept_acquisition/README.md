# Mirror 7 — Phase 57: Cross-View Raw Concept Acquisition

Phase 57 removes the single-byte-stream boundary from Phase 56.

The learner receives an **unlabeled bundle of raw views** per episode. A view
may be a 1-D integer sequence or a 2-D integer grid. The learner does not
receive modality names, concept names, task-family labels, or an expected
relation graph.

## Mechanism

```text
unlabeled raw views
        ↓
modality-agnostic relational atoms
        ↓
episode + cross-view support
        ↓
MDL-inspired compact concept selection
        ↓
view-invariant concept identities
        ↓
unordered cross-view relations
+ within-view transition events
        ↓
held-out transfer / recombination
```

The key change from Phase 56 is that the same latent relational pattern can be
recovered when it moves between different raw geometries or view positions.
Raw symbol identity is ignored; the learner retains only local equality
structure within bounded windows/patches.

## Acceptance gate

- 3 progressively harder concept families × 3 seeds;
- held-out view permutation + re-encoding;
- held-out cross-geometry transfer;
- 3 adversarial controls;
- deterministic regression;
- scaling guard;
- malformed-input rejection.

Run:

```bash
python -m phase57_multiview_concept_acquisition.run_phase57_gate
python -m pytest phase57_multiview_concept_acquisition/test_phase57.py -q
```

Expected:

```text
PHASE 57 ACCEPTANCE GATE: PASS
progressive: 3 task families × 3 seeds
held-out: 2/2
adversarial: 3/3
scaling: 1/1
regression: 10/10
```

## Boundary

Phase 57 establishes a bounded cross-view structural concept mechanism. It is
stronger than Phase 56 because the representation is shared across 1-D and 2-D
raw views and is tested under view permutation, raw re-encoding, view dropout,
and a basic scaling guard.

It is **not** semantic multimodal understanding, open-world perception,
real-world sensory grounding, or AGI. The next frontier is broader raw data,
learned hierarchical abstractions, long-horizon prediction, and independent
external evaluation.
