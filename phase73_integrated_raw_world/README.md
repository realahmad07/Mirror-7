# Phase 73 — Integrated Raw-to-World Learning

Phase 73 connects the Phase 71 raw numeric representation layer to the Phase 70 bounded world model.

## Flow

raw numeric stream -> discovered segments/signatures -> fixed-width structured state -> transition learning -> prediction/counterfactual/planning

## Guarantees

- No hand-supplied state schema is required.
- Representation width is bounded and stable through fixed segment slots.
- Unknown actions fail closed through the existing world-model gate.
- The scope is intentionally numeric raw streams; this is not general vision/language perception.

## Acceptance tests

The included tests cover fixed-width encoding, raw-to-world learning, held-out raw patterns, counterfactual chaining, unknown-action rejection, bounded partial input, and stable dimensionality.

These are implementation tests, not yet the full 3-seed/held-out/adversarial certification gate.
