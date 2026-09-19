# Phases 178–180 — External Dataset Performance

## Status

**IMPLEMENTED — awaiting repository CI execution for empirical promotion.**

### Phase 178 — UCI Wine
- Official UCI Wine dataset: 178 instances, 13 numeric features, 3 classes.
- Real download at test time from the UCI repository.
- Explicit prototype/exemplar memory adapter; no neural weights or matrix multiplication.
- Held-out accuracy gate: >= 80%.
- Sealed task-pack boundary remains separate from the learner.

### Phase 179 — UCI Wine Quality
- Official UCI Wine Quality dataset, red variant: 1,599 instances, 11 real-valued inputs, quality target.
- Held-out accuracy gate: >= 45%.
- Noise robustness gate: >= 40% under deterministic small perturbations.

### Phase 180 — Cross-domain boundary control
- Confirms the classifier can process a different feature width only when compatible with its fitted schema; no zero-shot cross-domain result is counted as success.
- External evaluator plumbing is retained without exposing labels.

## Important claim boundary

These phases measure a small explicit external-task adapter, not AGI and not unrestricted Mirror 7 intelligence. The adapter is deliberately isolated so benchmark performance cannot silently change the claims about the core cognitive runtime.
