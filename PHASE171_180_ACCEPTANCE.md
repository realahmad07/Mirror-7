# Phase 171-180 Acceptance — External Evaluation Foundation

## Implemented

This block begins the transition from repository-internal benchmark families to a transport-neutral external task boundary.

- 171 external task-pack protocol
- 172 sealed per-episode agent view
- 173 contamination-field rejection
- 174 deterministic public task fingerprint
- 175 evaluator as a separate process with no Mirror 7 runtime import
- 176 evaluator negative-control enforcement

## Verification

The workflow runs the new tests, then re-runs the 161-170, 153-160, and 141-152 gates, plus positive and negative evaluator controls.

No existing acceptance threshold was weakened.

## Boundary

This is infrastructure and validation hardening, not evidence of unrestricted external generalization. Real external datasets and externally supplied task packs are still required for the next research step.

## Intended flow

external provider -> task pack -> sealed agent view -> Mirror 7 learner -> episode results -> independent evaluator -> fingerprint/success/contamination gate

The evaluator-only identity remains outside the learner view.
