# Phase 9 — Evidence-Gated Capability Learning

## Purpose

Phase 9 converts repeated successful compositions into a reusable composite capability. It learns only structural patterns that have already worked and been independently checked; it does not infer correctness from a single answer or from execution success alone.

## Admission gate

`CapabilityLearningEngine` requires at least three observations by default. Each observation must bind the same task, plan, execution fingerprint, and a fully verified report. All executed steps must have passing independent checks, every capability must have a passing verification record, and no failed checks or blockers are permitted.

The same normalized composition structure must recur: selected capability identifiers and fingerprints, dependency edges, typed bindings, and expected postconditions must match. Task identifiers and task fingerprints must be distinct. At least two distinct independent verifier identities must appear across the observations.

## Learned capability

The learned capability's external inputs are derived only from explicit task-input/context bindings. Its outputs are the selected operations' outputs that are not consumed by another step. Incompatible or ambiguous port contracts block learning. Preconditions and postconditions are retained; the composite carries independently reproduced evidence tied to the named checks and verifier identities.

The proposal requires explicit promotion. Calling `CapabilityLearningEngine.persist(vault, proposal)` stores it through the existing capability vault API. No execution or verification event silently writes a new capability to durable memory. A newly learned composite still needs an explicitly registered native handler before the execution engine can run it directly.

## Boundary

Phase 9 creates a reusable, evidence-backed capability proposal from repeated validated compositions. It does not synthesize arbitrary source code, remove the need for a runtime handler, bypass independent verification, or promote failed observations.
