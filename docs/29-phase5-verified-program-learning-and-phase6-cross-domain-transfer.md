# Phase 5–6: Verified Program Learning and Controlled Cross-Domain Transfer

## Phase 5 — evidence-gated UOL program learning

`mirror_crystal/verified_program_learning.py` closes the loop from a synthesized UOL candidate to a reusable native capability.

The learning gate requires:
- at least three distinct task identifiers and execution fingerprints by default;
- one successful execution per observation, each containing exactly one candidate step;
- a fully verified, execution-fingerprint-bound `DomainVerificationReport` for every run;
- every required evidence artifact to pass and cover the same declared postconditions;
- at least two distinct verifier identities per observation;
- replay of the exact same immutable UOL program over each observed input, matching the recorded output;
- compatible typed input/output contracts and a process-serializable handler.

Only after all requirements pass does the engine create a capability with `INDEPENDENTLY_REPRODUCED` evidence. Learning is still a proposal, not a silent registry mutation. `VerifiedProgramLearningEngine.promote` explicitly adds the capability and its pickle-safe UOL handler to the registries; `persist` is a separate explicit vault call.

The generated handler executes the saved UOL program and its fixed rule catalog. It does not compile or execute Python source. The reserved `__mirror_verification__` input is optional and stripped before UOL program execution.

## Candidate verification without premature promotion

The domain engine now exposes `verify_candidate` for unpromoted candidates whose provenance identifies the UOL synthesizer or cross-domain transfer engine. This permits independent verification of a proposal while leaving its `UNVERIFIED` evidence unchanged. Ordinary `verify` still blocks an unverified candidate. Repeated passing candidate reports are required by Phase 5 before a capability becomes executable.

## Phase 6 — controlled cross-domain transfer

`mirror_crystal/cross_domain_transfer.py` transfers an already independently reproduced UOL program into a different target domain only when:
- the source capability is bound to the exact source program fingerprint;
- the target differs from the source domain;
- every referenced rule dependency is `VERIFIED` and globally scoped (`general` or `*`); source-domain-only rules are rejected;
- source input/output port names, types, units and dimensions are preserved, with no implicit conversions or invented mappings;
- the transferred procedure passes at least three new target training examples and two separate target held-out examples by default;
- target verification postconditions are explicitly supplied.

The result is a new target-domain capability candidate with `UNVERIFIED` evidence. Source-domain verification is never copied to the target. The candidate must pass target-domain `verify_candidate` reports and the Phase 5 repeated-observation learning gate before it can be promoted.

## Accepted scope and boundaries

These phases demonstrate a complete, bounded learning-and-transfer path: repeated independent verification → reusable UOL capability → explicit promotion → fresh target-domain transfer tests → target-specific verification → a separately learned target capability.

They do not establish unrestricted domain analogy, semantic equivalence of arbitrary concepts, automatic unit conversion, or general scientific discovery. A procedure is transferable only when its UOL dependencies and typed interface meet the explicit gates and the target task supplies meaningful regression examples and independent verification criteria. The host remains responsible for the quality and authenticity of those target examples and verifier registrations.

## Acceptance tests

Focused tests cover repeated distinct verification, output replay, explicit handler promotion, failure rejection, duplicate observations, target-domain held-out regression, input disjointness, domain-scoped dependency rejection, candidate verification, and target-domain re-verification. The complete repository CI and Security Guard workflows remain the acceptance gate.
