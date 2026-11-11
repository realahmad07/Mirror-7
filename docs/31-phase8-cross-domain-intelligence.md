# Phase 8 — Sealed Cross-Domain Intelligence Benchmark

## Purpose

Phase 8 moves beyond a single controlled transfer example. It supplies a frozen transfer matrix that exercises multiple source/target domains, requires separate target training and held-out examples, includes adversarial negative controls, and demands target-domain evidence before a transfer is counted as successful.

Implementation: `mirror_crystal/phase8_cross_domain_intelligence.py`.

## Acceptance contract

A `CrossDomainBenchmarkManifest` must contain:
- unique case IDs and a stable manifest fingerprint;
- at least three positive transfer cases and one negative control by default;
- at least two source domains, two target domains and two distinct source/target pairs by default;
- typed, non-empty training and held-out examples for every case;
- explicit target postconditions for positive cases;
- an expected rejection fragment for every negative control.

For a positive case, the runner:
1. invokes the existing Phase 6 transfer engine, which checks independently reproduced source capability evidence, globally verified rule dependencies, unchanged typed/unit/dimension ports, and disjoint target train/held-out examples;
2. requires a host-registered target-verifier factory;
3. obtains execution-bound `VerifiedProgramObservation` records from that factory;
4. passes those observations through the existing Phase 5 learning gate, including distinct tasks/executions, successful target verification, multiple verifier identities and exact program replay.

A transfer counts as passed only after target-specific repeated verification and learning succeed. The resulting proposal is returned for explicit promotion; Phase 8 does not write capability registries.

Negative controls count as passed only when the transfer engine rejects them for the expected reason. An unexpected transfer, missing verifier, rejected evidence or verifier exception fails the case and the full benchmark.

## Reproduction

```bash
python -m pytest -q mirror_crystal/tests/test_cross_domain_transfer.py
python -m pytest -q mirror_crystal mirror_eval
```

## Boundaries

The benchmark runner enforces the supplied manifest and evidence contracts. It cannot prove that host-provided verifier organizations are truly independent, that target labels are scientifically authentic, or that a finite benchmark establishes unrestricted semantic analogy. The current native verifiers cover declared structured invariants, not general-purpose domain science. A passed benchmark is bounded evidence for the tested domain pairs, not general cross-domain intelligence.
