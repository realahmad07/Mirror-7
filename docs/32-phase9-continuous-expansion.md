# Phase 9 — Evidence-Gated Continuous Expansion

## Purpose

Phase 9 adds a multi-cycle admission gate so a single successful transfer run cannot be presented as continuous learning. It requires repeated Phase 8 benchmark success against a frozen baseline, disjoint task and unseen-case identifiers, passed negative controls, multiple target domains and independent review identities.

Implementation: `mirror_crystal/phase9_continuous_expansion.py`.

## Acceptance contract

The default gate requires at least three cycles. Every cycle must contain:
- a passing Phase 8 report with at least one passed adversarial negative control;
- at least three distinct task IDs and one or more distinct unseen-case IDs;
- at least two distinct reviewer IDs;
- a SHA-256 fingerprint for the frozen baseline;
- no overlap in task IDs or unseen-case IDs with other cycles;
- fresh verified observation-task IDs, execution fingerprints and evidence fingerprints across cycles, all bound to the learned proposals.

Across the cycle set, the engine requires:
- distinct frozen benchmark fingerprints;
- the same baseline fingerprint in every cycle;
- at least two target domains;
- no conflicting capability identifiers;
- one or more independently learned target capabilities from the Phase 8 reports.

The result is a deterministic `ContinuousExpansionProposal` with a generation fingerprint, cycle lineage, benchmark fingerprints, task/case identifiers, target domains, reviewer IDs and learned capability proposals.

## Trust boundary

The engine **does not** automatically promote candidates, mutate registries, overwrite prior knowledge, or treat a caller-supplied ID as proof that a task is genuinely novel. The caller is responsible for supplying authentic unseen-case identities, trustworthy reviewer identities and a genuinely frozen baseline. Promotion remains a separate explicit host action through the existing Phase 5 promotion API.

This closes a bounded, repeatable evidence-admission mechanism. It does not yet implement an autonomous open-world task generator, a persistent append-only expansion ledger, independent external reviewer authentication, automatic rollback, or unrestricted scientific discovery. Therefore the original strategic Phase 9 is **PARTIAL (bounded admission gate implemented)** until those external and persistence requirements are independently accepted.

## Reproduction

```bash
python -m pytest -q mirror_crystal/tests/test_cross_domain_transfer.py
python -m pytest -q mirror_crystal mirror_eval
```
