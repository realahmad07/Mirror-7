# Canonical Blueprint and Rule Architecture

## Purpose

Mirror's knowledge foundation should be a composable system of blueprints and rules. A blueprint describes a capability or transformation; rules provide executable or retrievable pieces used to realize it. Stored examples, retrieval similarity and legacy trust labels are not proof of truth.

This page defines the first stable contract and a safe migration path. It does not claim that every current subsystem has already migrated.

## Canonical blueprint

The new `mirror_crystal.blueprint_contracts` module defines a versioned `Blueprint` with identity/type, domain and intent, prompt pattern and response template, slots, preconditions/postconditions, dependency IDs, provenance, license label, content digest, and verification evidence.

The `BlueprintDependencyGraph` resolves dependencies in deterministic order and rejects missing nodes and cycles. It is deliberately independent of retrieval and storage.

### Evidence levels

| Level | Meaning | Does not mean |
|---|---|---|
| `unverified` | No attached validation evidence | Not a correctness claim |
| `curated` | A documented curation review occurred | Does not prove factual correctness |
| `execution_tested` | Named executable checks passed | Does not prove behavior outside test scope |
| `formally_verified` | A named formal verifier checked an explicit claim | Does not prove unrelated claims |
| `independently_reproduced` | A named independent reproduction passed | Does not imply broad generalization |

Legacy `CrystalRule` and roadmap records are adapted with `unverified` evidence. A legacy `trust_tier=1`, the presence of a test string, or successful ingestion is not silently promoted to verification.

## Intended runtime flow

1. Parse request into intent and explicit constraints.
2. Retrieve candidate blueprints and rule fragments.
3. Check preconditions, domain, and entity coverage.
4. Resolve dependencies in dependency-first order.
5. Execute bounded deterministic transformations or tested procedures.
6. Validate postconditions and response invariants.
7. Compose without adding unsupported claims.
8. Return answer with evidence scope and trace, or clarify/abstain.

Retrieval, storage, composition, execution, and response realization should be independently testable components.

## Bottlenecks found

- `engine.py` combines ingestion, state, retrieval, chaining, composition, and abstention.
- `disk_vault.py` combines SQLite/index storage, ingestion, candidate retrieval, response policy, maintenance, and analytics.
- The in-memory and disk-backed engines duplicate substantial response logic.
- The disk vault imported response/session helpers from `engine.py`, an unnecessary dependency on the in-memory engine.
- Retrieval similarity and multi-hop fact chaining were previously labeled `verified_exact`; neither is independent proof.
- The Python AST judge is an in-process restricted execution harness, not an OS/container-isolated security boundary.

## Changes in this increment

1. Extracted shared response types, session memory, and compound-intent splitting. The disk vault now imports these shared primitives directly.
2. Added a canonical blueprint contract and dependency graph, plus conservative legacy adapters.
3. Centralized epistemic tier policy. Deterministic operations and test-passing synthesis can be exact; retrieval-only facts and factual chains cannot.
4. Added focused regression tests for contract round-trips, evidence invariants, dependency resolution, module compatibility, and epistemic labeling.

## Remaining migration plan

1. **Storage boundary:** move SQLite schema, transactions, LRU and index persistence behind a repository interface.
2. **Retrieval boundary:** implement a common candidate interface so memory and disk engines share matching and abstention policy.
3. **Composition boundary:** replace implicit rule chaining with explicit dependencies, preconditions, and postconditions.
4. **Execution boundary:** isolate deterministic formula interpreters and replace in-process code execution with an OS-level sandbox for untrusted programs.
5. **Answer validation:** attach source/evidence pointers to factual claims or mark them as inference; add contradiction and freshness checks.
6. **Evaluation:** add external, held-out, adversarial, multilingual and regression suites. Report accuracy, unsupported-answer rate, abstention quality, calibration and latency separately.

## Current capability boundary

The Crystal track can retrieve stored examples, generalize a bounded set of numeric formula patterns, perform selected deterministic CAS operations, recall explicit session constraints, compose some rule chains, and synthesize code against supplied tests. It is not yet a general-purpose factual knowledge engine. Stored-rule count and internal benchmark results do not establish broad answer correctness.
