# Stage 11.8–11.13 — Integration, Security Closure & Release Gates

**Status:** work in progress. This document is the acceptance contract; it is not a completion certificate. A phase is complete only when its required tests run on the final candidate commit and the evidence is recorded.

## 11.8 — Integrated verification

**Purpose:** establish that Stages 1–11.7 work together on a clean checkout, not merely in isolated tests.

Required gates:
- Run the full fast CI workflow on the exact candidate SHA, including Crystal/evaluation, backend, release-boundary, training, evaluator and package-build jobs.
- Run the full historical regression workflow with full Git history and required LFS artifacts available.
- Run the frozen Stage 11 benchmark with fixed inputs/seeds and retain per-case results.
- Exercise one end-to-end path through task interpretation, capability matching, guarded execution, independent verification, and evidence-gated learning.
- Repeat on a clean environment with no developer-local state or undeclared secrets.
- Any failure blocks promotion; fix the smallest evidenced defect and rerun the full affected regression.

**Current evidence:** the review branch's latest CI run is being checked. The earlier run exposed a path-containment diagnostic compatibility failure; the message was corrected while retaining resolved-path containment. The current run must finish before this phase can pass.

## 11.9 — Security closure

**Purpose:** separate static policy checks from actual execution controls.

Required gates:
- AST inspection is advisory only and never grants permission to execute code in-process.
- Route untrusted code through the isolated child-process runner or the configured container backend. The production profile must use a non-root container, read-only base filesystem, no network by default, bounded CPU/memory/process/file/output resources, and reliable process-tree termination.
- POSIX resource limits and subprocess timeout must be tested; unsupported platforms must fail closed or use the container backend rather than silently claiming equivalent isolation.
- Verify workspace path containment after resolving symlinks, including absolute paths, drive/UNC paths, and traversal attempts.
- Scan all tracked files and reachable Git history for credentials; report detections and revocation status. Do not equate a narrow source scan with a repository-wide secret audit.
- Keep prompt-injection and ReDoS detection labeled as heuristic. Neither is a complete prevention mechanism.

**Current boundary:** the repository contains a separate OS child-process runner with POSIX resource limits, but that alone is not a hardened container/VM boundary and does not provide a network namespace. The AST checker is not a sandbox. Full production-profile isolation remains open until the container policy and adversarial tests prove it.

## 11.10 — Mathematical and reasoning validation

**Purpose:** reject invalid domains and test exact calculations against independent oracles.

Implemented on the review branch:
- Exact rational helpers reject float and boolean coercion.
- Probability inputs are constrained to [0, 1].
- Binomial and hypergeometric domains are validated.
- Combinatorics rejects invalid negative populations and booleans.
- Matrix multiplication rejects empty and ragged matrices; trace requires a square matrix.
- Dedicated boundary tests cover these cases.

Still required:
- Extend exact-input validation to every public oracle entry point (including polynomial coefficient powers and all matrix routines).
- Add singular, degenerate, boundary, malformed, and adversarial cases for every oracle family.
- Verify with independent test oracles and unseen generated cases, not only self-consistency checks.
- Replace string-pattern fallacy checks with a sound, explicitly scoped inference engine before making formal-logic claims. Keep the current component described as a heuristic demonstrator.

## 11.11 — Grounding and memory validation

Required gates:
- Test claim-level entailment and contradiction, including negation, scope, dates, quantities, and conflicting sources.
- Bind every factual claim to source provenance and preserve source identity through retrieval and answer construction.
- Test memory write/read across process restart, tenant isolation, duplicate IDs, rollback, mutation isolation, and explicit forgetting.
- Verify that forgetting removes the intended memory from all relevant indexes and derived caches, with an auditable result.
- Expand abstention and contradiction benchmarks beyond the current tiny synthetic negative-control set; empty evaluation sets must never report perfect scores.

**Current boundary:** memory mutation isolation and duplicate-ID handling have review-branch fixes. Lexical token/n-gram overlap is not semantic entailment; broader grounding and persistence/forgetting verification remain open.

## 11.12 — Benchmark and audit reconciliation

Required gates:
- Reconcile historical test totals using raw CI logs for the exact commit and exact test command. Until then, do not repeat conflicting totals as verified.
- Freeze code SHA, vault digest, source manifest, benchmark task files, generator seeds, Python/runtime versions, and environment metadata.
- Record per-case pass/fail, expected/actual output, abstention, latency, and resource usage.
- Keep training/tuning data separate from held-out and adversarial evaluation data.
- Publish a limitations section alongside every score; do not extrapolate small synthetic suites to general intelligence.

## 11.13 — Release candidate

Required gates:
- Build distributable packages from the tagged candidate and verify hashes.
- Test install, CLI, API, and documented quick-start on clean Linux and Windows environments.
- Test configured vault discovery, missing-vault behavior, permissions, backup/restore, and graceful shutdown.
- Run API/CLI contract and smoke tests against the built package, not only the source checkout.
- Verify dependency/license metadata, security policy, release notes, known limitations, and rollback instructions.
- Tag only after all previous phase gates pass; attach benchmark and CI evidence to the release.

## Promotion rule

No phase is marked complete by documentation changes alone. Any failed test stops progression. A fix requires the focused regression plus the complete relevant suite. Do not merge a release-candidate branch to `main` while required checks are pending or failing. If a required isolation or verification guarantee is unavailable, state the limitation and leave the gate open.
