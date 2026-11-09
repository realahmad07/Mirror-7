# Canonical Blueprint and Rule Architecture

## Purpose
Mirror's knowledge foundation is a composable system of blueprints and rules. A blueprint describes a capability or transformation; rules provide executable or retrievable pieces used to realize it. Stored examples, retrieval similarity and legacy trust labels are not proof of truth.

## Canonical contract and storage
The versioned contract in \`blueprint_contracts.py\` records kind, domain, intent, prompt/response templates, slots, preconditions, postconditions, dependencies, provenance, licensing and verification evidence. Legacy adapters preserve prior trust labels as metadata but always emit \`unverified\` evidence.

\`DiskCrystalVault\` remains the compatibility facade with storage, indexing, retrieval and response logic separated into focused modules. SQLite stores canonical blueprints beside legacy rules without destructively rewriting existing vault files. The ordinary \`respond()\` path checks indexed canonical prompt patterns first and abstains when more than one blueprint matches.

## Dependency and verification policy
Composition is deterministic and dependency-first. Missing dependencies, cycles, missing template values, unmet prerequisites, missing postcondition verifiers and empty outputs block a chain. Arbitrary derived expressions require an explicit executor.

Fact-blueprint reliability now has a second gate: every factual dependency must have a frozen atomic-claim scope with two distinct independent source reviewers per claim. A reviewer decision includes an HTTPS source, locator, quoted excerpt, excerpt digest, retrieval date, license, reviewer identity, method and review timestamp. Legacy trust tiers are never upgraded automatically.

The metadata gate checks freshness, source references, content digests and conflicting records. This is auditable evidence policy, not automatic external fact-checking: reviewers must inspect whether the cited passage actually entails the claim.

## Legacy migration
Use \`scripts/migrate_legacy_blueprints.py\` in dry-run mode before applying any migration. It backfills in bounded pages and preserves original rule rows. Without an approved review manifest, converted blueprints remain unverified. Promotion requires two independent approvals for each pre-registered claim.

See [docs/15-source-evidence-benchmark-and-sandbox.md](15-source-evidence-benchmark-and-sandbox.md) for the manifest contract, CLI usage and review procedure.

## Evaluation
The evaluation-only factual v1 dataset is source-linked and digest-pinned but currently pending independent review. It contains no fabricated reviewer records and scoring is blocked until the review gate is satisfied. The harness measures accuracy, unsupported-answer rate, claim support, abstention precision/recall, coverage, latency and calibration over answered responses. It archives raw outputs with hashes. Its built-in lexical judge is a baseline, not semantic entailment.

Synthetic tests remain regression and negative controls, not a substitute for a broad independent factual benchmark. Add licensed, independently annotated, multilingual, adversarial and freshness-sensitive examples before drawing general reliability conclusions.

## Execution isolation
Production code verification now defaults to a Docker/Podman container with network disabled, a read-only filesystem, dropped capabilities, no-new-privileges, non-root user and CPU/memory/PID/file limits. Missing container runtime or image causes a fail-closed rejection.

The explicit \`MIRROR_SANDBOX_MODE=process\` mode is a lower-isolation test/development override only. A container is not a VM and does not eliminate all kernel/container-runtime risks; high-risk multi-tenant execution should move to a dedicated VM/microVM boundary with host hardening.

## Current capability boundary
Crystal can retrieve stored examples, generalize a bounded set of numeric patterns, perform selected deterministic CAS operations, recall explicit session constraints, compose explicit blueprint chains, and execute candidate code inside the configured boundary. It is not a general-purpose factual knowledge engine or established general intelligence.