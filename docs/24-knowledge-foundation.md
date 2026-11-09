# Phase 1 — Universal Knowledge Foundation

## Purpose

Phase 1 establishes what Mirror is allowed to treat as knowledge before the universal representation and task/capability pipeline consumes it. One canonical record format applies to mathematics, programming, physics, biology, chemistry, engineering, and future domains.

The phase makes a strict distinction between:
- **Content:** what a record says.
- **Epistemic status:** asserted, observed, derived, assumed, disputed, or unknown.
- **Verification evidence:** what checks have actually passed or failed.
- **Provenance:** where the content came from, its source reference, license, digest, capture time, and lineage.

A record's presence in memory or retrieval results does not make it true.

## Canonical knowledge records

`KnowledgeRecord` stores a typed knowledge item using a normalized subject–predicate–object claim, statement, domain, kind, tags, arbitrary JSON-compatible supporting content, epistemic status, provenance, verification evidence, and optional typed procedure contract.

Supported kinds include facts, definitions, rules, formulas, procedures, constraints, observations, models, assumptions, and heuristics. Records are immutable, versioned, canonically serialized, fingerprinted, and reject malformed JSON, missing identity fields, invalid provenance, or epistemic-status mismatches.

Legacy Crystal rules and canonical blueprints can be adapted, but their evidence levels and provenance are preserved. In particular, a legacy trust tier, retrieval match, or test-code field is never silently upgraded into proof.

## Admission and trust boundaries

A record is not automatically an assertable fact. Fact-like records are eligible for factual assertion only when they:
1. are not assumptions, disputed, or epistemically unknown;
2. carry formally verified or independently reproduced evidence with named passing checks and no failed checks;
3. carry a source identity, source reference, and content digest; and
4. are not part of an unresolved contradiction set.

Blueprint facts have an additional claim-level boundary: the blueprint adapter does not mark a fact assertable merely from blueprint-level metadata. The vault adapter includes a reliable `ClaimEvidenceReport` only after the frozen claim scope is reviewed under the existing two-independent-reviewer policy.

Assumptions use `KnowledgeKind.ASSUMPTION` and `EpistemicStatus.ASSUMED`. They remain available as explicitly labelled working assumptions but cannot be returned as verified facts. Unverified knowledge can be stored and inspected without becoming trusted.

## Contradiction handling

The deterministic conflict detector groups records by normalized domain, subject, and predicate. If the same proposition key has different object values, the records remain stored and the registry reports a `KnowledgeConflict`; it never picks a winner by recency, lexical score, trust label, or insertion order.

The DiskCrystalVault adapter writes explicit `contradicts` relations between the conflicting knowledge nodes in the Phase 2 representation graph. Search results identify conflicted records and force their `assertable` value to false until the conflict is resolved by a separate review process.

## Retrieval and cross-phase bridge

`KnowledgeRegistry.query` and `DiskCrystalVault.search_knowledge` provide deterministic lexical retrieval with visible match scores. This is not semantic entailment, source fetching, or proof; callers must inspect the returned evidence and conflict state.

An explicitly typed procedure can become a Phase 4 `CapabilitySpec` only when its record includes declared input/output ports, domains, intents, and passing operation-level evidence. The bridge preserves provenance and verification. It does not generate a native handler: execution still requires an explicitly registered host operation, and every result still needs independent verification.

Knowledge is stored in the Phase 2 representation graph as typed nodes, reusing its transactional SQLite persistence and lineage. It is not automatically ingested from private user requests.

## Tests and acceptance

Phase 1 tests cover canonical round trips, assumptions versus verified facts, blueprint evidence preservation, claim-review gating, identifier conflicts, contradictory claims, lexical retrieval, typed procedure conversion, representation-graph persistence, and durable contradiction links.

The combined Phase 1–9 acceptance test also exercises the complete path: knowledge → representation → task understanding → capability matching → composition → controlled execution → independent verification → repeated cross-task capability learning. Phase completion is recorded only after the full CI run passes.
