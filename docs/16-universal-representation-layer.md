# Phase 2 — Universal Representation Layer

## Purpose

Mirror needs one shared way to represent knowledge and structure across programming, mathematics, physics, biology, engineering, and future domains. A domain may add vocabulary and content, but it must not require a separate representation engine.

This phase establishes typed data contracts and durable graph storage only. It does not interpret natural-language tasks, decompose goals, choose capabilities, execute procedures, compose solutions, or learn from outcomes. Those are later layers.

## Shared representation model

The implementation is in:

- mirror_crystal/universal_representation.py
- mirror_crystal/vault_representation.py
- mirror_crystal/vault_storage.py
- mirror_crystal/disk_vault.py

A RepresentationNode records a stable identifier, kind, domain, label, typed content, optional unit and dimensional exponents, extensible attributes, provenance, evidence, epistemic status, and uncertainty.

Node kinds include entities, concepts, facts, rules, constraints, formulas, procedures, operations, structures, observations, measurements, evidence, goals, and artifacts.

A RepresentationRelation connects two node identifiers with a typed predicate. It has a domain, direction, epistemic status, provenance, verification evidence, uncertainty, and attributes. The graph rejects relations whose endpoints are missing.

## Knowledge is not proof

Epistemic status and verification evidence are independent fields. An item can be asserted while remaining unverified; an assumption does not become an observation; disputed evidence remains explicitly disputed.

The representation reuses Mirror's existing verification evidence contract. Legacy trust tiers, retrieval similarity, and the presence of a test-code string do not upgrade evidence.

## Durable persistence

DiskCrystalVault adds two SQLite tables:

- representation_nodes
- representation_relations

Existing rules, causal edges, canonical blueprints, and claim-evidence records are left intact. Opening an existing vault creates the new tables if absent; no existing row is rewritten.

Storage is idempotent for identical canonical records and rejects identifier reuse with different content. Relation endpoints must already exist. Query APIs support domain, kind, epistemic status, and relation predicate. Graph loading is bounded to at most 400 nodes per request so a large disk-backed vault is not materialized wholesale in memory.

## Existing blueprint adapter

RepresentationNode.from_blueprint and DiskCrystalVault.ingest_blueprints_as_representations expose existing canonical blueprints through the shared model. The adapter preserves the domain, intent, source, source reference, license, content digest, capture time, and verification evidence. Blueprint dependencies are represented as depends_on relations when both endpoints are available.

This adapter is additive: it does not rewrite blueprint records or alter the existing Crystal response path. Unverified source material remains unverified.

## Completion boundary

Phase 2 is a shared schema and persistence foundation. It does not itself supply chemistry, physics, biology, engineering, or programming knowledge. It does not yet understand user requests, select operations, compose a solution, or prove that a claim is true. Those responsibilities remain assigned to subsequent layers.
