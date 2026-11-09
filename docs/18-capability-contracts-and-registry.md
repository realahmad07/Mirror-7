# Phase 4 — Universal Capability Contracts and Registry

## Purpose

Phase 4 defines what a reusable capability is. A capability is a domain-neutral operation with declared input/output ports, supported domains and task intents, prerequisites, dependencies, provenance, and evidence. The same contract can describe a mathematical transformation, a program construction operation, a scientific analysis operation, or a future domain capability.

## Contracts

`PortContract` describes a named port, its value type, requiredness, physical unit, and dimensional exponents. Runtime value checks distinguish booleans from integers. Port-to-port compatibility is conservative: names and declared units/dimensions must agree, and the type relation must be supported.

`CapabilitySpec` is immutable and has deterministic canonical serialization and a SHA-256 fingerprint. Capability evidence is reused from the existing verification contract. Only execution-tested, formally verified, or independently reproduced capabilities with named passing checks and no failed checks are marked executable. Curated or unverified records remain catalogued but are not execution-eligible.

`CapabilityRegistry` provides deterministic in-memory registration, lookup, domain filtering, idempotent re-registration, and conflict detection. Persistence is explicit through `DiskCrystalVault.store_capability`; records are stored as operation nodes in the Phase 2 representation graph and can be reloaded after reopening the vault.

## Existing blueprint adapter

`CapabilitySpec.from_blueprint` exposes an existing blueprint through the new port contract. It preserves its evidence and provenance and does not upgrade legacy records. The adapter is a bridge, not a claim that every blueprint is an executable operation.

## Boundary

This phase defines and stores capabilities. It does not identify which capabilities a particular task needs, compose them, execute them, verify their outcomes, or learn new capabilities.
