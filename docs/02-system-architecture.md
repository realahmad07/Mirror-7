# System Architecture

## Layers

| Layer | Responsibility |
|---|---|
| Research phases | Bounded hypotheses, mechanisms, integrations, and evaluators. |
| Universal representation | Typed cross-domain knowledge nodes and relations with epistemic status, provenance, uncertainty, and durable storage. |
| Universal knowledge foundation | Canonical, domain-neutral facts, assumptions, rules, formulas, procedures, source provenance, evidence state, deterministic lexical lookup, and contradiction links. |
| Universal task understanding | Converts user requests into a domain-neutral TaskSpec with explicit goals, knowns, unknowns, constraints, outputs, and subtask dependencies. |
| Capability contracts and registry | Stores reusable domain-neutral operations with typed input/output ports, preconditions, dependencies, provenance, and explicit verification evidence. |
| Capability matching | Matches each task unit to domain/intent-compatible capabilities and reports missing inputs and evidence eligibility without executing anything. |
| Capability composition planning | Builds a bounded typed dependency plan, rejects unverified or unbound operations, and records postconditions as future verification requirements. |
| Controlled capability execution | Executes only explicitly registered native handlers in bounded child processes, validates typed inputs/outputs, and stops on first failure. |
| Independent outcome verification | Uses separately registered checks to verify each executed step and declared postconditions; execution success alone is not proof. |
| Evidence-gated capability learning | Proposes reusable composite capabilities only after repeated structurally identical plans succeed on distinct tasks and pass multiple independent verifiers. |
| Mirror runtime | Session-level state and research-engine orchestration. |
| Backend service | Lifecycle, persistence, actions, metrics, response generation, and capacity. |
| HTTP boundary | Stable JSON API for the UI/client. |
| Response generation | Converts a verified realization contract into user-facing text. |
| Training | Learned model experiments and dataset tooling. |
| UI | External presentation layer, intentionally decoupled. |

## Backend orchestration

mirror7_backend/service.py is the main backend facade.

It handles session registration, capacity limits, active-step accounting, checkpoint save/restore, action gateway access, response-generation configuration, response-model lifecycle, and backend metrics.

mirror7_backend/runtime.py owns the session-level Mirror interaction and builds the realization contract from semantic state.

## Response-generation boundary

| Component | Role |
|---|---|
| phase355_backend_realization_contract | Defines the structured realization payload. |
| phase357_response_generation_interface | Defines the generation request/call contract. |
| phase359_response_generation_adapter | Adapts external models to Mirror. |
| phase362_response_generation_config | Stores generation policy. |
| phase364_response_generation_validation | Validates generated output. |
| phase366_pretrained_response_realizer | Optional pretrained language renderer. |

The adapter boundary means the current pretrained model can later be replaced without rewriting the state/runtime system.

## Verified fact path

Phase 366 contains a verified-fact fast path. When the realization contract carries a verified fact, that value is returned directly instead of being entrusted to free-form language generation.

The corresponding regression test verifies that a failing model is not called for that path.

## HTTP boundary

mirror7_backend/http_server.py provides health/status endpoints, session creation and stepping, checkpoint save/restore, bounded action execution, and session/checkpoint deletion.

The server validates JSON request bodies, caps request size at 1 MiB, validates session identifiers, exposes configurable CORS, sends cache-control and common hardening headers, and maps expected failures to explicit HTTP statuses.

## Persistence and actions

Persistence is isolated from the session runtime so storage concerns do not become part of the cognitive mechanism.

Actions are passed through a bounded gateway rather than exposing arbitrary backend methods to an HTTP client.

## Research/training boundary

The phase tree and training tree are not one mandatory runtime path. A phase may represent a historical hypothesis, a bounded mechanism, an integration, or an acceptance boundary.

This repository preserves provenance instead of pretending that every historical experiment is simultaneously active in every request.
