# System Architecture

## Layers

| Layer | Responsibility |
|---|---|
| Research phases | Bounded hypotheses, mechanisms, integrations, and evaluators. |
| Mirror runtime | Session-level state and research-engine orchestration. |
| Backend service | Lifecycle, persistence, actions, metrics, response generation, and capacity. |
| HTTP boundary | Stable JSON API for the UI/client. |
| Response generation | Converts a verified realization contract into user-facing text. |
| Training | Learned model experiments and dataset tooling. |
| UI | External presentation layer, intentionally decoupled. |

## Universal compositional intelligence

Mirror now has a domain-neutral representation and capability engine under
`mirror_crystal/universal_representation.py` and
`mirror_crystal/universal_intelligence.py`.

| Shared component | Responsibility |
|---|---|
| Universal Representation | Typed facts, rules, constraints, formulas, procedures, capabilities, observations, evidence, goals, and artifacts with provenance. |
| Task Interpreter | Converts recognized requests or domain-adapter output into a versioned TaskSpec. |
| Capability Registry | Separates searchable knowledge from trusted executable capabilities. |
| Task Planner | Builds dependency-ordered task graphs and matches typed capability contracts. |
| Execution Engine | Binds inputs and prior outputs, validates types, and invokes local Mirror operators. |
| Universal Verifier | Enforces output schemas/acceptance criteria and invokes domain-specific independent verification. |
| Learning Engine | Promotes replayable compositions only after multi-seed, held-out, adversarial independent evaluation. |
| Transfer Engine | Copies capability structure to a new domain without copying its proof; target-domain re-verification is required. |

The first native operators cover arithmetic, comparisons, boolean logic, sequences,
and text. Native Python synthesis builds calculator and bubble-sort programs from
typed ProgramIR (including comparisons, assignments, conditionals, and bounded
loops), emits Python AST, and sends the result to Mirror's existing TDD sandbox.
This path does not require an external code-generation model.

Legacy blueprints are indexed as knowledge but do not become executors
automatically. Unknown or unsupported tasks return an explicit capability gap.

See [Universal Compositional Intelligence](./18-universal-compositional-intelligence.md)
for the contracts, runtime flow, API, tests, and current scope.

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
