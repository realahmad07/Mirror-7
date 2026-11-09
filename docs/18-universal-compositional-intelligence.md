# Universal Representation and Compositional Intelligence

## Objective

Mirror uses one shared intelligence engine across domains. A domain contributes
knowledge, typed operators, task interpretation rules, and an independent
verification adapter; it does not create a separate reasoning architecture.

The intended cycle is:

```text
Knowledge
  -> Capabilities
  -> Composition
  -> Execution
  -> Independent Verification
  -> Reusable Capability Learning
  -> Future Composition
```

This is a compositional system: it should construct a new solution from known
operations and contracts, not require a stored answer for every possible request.

## Shared representation layer

`mirror_crystal/universal_representation.py` defines versioned, typed contracts:

- `RepresentationNode`: entities, facts, relations, rules, constraints, formulas,
  procedures, capabilities, observations, evidence, goals, and artifacts.
- `ProvenanceRecord`: source, content digest, license, capture time, and lineage.
- `EvidenceRecord`: curated, execution-tested, formally verified, or independently
  reproduced evidence, kept separate from the underlying knowledge.
- `TaskSpec` and `SubtaskSpec`: goal, domain, typed inputs/outputs, constraints,
  dependencies, and acceptance criteria.
- `CapabilitySpec`: domains, typed inputs/outputs, tags, pre/postconditions,
  dependencies, implementation graph, and evidence.
- `TaskPlan`, `VerificationReport`, and `LearningGateReport`: planning,
  execution, validation, and learning outcomes.

Legacy Crystal blueprints are imported as knowledge nodes. Their old trust tier,
text, or test-code field is not enough to turn them into executable capabilities.

## Shared intelligence engine

`mirror_crystal/universal_intelligence.py` provides:

1. `CapabilityRegistry`: stores knowledge nodes, typed capability specs, and
   locally registered executors. Only trusted executable capabilities can run.
2. `UniversalTaskInterpreter`: maps recognized requests to `TaskSpec`; domain
   adapters can add native interpretation rules.
3. `UniversalTaskPlanner`: dependency ordering, typed capability matching,
   domain checks, ambiguity rejection, and capability-gap reporting.
4. `UniversalExecutionEngine`: input binding, runtime type checks, dependency
   output wiring, execution, and output schema enforcement.
5. `UniversalVerifier`: shared mandatory acceptance/output checks plus
   domain-specific verification. A generic handler returning successfully is not
   proof of correctness.
6. `CapabilityLearningEngine`: admits only independently evaluated reusable
   compositions; failed gates do not change the trusted registry.
7. `CrossDomainTransferEngine`: transfer starts unverified and must pass fresh
   target-domain evaluation.
8. `UniversalTaskCurriculum`: lists domains with missing trusted executors.

## First native capabilities

Mirror currently installs its own local primitives for numeric add/subtract/
multiply/divide, comparisons, boolean logic, sequence length/sum, and text
concatenation. Each built-in gets execution-tested status only after its own
deterministic registration self-test.

The first native code-generation examples are in
`mirror_crystal/native_program_synthesis.py`. A typed `ProgramIR` is compiled
into Python AST nodes, emitted as Python source, and checked by Mirror's existing
TDD sandbox. The calculator composes arithmetic dispatch and error handling.
The bubble-sort program composes local list copying, nested bounded loops,
comparisons, conditional branches, tuple assignment, and a return operation.
Both are constructed without an external code-generation model.

## Use

Python:

```python
from mirror_crystal import UniversalIntelligenceEngine

mirror = UniversalIntelligenceEngine()
result = mirror.solve_prompt("What is 19 plus 23?")
assert result["verified"] is True
assert result["answer"] == "42.0"

program = mirror.solve_prompt(
    "Build a Python calculator supporting addition, subtraction, "
    "multiplication and division."
)
assert program["verified"] is True
print(program["answer"])
```

HTTP:

```http
POST /api/universal/solve
Content-Type: application/json

{"prompt": "What is 19 plus 23?"}
```

A structured `TaskSpec` can be supplied instead of a prompt when the caller has
already constructed subtask dependencies and acceptance criteria.

## Add a new domain

A domain pack registers its own `CapabilitySpec` + local executor, optional
native interpreter, and independent verifier through
`UniversalIntelligenceEngine.register_domain_adapter()`. It uses the same
representation, planner, executor, registry, and learning gate as every other
domain. Domain-specific verification is additional to the shared output-type and
acceptance-criteria checks; it cannot bypass them.

A domain without a trusted executable capability is reported as
`capability_gap`. The engine does not fabricate an answer just because the
domain name is known or knowledge text exists.

## Composition and learning

A task can contain multiple subtasks with explicit dependencies and typed
bindings, for example `step:add.result` feeding `compare.left`. The planner
orders the graph, resolves a compatible capability for every step, and refuses
cycles, missing dependencies, ambiguous matches, and incompatible schemas.

After a solution passes its domain verifier and the held-out/adversarial,
multi-seed learning gate, it can be turned into a replayable composition
capability. Mirror executes the base, held-out, and adversarial suites itself;
it attests the report with a per-engine HMAC and records the candidate, runner,
and independent-verifier fingerprints. A report is bound to the exact
composition/transfer being promoted. Promotion rejects hand-constructed,
edited, stale, cross-engine, or different-candidate reports.

For durable learned-capability persistence across process restarts, set
`MIRROR_CAPABILITY_ATTESTATION_KEY` to a secret of at least 32 characters in
the deployment secret manager. Mirror signs the capability payload before
writing it to Crystal, and validates that signature before loading an executable
learned capability. Without this key, local in-memory learning can work, but
persistent promotion/reload is deliberately unavailable. Keep this key stable
for the lifetime of the persisted capability store; rotate it only with an
explicit re-signing/migration process. The persisted implementation is a typed Mirror composition graph,
not a copy of one answer. Replays use new inputs and invoke the same capability
registry and verification pipeline. Transferred capabilities lose their previous
evidence and require target-domain re-verification.

## Current scope and honest limitations

This is a foundation, not completed general intelligence.

- Implemented and tested today: typed shared representation, native numeric and
  logic/sequence/text primitives, explicit multi-step composition, arithmetic
  independent recomputation, native calculator AST synthesis + TDD, pluggable
  domain adapters, capability-gap handling, gated replayable learning contracts,
  and cross-domain transfer proposals.
- Not yet implemented as production domain packs: physics, biology, chemistry,
  and engineering operators, their interpreters, and independent scientific/
  engineering verifiers. A universal interface is not evidence that those domains
  already work.
- The initial native request interpreter recognizes a narrow set of forms. Rich
  unrestricted natural-language decomposition still needs broader native rules
  and evaluation.
- The first ProgramIR compiler is deliberately small. General loops, classes,
  data structures, multi-file projects, and broader program constructs need
  additional typed nodes and tests.
- Generic domain verification requires explicit acceptance criteria and is
  only contract verification. It does not establish scientific truth.
- Capability promotion requires real, independent cases, seeds, held-out and
  adversarial evaluations. A test passing on one example does not justify
  learning a capability.
- Passing the repository test suite confirms software contracts, not AGI or
  broad human-level reasoning.

## Nine-phase roadmap status

| Phase | Current implementation | Remaining work |
|---|---|---|
| 1. Knowledge foundation | Provenance/evidence records; legacy blueprints indexed separately from executable capabilities; trust tiers retained. | Broader source review, curation and contradiction resolution across all existing stores. |
| 2. Universal representation | Shared typed nodes, TaskSpec, SubtaskSpec, CapabilitySpec, provenance, evidence and execution contracts. | Evolve schema versions and add domain-specific quantity/units/uncertainty structures as needed. |
| 3. Task understanding | Rule-based native request interpretation, arithmetic AST decomposition, typed DAG tasks, and domain-interpreter hooks. | Broaden natural-language decomposition and connect native Mirror language reasoning to typed TaskSpec proposal/repair. |
| 4. Capability system | Native arithmetic, comparison, boolean, sequence and text primitives; calculator and bubble-sort synthesizers; typed domain adapter API. | More local operators and real domain packs for physics, biology, chemistry, engineering and other fields. |
| 5. Composition | Dependency-aware task DAG, typed binding, ambiguity and cycle rejection, expression-tree decomposition, reusable composition replay. | General capability search/backtracking, cost-aware planning, richer subgoal discovery and large plan optimization. |
| 6. Verification | Independent Decimal recomputation, TDD sandbox for programs, mandatory output and acceptance contracts, pluggable domain verifiers. | Independent scientific/empirical verifiers for each domain; current generic checks are not scientific proof. |
| 7. Self-improvement | Executed multi-seed/held-out/adversarial gate, engine-instance attestation, replayable capability promotion and Crystal persistence. | Automatically generate candidate abstractions and test suites; broader reviewer governance and revocation/rollback workflows. |
| 8. Cross-domain intelligence | Transfer proposals strip source evidence and require a new target-domain gate. | Domain-tested analogical transfer and validated structural correspondences, not just compatible operator reuse. |
| 9. Continuous expansion | Capability-gap curriculum reports and persistent learned capability blueprints. | Autonomous curriculum scheduling, frontier prioritization and continuous held-out evaluation. |

The roadmap is therefore an integrated foundation with working end-to-end examples—not a claim that every phase is complete for every domain. The highest-confidence current end-to-end paths are arithmetic composition and two native Python synthesis examples (calculator and bubble sort).

## Next expansion

Extend the shared ProgramIR and operator algebra (variables, conditionals,
loops, functions, data structures); then add physics, biology, chemistry, and
engineering adapters one at a time. Every adapter must contribute typed
capabilities, independent verification, held-out/adversarial tests, and fresh
transfer evidence before it can enter the trusted capability registry.
