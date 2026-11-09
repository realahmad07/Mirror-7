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

The first native code-generation example is
`mirror_crystal/native_program_synthesis.py`. A typed `ProgramIR` is compiled
into Python AST nodes, emitted as Python source, and checked by Mirror's existing
TDD sandbox. The calculator example builds addition, subtraction, multiplication,
and division (including division-by-zero handling) without any external model.

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
capability. The persisted implementation is a typed Mirror composition graph,
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

## Next expansion

Extend the shared ProgramIR and operator algebra (variables, conditionals,
loops, functions, data structures); then add physics, biology, chemistry, and
engineering adapters one at a time. Every adapter must contribute typed
capabilities, independent verification, held-out/adversarial tests, and fresh
transfer evidence before it can enter the trusted capability registry.
