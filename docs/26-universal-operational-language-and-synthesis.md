# Universal Operational Language and Capability Synthesis — v1

## Purpose

The Universal Operational Language (UOL) gives Mirror a common, domain-neutral way to describe and execute bounded operations. The capability synthesizer searches this language for new candidate procedures from examples and structured rules supplied to Mirror.

This is a new construction layer above the existing knowledge and capability contracts. It does not replace the universal representation, task parser, planner, controlled executor, verifier, or evidence-gated learning system.

## UOL representation

`mirror_crystal/operational_language.py` provides a typed expression tree and a versioned program contract:

- typed inputs and variables;
- literals, primitive calls, conditionals, arrays and records;
- field/index access and local bindings;
- bounded map and fold transformations;
- explicit preconditions and postconditions;
- domain-scoped structured rules with provenance and evidence status;
- stable serialization and fingerprints.

The v1 primitive set includes arithmetic, comparison, boolean, text, collection length, sum and product operations. The interpreter evaluates the expression tree directly. It does not evaluate Python source, import modules, call arbitrary host functions, access the filesystem or network, or construct shell commands. Step, depth, collection and numeric magnitude limits constrain execution.

## Supplied rules

A rule can be added directly as a typed `OperationalRule`, or adapted from a canonical `KnowledgeRecord` only when its `content.uol_rule` field contains a structured input schema and serialized UOL expression. Natural-language rule descriptions are not silently compiled into executable code.

The adapter preserves provenance and epistemic status. Disputed or unknown rules cannot execute; assumed rules may be used as explicitly unverified hypotheses. A successful synthesis never upgrades the evidence of a rule it used.

## Capability synthesizer

`mirror_crystal/capability_synthesizer.py` implements bounded, deterministic example-guided search:

1. Build typed terminals from declared task inputs and explicitly supplied constants.
2. Enumerate type-correct primitive expressions and eligible structured rule calls within a caller-defined depth and candidate budget.
3. Evaluate candidates against all training examples.
4. Require a separate, disjoint held-out validation set to pass every case.
5. Return a serialized UOL program candidate or an explicit `NO_SOLUTION` / `BUDGET_EXHAUSTED` result.

The v1 search grammar discovers primitive-expression combinations and compositions that call supplied rules. UOL itself can interpret conditionals, records, maps and folds, but the synthesizer does not yet discover arbitrary conditional programs, loops, scientific models, or unrestricted source code. Search is deliberately bounded and deterministic.

## Existing Mirror capability boundary

A successful UOL program can be adapted to the existing `CapabilitySpec` schema. The adapter marks every synthesized capability `UNVERIFIED`. It is therefore storable and reviewable, but it is not eligible for ordinary capability execution or promotion until Mirror's independent verification and evidence-gated learning path supplies appropriate evidence.

The UOL native-handler adapter invokes only the bounded UOL interpreter. It does not create or execute generated Python code, and execution success is not verification.

## Current acceptance

Focused tests cover arithmetic construction, user-rule composition, held-out validation, disjoint example splits, type checks, rule status/domain gates, recursive-rule rejection, pre/postconditions, map/fold behavior, serialization, resource limits, and unverified candidate admission.

The initial isolated suite passes 15 tests. Full repository CI is the next acceptance gate. This phase does not claim unrestricted natural-language-to-code synthesis, arbitrary algorithm invention, general cross-domain transfer, or AGI.

## Minimal use

```python
from mirror_crystal import (
    CapabilitySynthesizer, Example, SynthesisGoal,
    UOL, VariableSpec,
)

goal = SynthesisGoal(
    name="increment-and-scale",
    domain="mathematics",
    inputs=(VariableSpec("x", "integer"),),
    output_type="integer",
    training=(Example({"x": 1}, 6), Example({"x": 2}, 9)),
    validation=(Example({"x": 4}, 15),),
    constants=(2, 3),
)
synthesizer = CapabilitySynthesizer(UOL())
report = synthesizer.synthesize(goal)
if report.succeeded:
    candidate = synthesizer.to_candidate_capability(report)
    assert not candidate.executable  # independent verification is still required
```
