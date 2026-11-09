# Phase 3–4: Advanced Capability Synthesis and Native Domain Verification

## Phase 3 — advanced capability synthesis

The UOL synthesizer now searches beyond flat primitive expressions. The bounded grammar can construct:

- conditional selection using a boolean guard and compatible result branches;
- array transformations using UOL map expressions;
- array reductions using UOL fold expressions;
- compositions that contain supplied structured rules inside nested expressions, with rule references collected recursively.

Every candidate still has to pass all supplied training examples and a disjoint held-out validation set. The existing CEGIS loop can then add oracle-discovered counterexamples and search again. Candidate-generation budgets and the CEGIS round ceiling remain explicit. The UOL interpreter remains the execution semantics; the synthesizer does not emit or execute Python source.

The search is intentionally finite. Map element variables are currently inferred as UOL any and are checked concretely during bounded execution against examples. The engine does not yet synthesize unrestricted loops, recursion, arbitrary native code, scientific models or open-ended data structures. A candidate that fits examples is not thereby proven correct.

## Phase 4 — native domain verification engines

`mirror_crystal/native_verifiers.py` supplies two concrete verifier paths for each of the five initial domains. They register into the existing `DomainVerifierRegistry` and return `DomainEvidenceArtifact` objects bound to the exact step, capability and execution fingerprint. Each artifact contains a digest of the structured evidence and the verifier's observations.

| Domain | Independent check 1 | Independent check 2 |
|---|---|---|
| Mathematics | Exact polynomial normalization over rational coefficients. | Exact rational evaluation over separately supplied sample points. |
| Programming | A bounded reference interpreter replays serialized UOL programs without calling the production UOL interpreter. | Input/output port names, runtime types and finite JSON values are checked against the capability contract. |
| Physics | Equation terms are checked for dimensional homogeneity using dimension vectors. | A declared conserved scalar is checked before/after using an explicit tolerance. |
| Biology | Measurements are checked against declared numeric ranges and units. | Component quantities are checked against an explicit mass-balance total and tolerance. |
| Engineering | Reported stress is independently recomputed from load/area in N/mm² (MPa). | Reported safety factor is recomputed and compared to the required minimum. |

### Using the built-in suite

```python
from mirror_crystal import (
    DomainVerificationEngine,
    NativeDomainVerifierSuite,
)

suite = NativeDomainVerifierSuite()
report = DomainVerificationEngine(suite.registry).verify(
    execution_report,
    capability_specs,
    requirements={
        capability_id: suite.requirement("mathematics"),
    },
    required_checks={
        capability_id: suite.required_checks("mathematics"),
    },
)
```

The executed step must include a declared `__mirror_verification__` input with `schema_version: 1`, the matching domain, and a domain-specific payload. The suite does not trust a boolean callback or a self-asserted “verified” flag. Invalid or missing evidence fails closed. The capability's postconditions must include the two condition identifiers returned by `suite.postconditions(domain)`.

## Evidence boundary

These are real, deterministic checkers for the explicitly named mathematical, programming and numeric/scientific invariants—not universal proof assistants, compilers, physics simulators, biological experiment reviewers or engineering certification systems. The checks validate only the supplied structured claim and execution context. A passed dimensional check does not prove a physical model; a passed biological range check does not diagnose a sample; a load/area safety-factor calculation does not certify a real structure.

Each check has its own algorithm and verifier identity, and the existing domain engine still enforces execution binding, distinct verifier identities, declared postconditions and evidence completeness. Source authenticity and suitability of supplied inputs remain host responsibilities. Cross-domain transfer and broad scientific validation remain future work.

## Acceptance tests

Focused tests exercise:
- conditional selection, map and fold synthesis on held-out examples;
- the advanced candidate-budget stop;
- all ten native domain checkers through the real `DomainVerificationEngine`;
- valid mathematical, programming, physics, biology and engineering evidence;
- adversarial mismatches and missing evidence that must not pass.

The full repository CI and Security Guard workflows are the acceptance gate for this increment.
