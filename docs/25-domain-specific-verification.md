# Domain-specific verification contracts

## Purpose

Strategic Phase 6 requires Mirror to evaluate an outcome using verification methods
appropriate to the domain. A programming test, mathematical proof, physical
simulation, biological observation and engineering tolerance check are not
interchangeable evidence.

`mirror_crystal/domain_verification.py` adds the shared evidence contract and
policy engine. It extends the existing execution-fingerprint and independent
checker boundary with evidence-kind and domain requirements.

## Core flow

1. A capability declares the domain and its postconditions.
2. A host configures a `DomainVerificationRequirement` for that capability,
   listing required evidence kinds and a minimum number of distinct verifier IDs.
3. Each required evidence kind maps to one allow-listed
   `DomainVerificationCheck`.
4. The check receives a `VerificationContext` and must return a
   `DomainEvidenceArtifact`, not a boolean.
5. Mirror validates artifact binding, evidence reference, canonical payload
   digest, declared conditions, evidence decision and verifier identity.
6. All required kinds, all postconditions and the independent-verifier floor
   must pass for that execution step.

## Example evidence profiles

These are policy examples, not built-in solvers or claims of production
verification coverage.

| Domain | Example required evidence kinds |
|---|---|
| Programming | `test_suite`, `static_analysis`, optionally `property_test` |
| Mathematics | `exact_computation`, `formal_proof` |
| Physics | `dimensional_analysis`, `simulation`, `observation` |
| Biology | `source_corroboration`, `experimental_observation`, `reproducibility` |
| Engineering | `simulation`, `tolerance_analysis`, `safety_constraint_check` |

Policy owners must choose kinds appropriate to the actual claim. A software
test does not establish a theorem; a simulation does not by itself establish
that a physical model is valid; a source citation does not prove an experimental
result is reproducible.

## Failure semantics

- **PASS** — the evidence artifact is correctly bound and covers one or more
  declared postconditions.
- **FAIL** — the verifier rejects the claim, returns malformed evidence, or
  attempts to verify an undeclared condition.
- **BLOCKED** — a verifier/policy is missing, the capability is not evidence
  eligible, there are too few distinct verifier identities, or evidence is
  inconclusive.
- A failed or blocked report can never be promoted as a verified result.
- Evidence is bound to the exact execution fingerprint. Replaying an artifact
  against another execution fails closed.

Evidence payloads are frozen JSON and receive a deterministic SHA-256 digest.
The `evidence_ref` is an audit locator (for example a report ID, proof bundle,
dataset record or simulation artifact); the domain verifier must authenticate
and interpret the referenced material. Presence of a locator alone is not
authentication.

## Current acceptance boundary

The policy enforcement tests cover five domain labels and adversarial binding
cases. Their verifier callbacks are deterministic test doubles. This proves
that the contract enforces declared evidence requirements; it does not supply
compiler infrastructure, formal proof engines, simulation software, or
scientific/experimental review systems. Strategic Phase 6 remains PARTIAL until
those real adapters pass frozen held-out and adversarial benchmarks.

## Minimal API sketch

```python
from mirror_crystal import (
    DomainVerificationCheck,
    DomainVerificationEngine,
    DomainVerificationRequirement,
    DomainVerifierRegistry,
)

registry = DomainVerifierRegistry()
registry.register(
    DomainVerificationCheck(
        check_id="test-suite-v1",
        verifier_id="isolated-test-runner",
        domain="programming",
        evidence_kind="test_suite",
        callback=run_tests_and_return_evidence,
    ),
    independent=True,
)

report = DomainVerificationEngine(registry).verify(
    execution,
    capabilities,
    requirements={
        "compile-module": DomainVerificationRequirement(
            domain="programming",
            evidence_kinds=("test_suite", "static_analysis"),
            minimum_independent_verifiers=2,
        ),
    },
    required_checks={
        "compile-module": {
            "test_suite": "test-suite-v1",
            "static_analysis": "static-analysis-v1",
        },
    },
)
```

The callback above is intentionally a host-supplied integration point; it is
not included as a pretend built-in test runner.
