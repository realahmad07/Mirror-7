# Strategic Phase 8 — Cross-Domain Transfer

**Implementation:** `mirror_crystal/phase8_cross_domain_acceptance.py`  
**Tests:** `mirror_crystal/tests/test_phase8_cross_domain_acceptance.py`  
**Status:** bounded acceptance implemented; all target candidates remain subject to fresh target-domain verification and learning.

## Goal

Demonstrate that an independently reproduced UOL operation can move from one source domain into multiple distinct target domains without copying its source trust label. Transfer is restricted to domain-neutral UOL dependencies and an unchanged typed interface.

## Acceptance flow

1. Validate the source learning proposal: its capability must be executable from independently reproduced evidence, its provenance must bind the exact source program, and its saved handler must contain that exact program and the same recursively referenced verified rule definitions as the target UOL catalog.
2. Require at least two unique target domains by default (configurable within a maximum of five); the source domain cannot be a target.
3. For every target, require separate training and held-out examples. The existing transfer engine checks a minimum of three training and two held-out examples by default, enforces disjoint input sets, preserves typed ports/units/dimensions, and rejects source-domain-only rule dependencies.
4. Obtain fresh target-domain verification observations from the host-supplied target verifier callback. Their task IDs and execution fingerprints must be distinct across targets. Each observation contains one execution step, and its input must be disjoint from target transfer training/held-out inputs and from other target verification probes.
5. Pass the target candidate through the existing execution-bound domain-verification and repeated-observation learning gate (three distinct tasks and at least two verifier identities by default).
6. Report PASS only when every requested target has passed the transfer, fresh probe, independent verification, output replay and learning gates.

The benchmark manifest is a canonical SHA-256 digest over the source program identity, target domains, train/held-out example identities and declared target postconditions. The report retains the per-domain transfer and learned proposals for inspection.

## API

```python
from mirror_crystal import (
    CrossDomainTransferBenchmarkEngine,
    TransferTargetBenchmark,
)

report = CrossDomainTransferBenchmarkEngine().evaluate(
    "sealed-transfer-benchmark-v1",
    verified_source_proposal,
    uol_language,
    (
        TransferTargetBenchmark(
            domain="mathematics",
            training=math_train,
            held_out=math_held_out,
            postconditions=math_postconditions,
            observations=math_verifier_callback,
        ),
        TransferTargetBenchmark(
            domain="physics",
            training=physics_train,
            held_out=physics_held_out,
            postconditions=physics_postconditions,
            observations=physics_verifier_callback,
        ),
    ),
)
```

A target-domain transfer candidate is always `UNVERIFIED` and not ordinarily executable. Its separately verified learning proposal is returned only after the Phase 5 gate; it is still not registered until the host explicitly calls `VerifiedProgramLearningEngine.promote`.

## Tests and negative controls

The Phase 8 test module covers:
- successful transfer and separate re-learning in two different target domains;
- source-domain equality and duplicate target-domain rejection;
- absent target verification observations;
- counterfeit source-handler/program binding;
- target held-out failure;
- verifier-probe leakage into the target training set.

## What this phase does not establish

This is **bounded cross-domain transfer of typed UOL procedures**, not unrestricted semantic analogy or general intelligence. Transfer requires the source operation, globally scoped verified rule dependencies, explicit target examples, postconditions and host-supplied target verifiers. The native verifier suite contains deterministic bounded checks; it is not a general-purpose physics simulator, biological experiment platform, formal theorem prover or source-authentication authority. Host-supplied verifier identities and callbacks remain operator trust responsibilities. A passing finite benchmark applies only to its frozen cases and declared invariants.
