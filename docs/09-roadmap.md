# Roadmap After the 75+ Hour Build

The first 75+ hour construction phase is complete. The next work should be treated as a new research program rather than a continuation of arbitrary numbered phases.

## Track A — Public engineering surface

| Goal | State |
|---|:---:|
| Clean active checks | ✅ |
| Documentation hierarchy | ✅ |
| Backend smoke gate | ✅ |
| Release workflow | ✅ |
| Security guard | ✅ |
| UI/backend contract | ✅ |

## Track B — Native language realization

This is the highest-value unresolved model problem.

The next work should:

1. define a stronger held-out response benchmark;
2. freeze evaluator and data fingerprints;
3. test latent/state-prediction objectives rather than only raw byte prediction;
4. compare recurrent, state-space, and transformer alternatives under the same compute budget;
5. measure free-running generation directly;
6. retain a frozen baseline after every successful experiment.

## Track C — Native semantic-state scaling

The next question after Experiments 6–7 is not simply “make it bigger”.

The next question is whether learned state and transformation behavior can move from controlled synthetic operations to richer naturally occurring semantic relations without losing compositional generalization.

That requires new benchmarks, not just more epochs.

## Track D — Independent evaluation

Before making broad public intelligence claims:

- freeze the benchmark;
- publish the evaluator hash;
- separate training and evaluation artifacts;
- invite an external evaluator to reproduce or challenge the result.

## Track E — Product integration

The UI can be built against the current backend without waiting for the native model to finish.

## Track F — Compute strategy

A100 time should be spent only after an experiment has a falsifiable hypothesis.

Preferred sequence:

small pilot → negative controls → fixed baseline → multi-seed confirmation → scale.

Avoid another broad sweep of the failed scratch ByteGRU route without a new mechanism that directly addresses free-running instability.

## Next milestone

The next major milestone should be a measurable improvement in native response generation or a latent-state alternative that reduces dependence on raw autoregressive byte prediction while improving held-out free-running behavior.


## Native construction milestone — Phase 1 and Phase 2

The next backend increment adds a knowledge-to-operation bridge and counterexample-guided candidate refinement:

- [Phase 1–2 implementation and boundaries](27-phase1-knowledge-to-uol-and-phase2-cegis.md)
- Phase 1 imports only reviewed, typed UOL rule payloads from canonical knowledge records; it keeps source/evidence lineage and fails closed on contradictions or identifier collisions.
- Phase 2 uses a host-supplied oracle to generate counterexamples, adds them as constraints, and repeats bounded synthesis.
- Candidates remain unverified. The next unresolved work is expanding the synthesis grammar and connecting successful candidate programs to real, independently validated domain verifiers.


## Native construction milestone — Phase 3 and Phase 4

- Phase 3 expands UOL search to conditionals, map and fold transformations, retaining explicit candidate budgets and held-out checks.
- Phase 4 adds native deterministic checkers for a deliberately bounded set of mathematical, programming, physics, biology and engineering invariants.
- [Implementation details and evidence boundaries](28-phase3-advanced-synthesis-and-phase4-native-verification.md).

These increments do not make the broader original strategic Phase 6 complete: general-purpose domain simulators, source-authentication, formal proof tools and experimental validation remain separate work.


## Native construction milestone — Phase 5 and Phase 6

- Phase 5 learns a reusable native UOL capability only from repeated distinct executions, independently verified evidence and exact program-output replay. Capability and handler registration remain explicit.
- Phase 6 transfers a verified program only through globally verified dependencies, unchanged typed/unit/dimension ports, and fresh target-domain regression examples. Transfer candidates remain unverified until target verifiers and the Phase 5 learning gate pass.
- [Phase 5–6 implementation and acceptance boundaries](29-phase5-verified-program-learning-and-phase6-cross-domain-transfer.md).

The remaining broader research goal is not unrestricted transfer: it is to establish reliable semantic correspondence and cross-domain generalization on frozen, held-out, adversarial benchmarks without inheriting source-domain trust.


## Native construction milestone — Phase 7

- Phase 7 adds typed failure diagnosis across composition, synthesis, execution and domain verification, with failed-step and rejected-evidence localization.
- For UOL programs only, fresh host-oracle counterexamples can drive bounded CEGIS to propose a replacement that is replayed against the original train/held-out examples and new probes.
- Repair results remain UNVERIFIED and require independent domain verification plus the existing repeated-observation learning and explicit promotion gates. Runtime handlers are not blindly retried.
- [Phase 7 implementation and acceptance boundaries](30-phase7-failure-diagnosis-and-targeted-repair.md).

This closes a bounded failure-feedback loop; it does not establish unrestricted self-modifying software, arbitrary code repair, general scientific discovery or AGI.

## Operational governance milestone — Stage 22

The operational path now complements the native-language research track:

- [Stage 22 design and acceptance boundaries](37-stage22-trusted-real-world-execution.md).
- Host adapters are explicit, immutable and action-scoped; no generic network or shell bridge is exposed.
- Reviewer identity is authenticated by host-issued signed assertions; approvals are short-lived, action-scoped and single-use.
- External actions require adapter-signed observations, a durable tamper-evident event chain and transactional idempotency/replay controls.
- Interrupted outcomes are reconciled rather than blindly retried; reversible/consequential actions require signed, independently observed recovery checks.
- Feedback is separately reviewed and never silently promoted into trusted knowledge.

The native model roadmap remains a separate research track: operational governance does not imply stronger free-running language generation or general intelligence.
