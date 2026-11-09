# Phase 7 — Failure Diagnosis, Targeted Repair and Bounded Self-Improvement

## Purpose

Phase 7 adds the missing feedback path after composition, synthesis, execution and
independent verification. It does not replace those systems. It identifies the
smallest failure location supported by a typed report and proposes a constrained
repair action.

Implementation: `mirror_crystal/failure_repair.py`.

## Diagnosis coverage

The diagnostic engine accepts:
- a blocked `CompositionPlan`, preserving planner blockers;
- a failed, budget-exhausted or counterexample-limited `SynthesisReport`;
- a blocked or failed `PlanExecutionReport`, identifying the first failed step and
  exact capability/handler;
- a failed or blocked `DomainVerificationReport`, distinguishing rejected
  evidence from missing or inconclusive verification.

Recommendations are intentionally conservative. Execution failures are not
automatically retried because a handler may have side effects. Verification
thresholds are never relaxed to manufacture a pass.

## Targeted program repair

`FailureRepairEngine.propose_program_repair` accepts an existing UOL program,
the original typed synthesis goal, fresh probe inputs and an explicit
host-supplied oracle. It:

1. verifies that the old program matches the goal's domain and input interface;
2. rejects malformed, duplicate or train/held-out-overlapping probes before
   calling the oracle;
3. records only probes on which the old program demonstrably disagrees with the
   oracle or cannot execute;
4. uses the existing bounded counterexample-guided synthesis (CEGIS) engine to
   search a replacement;
5. replays the result on the original training set, held-out set and fresh probes;
6. returns a candidate with `UNVERIFIED` evidence.

The oracle is the source of expected values and remains host-supplied. The engine
does not invent truth labels or claim that finite probes prove general
correctness.

## Trust and autonomy boundary

A successful repair is a **proposal**, not a registry mutation. It is not
automatically executable, learned, persisted or promoted. It must pass the
independent domain-verification process and repeated-observation learning gate
from Phases 5–6 before it can become a reusable capability.

No automatic repair is attempted for arbitrary Python, native handlers,
scientific models or external side effects. Phase 7 repairs bounded UOL programs
only. Domain-specific verifier callbacks, the quality/authenticity of host
oracles, and independent review remain external responsibilities.

## Acceptance coverage

Tests cover exact failed-step localization, blocked composition, rejected domain
evidence, search-budget/counterexample-limit diagnosis, a deliberately underfit
program repaired through CEGIS, independent replay over original and fresh
examples, refusal when no failure is evidenced, probe-leakage rejection before
oracle invocation, and interface mismatch rejection.

Phase 7 acceptance is bounded to these deterministic contracts. It is not
evidence of unrestricted self-modifying software or autonomous open-world
scientific discovery.
