# Phase 8 — Independent Outcome Verification

## Purpose

Phase 8 evaluates execution results using checker callbacks registered separately from operation handlers. It distinguishes "the operation returned" from "the requested result was independently checked."

## Independent checks

The host registers `IndependentCheck` objects with a checker identity, a check identifier, and the exact postconditions the check is authorized to verify. Registration requires the host to explicitly mark the checker independent. The verification request supplies the required check identifiers for each capability.

The engine binds every check to the task, plan, step, capability contract, and execution fingerprint. Missing or ambiguous checks, exceptions, non-boolean results, false results, and attempts to claim undeclared postconditions fail closed. All declared postconditions for each executed capability must be covered by passing checks. A failed or blocked execution cannot become verified.

The report records each check, its checker identity, pass/fail result, verified conditions, and the exact execution fingerprint. It does not mutate the underlying CapabilitySpec.

## Trust boundary

Independence is an explicit trust assignment made by the host/operator; a label alone cannot prove that two implementations are genuinely independent. Checkers should use a distinct method, implementation, or evidence source from the operation being checked.

## Boundary

Phase 8 produces verification evidence. It does not rewrite capability definitions, automatically mark a result true, or learn a new capability.
