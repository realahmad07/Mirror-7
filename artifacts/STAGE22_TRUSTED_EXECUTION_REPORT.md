# Stage 22 — Trusted Real-World Execution & Operational Governance Acceptance Report

**Status:** Implementation committed on the Stage 22 branch; final acceptance pending CI, Deep Regression and Security Guard.  
**Implementation:** `mirror_crystal/stage22_trusted_execution.py`  
**Tests:** `mirror_crystal/tests/test_stage22_trusted_execution.py`  
**Focused test cases:** 21

## Delivered

- Explicit, immutable host adapter registry mapping Stage 21 action IDs to reviewed adapter implementations and Ed25519 verification keys.
- Four distinct adapter operations: signed dry-run, execute, external observe and compensate.
- Host-issued HMAC-signed reviewer assertions with trusted issuer, audience, subject, exact scopes, nonce, issued-at and expiry validation.
- Short-lived HMAC-signed approval tokens bound to the exact Stage 21 proposal, verification, action and adapter; one-use transactional consumption; proposer self-approval rejection.
- SQLite WAL event ledger with FULL synchronous writes, immutable event triggers, SHA-256 previous-hash chain, HMAC authentication and optional externally pinned head verification.
- Transactional operation reservation, one proposal/one execution, unique idempotency keys, replay rejection and per-adapter/action rate limiting.
- Mandatory signed adapter preflight; no side effect in dry-run mode.
- Signed external observation compared against all declared Stage 21 postconditions.
- Compensating execution and independently signed post-rollback observation against registered recovery postconditions.
- UNKNOWN outcome tracking and reconciliation by idempotency key; no blind re-execution after ambiguous timeout.
- Independent feedback review with action-specific scope and reviewer separation; no automatic mutation of trusted evidence or knowledge.
- Package exports and Stage 22 README, STATUS, roadmap, documentation index and design/report pages.

## Focused verification coverage

The 21 focused test cases cover:

- Identity assertion signature, trusted issuer, scope, audience and expiry.
- Adapter allow-list immutability and required host methods.
- Dry-run isolation and audit recording; mandatory approval for real execution.
- Exact approval scopes, self-approval rejection, proposal/verification/adapter binding, token expiry and one-use replay blocking.
- Stale proposal rejection, transactional rate limits and durable idempotency.
- Signed adapter outcomes, durable ledger restart, event immutability, hash-chain tamper detection and external head-anchor mismatch.
- Postcondition mismatch, compensating action, positive recovery observation and failed recovery observation.
- Uncertain external outcome and recovery without re-execution.
- Invalid adapter signatures and independent feedback reviewer separation.
- Read-only mismatch is not mislabeled as rolled back.

## Security invariants

1. Stage 22 never accepts a new or modified Stage 21 action definition; the exact definition fingerprint remains in the Stage 21 host registry.
2. Only the host-controlled adapter registry can map an action ID to an adapter implementation.
3. A real operation requires an authenticated reviewer, action-specific scope, a token signed by the gateway, and exact proposal/verification/adapter binding.
4. Approval consumption, execution reservation, idempotency and rate-limit checks are serialized in SQLite before an adapter side effect.
5. External success is based on a signed observation, not solely on the adapter's execution acknowledgment.
6. Compensation is not called verified until a signed observation proves the registered recovery postconditions.
7. Unknown results are reconciled; the gateway does not issue an automatic second execute call.
8. Feedback review is independently authorized and has no implicit knowledge-promotion side effect.

## Explicit deployment boundaries

This repository does not include a production device/service adapter or any credentials. Adapters are trusted host code and require independent code review, limited permissions and their own service-level authorization. Identity, adapter, approval and ledger keys must be stored outside the repository/database. An external ledger head anchor is required for rollback/truncation detection. A remote service may remain unavailable or non-compensatable; in that case the operation remains UNKNOWN or RECOVERY_FAILED and must be handled by the host operator.

## Acceptance rule

Stage 22 is accepted only when:

1. All 21 focused tests pass.
2. The complete `mirror_crystal` / `mirror_eval` and backend suites pass in Fast CI.
3. Full historical `python -m pytest -q` Deep Regression passes.
4. Security Guard passes on the exact final PR head.
5. The PR is merged and `main` is confirmed at the resulting merge commit.

No unrestricted autonomous execution or universal real-world safety claim is made.
