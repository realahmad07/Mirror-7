# Stage 22 — Trusted Real-World Execution & Operational Governance

## Goal

Stage 21 proves that a bounded action proposal is grounded in authenticated Stage 20 evidence, passes two independent policy/provenance checkers, and has explicit predicted postconditions. Stage 22 adds a host-controlled bridge from that verified proposal to a narrowly allow-listed external operation.

This is a gateway framework, not a general-purpose autonomous actuator. Stage 22 does not expose arbitrary URLs, shell commands, dynamic imports, raw credentials, host filesystem mounts or a generic network client to action proposals. A host must deliberately register a reviewed adapter implementation for each permitted action.

Implementation: `mirror_crystal/stage22_trusted_execution.py`  
Tests: `mirror_crystal/tests/test_stage22_trusted_execution.py`  
Stage 21 dependency: `mirror_crystal/stage21_grounded_action.py`

## 1. Trusted action gateway and adapter allow-list

`TrustedActionAdapterRegistry` maps a Stage 21 `action_id` to one immutable `TrustedActionAdapter` implementation, adapter ID, action allow-list, Ed25519 result-verification public key and (for reversible/consequential operations) required recovery postconditions.

An adapter must implement four separate host methods:

- `dry_run(request)`: validate the exact operation without applying a side effect.
- `execute(request)`: submit the exact approved operation using its idempotency key.
- `observe(request, operation_id)`: independently read the external state after an operation; on recovery, an adapter may resolve an unknown operation by idempotency key.
- `compensate(request, operation_id)`: attempt a compensating action for an identified partial or mismatched operation.

An adapter is registered by the trusted host. Duplicate adapter registrations and overlapping action IDs are rejected. No adapter is loaded from a proposal, user-controlled path, URL or package name. The adapter public key authenticates every returned receipt, binding adapter ID, action ID, phase, idempotency key, operation ID, success status, output and observation time.

The gateway always reuses the exact Stage 21 proposal and passing verification fingerprint. It checks that the proposal's action definition still matches the immutable Stage 21 registry entry and is mapped to the expected host adapter.

## 2. Authenticated identity and scoped approval

`ReviewerAuthenticator` trusts an explicit map of identity-provider issuer IDs to HMAC keys. The issuer signs a short-lived assertion containing:

- issuer and reviewer subject;
- a unique nonce and audience;
- an explicit list of action scopes;
- issued-at and expiry timestamps.

The gateway requires the exact `action:approve:<action_id>` scope. The proposer cannot approve their own proposal. Identity assertions are verified for signature, issuer, audience, scope syntax, issue time, expiry and a 15-minute maximum lifetime.

After successful authentication, the gateway issues a separate HMAC-signed approval token bound to the exact proposal fingerprint, Stage 21 verification fingerprint, adapter ID, action ID, reviewer subject, action scopes and expiry. The default token lifetime is five minutes and the configured maximum is ten minutes. Approval tokens are stored by a unique token ID and proposal ID; they can be consumed once only. A token for another proposal, checker result, adapter, action, reviewer or audience is rejected.

The HMAC keys for the identity provider and gateway are separate secrets. They belong in a host secret manager, not in source code, the audit database or action input.

## 3. Durable, tamper-evident audit ledger

`DurableAuditLedger` stores events in SQLite with WAL mode and `synchronous=FULL`. Each append-only event binds:

- monotonically increasing sequence number;
- event type, aggregate ID and UTC timestamp;
- canonical JSON payload;
- previous event hash;
- SHA-256 event hash;
- HMAC-SHA-256 authentication code.

SQLite triggers reject normal event-row updates and deletes. Verification recomputes sequence, links, hashes, MACs and JSON payload validity. The HMAC key is supplied separately from the database. A host should pin the ledger head hash in a separate trusted system; without an external head anchor, a restored older but internally valid database cannot be distinguished from the current history.

The ledger also keeps small transactional projections for approvals, one-use operation reservations and independent feedback decisions. Those projections support concurrency/replay controls; the authoritative historical record is the append-only event chain.

Important event classes include proposal verification/rejection, approval issuance/consumption, dry-run, execution start, adapter acceptance, observed success, mismatch, unknown outcome, compensation/recovery, and independent feedback review.

## 4. Idempotency, expiry, rate limits and dry-run

Before any side effect, the gateway:

1. checks that the proposal and Stage 21 verification are within their configured age limits;
2. verifies a successful adapter dry-run receipt;
3. verifies the signed approval token and its action/adapter/proposal/verification scope;
4. transactionally consumes the approval token, checks the per-adapter/action rate window, and reserves one operation and one idempotency key;
5. only then calls the registered adapter's `execute` method.

The default idempotency key is a SHA-256 digest of the exact proposal ID. A caller may supply a safe bounded key. The database enforces unique keys and one reserved execution per proposal. Concurrent attempts serialize through `BEGIN IMMEDIATE`; replays fail closed.

Dry-run can be requested without a human approval because its contract is non-mutating. It still uses the signed adapter receipt and is durably audited. A host adapter must keep its dry-run method genuinely side-effect-free.

## 5. External observation, rollback and recovery

The return value from `execute` is not by itself treated as proof that an external operation succeeded. The gateway requires a signed execution receipt, then calls `observe` to obtain a separately signed external-state observation. It compares that observation against every Stage 21 expected postcondition.

If expected postconditions do not match:

- a read-only action is recorded as a mismatch; no rollback is claimed;
- reversible/consequential actions may call the registered adapter's compensating method;
- compensation must return a valid signed receipt;
- the gateway then performs another signed observation;
- rollback is considered successful only when the post-compensation observation satisfies all registered recovery postconditions.

If the adapter times out or an observation cannot be authenticated, the operation is recorded as `unknown` or `recovery_failed`. The gateway does not blindly repeat the external mutation. `recover(proposal, verification)` asks the same trusted adapter to reconcile state by idempotency key/known operation ID, then either confirms the predicted state or runs the bounded compensating/recovery sequence.

Adapters must implement their own service/device-level idempotency and compensation semantics. The gateway cannot guarantee that a remote service honors an idempotency key unless that adapter and service contract actually do so.

## 6. Independently reviewed feedback

A separate reviewer with `action:feedback-review:<action_id>` scope records an accept/reject decision for a finished or reconciled operation. The reviewer must be different from both the proposal author and the execution approver. The decision and reason are recorded in the durable ledger.

Feedback acceptance does not mutate Stage 20 observations, the knowledge registry, capability registry, action definitions or trusted facts. It is only a review record. A future knowledge-admission process must independently validate any proposed learning from the outcome.

## Public API

```python
from mirror_crystal import (
    DurableAuditLedger,
    ReviewerAuthenticator,
    TrustedActionAdapterRegistry,
    TrustedActionGateway,
)

# Host supplies its trusted identity-provider keys and an adapter registry.
authenticator = ReviewerAuthenticator({"corporate-idp": identity_provider_hmac_key})
adapters = TrustedActionAdapterRegistry()
adapters.register(
    reviewed_adapter,
    verification_public_key=adapter_ed25519_public_key,
    action_ids=["prepare_safe_state"],
    recovery_postconditions={"status": "idle"},
)
ledger = DurableAuditLedger("/var/lib/mirror/stage22-audit.sqlite", audit_hmac_key)

gateway = TrustedActionGateway(
    stage21_engine,
    adapters,
    authenticator,
    ledger,
    approval_signing_key=gateway_approval_hmac_key,
)

verification = gateway.verify(stage21_proposal, stage20_grounding_report)
approval_token = gateway.issue_approval_token(
    stage21_proposal,
    verification,
    reviewer_assertion=signed_reviewer_assertion,
    reason="Reviewed exact proposal, evidence and predicted postconditions.",
)
result = gateway.execute(
    stage21_proposal,
    verification,
    approval_token=approval_token,
)
assert gateway.verify_audit_ledger().valid
```

The code above deliberately leaves adapter code, all secret material, identity assertions and the Stage 21 proposal/report as host-supplied objects. No production keys or device/service adapters are embedded in this repository.

## Acceptance gates

The focused suite must cover identity issuer/signature/audience/scope/expiry, self-approval, token binding and expiry, adapter allow-list replacement, signed receipt validation, dry-run isolation, one-use approval and execution, idempotency, rate limiting, stale proposals, signed external observation, audit mutation/hash/HMAC/anchor checks, partial failure, compensation, post-rollback observation, unknown-outcome reconciliation, and independent feedback review.

Repository acceptance additionally requires Fast CI, full historical Deep Regression and Security Guard to pass on the exact final PR head. A passing unit suite alone does not authorize connection to a real production service.

## Explicit limitations and deployment responsibilities

- The host must code-review each adapter and limit its operations to a specific external system/service/device and action allow-list.
- Identity-provider, adapter-signing, gateway-approval and audit HMAC keys must be independently generated, stored and rotated by the host.
- The audit database path, file permissions, backups, retention, monitoring and external head-hash anchor are host deployment responsibilities.
- The adapter must enforce its own service-level authorization, idempotency and compensation semantics. A valid Mirror approval is not a substitute for the external service's access control.
- Dry-run safety is part of the adapter's trusted contract and must be tested by the adapter owner.
- A remote timeout can leave an external result unknown; Stage 22 records this uncertainty and offers reconciliation, but cannot promise rollback if the external service is unavailable or not compensatable.
- No default production adapter, service credential, device driver, arbitrary HTTP endpoint, shell bridge or general host-control interface is included.
- The ledger's HMAC chain detects alteration while its key remains secret. Detecting database rollback/truncation also requires an independently stored expected head hash.
- Review of feedback does not automatically admit external outcomes into trusted knowledge.

Stage 22 is therefore bounded PASS only after the focused and repository-wide acceptance gates succeed. It establishes a controlled external-execution boundary, not unrestricted autonomy or a claim of safe operation in every real-world environment.
