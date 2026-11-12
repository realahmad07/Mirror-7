from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mirror_crystal.stage20_multimodal_grounding import RealWorldGroundingEngine, SourceTrustRegistry
from mirror_crystal.stage21_grounded_action import (
    ActionRisk, ActionSandboxResult, GroundedActionDefinition,
    GroundedActionEngine, GroundedActionProposal, GroundedActionRegistry,
)
from mirror_crystal.stage22_trusted_execution import (
    ActionAdapterRequest, AdapterPhase, AdapterReceipt, DurableAuditLedger,
    GatewayExecutionStatus, ReviewerAuthenticator, Stage22Error,
    TrustedActionAdapter, TrustedActionAdapterRegistry, TrustedActionGateway,
)


ACTION_SOURCE = '''
def run(inputs):
    return {"status": "prepared", "target": inputs["temperature_k"]}
'''
SCHEMA = {
    "type": "object",
    "properties": {
        "temperature_k": {"type": "number", "minimum": 0, "maximum": 500},
    },
    "required": ["temperature_k"],
    "additionalProperties": False,
}
IDENTITY_KEY = b"identity-provider-key-material-32-bytes-minimum"
AUDIT_KEY = b"durable-audit-key-material-32-bytes-minimum"
APPROVAL_KEY = b"gateway-approval-key-material-32-bytes-minimum"


def _signature(private_key: Ed25519PrivateKey, payload: bytes) -> str:
    return base64.b64encode(private_key.sign(payload)).decode("ascii")


def grounded_report():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    trust = SourceTrustRegistry()
    trust.register("document-owner", public)
    trust.register("device-a", public)
    engine = RealWorldGroundingEngine(trust)
    raw = (
        b"entity_key,attribute,value,unit,observed_at\n"
        b"sensor-a,temperature,23.0,C,2026-11-11T10:00:00Z\n"
    )
    engine.ingest_document(
        "signed-report", "sensor.csv", raw,
        signer_id="document-owner", signature_b64=_signature(private, raw),
    )
    sample = {
        "sample_id": "sample-001",
        "entity_key": "sensor-a",
        "attribute": "temperature",
        "value": 23.0,
        "unit": "C",
        "observed_at": "2026-11-11T10:00:00Z",
    }
    signature = _signature(private, engine.telemetry_signing_bytes(sample))
    engine.ingest_telemetry("telemetry-001", sample, device_id="device-a", signature_b64=signature)
    report = engine.analyze(absolute_tolerance=0.02)
    ids = tuple(item.observation_id for item in report.observations if item.attribute == "temperature")
    return report, ids


def action_definition(risk: ActionRisk = ActionRisk.REVERSIBLE):
    return GroundedActionDefinition(
        action_id="prepare_safe_state",
        description="Apply one approved, reversible safe-state transition through a host adapter.",
        source=ACTION_SOURCE,
        input_schema=SCHEMA,
        risk=risk,
    )


def make_proposal(action, report, ids, created_at):
    return GroundedActionProposal.create(
        action,
        inputs={"temperature_k": 296.15},
        input_evidence_bindings={"temperature_k": ids},
        evidence_report_fingerprint=report.report_fingerprint,
        evidence_observation_ids=ids,
        expected_postconditions={"status": "prepared", "target": 296.15},
        created_by="planner-a",
        created_at=created_at,
    )


class FakeSandboxExecutor:
    def execute(self, definition, inputs):
        return ActionSandboxResult(
            {"status": "prepared", "target": inputs["temperature_k"]},
            "docker", 1, 0,
        )


class FakeHostAdapter(TrustedActionAdapter):
    adapter_id = "plant.safe-state"

    def __init__(self, private_key, *, mismatch=False, fail_after_effect=False,
                 fail_rollback=False, tamper_observation=False):
        self.private_key = private_key
        self.mismatch = mismatch
        self.fail_after_effect = fail_after_effect
        self.fail_rollback = fail_rollback
        self.tamper_observation = tamper_observation
        self.state = {"status": "idle"}
        self.operation_id = "operation-001"
        self.calls = []

    def _receipt(self, request, phase, operation_id, ok, output):
        observed_at = datetime.now(timezone.utc).isoformat()
        payload = AdapterReceipt.signing_bytes(
            request.adapter_id, request.action_id, phase,
            request.idempotency_key, operation_id, ok, output, observed_at,
        )
        signature = _signature(self.private_key, payload)
        if self.tamper_observation and phase is AdapterPhase.OBSERVE:
            signature = _signature(self.private_key, b"tampered")
            self.tamper_observation = False
        return AdapterReceipt(
            request.adapter_id, request.action_id, phase,
            request.idempotency_key, operation_id, ok, output, observed_at, signature,
        )

    def dry_run(self, request: ActionAdapterRequest) -> AdapterReceipt:
        self.calls.append("dry_run")
        return self._receipt(request, AdapterPhase.DRY_RUN, "preview-001", True, {"preflight": "ready"})

    def execute(self, request: ActionAdapterRequest) -> AdapterReceipt:
        self.calls.append("execute")
        self.state = (
            {"status": "wrong", "target": 296.15}
            if self.mismatch else {"status": "prepared", "target": 296.15}
        )
        if self.fail_after_effect:
            self.fail_after_effect = False
            raise TimeoutError("response lost after external commit")
        return self._receipt(request, AdapterPhase.EXECUTE, self.operation_id, True, {"accepted": True})

    def observe(self, request: ActionAdapterRequest, operation_id: str | None) -> AdapterReceipt:
        self.calls.append("observe")
        return self._receipt(request, AdapterPhase.OBSERVE, operation_id or self.operation_id, True, dict(self.state))

    def compensate(self, request: ActionAdapterRequest, operation_id: str) -> AdapterReceipt:
        self.calls.append("compensate")
        if not self.fail_rollback:
            self.state = {"status": "idle"}
        return self._receipt(
            request, AdapterPhase.COMPENSATE, operation_id, True,
            {"compensation": "accepted"},
        )


def make_world(tmp_path, *, risk=ActionRisk.REVERSIBLE, rate_limit_count=10,
               verification_max_age_seconds=600, mismatch=False,
               fail_after_effect=False, fail_rollback=False, tamper_observation=False):
    now = [datetime.now(timezone.utc).timestamp()]
    report, ids = grounded_report()
    action = action_definition(risk)
    registry = GroundedActionRegistry()
    registry.register(action)
    stage21 = GroundedActionEngine(
        registry, FakeSandboxExecutor(), allow_test_executor=True,
    )
    adapter_private = Ed25519PrivateKey.generate()
    adapter_public = adapter_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    adapter = FakeHostAdapter(
        adapter_private, mismatch=mismatch, fail_after_effect=fail_after_effect,
        fail_rollback=fail_rollback, tamper_observation=tamper_observation,
    )
    adapters = TrustedActionAdapterRegistry()
    adapters.register(
        adapter,
        verification_public_key=adapter_public,
        action_ids=["prepare_safe_state"],
        recovery_postconditions={"status": "idle"},
    )
    auth = ReviewerAuthenticator({"corporate-idp": IDENTITY_KEY})
    ledger = DurableAuditLedger(tmp_path / "stage22-audit.sqlite", AUDIT_KEY)
    gateway = TrustedActionGateway(
        stage21, adapters, auth, ledger,
        approval_signing_key=APPROVAL_KEY,
        verification_max_age_seconds=verification_max_age_seconds,
        rate_limit_count=rate_limit_count,
        clock=lambda: now[0],
    )
    timestamp = datetime.fromtimestamp(now[0], timezone.utc).isoformat()
    proposal = make_proposal(action, report, ids, timestamp)
    verification = gateway.verify(proposal, report)
    assert verification.passed
    return {
        "now": now, "report": report, "ids": ids, "action": action,
        "adapter": adapter, "ledger": ledger, "auth": auth,
        "gateway": gateway, "proposal": proposal, "verification": verification,
        "tmp_path": tmp_path,
    }


def identity(world, subject, scopes, *, ttl=120):
    now = world["now"][0]
    return ReviewerAuthenticator.sign_assertion(
        "corporate-idp", subject, scopes, IDENTITY_KEY,
        audience="mirror-stage22",
        issued_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        expires_at=datetime.fromtimestamp(now + ttl, timezone.utc).isoformat(),
        nonce=f"nonce-{subject}-{int(now)}",
    )


def approval(world, subject="operator-a", scopes=None):
    assertion = identity(
        world, subject,
        scopes or ["action:approve:prepare_safe_state", "action:feedback-review:prepare_safe_state"],
    )
    return world["gateway"].issue_approval_token(
        world["proposal"], world["verification"],
        reviewer_assertion=assertion,
        reason="Approved after reviewing grounded evidence and the exact action contract.",
    )


def test_stage22_identity_assertions_are_signed_scoped_audience_and_expiry_bound(tmp_path):
    world = make_world(tmp_path)
    token = identity(world, "reviewer-1", ["action:approve:prepare_safe_state"])
    verified = world["auth"].authenticate(token, now=world["now"][0])
    assert verified.subject == "reviewer-1"
    assert "action:approve:prepare_safe_state" in verified.scopes

    with pytest.raises(Stage22Error, match="signature"):
        world["auth"].authenticate(token[:-1] + ("A" if token[-1] != "A" else "B"), now=world["now"][0])
    expired = identity(world, "reviewer-2", ["action:approve:prepare_safe_state"], ttl=10)
    with pytest.raises(Stage22Error, match="expired"):
        world["auth"].authenticate(expired, now=world["now"][0] + 11)


def test_stage22_identity_nonce_is_single_use_across_distinct_approval_proposals(tmp_path):
    world = make_world(tmp_path)
    assertion = identity(world, "reviewer-nonce", ["action:approve:prepare_safe_state"])
    world["gateway"].issue_approval_token(
        world["proposal"], world["verification"],
        reviewer_assertion=assertion, reason="Approve exact first proposal.",
    )

    world["now"][0] += 1
    timestamp = datetime.fromtimestamp(world["now"][0], timezone.utc).isoformat()
    proposal2 = make_proposal(world["action"], world["report"], world["ids"], timestamp)
    verification2 = world["gateway"].verify(proposal2, world["report"])
    assert verification2.passed
    with pytest.raises(Stage22Error, match="nonce has already been consumed"):
        world["gateway"].issue_approval_token(
            proposal2, verification2,
            reviewer_assertion=assertion, reason="Replay the first signed identity assertion.",
        )


def test_stage22_feedback_identity_nonce_is_single_use_across_operations(tmp_path):
    world = make_world(tmp_path)
    first_token = approval(world)
    first = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=first_token,
    )
    assert first.status is GatewayExecutionStatus.SUCCEEDED

    feedback_assertion = identity(
        world, "independent-reviewer", ["action:feedback-review:prepare_safe_state"],
    )
    world["gateway"].review_feedback(
        world["proposal"], world["verification"],
        reviewer_assertion=feedback_assertion,
        accepted=True, reason="Reviewed the signed external observation.",
    )

    world["now"][0] += 1
    timestamp = datetime.fromtimestamp(world["now"][0], timezone.utc).isoformat()
    proposal2 = make_proposal(world["action"], world["report"], world["ids"], timestamp)
    verification2 = world["gateway"].verify(proposal2, world["report"])
    approval_assertion = identity(world, "operator-b", ["action:approve:prepare_safe_state"])
    second_token = world["gateway"].issue_approval_token(
        proposal2, verification2,
        reviewer_assertion=approval_assertion,
        reason="Approve the second proposal independently.",
    )
    second = world["gateway"].execute(
        proposal2, verification2, approval_token=second_token,
    )
    assert second.status is GatewayExecutionStatus.SUCCEEDED

    with pytest.raises(Stage22Error, match="nonce has already been consumed"):
        world["gateway"].review_feedback(
            proposal2, verification2,
            reviewer_assertion=feedback_assertion,
            accepted=True, reason="Replay the first feedback identity assertion.",
        )


def test_stage22_rejects_untrusted_or_incomplete_identity_issuers(tmp_path):
    with pytest.raises(Stage22Error, match="safe"):
        ReviewerAuthenticator({"bad issuer": IDENTITY_KEY})
    world = make_world(tmp_path)
    token = identity(world, "reviewer-1", ["action:approve:prepare_safe_state"])
    other = ReviewerAuthenticator({"other-idp": b"another-idp-key-material-32-bytes-min"})
    with pytest.raises(Stage22Error, match="not host-trusted"):
        other.authenticate(token, now=world["now"][0])


def test_stage22_adapter_allowlist_is_immutable_and_requires_all_host_methods(tmp_path):
    class IncompleteAdapter(TrustedActionAdapter):
        adapter_id = "incomplete-adapter"

    adapters = TrustedActionAdapterRegistry()
    with pytest.raises(Stage22Error, match="implement host method"):
        adapters.register(
            IncompleteAdapter(), verification_public_key=b"x" * 32,
            action_ids=["prepare_safe_state"],
        )


def test_stage22_dry_run_is_signed_audited_and_does_not_execute_side_effects(tmp_path):
    world = make_world(tmp_path)
    result = world["gateway"].execute(
        world["proposal"], world["verification"], dry_run=True,
    )
    assert result.status is GatewayExecutionStatus.DRY_RUN
    assert world["adapter"].calls == ["dry_run"]
    assert any(event["event_type"] == "execution.dry_run" for event in world["ledger"].events)
    assert world["ledger"].verify_chain().valid


def test_stage22_real_execution_requires_an_approval_token(tmp_path):
    world = make_world(tmp_path)
    with pytest.raises(Stage22Error, match="requires a single-use"):
        world["gateway"].execute(world["proposal"], world["verification"])
    assert world["adapter"].calls == ["dry_run"]
    assert world["ledger"].operation(world["proposal"].proposal_id) is None


def test_stage22_approval_requires_exact_scope_and_rejects_self_approval(tmp_path):
    world = make_world(tmp_path)
    under_scoped = identity(world, "reviewer", ["action:execute:prepare_safe_state"])
    with pytest.raises(Stage22Error, match="lacks required scope"):
        world["gateway"].issue_approval_token(
            world["proposal"], world["verification"],
            reviewer_assertion=under_scoped, reason="No approval scope.",
        )
    self_assertion = identity(world, "planner-a", ["action:approve:prepare_safe_state"])
    with pytest.raises(Stage22Error, match="cannot approve"):
        world["gateway"].issue_approval_token(
            world["proposal"], world["verification"],
            reviewer_assertion=self_assertion, reason="Self approval is forbidden.",
        )


def test_stage22_short_lived_approval_token_is_bound_to_proposal_verification_and_adapter(tmp_path):
    world = make_world(tmp_path)
    token = approval(world)
    claims = world["gateway"]._verify_approval_token(
        token, world["proposal"], world["verification"], "plant.safe-state",
    )
    assert claims["proposal_id"] == world["proposal"].proposal_id
    assert claims["verification_fingerprint"] == world["verification"].fingerprint
    assert claims["adapter_id"] == "plant.safe-state"
    world["now"][0] += 301
    with pytest.raises(Stage22Error, match="expired"):
        world["gateway"]._verify_approval_token(
            token, world["proposal"], world["verification"], "plant.safe-state",
        )


def test_stage22_proposal_and_verification_expiry_are_enforced(tmp_path):
    world = make_world(tmp_path)
    world["now"][0] += 3601
    with pytest.raises(Stage22Error, match="proposal has expired"):
        approval(world)


def test_stage22_signed_adapter_observation_proves_real_outcome_and_persists_durable_ledger(tmp_path):
    world = make_world(tmp_path)
    token = approval(world)
    result = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert result.status is GatewayExecutionStatus.SUCCEEDED
    assert result.observed_output == {"status": "prepared", "target": 296.15}
    assert result.matched_conditions == ("status", "target")
    events = [event["event_type"] for event in world["ledger"].events]
    assert "approval.issued" in events
    assert "approval.consumed" in events
    assert "execution.started" in events
    assert "execution.verified" in events
    head = world["ledger"].head_hash
    assert world["ledger"].verify_chain(expected_head_hash=head).valid

    world["ledger"].close()
    reopened = DurableAuditLedger(tmp_path / "stage22-audit.sqlite", AUDIT_KEY)
    assert reopened.verify_chain(expected_head_hash=head).valid
    assert len(reopened.events) == len(events)
    reopened.close()


def test_stage22_approval_token_and_proposal_execution_are_single_use(tmp_path):
    world = make_world(tmp_path)
    token = approval(world)
    first = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert first.status is GatewayExecutionStatus.SUCCEEDED
    with pytest.raises(Stage22Error, match="consumed|reserved execution|replay"):
        world["gateway"].execute(
            world["proposal"], world["verification"], approval_token=token,
        )
    assert world["adapter"].calls.count("execute") == 1


def test_stage22_action_rate_limit_is_durable_across_proposals(tmp_path):
    world = make_world(tmp_path, rate_limit_count=1)
    token = approval(world)
    world["gateway"].execute(world["proposal"], world["verification"], approval_token=token)

    timestamp = datetime.fromtimestamp(world["now"][0] + 1, timezone.utc).isoformat()
    proposal2 = make_proposal(world["action"], world["report"], world["ids"], timestamp)
    verification2 = world["gateway"].verify(proposal2, world["report"])
    token2 = world["gateway"].issue_approval_token(
        proposal2, verification2,
        reviewer_assertion=identity(world, "operator-b", ["action:approve:prepare_safe_state"]),
        reason="Second approved operation for rate-limit test.",
    )
    with pytest.raises(Stage22Error, match="rate limit"):
        world["gateway"].execute(proposal2, verification2, approval_token=token2)


def test_stage22_external_postcondition_mismatch_triggers_compensation_and_verified_recovery(tmp_path):
    world = make_world(tmp_path, mismatch=True)
    token = approval(world)
    result = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert result.status is GatewayExecutionStatus.ROLLED_BACK
    assert result.rollback_status == "verified"
    assert result.observed_output == {"status": "idle"}
    assert world["adapter"].calls.count("compensate") == 1
    assert any(event["event_type"] == "execution.compensated" for event in world["ledger"].events)


def test_stage22_partial_compensation_requires_independent_recovery_observation(tmp_path):
    world = make_world(tmp_path, mismatch=True, fail_rollback=True)
    token = approval(world)
    result = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert result.status is GatewayExecutionStatus.RECOVERY_FAILED
    assert result.rollback_status == "failed"
    assert any(event["event_type"] == "execution.recovery_failed" for event in world["ledger"].events)


def test_stage22_unknown_outcome_is_reconciled_by_idempotency_key_without_reexecution(tmp_path):
    world = make_world(tmp_path, fail_after_effect=True)
    token = approval(world)
    result = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert result.status is GatewayExecutionStatus.UNKNOWN
    assert world["ledger"].operation(world["proposal"].proposal_id)["status"] == "unknown"

    # Reconciliation remains permitted after the new-execution freshness window,
    # because it resolves an already reserved/possibly committed external operation.
    world["now"][0] += 3601
    recovered = world["gateway"].recover(world["proposal"], world["verification"])
    assert recovered.status is GatewayExecutionStatus.SUCCEEDED
    assert recovered.observed_output == {"status": "prepared", "target": 296.15}
    assert world["adapter"].calls.count("execute") == 1
    assert any(event["event_type"] == "execution.recovered" for event in world["ledger"].events)


def test_stage22_tampered_adapter_telemetry_fails_closed_and_is_recoverable(tmp_path):
    world = make_world(tmp_path, tamper_observation=True)
    token = approval(world)
    result = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert result.status is GatewayExecutionStatus.UNKNOWN
    assert "recovery_required" in result.rollback_status
    recovered = world["gateway"].recover(world["proposal"], world["verification"])
    assert recovered.status is GatewayExecutionStatus.SUCCEEDED


def test_stage22_unauthenticated_or_misbound_adapter_receipt_cannot_be_accepted(tmp_path):
    world = make_world(tmp_path, tamper_observation=True)
    token = approval(world)
    result = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert result.status is GatewayExecutionStatus.UNKNOWN
    assert "signature verification failed" in result.error


def test_stage22_feedback_review_requires_an_independent_scoped_reviewer(tmp_path):
    world = make_world(tmp_path)
    token = approval(world, subject="operator-a")
    result = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert result.status is GatewayExecutionStatus.SUCCEEDED
    # Feedback can be independently reviewed after the proposal's execution TTL.
    world["now"][0] += 3601

    with pytest.raises(Stage22Error, match="independent"):
        world["gateway"].review_feedback(
            world["proposal"], world["verification"],
            reviewer_assertion=identity(world, "operator-a", ["action:feedback-review:prepare_safe_state"]),
            accepted=True, reason="Approver cannot review its own outcome.",
        )
    reviewed = world["gateway"].review_feedback(
        world["proposal"], world["verification"],
        reviewer_assertion=identity(world, "independent-reviewer", ["action:feedback-review:prepare_safe_state"]),
        accepted=True, reason="Signed external observation matched the expected conditions.",
    )
    assert reviewed.accepted is True
    assert reviewed.trusted_knowledge_changed is False
    event = world["ledger"].events[-1]
    assert event["event_type"] == "feedback.reviewed"
    assert event["payload"]["trusted_knowledge_changed"] is False
    with pytest.raises(Stage22Error, match="already has"):
        world["gateway"].review_feedback(
            world["proposal"], world["verification"],
            reviewer_assertion=identity(world, "second-reviewer", ["action:feedback-review:prepare_safe_state"]),
            accepted=False, reason="Duplicate decision must be rejected.",
        )


def test_stage22_feedback_reviewer_must_be_distinct_from_proposer(tmp_path):
    world = make_world(tmp_path)
    token = approval(world)
    world["gateway"].execute(world["proposal"], world["verification"], approval_token=token)
    with pytest.raises(Stage22Error, match="independent"):
        world["gateway"].review_feedback(
            world["proposal"], world["verification"],
            reviewer_assertion=identity(world, "planner-a", ["action:feedback-review:prepare_safe_state"]),
            accepted=True, reason="Proposal author cannot certify their own outcome.",
        )


def test_stage22_audit_events_are_immutable_and_hash_chain_detects_tampering(tmp_path):
    world = make_world(tmp_path)
    world["ledger"].append("test.event", "aggregate-1", {"value": 1})
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        world["ledger"]._conn.execute("UPDATE ledger_events SET payload_json='{}' WHERE sequence=1")

    # Simulate a privileged database-file editor removing the SQLite trigger.
    world["ledger"]._conn.execute("DROP TRIGGER ledger_events_no_update")
    world["ledger"]._conn.execute("UPDATE ledger_events SET payload_json='{}' WHERE sequence=1")
    verified = world["ledger"].verify_chain()
    assert verified.valid is False
    assert "hash mismatch" in verified.error


def test_stage22_audit_chain_detects_missing_or_changed_external_anchor(tmp_path):
    world = make_world(tmp_path)
    world["ledger"].append("event.one", "one", {"ok": True})
    world["ledger"].append("event.two", "two", {"ok": True})
    correct = world["ledger"].head_hash
    assert world["ledger"].verify_chain(expected_head_hash=correct).valid
    assert not world["ledger"].verify_chain(expected_head_hash="0" * 64).valid


def test_stage22_read_only_action_does_not_claim_rollback_capability(tmp_path):
    world = make_world(tmp_path, risk=ActionRisk.READ_ONLY)
    token = approval(world)
    world["adapter"].mismatch = True
    result = world["gateway"].execute(
        world["proposal"], world["verification"], approval_token=token,
    )
    assert result.status is GatewayExecutionStatus.POSTCONDITION_MISMATCH
    assert result.rollback_status == "not_supported"
    assert world["adapter"].calls.count("compensate") == 0

def test_stage22_audit_ledger_creates_missing_parent_directories(tmp_path):
    nested_path = tmp_path / "audit" / "nested" / "events.sqlite"
    ledger = DurableAuditLedger(nested_path, AUDIT_KEY)
    ledger.append("ledger.initialized", "deployment-1", {"created": True})
    assert nested_path.is_file()
    assert ledger.verify_chain().valid
    ledger.close()

