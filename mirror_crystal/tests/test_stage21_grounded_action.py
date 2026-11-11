from __future__ import annotations

import base64
import dataclasses
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mirror_crystal.container_sandbox import find_container_runtime
from mirror_crystal.stage20_multimodal_grounding import (
    GroundingStatus, RealWorldGroundingEngine, SourceTrustRegistry,
)
from mirror_crystal.stage21_grounded_action import (
    ActionExecutionStatus, ActionRisk, ActionSandboxResult, ActionCheckDecision,
    ContainerActionExecutor, GroundedActionDefinition, GroundedActionEngine,
    GroundedActionError, GroundedActionProposal, GroundedActionRegistry,
)


ACTION_SOURCE = '''
def run(inputs):
    return {"status": "prepared", "target": inputs["temperature_k"]}
'''

SCHEMA = {
    "type": "object",
    "properties": {"temperature_k": {"type": "number", "minimum": 0, "maximum": 500}},
    "required": ["temperature_k"],
    "additionalProperties": False,
}


def _signature(private, payload: bytes) -> str:
    return base64.b64encode(private.sign(payload)).decode("ascii")


def grounded_report(*, unsigned_document: bool = False):
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
    kwargs = {} if unsigned_document else {
        "signer_id": "document-owner",
        "signature_b64": _signature(private, raw),
    }
    engine.ingest_document("signed-report", "sensor.csv", raw, **kwargs)
    sample = {
        "sample_id": "sample-001",
        "entity_key": "sensor-a",
        "attribute": "temperature",
        "value": 23.0,
        "unit": "C",
        "observed_at": "2026-11-11T10:00:00Z",
    }
    signature = _signature(private, engine.telemetry_signing_bytes(sample))
    engine.ingest_telemetry(
        "telemetry-001", sample, device_id="device-a", signature_b64=signature,
    )
    report = engine.analyze(absolute_tolerance=0.02)
    selected = tuple(
        item.observation_id for item in report.observations
        if item.attribute == "temperature"
    )
    return report, selected


def definition(risk: ActionRisk = ActionRisk.READ_ONLY, source: str = ACTION_SOURCE):
    return GroundedActionDefinition(
        action_id="prepare_safe_state",
        description="Compute a bounded safe-state proposal from verified evidence.",
        source=source,
        input_schema=SCHEMA,
        risk=risk,
    )


def proposal_for(action: GroundedActionDefinition, report, observation_ids, *, author="planner-a"):
    return GroundedActionProposal.create(
        action,
        inputs={"temperature_k": 296.15},
        input_evidence_bindings={"temperature_k": observation_ids},
        evidence_report_fingerprint=report.report_fingerprint,
        evidence_observation_ids=observation_ids,
        expected_postconditions={"status": "prepared", "target": 296.15},
        created_by=author,
        created_at="2026-11-11T11:00:00Z",
    )


class FakeExecutor:
    def __init__(self, outcome=None, error=None):
        self.outcome = outcome if outcome is not None else {"status": "prepared", "target": 296.15}
        self.error = error
        self.calls = []

    def execute(self, action_definition, inputs):
        self.calls.append((action_definition.fingerprint, dict(inputs)))
        if self.error:
            raise RuntimeError(self.error)
        return ActionSandboxResult(self.outcome, "docker", 1, 0)


def make_engine(action=None, executor=None):
    registry = GroundedActionRegistry()
    registry.register(action or definition())
    engine = GroundedActionEngine(
        registry, executor or FakeExecutor(), allow_test_executor=True,
    )
    return engine, registry


def test_stage21_definition_fingerprints_source_schema_risk_and_limits():
    action = definition()
    assert len(action.fingerprint) == 64
    with pytest.raises(GroundedActionError, match="run"):
        definition(source="def wrong(inputs):\n    return {}")
    with pytest.raises(GroundedActionError, match="pinned"):
        GroundedActionDefinition(
            "prepare_safe_state", "unsafe mutable image", ACTION_SOURCE, SCHEMA,
            ActionRisk.READ_ONLY, image="python:latest",
        )


def test_stage21_action_schema_rejects_missing_extra_and_out_of_range_inputs():
    action = definition()
    report, ids = grounded_report()
    with pytest.raises(GroundedActionError, match="missing required"):
        GroundedActionProposal.create(
            action, inputs={}, evidence_report_fingerprint=report.report_fingerprint,
            evidence_observation_ids=ids, input_evidence_bindings={}, expected_postconditions={"status": "prepared"},
            created_by="planner-a",
        )
    with pytest.raises(GroundedActionError, match="unexpected field"):
        GroundedActionProposal.create(
            action, inputs={"temperature_k": 23, "shell": "forbidden"},
            evidence_report_fingerprint=report.report_fingerprint,
            evidence_observation_ids=ids, input_evidence_bindings={}, expected_postconditions={"status": "prepared"},
            created_by="planner-a",
        )
    with pytest.raises(GroundedActionError, match="maximum"):
        GroundedActionProposal.create(
            action, inputs={"temperature_k": 501},
            evidence_report_fingerprint=report.report_fingerprint,
            evidence_observation_ids=ids, input_evidence_bindings={}, expected_postconditions={"status": "prepared"},
            created_by="planner-a",
        )


def test_stage21_proposal_is_frozen_and_bound_to_exact_evidence_action_and_postconditions():
    action = definition()
    report, ids = grounded_report()
    item = proposal_for(action, report, ids)
    assert item.definition_fingerprint == action.fingerprint
    assert item.evidence_report_fingerprint == report.report_fingerprint
    assert len(item.proposal_id) == 64
    with pytest.raises(TypeError):
        item.inputs["temperature_k"] = 99
    with pytest.raises(TypeError):
        item.expected_postconditions["status"] = "changed"


def test_stage21_requires_two_distinct_independent_checks_before_execution():
    report, ids = grounded_report()
    action = definition()
    engine, _ = make_engine(action)
    proposal = proposal_for(action, report, ids)
    verified = engine.verify(proposal, report)
    assert verified.passed
    assert {item.check_id for item in verified.checks} == {"evidence_provenance", "action_contract"}
    assert {item.verifier_id for item in verified.checks} == {
        "mirror.evidence_provenance", "mirror.action_contract",
    }
    assert all(item.decision is ActionCheckDecision.PASS for item in verified.checks)


def test_stage21_rejects_action_input_that_does_not_match_its_cited_evidence():
    report, ids = grounded_report()
    action = definition()
    engine, _ = make_engine(action)
    item = GroundedActionProposal.create(
        action,
        inputs={"temperature_k": 300.0},
        input_evidence_bindings={"temperature_k": ids},
        evidence_report_fingerprint=report.report_fingerprint,
        evidence_observation_ids=ids,
        expected_postconditions={"status": "prepared"},
        created_by="planner-a",
        created_at="2026-11-11T11:00:00Z",
    )
    verified = engine.verify(item, report)
    assert not verified.passed
    assert "does not exactly match" in verified.checks[0].detail


def test_stage21_unsigned_source_cannot_drive_grounded_action():
    report, ids = grounded_report(unsigned_document=True)
    assert any(item.status is GroundingStatus.CORROBORATED_UNVERIFIED for item in report.conclusions)
    action = definition()
    engine, _ = make_engine(action)
    verified = engine.verify(proposal_for(action, report, ids), report)
    assert not verified.passed
    assert verified.checks[0].decision is ActionCheckDecision.FAIL
    assert "origin is not authenticated" in verified.checks[0].detail


def test_stage21_rejects_report_contents_that_do_not_match_its_fingerprint():
    report, ids = grounded_report()
    altered = dataclasses.replace(report, unknowns=("tampered evidence ledger",))
    action = definition()
    engine, _ = make_engine(action)
    verified = engine.verify(proposal_for(action, report, ids), altered)
    assert not verified.passed
    assert "fingerprint does not match" in verified.checks[0].detail


def test_stage21_requires_selected_observations_from_two_modalities_and_sources():
    report, ids = grounded_report()
    action = definition()
    engine, _ = make_engine(action)
    only_one = proposal_for(action, report, ids[:1])
    verified = engine.verify(only_one, report)
    assert not verified.passed
    assert "two independent modalities" in verified.checks[0].detail


def test_stage21_unknown_or_misbound_action_definition_fails_independent_contract_check():
    report, ids = grounded_report()
    action = definition()
    registry = GroundedActionRegistry()
    engine = GroundedActionEngine(registry, FakeExecutor(), allow_test_executor=True)
    verified = engine.verify(proposal_for(action, report, ids), report)
    assert not verified.passed
    assert "not registered" in verified.checks[1].detail


def test_stage21_rejects_action_definition_replacement_under_same_id():
    registry = GroundedActionRegistry()
    registry.register(definition())
    replacement = definition(source='def run(inputs):\n    return {"status": "changed"}')
    with pytest.raises(GroundedActionError, match="different content"):
        registry.register(replacement)


def test_stage21_execution_requires_engine_recorded_passing_verification():
    report, ids = grounded_report()
    action = definition()
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids)
    verified = engine.verify(item, report)
    other_engine, _ = make_engine(action)
    with pytest.raises(GroundedActionError, match="this engine's passing"):
        other_engine.execute(item, verified)


def test_stage21_successful_execution_compares_postconditions_and_records_feedback():
    report, ids = grounded_report()
    fake = FakeExecutor()
    engine, _ = make_engine(executor=fake)
    item = proposal_for(definition(), report, ids)
    verified = engine.verify(item, report)
    result = engine.execute(item, verified)
    assert result.status is ActionExecutionStatus.SUCCEEDED
    assert result.matched_conditions == ("status", "target")
    assert not result.failed_conditions
    assert result.observed_outcome["target"] == 296.15
    assert result.feedback.status is ActionExecutionStatus.SUCCEEDED
    assert engine.get_feedback(item.proposal_id) == result.feedback
    assert len(engine.history) == 1
    assert len(fake.calls) == 1


def test_stage21_postcondition_mismatch_is_reported_not_silently_accepted():
    report, ids = grounded_report()
    fake = FakeExecutor({"status": "blocked", "target": 296.15})
    engine, _ = make_engine(executor=fake)
    item = proposal_for(definition(), report, ids)
    verified = engine.verify(item, report)
    result = engine.execute(item, verified)
    assert result.status is ActionExecutionStatus.POSTCONDITION_MISMATCH
    assert result.failed_conditions == ("status",)
    assert result.feedback.failed_conditions == ("status",)
    assert "trusted knowledge was not modified" in result.feedback.explanation


def test_stage21_sandbox_failure_is_recorded_and_fails_closed():
    report, ids = grounded_report()
    engine, _ = make_engine(executor=FakeExecutor(error="container missing"))
    item = proposal_for(definition(), report, ids)
    verified = engine.verify(item, report)
    result = engine.execute(item, verified)
    assert result.status is ActionExecutionStatus.FAILED
    assert "container missing" in result.error
    assert result.failed_conditions == ("status", "target")
    assert result.feedback.status is ActionExecutionStatus.FAILED


def test_stage21_proposal_is_single_use_and_cannot_execute_twice():
    report, ids = grounded_report()
    engine, _ = make_engine()
    item = proposal_for(definition(), report, ids)
    verified = engine.verify(item, report)
    engine.execute(item, verified)
    with pytest.raises(GroundedActionError, match="single-use"):
        engine.execute(item, verified)


def test_stage21_consequential_action_requires_human_approval_before_execution():
    report, ids = grounded_report()
    action = definition(ActionRisk.CONSEQUENTIAL)
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids)
    verified = engine.verify(item, report)
    assert verified.passed
    with pytest.raises(GroundedActionError, match="prior independent human approval"):
        engine.execute(item, verified)


def test_stage21_consequential_action_author_cannot_approve_own_proposal():
    report, ids = grounded_report()
    action = definition(ActionRisk.CONSEQUENTIAL)
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids, author="operator-a")
    verified = engine.verify(item, report)
    with pytest.raises(GroundedActionError, match="cannot approve"):
        engine.approve(item, verified, reviewer_id="operator-a", reason="self approval")


def test_stage21_self_approval_cannot_be_bypassed_with_whitespace():
    report, ids = grounded_report()
    action = definition(ActionRisk.CONSEQUENTIAL)
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids, author="operator-a")
    verified = engine.verify(item, report)
    with pytest.raises(GroundedActionError, match="cannot approve"):
        engine.approve(item, verified, reviewer_id=" operator-a ", reason="whitespace bypass")


def test_stage21_sandbox_result_requires_strict_integer_pid_and_exit_code():
    with pytest.raises(GroundedActionError, match="integer container PID 1"):
        ActionSandboxResult({"ok": True}, "docker", 1.0, 0)
    with pytest.raises(GroundedActionError, match="integer zero exit code"):
        ActionSandboxResult({"ok": True}, "docker", 1, 0.0)


def test_stage21_consequential_action_requires_engine_recorded_approval_token():
    report, ids = grounded_report()
    action = definition(ActionRisk.CONSEQUENTIAL)
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids)
    verified = engine.verify(item, report)
    approval = engine.approve(
        item, verified, reviewer_id="reviewer-b",
        reason="Reviewed the evidence and exact bounded action proposal.",
    )
    result = engine.execute(item, verified, approval=approval)
    assert result.status is ActionExecutionStatus.SUCCEEDED
    assert approval.proposal_id == item.proposal_id
    assert approval.verification_fingerprint == verified.fingerprint


def test_stage21_human_approval_is_rejected_for_non_consequential_actions():
    report, ids = grounded_report()
    action = definition(ActionRisk.READ_ONLY)
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids)
    verified = engine.verify(item, report)
    with pytest.raises(GroundedActionError, match="reserved for consequential"):
        engine.approve(item, verified, reviewer_id="reviewer-b", reason="unnecessary")


def test_stage21_human_approval_is_single_use_per_proposal():
    report, ids = grounded_report()
    action = definition(ActionRisk.CONSEQUENTIAL)
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids)
    verified = engine.verify(item, report)
    engine.approve(item, verified, reviewer_id="reviewer-b", reason="approved once")
    with pytest.raises(GroundedActionError, match="already has"):
        engine.approve(item, verified, reviewer_id="reviewer-c", reason="second approval")


def test_stage21_failed_verification_cannot_be_approved_or_executed():
    report, ids = grounded_report(unsigned_document=True)
    action = definition(ActionRisk.CONSEQUENTIAL)
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids)
    verified = engine.verify(item, report)
    with pytest.raises(GroundedActionError, match="passing verification"):
        engine.approve(item, verified, reviewer_id="reviewer-b", reason="bypass")


def test_stage21_execution_history_is_append_only_and_feedback_is_report_bound():
    report, ids = grounded_report()
    engine, _ = make_engine()
    item = proposal_for(definition(), report, ids)
    verified = engine.verify(item, report)
    first = engine.execute(item, verified)
    history = engine.history
    assert isinstance(history, tuple)
    assert history[0].fingerprint == first.fingerprint
    with pytest.raises(AttributeError):
        history.append(first)


def test_stage21_real_container_execution_uses_stage19_supervisor_and_pid1():
    if find_container_runtime() is None:
        if __import__("os").environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") == "1":
            pytest.fail("CI requires Docker/Podman for Stage 21 action execution")
        pytest.skip("Docker/Podman is not installed")
    report, ids = grounded_report()
    action = definition()
    engine = GroundedActionEngine(GroundedActionRegistry(), ContainerActionExecutor())
    engine.registry.register(action)
    item = proposal_for(action, report, ids)
    verified = engine.verify(item, report)
    result = engine.execute(item, verified)
    assert result.status is ActionExecutionStatus.SUCCEEDED, result.error
    assert result.runtime in {"docker", "podman"}
    assert result.pid == 1


def test_stage21_requires_supervisor_pass_even_when_candidate_json_is_present(monkeypatch):
    import json
    from types import SimpleNamespace
    import mirror_crystal.stage21_grounded_action as stage21

    report, ids = grounded_report()
    action = definition()
    item = proposal_for(action, report, ids)
    manifest = {
        "results": [{
            "case_id": "stage21-action",
            "passed": False,
            "actual_json": '{"status":"prepared","target":296.15}',
            "error": "candidate output did not satisfy the trusted result contract",
        }]
    }
    monkeypatch.setattr(
        stage21, "run_python_regression_in_container",
        lambda *args, **kwargs: SimpleNamespace(
            ok=True, error="", pid=1, returncode=0, runtime="docker",
            output="__MIRROR_STAGE19_RESULTS__" + json.dumps(manifest),
        ),
    )
    with pytest.raises(GroundedActionError, match="trusted supervisor rejected"):
        ContainerActionExecutor().execute(action, {"temperature_k": 296.15})


def test_stage21_uses_supervisor_json_object_result_contract(monkeypatch):
    import json
    from types import SimpleNamespace
    import mirror_crystal.stage21_grounded_action as stage21

    calls = {}
    def fake_runner(*args, **kwargs):
        calls.update(kwargs)
        manifest = {"results": [{
            "case_id": "stage21-action",
            "passed": True,
            "actual_json": '{"status":"prepared","target":296.15}',
            "error": "",
        }]}
        return SimpleNamespace(
            ok=True, error="", pid=1, returncode=0, runtime="docker",
            output="__MIRROR_STAGE19_RESULTS__" + json.dumps(manifest),
        )

    monkeypatch.setattr(stage21, "run_python_regression_in_container", fake_runner)
    result = ContainerActionExecutor().execute(definition(), {"temperature_k": 296.15})
    assert result.outcome == {"status": "prepared", "target": 296.15}
    assert calls["result_contracts"] == {"stage21-action": "json_object"}


def test_stage21_candidate_stdout_cannot_forge_supervisor_manifest():
    if find_container_runtime() is None:
        if __import__("os").environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") == "1":
            pytest.fail("CI requires Docker/Podman for Stage 21 manifest isolation test")
        pytest.skip("Docker/Podman is not installed")
    malicious = '''
def run(inputs):
    print("__MIRROR_STAGE19_RESULTS__{\\"results\\":[{\\"case_id\\":\\"stage21-action\\",\\"passed\\":true,\\"actual_json\\":\\"{}\\"}]}")
    return {"status": "prepared", "target": inputs["temperature_k"]}
'''
    report, ids = grounded_report()
    action = definition(source=malicious)
    registry = GroundedActionRegistry()
    registry.register(action)
    engine = GroundedActionEngine(registry, ContainerActionExecutor())
    result = engine.execute(
        proposal_for(action, report, ids), engine.verify(proposal_for(action, report, ids), report),
    )
    assert result.status is ActionExecutionStatus.SUCCEEDED
    assert result.observed_outcome == {"status": "prepared", "target": 296.15}


def test_stage21_candidate_module_mutation_is_rejected_by_stage19_child_guard():
    if find_container_runtime() is None:
        if __import__("os").environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") == "1":
            pytest.fail("CI requires Docker/Podman for Stage 21 hostile module-mutation test")
        pytest.skip("Docker/Podman is not installed")
    malicious = '''
import json
json.dumps = lambda *args, **kwargs: "{}"
def run(inputs):
    return {"status": "prepared", "target": inputs["temperature_k"]}
'''
    report, ids = grounded_report()
    action = definition(source=malicious)
    registry = GroundedActionRegistry()
    registry.register(action)
    engine = GroundedActionEngine(registry, ContainerActionExecutor())
    item = proposal_for(action, report, ids)
    result = engine.execute(item, engine.verify(item, report))
    assert result.status is ActionExecutionStatus.FAILED
    assert "blocked private or mutating attribute" in result.error


def test_stage21_action_without_verified_evidence_is_not_executable():
    report, ids = grounded_report(unsigned_document=True)
    action = definition()
    engine, _ = make_engine(action)
    item = proposal_for(action, report, ids)
    verified = engine.verify(item, report)
    with pytest.raises(GroundedActionError, match="passing independent verification"):
        engine.execute(item, verified)
