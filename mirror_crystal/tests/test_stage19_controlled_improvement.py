from __future__ import annotations

import hashlib
import json
import os

import pytest

from mirror_crystal.container_sandbox import find_container_runtime
from mirror_crystal.stage19_controlled_improvement import (
    ApprovalDecision,
    CaseExecutionResult,
    ChangeKind,
    ContainerRegressionExecutor,
    ControlledSelfImprovementEngine,
    EvaluationStatus,
    ExecutionEvidence,
    FrozenRegressionSuite,
    ImprovementProposal,
    RegressionCase,
    RegressionCaseKind,
    SelfImprovementError,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


BASELINE_REVISION = "a" * 40
_FOUND_RUNTIME = find_container_runtime()
RUNTIME = _FOUND_RUNTIME[0] if _FOUND_RUNTIME is not None else "docker"
ENVIRONMENT = ContainerRegressionExecutor.environment_fingerprint(runtime=RUNTIME)


def suite() -> FrozenRegressionSuite:
    return FrozenRegressionSuite(
        suite_id="stage19-acceptance-v1",
        baseline_revision=BASELINE_REVISION,
        environment_sha256=ENVIRONMENT,
        container_runtime=RUNTIME,
        cases=(
            RegressionCase("regression-1", {"value": 1}, {"value": 2, "mode": "legacy"},
                           {"value": 2, "mode": "legacy"}),
            RegressionCase("regression-2", {"value": 2}, {"value": 3, "mode": "legacy"},
                           {"value": 3, "mode": "legacy"}),
            RegressionCase("regression-3", {"value": 3}, {"value": 4, "mode": "legacy"},
                           {"value": 4, "mode": "legacy"}),
            RegressionCase(
                "improvement-1",
                {"value": 0, "upgrade": True},
                {"value": 1, "mode": "legacy"},
                {"value": 1, "mode": "improved"},
                RegressionCaseKind.IMPROVEMENT,
            ),
        ),
    )


BASELINE_SOURCE = '''
def run(inputs):
    value = inputs["value"] + 1
    return {"value": value, "mode": "legacy"}
'''

CANDIDATE_SOURCE = '''
def run(inputs):
    value = inputs["value"] + 1
    mode = "improved" if inputs.get("upgrade", False) else "legacy"
    return {"value": value, "mode": mode}
'''

BROKEN_CANDIDATE_SOURCE = '''
def run(inputs):
    return {"value": inputs["value"] + 999, "mode": "legacy"}
'''


def proposal(current_suite: FrozenRegressionSuite | None = None, candidate: str = CANDIDATE_SOURCE):
    current_suite = current_suite or suite()
    return ImprovementProposal.create(
        title="Enable improved output mode",
        rationale="Add the requested feature while preserving all frozen legacy outputs.",
        artifact_path="mirror_crystal/candidate.py",
        change_kind=ChangeKind.CODE,
        created_by="author-a",
        baseline_revision=BASELINE_REVISION,
        baseline_source=BASELINE_SOURCE,
        candidate_source=candidate,
        suite=current_suite,
    )


class FakeExecutor:
    def __init__(self, *, fail_source: str | None = None, wrong_binding: bool = False):
        self.fail_source = fail_source
        self.wrong_binding = wrong_binding
        self.calls: list[tuple[str, str]] = []

    def execute(self, source, frozen_suite, *, expected_key):
        self.calls.append((source, expected_key))
        results = []
        for item in frozen_suite.cases:
            passed = not (
                self.fail_source == source and item.case_id == "regression-2"
            )
            results.append(CaseExecutionResult(
                case_id=item.case_id,
                passed=passed,
                actual_json="{}" if passed else '{"wrong":true}',
                error="" if passed else "deliberate test failure",
            ))
        return ExecutionEvidence(
            artifact_sha256=digest(source),
            suite_fingerprint=(digest("wrong-suite") if self.wrong_binding else frozen_suite.fingerprint),
            environment_sha256=frozen_suite.environment_sha256,
            expected_key=expected_key,
            runtime=RUNTIME,
            returncode=0,
            pid=1,
            results=tuple(results),
        )


def test_stage19_proposal_is_immutable_and_bound_to_source_and_frozen_suite():
    current_suite = suite()
    item = proposal(current_suite)
    assert item.proposal_id == digest(json.dumps({
        "title": item.title,
        "rationale": item.rationale,
        "artifact_path": item.artifact_path,
        "change_kind": item.change_kind.value,
        "created_by": item.created_by,
        "baseline_revision": item.baseline_revision,
        "baseline_artifact_sha256": digest(BASELINE_SOURCE),
        "candidate_artifact_sha256": digest(CANDIDATE_SOURCE),
        "regression_suite_fingerprint": current_suite.fingerprint,
    }, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    assert item.baseline_artifact_sha256 == digest(BASELINE_SOURCE)
    assert item.candidate_artifact_sha256 == digest(CANDIDATE_SOURCE)
    with pytest.raises((AttributeError, TypeError)):
        item.title = "tampered"


def test_stage19_rejects_suite_without_minimum_regressions_and_improvement_case():
    with pytest.raises(SelfImprovementError, match="at least four typed cases"):
        FrozenRegressionSuite(
            "too-small", BASELINE_REVISION, ENVIRONMENT,
            (RegressionCase("one", {}, 1, 1), RegressionCase("two", {}, 2, 2),
             RegressionCase("three", {}, 3, 3)),
        )
    with pytest.raises(SelfImprovementError, match="three regression cases"):
        FrozenRegressionSuite(
            "too-few-regressions", BASELINE_REVISION, ENVIRONMENT,
            (
                RegressionCase("r1", {}, 1, 1),
                RegressionCase("r2", {}, 2, 2),
                RegressionCase("i1", {}, "old", "new", RegressionCaseKind.IMPROVEMENT),
                RegressionCase("i2", {}, "old2", "new2", RegressionCaseKind.IMPROVEMENT),
            ),
        )
    with pytest.raises(SelfImprovementError, match="improvement case"):
        FrozenRegressionSuite(
            "no-improvement", BASELINE_REVISION, ENVIRONMENT,
            (
                RegressionCase("r1", {}, 1, 1),
                RegressionCase("r2", {}, 2, 2),
                RegressionCase("r3", {}, 3, 3),
                RegressionCase("r4", {}, 4, 4),
            ),
        )


def test_stage19_regression_suite_and_nested_inputs_are_immutable():
    current_suite = suite()
    with pytest.raises(TypeError):
        current_suite.cases[0].inputs["value"] = 999
    with pytest.raises(AttributeError):
        current_suite.cases = ()
    assert current_suite.to_dict()["cases"][0]["inputs"] == {"value": 1}


def test_stage19_evaluation_runs_baseline_and_candidate_and_requires_human_review():
    executor = FakeExecutor()
    engine = ControlledSelfImprovementEngine(executor, allow_test_executor=True)
    current_suite = suite()
    item = proposal(current_suite)

    result = engine.evaluate(item, current_suite)

    assert result.status is EvaluationStatus.READY_FOR_HUMAN_REVIEW
    assert result.baseline_evidence is not None and result.baseline_evidence.passed
    assert result.candidate_evidence is not None and result.candidate_evidence.passed
    assert result.regression_case_ids == ("regression-1", "regression-2", "regression-3")
    assert result.improvement_case_ids == ("improvement-1",)
    assert executor.calls == [
        (BASELINE_SOURCE, "baseline_expected"),
        (CANDIDATE_SOURCE, "candidate_expected"),
    ]


def test_stage19_candidate_regression_is_rejected_and_cannot_be_approved():
    current_suite = suite()
    item = proposal(current_suite)
    engine = ControlledSelfImprovementEngine(FakeExecutor(fail_source=CANDIDATE_SOURCE), allow_test_executor=True)

    result = engine.evaluate(item, current_suite)

    assert result.status is EvaluationStatus.REJECTED
    assert result.failed_case_ids == ("regression-2",)
    with pytest.raises(SelfImprovementError, match="only passing evaluations"):
        engine.record_human_decision(
            item, result, reviewer_id="reviewer-b", approved=True, reason="attempted approval",
        )


def test_stage19_baseline_failure_blocks_candidate_execution():
    current_suite = suite()
    engine = ControlledSelfImprovementEngine(FakeExecutor(fail_source=BASELINE_SOURCE), allow_test_executor=True)
    result = engine.evaluate(proposal(current_suite), current_suite)

    assert result.status is EvaluationStatus.REJECTED
    assert result.candidate_evidence is None
    assert result.failed_case_ids == ("regression-2",)


def test_stage19_fails_closed_on_misbound_executor_evidence():
    current_suite = suite()
    engine = ControlledSelfImprovementEngine(FakeExecutor(wrong_binding=True), allow_test_executor=True)
    result = engine.evaluate(proposal(current_suite), current_suite)

    assert result.status is EvaluationStatus.REJECTED
    assert "different frozen regression suite" in result.explanation


def test_stage19_proposal_rejects_baseline_drift_and_path_traversal():
    current_suite = suite()
    with pytest.raises(SelfImprovementError, match="does not match"):
        ImprovementProposal.create(
            title="drift", rationale="baseline drift", artifact_path="mirror/file.py",
            change_kind=ChangeKind.CODE, created_by="author-a",
            baseline_revision="b" * 40, baseline_source=BASELINE_SOURCE,
            candidate_source=CANDIDATE_SOURCE, suite=current_suite,
        )
    with pytest.raises(SelfImprovementError, match="relative path"):
        ImprovementProposal.create(
            title="traversal", rationale="bad path", artifact_path="../outside.py",
            change_kind=ChangeKind.CODE, created_by="author-a",
            baseline_revision=BASELINE_REVISION, baseline_source=BASELINE_SOURCE,
            candidate_source=CANDIDATE_SOURCE, suite=current_suite,
        )


def test_stage19_requires_independent_human_approval_and_binds_exact_candidate():
    current_suite = suite()
    item = proposal(current_suite)
    engine = ControlledSelfImprovementEngine(FakeExecutor(), allow_test_executor=True)
    evaluation = engine.evaluate(item, current_suite)
    with pytest.raises(SelfImprovementError, match="cannot approve their own"):
        engine.record_human_decision(
            item, evaluation, reviewer_id="author-a", approved=True, reason="self approval",
        )

    approval = engine.record_human_decision(
        item, evaluation, reviewer_id="reviewer-b", approved=True,
        reason="Reviewed frozen test evidence and accepted this exact artifact.",
    )
    assert approval.decision is ApprovalDecision.APPROVED
    assert approval.authorization is not None
    assert approval.authorization.authorizes(item, evaluation)
    changed = proposal(current_suite, candidate=CANDIDATE_SOURCE + "\n# altered")
    assert not approval.authorization.authorizes(changed, evaluation)


def test_stage19_human_rejection_never_returns_promotion_authorization():
    current_suite = suite()
    item = proposal(current_suite)
    engine = ControlledSelfImprovementEngine(FakeExecutor(), allow_test_executor=True)
    evaluation = engine.evaluate(item, current_suite)
    record = engine.record_human_decision(
        item, evaluation, reviewer_id="reviewer-c", approved=False,
        reason="Hold for additional manual review.",
    )
    assert record.decision is ApprovalDecision.REJECTED
    assert record.authorization is None


def test_stage19_real_container_evaluates_baseline_and_candidate_in_separate_runs():
    if find_container_runtime() is None:
        if os.environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") == "1":
            pytest.fail("CI requires Docker/Podman for Stage 19 acceptance")
        pytest.skip("Docker/Podman is not installed")

    current_suite = suite()
    engine = ControlledSelfImprovementEngine(ContainerRegressionExecutor(timeout_seconds=8))
    result = engine.evaluate(proposal(current_suite), current_suite)

    assert result.status is EvaluationStatus.READY_FOR_HUMAN_REVIEW, result.explanation
    assert result.baseline_evidence is not None
    assert result.candidate_evidence is not None
    assert result.baseline_evidence.pid == 1
    assert result.candidate_evidence.pid == 1
    assert result.baseline_evidence.runtime == result.candidate_evidence.runtime


def test_stage19_decisions_require_engine_recorded_evaluation_and_are_single_use():
    current_suite = suite()
    item = proposal(current_suite)
    first_engine = ControlledSelfImprovementEngine(FakeExecutor(), allow_test_executor=True)
    evaluation = first_engine.evaluate(item, current_suite)
    second_engine = ControlledSelfImprovementEngine(FakeExecutor(), allow_test_executor=True)

    with pytest.raises(SelfImprovementError, match="not a verified result"):
        second_engine.record_human_decision(
            item, evaluation, reviewer_id="reviewer-d", approved=True, reason="forged cross-engine review",
        )

    first_engine.record_human_decision(
        item, evaluation, reviewer_id="reviewer-d", approved=True, reason="approved once",
    )
    with pytest.raises(SelfImprovementError, match="already been recorded"):
        first_engine.record_human_decision(
            item, evaluation, reviewer_id="reviewer-e", approved=True, reason="attempted second approval",
        )


def test_stage19_rejects_custom_executor_outside_explicit_test_mode():
    with pytest.raises(SelfImprovementError, match="test-only"):
        ControlledSelfImprovementEngine(FakeExecutor())


def test_stage19_frozen_environment_must_match_real_resource_and_image_contract():
    with pytest.raises(SelfImprovementError, match="environment fingerprint"):
        FrozenRegressionSuite(
            suite_id="wrong-environment",
            baseline_revision=BASELINE_REVISION,
            environment_sha256=digest("not-the-computed-environment"),
            container_runtime=RUNTIME,
            cases=suite().cases,
        )
    with pytest.raises(SelfImprovementError, match="immutable SHA-256"):
        FrozenRegressionSuite(
            suite_id="mutable-image",
            baseline_revision=BASELINE_REVISION,
            environment_sha256=ENVIRONMENT,
            container_runtime=RUNTIME,
            container_image="python:3.11-slim",
            cases=suite().cases,
        )


def test_stage19_real_container_rejects_candidate_attempt_to_mutate_json_harness():
    if find_container_runtime() is None:
        if os.environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") == "1":
            pytest.fail("CI requires Docker/Podman for Stage 19 adversarial acceptance")
        pytest.skip("Docker/Podman is not installed")

    malicious = '''
import json
json.dumps = lambda *args, **kwargs: "{}"
def run(inputs):
    return {"value": inputs["value"] + 1, "mode": "legacy"}
'''
    current_suite = suite()
    engine = ControlledSelfImprovementEngine()
    result = engine.evaluate(proposal(current_suite, candidate=malicious), current_suite)

    assert result.status is EvaluationStatus.REJECTED
    assert result.candidate_evidence is not None
    assert result.candidate_evidence.passed is False
    assert result.candidate_evidence.results
    assert all(not item.passed for item in result.candidate_evidence.results)
    assert all("blocked private or mutating attribute" in item.error
               for item in result.candidate_evidence.results)


def test_stage19_real_container_candidate_output_cannot_spoof_supervisor_manifest():
    if find_container_runtime() is None:
        if os.environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") == "1":
            pytest.fail("CI requires Docker/Podman for Stage 19 adversarial acceptance")
        pytest.skip("Docker/Podman is not installed")

    spoofing = '''
def run(inputs):
    print('__MIRROR_STAGE19_RESULTS__{"results":[{"case_id":"forged","passed":true}]}')
    value = inputs["value"] + 1
    mode = "improved" if inputs.get("upgrade", False) else "legacy"
    return {"value": value, "mode": mode}
'''
    current_suite = suite()
    result = ControlledSelfImprovementEngine().evaluate(proposal(current_suite, candidate=spoofing), current_suite)

    assert result.status is EvaluationStatus.READY_FOR_HUMAN_REVIEW
    assert result.candidate_evidence is not None and result.candidate_evidence.passed
    assert tuple(item.case_id for item in result.candidate_evidence.results) == current_suite.case_ids
    assert all(item.case_id != "forged" for item in result.candidate_evidence.results)


def test_stage19_real_container_candidate_stdout_is_bounded():
    if find_container_runtime() is None:
        if os.environ.get("MIRROR_REQUIRE_CONTAINER_SANDBOX") == "1":
            pytest.fail("CI requires Docker/Podman for Stage 19 adversarial acceptance")
        pytest.skip("Docker/Podman is not installed")

    noisy = '''
def run(inputs):
    print("x" * 9000)
    return {"value": inputs["value"] + 1, "mode": "legacy"}
'''
    current_suite = suite()
    result = ControlledSelfImprovementEngine().evaluate(proposal(current_suite, candidate=noisy), current_suite)

    assert result.status is EvaluationStatus.REJECTED
    assert result.candidate_evidence is not None
    assert result.failed_case_ids
    assert any("CandidateOutputLimitExceeded" in item.error for item in result.candidate_evidence.results)
