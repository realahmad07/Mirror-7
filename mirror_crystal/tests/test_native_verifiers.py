from __future__ import annotations

import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.domain_verification import (
    DomainVerificationEngine, DomainVerificationStatus,
)
from mirror_crystal.execution import ExecutionStatus, PlanExecutionReport, StepExecutionResult
from mirror_crystal.native_verifiers import NativeDomainVerifierSuite


def run_native(domain, payload, *, step_inputs=None, outputs=None, output_ports=None):
    suite = NativeDomainVerifierSuite()
    input_values = dict(step_inputs or {})
    input_values["__mirror_verification__"] = {
        "schema_version": 1, "domain": domain, "payload": payload,
    }
    outputs = dict(outputs or {"result": 1})
    cap = CapabilitySpec(
        capability_id=f"native:{domain}",
        name=f"native {domain} calculation",
        description="Bounded domain verification fixture",
        domains=(domain,),
        intents=("verify",),
        inputs=tuple(
            PortContract(name, "object" if name == "__mirror_verification__" else
                         "integer" if isinstance(value, int) and not isinstance(value, bool) else "number")
            for name, value in input_values.items()
        ),
        outputs=tuple(
            PortContract(name, "number" if isinstance(value, (int, float)) and not isinstance(value, bool)
                         else "object" if isinstance(value, dict) else "array")
            for name, value in outputs.items()
        ),
        postconditions=NativeDomainVerifierSuite.postconditions(domain),
        verification=VerificationEvidence(
            level=EvidenceLevel.EXECUTION_TESTED,
            checks_passed=("operation-contract-regression",),
            verifier="host-execution-tests",
        ),
    )
    step = StepExecutionResult(
        step_id="step-1", subtask_id="task-1", capability_id=cap.capability_id,
        success=True, inputs=input_values, outputs=outputs,
    )
    execution = PlanExecutionReport(
        plan_id="plan-1", task_id="task-1", status=ExecutionStatus.SUCCEEDED,
        steps=(step,),
    )
    return DomainVerificationEngine(suite.registry).verify(
        execution, (cap,),
        requirements={cap.capability_id: suite.requirement(domain)},
        required_checks={cap.capability_id: suite.required_checks(domain)},
    )


def polynomial(op, left, right=None):
    node = {"op": op, "left": left}
    if right is not None:
        node["right"] = right
    return node


def var(name):
    return {"op": "var", "name": name}


def lit(value):
    return {"op": "lit", "value": value}


@pytest.mark.parametrize("domain", NativeDomainVerifierSuite.supported_domains())
def test_native_suite_registers_two_distinct_real_verifiers_per_domain(domain):
    suite = NativeDomainVerifierSuite()
    checks = suite.registry.list(domain=domain)
    assert len(checks) == 2
    assert len({item.verifier_id for item in checks}) == 2
    assert all(item.check_id and item.evidence_kind for item in checks)


def test_mathematics_exact_polynomial_and_rational_sample_verification():
    left = polynomial("add", var("x"), var("x"))
    right = polynomial("mul", lit(2), var("x"))
    report = run_native("mathematics", {
        "mathematics": {
            "variables": ["x"], "left": left, "right": right,
            "samples": [
                {"variables": {"x": "0"}},
                {"variables": {"x": "3/2"}},
                {"variables": {"x": "-7"}},
            ],
        }
    })
    assert report.status is DomainVerificationStatus.VERIFIED
    assert report.verified_conditions == NativeDomainVerifierSuite.postconditions("mathematics")


def test_programming_reference_interpreter_and_contract_check():
    program = {
        "program_id": "p:increment", "name": "increment",
        "inputs": [{"name": "x", "value_type": "integer"}],
        "body": {
            "op": "call", "name": "add",
            "args": [{"op": "var", "name": "x"}, {"op": "lit", "value": 1}],
        },
        "domain": "programming", "preconditions": [], "postconditions": [], "rule_refs": [],
    }
    report = run_native(
        "programming",
        {"programming": {"program": program, "output_name": "result"}},
        step_inputs={"x": 4},
        outputs={"result": 5},
    )
    assert report.status is DomainVerificationStatus.VERIFIED
    assert len(report.verifier_ids) == 2


def test_physics_dimensional_homogeneity_and_conservation():
    report = run_native("physics", {
        "physics": {
            "terms": [
                {"dimensions": {"length": 1, "time": -2}},
                {"dimensions": {"length": 1, "time": -2}},
                {"dimensions": {"length": 1, "time": -2}},
            ],
            "before": [1, 2, 3], "after": [4, 2], "tolerance": 0,
        }
    })
    assert report.verified


def test_biology_measurement_bounds_and_mass_balance():
    report = run_native("biology", {
        "biology": {
            "measurements": [
                {"name": "pH", "value": 7.2, "minimum": 0, "maximum": 14, "unit": "pH"},
                {"name": "viability", "value": 0.98, "minimum": 0, "maximum": 1, "unit": "fraction"},
            ],
            "initial_mass": 100, "component_masses": [60, 25, 15], "mass_tolerance": 0.001,
        }
    })
    assert report.verified


def test_engineering_stress_and_safety_factor_calculation():
    report = run_native("engineering", {
        "engineering": {
            "load_n": 1000, "area_mm2": 100,
            "allowable_stress_mpa": 60, "required_safety_factor": 4,
        }
    }, outputs={"stress_mpa": 10, "safety_factor": 6})
    assert report.verified


@pytest.mark.parametrize(
    ("domain", "payload", "outputs", "expected_failure"),
    [
        ("mathematics", {"mathematics": {
            "variables": ["x"], "left": polynomial("add", var("x"), lit(1)),
            "right": var("x"), "samples": [{"variables": {"x": 2}}],
        }}, {"result": 1}, "polynomial"),
        ("physics", {"physics": {
            "terms": [{"dimensions": {"length": 1}}, {"dimensions": {"time": 1}}],
            "before": [1], "after": [1], "tolerance": 0,
        }}, {"result": 1}, "dimension"),
        ("biology", {"biology": {
            "measurements": [{"name": "pH", "value": 15, "minimum": 0, "maximum": 14, "unit": "pH"}],
            "initial_mass": 1, "component_masses": [1], "mass_tolerance": 0,
        }}, {"result": 1}, "bounds"),
        ("engineering", {"engineering": {
            "load_n": 1000, "area_mm2": 100,
            "allowable_stress_mpa": 60, "required_safety_factor": 7,
        }}, {"stress_mpa": 10, "safety_factor": 6}, "safety factor"),
    ],
)
def test_native_domain_engines_fail_closed_on_invalid_evidence(domain, payload, outputs, expected_failure):
    report = run_native(domain, payload, outputs=outputs)
    assert report.status is DomainVerificationStatus.FAILED
    assert any(expected_failure in item.detail.casefold() for item in report.results)


def test_programming_reference_replay_rejects_incorrect_execution_output():
    program = {
        "program_id": "p:increment", "name": "increment",
        "inputs": [{"name": "x", "value_type": "integer"}],
        "body": {"op": "call", "name": "add", "args": [
            {"op": "var", "name": "x"}, {"op": "lit", "value": 1},
        ]},
    }
    report = run_native(
        "programming", {"programming": {"program": program}},
        step_inputs={"x": 4}, outputs={"result": 999},
    )
    assert report.status is DomainVerificationStatus.FAILED
    assert any("disagrees" in item.detail for item in report.results)


def test_missing_verification_payload_never_passes():
    suite = NativeDomainVerifierSuite()
    # A direct call with no reserved envelope must produce failed evidence.
    from mirror_crystal.verification import VerificationContext
    from mirror_crystal.execution import StepExecutionResult
    cap = CapabilitySpec(
        capability_id="missing-evidence", name="Missing evidence", description="Test",
        domains=("mathematics",), intents=("verify",),
        inputs=(), outputs=(PortContract("result", "integer"),),
        postconditions=NativeDomainVerifierSuite.postconditions("mathematics"),
        verification=VerificationEvidence(level=EvidenceLevel.EXECUTION_TESTED,
                                          checks_passed=("operation-test",), verifier="host-tests"),
    )
    step = StepExecutionResult("s", "t", cap.capability_id, True, {}, {"result": 1})
    context = VerificationContext("t", "p", step, cap, "a" * 64)
    artifacts = [check.callback(context) for check in suite.registry.list(domain="mathematics")]
    assert all(item.decision.value == "fail" for item in artifacts)
