from __future__ import annotations

import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel
from mirror_crystal.capability_composition import CompositionPlan, PlanStatus
from mirror_crystal.capability_synthesizer import (
    Example, SynthesisGoal, SynthesisReport, SynthesisStatus,
)
from mirror_crystal.domain_verification import (
    DomainCheckResult, DomainDecision, DomainVerificationReport,
    DomainVerificationStatus,
)
from mirror_crystal.execution import (
    ExecutionStatus, PlanExecutionReport, StepExecutionResult,
)
from mirror_crystal.failure_repair import (
    FailureKind, FailureRepairEngine, FailureRepairError, RepairStatus,
)
from mirror_crystal.operational_language import Program, Var, VariableSpec, UOL


def square_goal() -> SynthesisGoal:
    return SynthesisGoal(
        "square-x",
        "mathematics",
        (VariableSpec("x", "integer"), VariableSpec("y", "integer")),
        "integer",
        training=(Example({"x": 0, "y": 0}, 0), Example({"x": 1, "y": 0}, 1)),
        validation=(Example({"x": 0, "y": 3}, 0),),
        constants=(0, 1),
        max_depth=2,
        candidate_budget=20000,
    )


def test_diagnosis_identifies_first_failed_execution_step_and_exact_capability():
    failed = StepExecutionResult(
        "step-2", "subtask-2", "cap:divide", False,
        {"numerator": 12, "denominator": 0}, {},
        "ZeroDivisionError: division by zero",
    )
    report = PlanExecutionReport(
        "plan-1", "task-1", ExecutionStatus.FAILED, (failed,),
        "execution stopped at step-2",
    )

    diagnosis = FailureRepairEngine().diagnose(report)

    assert diagnosis.kind is FailureKind.EXECUTION_FAILED
    assert diagnosis.step_id == "step-2"
    assert diagnosis.capability_id == "cap:divide"
    assert "ZeroDivisionError" in diagnosis.message
    assert "do not retry side effects automatically" in diagnosis.recommended_action


def test_diagnosis_does_not_treat_blocked_composition_as_a_repairable_success():
    plan = CompositionPlan(
        "plan-blocked", "task-2", PlanStatus.BLOCKED, (),
        blockers=("required input is unbound: mass",),
    )
    diagnosis = FailureRepairEngine().diagnose(plan)
    assert diagnosis.kind is FailureKind.COMPOSITION_BLOCKED
    assert "mass" in diagnosis.message


def test_diagnosis_separates_rejected_evidence_from_missing_verification():
    report = DomainVerificationReport(
        "plan-3", "task-3", DomainVerificationStatus.FAILED,
        (
            DomainCheckResult(
                "step-1", "cap:calc", "mathematics", "formal-proof",
                "checker-a", DomainDecision.FAIL, detail="proof obligation not discharged",
            ),
        ),
        (), (), ("proof obligation not discharged",), "a" * 64,
    )
    diagnosis = FailureRepairEngine().diagnose(report)
    assert diagnosis.kind is FailureKind.DOMAIN_EVIDENCE_REJECTED
    assert diagnosis.evidence_kinds == ("formal-proof",)
    assert "checker-a" in diagnosis.message
    assert "never lower" in diagnosis.recommended_action


def test_diagnosis_classifies_budget_and_counterexample_exhaustion():
    engine = FailureRepairEngine()
    budget = engine.diagnose(SynthesisReport(
        SynthesisStatus.BUDGET_EXHAUSTED, None, 20, 0, 0,
        blockers=("candidate budget exhausted",),
    ))
    assert budget.kind is FailureKind.SYNTHESIS_BUDGET_EXHAUSTED

    counterexamples = engine.diagnose(SynthesisReport(
        SynthesisStatus.COUNTEREXAMPLE_LIMIT, None, 50, 0, 0,
        blockers=("fresh held-out probe required",),
    ))
    assert counterexamples.kind is FailureKind.SYNTHESIS_COUNTEREXAMPLE_LIMIT


def test_phase7_repairs_underfit_program_with_fresh_cegis_evidence_and_keeps_candidate_unverified():
    goal = square_goal()
    language = UOL()
    original = Program(
        "program:naive-identity", "naive-identity",
        goal.inputs, Var("x"), "mathematics",
    )
    proposal = FailureRepairEngine(language).propose_program_repair(
        goal,
        original,
        counterexample_inputs=({"x": 0, "y": 4}, {"x": 2, "y": 0}),
        counterexample_oracle=lambda values: values["x"] ** 2,
        max_rounds=4,
    )

    assert proposal.status is RepairStatus.PROPOSED
    assert proposal.ready_for_independent_verification
    assert proposal.requires_explicit_promotion
    assert proposal.original_program_fingerprint == original.fingerprint
    assert proposal.program is not None
    assert proposal.program.fingerprint != original.fingerprint
    assert language.execute(proposal.program, {"x": 8, "y": 0}) == 64
    assert proposal.synthesis_report.succeeded
    assert proposal.synthesis_report.probes_passed == 2
    assert len(proposal.counterexamples) == 2
    assert proposal.capability_candidate is not None
    assert proposal.capability_candidate.verification.level is EvidenceLevel.UNVERIFIED
    assert not proposal.capability_candidate.executable


def test_phase7_refuses_to_repair_without_a_demonstrated_counterexample():
    goal = square_goal()
    original = Program(
        "program:identity", "identity", goal.inputs, Var("x"), "mathematics",
    )
    with pytest.raises(FailureRepairError, match="passed every supplied probe"):
        FailureRepairEngine().propose_program_repair(
            goal,
            original,
            counterexample_inputs=({"x": 2, "y": 0},),
            counterexample_oracle=lambda values: values["x"],
        )


def test_phase7_rejects_probe_leakage_before_calling_the_host_oracle():
    goal = square_goal()
    original = Program(
        "program:identity", "identity", goal.inputs, Var("x"), "mathematics",
    )
    calls = []

    def oracle(values):
        calls.append(dict(values))
        return values["x"] ** 2

    with pytest.raises(FailureRepairError, match="disjoint"):
        FailureRepairEngine().propose_program_repair(
            goal,
            original,
            counterexample_inputs=({"x": 0, "y": 0},),
            counterexample_oracle=oracle,
        )
    assert calls == []


def test_phase7_rejects_wrong_program_interface_before_repair():
    goal = square_goal()
    original = Program(
        "program:wrong-interface", "wrong-interface",
        (VariableSpec("other", "integer"),), Var("other"), "mathematics",
    )
    with pytest.raises(FailureRepairError, match="typed inputs"):
        FailureRepairEngine().propose_program_repair(
            goal,
            original,
            counterexample_inputs=({"x": 2, "y": 0},),
            counterexample_oracle=lambda values: values["x"] ** 2,
        )
