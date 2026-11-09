from __future__ import annotations

import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_model import CapabilityRegistry, CapabilitySpec, PortContract
from mirror_crystal.domain_verification import DomainVerificationEngine
from mirror_crystal.execution import (
    ExecutionStatus, NativeOperationRegistry, PlanExecutionReport, StepExecutionResult,
)
from mirror_crystal.native_verifiers import NativeDomainVerifierSuite
from mirror_crystal.operational_language import (
    Call, Lit, Program, UOL, Var, VariableSpec, program_dict,
)
from mirror_crystal.universal_representation import RepresentationProvenance
from mirror_crystal.verified_program_learning import (
    ProgramLearningError, VerifiedProgramLearningEngine, VerifiedProgramObservation,
)


def candidate(program, suite, *, domain="programming"):
    return CapabilitySpec(
        capability_id=f"candidate:{domain}:{program.program_id}",
        name=program.name,
        description="Unpromoted UOL candidate",
        domains=(domain,),
        intents=("transform",),
        inputs=(
            *(PortContract(item.name, item.value_type) for item in program.inputs),
            PortContract("__mirror_verification__", "object", required=False),
        ),
        outputs=(PortContract("result", "integer"),),
        postconditions=suite.postconditions(domain),
        provenance=RepresentationProvenance(
            source="mirror_uol_synthesizer", source_ref=program.fingerprint,
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )


def observation_set(program, cap, domain, values, *, bad_last=False):
    suite = NativeDomainVerifierSuite()
    engine = DomainVerificationEngine(suite.registry)
    observations = []
    for index, value in enumerate(values):
        program_input = {"x": value}
        result = UOL().execute(program, program_input)
        if bad_last and index == len(values) - 1:
            result += 100
        if domain == "programming":
            payload = {
                "programming": {
                    "program": program_dict(program),
                    "output_name": "result",
                }
            }
        else:
            payload = {
                "mathematics": {
                    "variables": ["x"],
                    "left": {
                        "op": "add", "left": {"op": "var", "name": "x"},
                        "right": {"op": "lit", "value": 1},
                    },
                    "right": {
                        "op": "add", "left": {"op": "lit", "value": 1},
                        "right": {"op": "var", "name": "x"},
                    },
                    "samples": [{"variables": {"x": str(value)}}, {"variables": {"x": str(value + 2)}}],
                }
            }
        envelope = {"schema_version": 1, "domain": domain, "payload": payload}
        step_inputs = {"x": value, "__mirror_verification__": envelope}
        step = StepExecutionResult(
            step_id=f"step-{index}", subtask_id=f"task-{index}",
            capability_id=cap.capability_id, success=True,
            inputs=step_inputs, outputs={"result": result},
        )
        execution = PlanExecutionReport(
            plan_id=f"plan-{index}", task_id=f"task-{index}",
            status=ExecutionStatus.SUCCEEDED, steps=(step,),
        )
        report = engine.verify_candidate(
            execution, cap,
            requirements={cap.capability_id: suite.requirement(domain)},
            required_checks={cap.capability_id: suite.required_checks(domain)},
        )
        observations.append(VerifiedProgramObservation(
            f"task-{index}", cap, execution, report,
        ))
    return tuple(observations)


def increment_program(domain="programming"):
    return Program(
        "program:increment", "increment",
        (VariableSpec("x", "integer"),),
        Call("add", (Var("x"), Lit(1))),
        domain,
    )


def test_repeated_domain_verified_program_becomes_reusable_and_explicitly_promotable():
    program = increment_program()
    suite = NativeDomainVerifierSuite()
    cap = candidate(program, suite)
    observations = observation_set(program, cap, "programming", (1, 4, 9))

    proposal = VerifiedProgramLearningEngine().propose(
        program, observations, UOL(), intents=("transform",),
    )
    assert not cap.executable
    assert cap.verification.level is EvidenceLevel.UNVERIFIED
    assert proposal.requires_explicit_promotion
    assert proposal.capability.executable
    assert proposal.capability.verification.level is EvidenceLevel.INDEPENDENTLY_REPRODUCED
    assert proposal.observation_task_ids == ("task-0", "task-1", "task-2")
    assert len(proposal.verifier_ids) == 2

    capabilities = CapabilityRegistry()
    operations = NativeOperationRegistry()
    assert capabilities.get(proposal.capability.capability_id) is None
    promoted = VerifiedProgramLearningEngine.promote(proposal, capabilities, operations)
    assert capabilities.get(promoted) == proposal.capability
    assert operations.get(promoted)({"x": 12}) == {"result": 13}


def test_program_learning_rejects_failed_domain_observation():
    program = increment_program()
    suite = NativeDomainVerifierSuite()
    cap = candidate(program, suite)
    observations = observation_set(program, cap, "programming", (1, 4, 9), bad_last=True)
    assert not observations[-1].verification.verified
    with pytest.raises(ProgramLearningError, match="fully verified"):
        VerifiedProgramLearningEngine().propose(program, observations, UOL())


def test_program_learning_rejects_reused_task_or_execution_evidence():
    program = increment_program()
    suite = NativeDomainVerifierSuite()
    cap = candidate(program, suite)
    observations = observation_set(program, cap, "programming", (1, 4, 9))
    duplicate = (observations[0], observations[0], observations[2])
    with pytest.raises(ProgramLearningError, match="distinct task"):
        VerifiedProgramLearningEngine().propose(program, duplicate, UOL())


def test_program_learning_refuses_mismatched_candidate_program():
    program = increment_program()
    suite = NativeDomainVerifierSuite()
    cap = candidate(program, suite)
    observations = observation_set(program, cap, "programming", (1, 4, 9))
    other = Program(
        "program:wrong", "wrong", (VariableSpec("x", "integer"),),
        Call("add", (Var("x"), Lit(2))), "programming",
    )
    with pytest.raises(ProgramLearningError):
        VerifiedProgramLearningEngine().propose(other, observations, UOL())
