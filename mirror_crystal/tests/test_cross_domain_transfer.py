from __future__ import annotations

from dataclasses import replace

import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.capability_synthesizer import Example
from mirror_crystal.domain_verification import DomainVerificationEngine
from mirror_crystal.execution import (
    ExecutionStatus, PlanExecutionReport, StepExecutionResult,
)
from mirror_crystal.native_verifiers import NativeDomainVerifierSuite
from mirror_crystal.operational_language import (
    Call, Lit, OperationalRule, Program, RuleStatus, UOL, Var, VariableSpec,
)
from mirror_crystal.universal_representation import RepresentationProvenance
from mirror_crystal.verified_program_learning import (
    UOLProgramHandler, UOLProgramLearningProposal, VerifiedProgramLearningEngine,
    VerifiedProgramObservation,
)
from mirror_crystal.cross_domain_transfer import (
    CrossDomainTransferEngine, CrossDomainTransferError,
)


def source_program():
    return Program(
        "program:increment", "increment",
        (VariableSpec("x", "integer"),),
        Call("add", (Var("x"), Lit(1))),
        "programming",
    )


def source_proposal():
    program = source_program()
    suite = NativeDomainVerifierSuite()
    capability = CapabilitySpec(
        capability_id="learned:increment:programming",
        name="Verified increment",
        description="Verified source-domain operation",
        domains=("programming",),
        intents=("transform",),
        inputs=(
            PortContract("x", "integer"),
            PortContract("__mirror_verification__", "object", required=False),
        ),
        outputs=(PortContract("result", "integer"),),
        postconditions=("program.output_matches_reference", "program.ports_match_contract"),
        provenance=RepresentationProvenance(
            source="mirror_verified_uol_program_learning",
            source_ref=program.fingerprint,
        ),
        verification=VerificationEvidence(
            level=EvidenceLevel.INDEPENDENTLY_REPRODUCED,
            checks_passed=("checker-a:reference_replay", "checker-b:contract_validation"),
            verifier="two independent programming verifiers",
        ),
    )
    handler = UOLProgramHandler(program, ())
    return UOLProgramLearningProposal(
        capability=capability,
        program=program,
        handler=handler,
        observation_task_ids=("source-task-1", "source-task-2", "source-task-3"),
        execution_fingerprints=("a" * 64, "b" * 64, "c" * 64),
        verifier_ids=("checker-a", "checker-b"),
        evidence_fingerprints=("d" * 64, "e" * 64, "f" * 64),
        minimum_successes=3,
        minimum_verifiers=2,
    )


def target_math_postconditions():
    return NativeDomainVerifierSuite.postconditions("mathematics")


def make_math_observations(program, candidate):
    suite = NativeDomainVerifierSuite()
    engine = DomainVerificationEngine(suite.registry)
    observations = []
    for index, value in enumerate((2, 5, 9)):
        payload = {
            "mathematics": {
                "variables": ["x"],
                "left": {"op": "add", "left": {"op": "var", "name": "x"}, "right": {"op": "lit", "value": 1}},
                "right": {"op": "add", "left": {"op": "lit", "value": 1}, "right": {"op": "var", "name": "x"}},
                "samples": [{"variables": {"x": str(value)}}, {"variables": {"x": str(value + 1)}}],
            }
        }
        envelope = {"schema_version": 1, "domain": "mathematics", "payload": payload}
        step = StepExecutionResult(
            f"target-step-{index}", f"target-task-{index}", candidate.capability_id, True,
            {"x": value, "__mirror_verification__": envelope},
            {"result": UOL().execute(program, {"x": value})},
        )
        execution = PlanExecutionReport(
            f"target-plan-{index}", f"target-task-{index}",
            ExecutionStatus.SUCCEEDED, (step,),
        )
        report = engine.verify_candidate(
            execution, candidate,
            requirements={candidate.capability_id: suite.requirement("mathematics")},
            required_checks={candidate.capability_id: suite.required_checks("mathematics")},
        )
        observations.append(VerifiedProgramObservation(
            f"target-task-{index}", candidate, execution, report,
        ))
    return tuple(observations)


def test_verified_domain_neutral_program_transfers_and_is_reverified_in_target_domain():
    source = source_proposal()
    source_input = PortContract("x", "integer", unit="count", dimensions={"count": 1})
    source_output = PortContract("result", "integer", unit="count", dimensions={"count": 1})
    source_capability = replace(
        source.capability,
        inputs=(source_input, source.capability.inputs[1]),
        outputs=(source_output,),
    )
    source = replace(source, capability=source_capability)
    language = UOL()
    proposal = CrossDomainTransferEngine().propose(
        source, language,
        target_domain="mathematics",
        training=(
            Example({"x": 2}, 3), Example({"x": 5}, 6), Example({"x": 9}, 10),
        ),
        validation=(Example({"x": 0}, 1), Example({"x": 12}, 13)),
        target_postconditions=target_math_postconditions(),
    )
    assert proposal.source_domain == "programming"
    assert proposal.target_domain == "mathematics"
    assert proposal.training_passed == 3 and proposal.validation_passed == 2
    assert proposal.capability.inputs[0] == source_input
    assert proposal.capability.outputs == (source_output,)
    assert not proposal.capability.executable
    assert proposal.capability.verification.level is EvidenceLevel.UNVERIFIED

    target_observations = make_math_observations(proposal.program, proposal.capability)
    learned = VerifiedProgramLearningEngine().propose(
        proposal.program, target_observations, language, intents=("transform",),
    )
    assert learned.capability.domains == ("mathematics",)
    assert learned.capability.executable
    assert len(learned.verifier_ids) == 2


def test_transfer_rejects_source_domain_and_target_domain_equality():
    with pytest.raises(CrossDomainTransferError, match="must differ"):
        CrossDomainTransferEngine().propose(
            source_proposal(), UOL(), target_domain="programming",
            training=(Example({"x": 1}, 2), Example({"x": 2}, 3), Example({"x": 3}, 4)),
            validation=(Example({"x": 4}, 5), Example({"x": 5}, 6)),
            target_postconditions=("target.checked",),
        )


def test_transfer_rejects_target_heldout_failure_and_input_overlap():
    engine = CrossDomainTransferEngine()
    with pytest.raises(CrossDomainTransferError, match="held-out"):
        engine.propose(
            source_proposal(), UOL(), target_domain="mathematics",
            training=(Example({"x": 1}, 2), Example({"x": 2}, 3), Example({"x": 3}, 4)),
            validation=(Example({"x": 4}, 999), Example({"x": 5}, 6)),
            target_postconditions=target_math_postconditions(),
        )
    with pytest.raises(CrossDomainTransferError, match="globally disjoint"):
        engine.propose(
            source_proposal(), UOL(), target_domain="mathematics",
            training=(Example({"x": 1}, 2), Example({"x": 2}, 3), Example({"x": 3}, 4)),
            validation=(Example({"x": 3}, 4), Example({"x": 5}, 6)),
            target_postconditions=target_math_postconditions(),
        )


def test_transfer_refuses_nonverified_domain_specific_rule_dependencies():
    source = source_proposal()
    rule = OperationalRule(
        "program-only", "program-only", (VariableSpec("n", "integer"),),
        Call("add", (Var("n"), Lit(1))), ("programming",), RuleStatus.VERIFIED,
    )
    program = Program(
        "program:increment-via-rule", "increment-via-rule",
        (VariableSpec("x", "integer"),),
        Call("program-only", (Var("x"),)), "programming",
        rule_refs=("program-only",),
    )
    cap = replace(
        source.capability,
        provenance=RepresentationProvenance(
            source="mirror_verified_uol_program_learning",
            source_ref=program.fingerprint,
        ),
    )
    source = replace(
        source, program=program, capability=cap,
        handler=UOLProgramHandler(program, (rule,)),
    )
    with pytest.raises(CrossDomainTransferError, match="domain-specific rule"):
        CrossDomainTransferEngine().propose(
            source, UOL((rule,)), target_domain="mathematics",
            training=(Example({"x": 1}, 2), Example({"x": 2}, 3), Example({"x": 3}, 4)),
            validation=(Example({"x": 4}, 5), Example({"x": 5}, 6)),
            target_postconditions=target_math_postconditions(),
        )


def test_transfer_accepts_only_verified_global_rule_dependencies():
    source = source_proposal()
    rule = OperationalRule(
        "general-increment", "general-increment",
        (VariableSpec("n", "integer"),),
        Call("add", (Var("n"), Lit(1))), ("general",), RuleStatus.VERIFIED,
    )
    program = Program(
        "program:increment-via-general-rule", "increment-via-general-rule",
        (VariableSpec("x", "integer"),),
        Call("general-increment", (Var("x"),)), "programming",
        rule_refs=("general-increment",),
    )
    capability = replace(
        source.capability,
        provenance=RepresentationProvenance(
            source="mirror_verified_uol_program_learning",
            source_ref=program.fingerprint,
        ),
    )
    source = replace(
        source, program=program, capability=capability,
        handler=UOLProgramHandler(program, (rule,)),
    )
    transferred = CrossDomainTransferEngine().propose(
        source, UOL((rule,)), target_domain="mathematics",
        training=(Example({"x": 1}, 2), Example({"x": 2}, 3), Example({"x": 3}, 4)),
        validation=(Example({"x": 4}, 5), Example({"x": 5}, 6)),
        target_postconditions=target_math_postconditions(),
    )
    assert transferred.rule_ids == ("general-increment",)
    assert transferred.capability.provenance.parent_ids == (
        source.capability.capability_id, "general-increment",
    )
