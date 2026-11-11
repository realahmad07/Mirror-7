from __future__ import annotations

from dataclasses import replace

import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.capability_synthesizer import Example
from mirror_crystal.domain_verification import DomainVerificationEngine
from mirror_crystal.execution import ExecutionStatus, PlanExecutionReport, StepExecutionResult
from mirror_crystal.native_verifiers import NativeDomainVerifierSuite
from mirror_crystal.operational_language import Call, Lit, Program, UOL, Var, VariableSpec, program_dict
from mirror_crystal.universal_representation import RepresentationProvenance
from mirror_crystal.verified_program_learning import (
    UOLProgramLearningProposal, VerifiedProgramLearningEngine, VerifiedProgramObservation,
)
from mirror_crystal.phase8_cross_domain_acceptance import (
    CrossDomainTransferBenchmarkEngine, TransferBenchmarkError,
    TransferBenchmarkStatus, TransferTargetBenchmark,
)


def increment_program(domain="programming"):
    return Program(
        "program:phase8-increment", "phase8-increment",
        (VariableSpec("x", "integer"),),
        Call("add", (Var("x"), Lit(1))),
        domain,
    )


def make_candidate(program, suite, domain):
    return CapabilitySpec(
        capability_id=f"phase8-candidate:{domain}:{program.program_id}",
        name=f"Increment candidate ({domain})",
        description="Unpromoted UOL transfer candidate",
        domains=(domain,),
        intents=("transform",),
        inputs=(
            PortContract("x", "integer"),
            PortContract("__mirror_verification__", "object", required=False),
        ),
        outputs=(PortContract("result", "integer"),),
        postconditions=suite.postconditions(domain),
        provenance=RepresentationProvenance(
            source="mirror_uol_synthesizer", source_ref=program.fingerprint,
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )


def build_observations(program, capability, domain, values):
    suite = NativeDomainVerifierSuite()
    verifier = DomainVerificationEngine(suite.registry)
    language = UOL()
    results = []
    for index, value in enumerate(values):
        expected = language.execute(program, {"x": value})
        if domain == "programming":
            payload = {
                "programming": {
                    "program": program_dict(program),
                    "output_name": "result",
                }
            }
        elif domain == "mathematics":
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
                    "samples": [{"variables": {"x": str(value)}}],
                }
            }
        elif domain == "physics":
            payload = {
                "physics": {
                    "terms": [
                        {"dimensions": {"length": 1}},
                        {"dimensions": {"length": 1}},
                    ],
                    "before": [value, 1],
                    "after": [expected],
                    "tolerance": 0,
                }
            }
        else:
            raise AssertionError(domain)
        envelope = {"schema_version": 1, "domain": domain, "payload": payload}
        step = StepExecutionResult(
            step_id=f"{domain}-step-{index}",
            subtask_id=f"{domain}-task-{index}",
            capability_id=capability.capability_id,
            success=True,
            inputs={"x": value, "__mirror_verification__": envelope},
            outputs={"result": expected},
        )
        execution = PlanExecutionReport(
            plan_id=f"{domain}-plan-{index}",
            task_id=f"{domain}-task-{index}",
            status=ExecutionStatus.SUCCEEDED,
            steps=(step,),
        )
        report = verifier.verify_candidate(
            execution,
            capability,
            requirements={capability.capability_id: suite.requirement(domain)},
            required_checks={capability.capability_id: suite.required_checks(domain)},
        )
        results.append(VerifiedProgramObservation(
            f"{domain}-task-{index}", capability, execution, report,
        ))
    return tuple(results)


def make_source_proposal():
    program = increment_program()
    suite = NativeDomainVerifierSuite()
    candidate = make_candidate(program, suite, "programming")
    observations = build_observations(program, candidate, "programming", (101, 107, 113))
    return VerifiedProgramLearningEngine().propose(
        program, observations, UOL(), intents=("transform",),
    )


def target(domain, *, offset):
    suite = NativeDomainVerifierSuite()
    train_values = (offset, offset + 2, offset + 4)
    held_values = (offset + 6, offset + 8)
    observation_values = (offset + 20, offset + 24, offset + 28)

    def observe(transfer):
        return build_observations(
            transfer.program, transfer.capability, domain, observation_values,
        )

    return TransferTargetBenchmark(
        domain=domain,
        training=tuple(Example({"x": value}, value + 1) for value in train_values),
        held_out=tuple(Example({"x": value}, value + 1) for value in held_values),
        postconditions=suite.postconditions(domain),
        observations=observe,
    )


def test_phase8_verifies_and_learns_transfers_in_multiple_target_domains():
    report = CrossDomainTransferBenchmarkEngine().evaluate(
        "phase8-multi-domain-v1",
        make_source_proposal(),
        UOL(),
        (target("mathematics", offset=200), target("physics", offset=300)),
    )
    assert report.status is TransferBenchmarkStatus.PASS
    assert report.passed_domains == ("mathematics", "physics")
    assert len(report.learning_proposals) == 2
    for result in report.targets:
        assert result.transfer is not None
        assert result.transfer.capability.verification.level is EvidenceLevel.UNVERIFIED
        assert not result.transfer.capability.executable
        assert result.learned is not None
        assert result.learned.capability.executable
        assert result.learned.capability.domains == (result.domain,)
        assert len(result.learned.verifier_ids) >= 2


def test_phase8_rejects_duplicate_target_domains_and_source_domain():
    engine = CrossDomainTransferBenchmarkEngine()
    source = make_source_proposal()
    with pytest.raises(TransferBenchmarkError, match="only once"):
        engine.evaluate(
            "duplicate-domain", source, UOL(),
            (target("mathematics", offset=400), target("mathematics", offset=500)),
        )
    with pytest.raises(TransferBenchmarkError, match="source domain"):
        engine.evaluate(
            "source-domain", source, UOL(),
            (target("programming", offset=600),),
            minimum_target_domains=1,
        )


def test_phase8_blocks_a_target_without_independent_verification():
    target_case = target("mathematics", offset=700)

    def no_evidence(_transfer):
        return ()

    target_case = replace(target_case, observations=no_evidence)
    report = CrossDomainTransferBenchmarkEngine().evaluate(
        "missing-target-evidence", make_source_proposal(), UOL(), (target_case,),
        minimum_target_domains=1,
    )
    assert report.status is TransferBenchmarkStatus.BLOCKED
    assert not report.learning_proposals
    assert "need at least" in report.targets[0].blocker


def test_phase8_rejects_a_counterfeit_source_handler_binding():
    source = make_source_proposal()
    other = increment_program("programming")
    other = replace(other, program_id="different-source-program")
    counterfeit = replace(source, handler=replace(source.handler, program=other))
    with pytest.raises(TransferBenchmarkError, match="handler is not bound"):
        CrossDomainTransferBenchmarkEngine().evaluate(
            "counterfeit-source", counterfeit, UOL(),
            (target("mathematics", offset=800),),
            minimum_target_domains=1,
        )


def test_phase8_rejects_transfer_that_fails_fresh_target_heldout_examples():
    good = target("mathematics", offset=900)
    bad = replace(
        good,
        held_out=(Example({"x": 906}, -999), Example({"x": 908}, 909)),
    )
    report = CrossDomainTransferBenchmarkEngine().evaluate(
        "target-heldout-negative-control", make_source_proposal(), UOL(), (bad,),
        minimum_target_domains=1,
    )
    assert report.status is TransferBenchmarkStatus.BLOCKED
    assert "held-out" in report.targets[0].blocker



def test_phase8_rejects_reused_verifier_probe_from_transfer_training_set():
    good = target("mathematics", offset=1000)

    def leaking_observations(transfer):
        # Generate valid, execution-bound verification evidence for a probe
        # deliberately reused from the target training set.
        return build_observations(
            transfer.program, transfer.capability, "mathematics",
            (1000, 1024, 1028),
        )

    leaking = replace(good, observations=leaking_observations)
    report = CrossDomainTransferBenchmarkEngine().evaluate(
        "verifier-probe-leakage",
        make_source_proposal(),
        UOL(),
        (leaking,),
        minimum_target_domains=1,
    )
    assert report.status is TransferBenchmarkStatus.BLOCKED
    assert "disjoint from transfer train/held-out" in report.targets[0].blocker
