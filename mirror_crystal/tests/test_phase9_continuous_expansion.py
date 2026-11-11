from __future__ import annotations

from dataclasses import replace
import sqlite3

import pytest

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_model import CapabilityRegistry, CapabilitySpec, PortContract
from mirror_crystal.capability_synthesizer import Example
from mirror_crystal.domain_verification import DomainVerificationEngine
from mirror_crystal.execution import (
    ExecutionStatus, NativeOperationRegistry, PlanExecutionReport, StepExecutionResult,
)
from mirror_crystal.native_verifiers import NativeDomainVerifierSuite
from mirror_crystal.operational_language import Call, Lit, Program, UOL, Var, VariableSpec
from mirror_crystal.universal_representation import RepresentationProvenance
from mirror_crystal.verified_program_learning import (
    UOLProgramHandler, UOLProgramLearningProposal, VerifiedProgramObservation,
)
from mirror_crystal.phase8_cross_domain_acceptance import (
    CrossDomainTransferBenchmarkEngine, TransferBenchmarkStatus, TransferTargetBenchmark,
)
from mirror_crystal.phase9_continuous_expansion import (
    ContinuousExpansionEngine, ContinuousExpansionError, ExpansionDecision,
    ExpansionLedger, FrozenRegressionCase,
)


def make_source():
    program = Program(
        "phase9:increment", "increment",
        (VariableSpec("x", "integer"),),
        Call("add", (Var("x"), Lit(1))), "programming",
    )
    capability = CapabilitySpec(
        capability_id="phase9:source:increment",
        name="Verified source increment",
        description="Source operation used by the strategic Phase 9 acceptance test",
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
            verifier="independent source verifiers: checker-a, checker-b",
        ),
    )
    return UOLProgramLearningProposal(
        capability=capability,
        program=program,
        handler=UOLProgramHandler(program, ()),
        observation_task_ids=("source-task-a", "source-task-b", "source-task-c"),
        execution_fingerprints=("a" * 64, "b" * 64, "c" * 64),
        verifier_ids=("checker-a", "checker-b"),
        evidence_fingerprints=("d" * 64, "e" * 64, "f" * 64),
        minimum_successes=3,
        minimum_verifiers=2,
    )


def make_report():
    source = make_source()
    suite = NativeDomainVerifierSuite()
    target_domain = "mathematics"
    program = Program(
        source.program.program_id + ":math",
        source.program.name,
        source.program.inputs,
        source.program.body,
        target_domain,
    )
    training_values = (201, 203, 205)
    held_values = (207, 209)
    observation_values = (221, 223, 225)

    def observe(transfer):
        verifier_suite = NativeDomainVerifierSuite()
        verifier = DomainVerificationEngine(verifier_suite.registry)
        observations = []
        for index, value in enumerate(observation_values):
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
            envelope = {"schema_version": 1, "domain": target_domain, "payload": payload}
            output = UOL().execute(transfer.program, {"x": value})
            step = StepExecutionResult(
                f"phase9-step-{index}", f"phase9-task-{index}",
                transfer.capability.capability_id, True,
                {"x": value, "__mirror_verification__": envelope},
                {"result": output},
            )
            execution = PlanExecutionReport(
                f"phase9-plan-{index}", f"phase9-task-{index}",
                ExecutionStatus.SUCCEEDED, (step,),
            )
            verification = verifier.verify_candidate(
                execution,
                transfer.capability,
                requirements={transfer.capability.capability_id: verifier_suite.requirement(target_domain)},
                required_checks={transfer.capability.capability_id: verifier_suite.required_checks(target_domain)},
            )
            observations.append(VerifiedProgramObservation(
                f"phase9-task-{index}", transfer.capability, execution, verification,
            ))
        return tuple(observations)

    target = TransferTargetBenchmark(
        target_domain,
        tuple(Example({"x": value}, value + 1) for value in training_values),
        tuple(Example({"x": value}, value + 1) for value in held_values),
        suite.postconditions(target_domain),
        observe,
    )
    return CrossDomainTransferBenchmarkEngine().evaluate(
        "phase9-transfer-benchmark-v1", source, UOL(), (target,),
        minimum_target_domains=1,
    )


def regression_cases():
    return (
        FrozenRegressionCase(
            "legacy-increment-case-1", "legacy:increment", {"x": 10}, 11,
        ),
        FrozenRegressionCase(
            "legacy-increment-case-2", "legacy:increment", {"x": 49}, 50,
        ),
    )


def baseline_runner(capability_id, inputs):
    assert capability_id == "legacy:increment"
    return inputs["x"] + 1


def test_phase9_accepts_novel_epoch_after_all_frozen_regressions_and_requires_explicit_promotion():
    benchmark = make_report()
    assert benchmark.status is TransferBenchmarkStatus.PASS
    ledger = ExpansionLedger()
    engine = ContinuousExpansionEngine(ledger)
    result = engine.run_epoch(
        "epoch-0001", benchmark, regression_cases(), baseline_runner,
    )

    assert result.accepted
    assert result.decision is ExpansionDecision.ACCEPTED
    assert result.checked_regressions == 2
    assert len(result.proposals) == 1
    assert ledger.verify_chain()
    assert result.ledger_entry.sequence == 1
    assert result.ledger_entry.previous_digest == ledger.GENESIS

    proposal = result.proposals[0]
    capabilities = CapabilityRegistry()
    operations = NativeOperationRegistry()
    assert capabilities.get(proposal.capability.capability_id) is None
    promoted = engine.promote(result, "mathematics", capabilities, operations)
    assert promoted == proposal.capability.capability_id
    assert capabilities.get(promoted) == proposal.capability
    assert operations.get(promoted)({"x": 30}) == {"result": 31}


def test_phase9_rejects_epoch_when_any_frozen_regression_fails():
    ledger = ExpansionLedger()
    engine = ContinuousExpansionEngine(ledger)
    result = engine.run_epoch(
        "epoch-regression-failure",
        make_report(),
        regression_cases(),
        lambda _capability_id, inputs: inputs["x"] - 1,
    )

    assert not result.accepted
    assert not result.proposals
    assert result.decision is ExpansionDecision.REJECTED
    assert any("frozen regression failed" in item for item in result.blockers)
    assert ledger.entries()[0].decision is ExpansionDecision.REJECTED
    with pytest.raises(ContinuousExpansionError, match="cannot be promoted"):
        engine.promote(result, "mathematics", CapabilityRegistry(), NativeOperationRegistry())


def test_phase9_rejects_reused_task_evidence_across_epochs():
    ledger = ExpansionLedger()
    engine = ContinuousExpansionEngine(ledger)
    first = engine.run_epoch(
        "epoch-first", make_report(), regression_cases(), baseline_runner,
    )
    assert first.accepted

    second = engine.run_epoch(
        "epoch-second", make_report(), regression_cases(), baseline_runner,
    )
    assert not second.accepted
    assert any("already consumed" in item for item in second.blockers)
    assert second.ledger_entry.sequence == 2
    assert ledger.verify_chain()


def test_phase9_requires_unique_epoch_ids_and_nonempty_frozen_regression():
    ledger = ExpansionLedger()
    engine = ContinuousExpansionEngine(ledger)
    first = engine.run_epoch(
        "epoch-unique", make_report(), regression_cases(), baseline_runner,
    )
    assert first.accepted
    with pytest.raises(ContinuousExpansionError, match="already exists"):
        engine.run_epoch(
            "epoch-unique", make_report(), regression_cases(), baseline_runner,
        )
    with pytest.raises(ContinuousExpansionError, match="requires 1"):
        engine.run_epoch(
            "epoch-no-regression", make_report(), (), baseline_runner,
        )


def test_phase9_ledger_is_append_only_and_detects_out_of_band_mutation():
    ledger = ExpansionLedger()
    result = ContinuousExpansionEngine(ledger).run_epoch(
        "epoch-ledger", make_report(), regression_cases(), baseline_runner,
    )
    assert result.accepted
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger._connection.execute(
            "UPDATE expansion_epochs SET decision='rejected' WHERE epoch_id='epoch-ledger'"
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger._connection.execute("DELETE FROM expansion_epochs")
    assert ledger.verify_chain()

    ledger._connection.execute("DROP TRIGGER expansion_epochs_no_update")
    ledger._connection.execute(
        "UPDATE expansion_epochs SET payload_json='{}' WHERE epoch_id='epoch-ledger'"
    )
    assert not ledger.verify_chain()


def test_phase9_baseline_runner_exception_fails_closed_and_is_recorded():
    ledger = ExpansionLedger()
    result = ContinuousExpansionEngine(ledger).run_epoch(
        "epoch-baseline-error",
        make_report(),
        regression_cases(),
        lambda _capability_id, _inputs: (_ for _ in ()).throw(RuntimeError("baseline unavailable")),
    )
    assert not result.accepted
    assert result.checked_regressions == 2
    assert any("could not run" in item for item in result.blockers)
    assert ledger.entries()[0].decision is ExpansionDecision.REJECTED
