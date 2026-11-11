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
    Call, Lit, OperationalRule, Program, RuleStatus, UOL, Var, VariableSpec, program_dict,
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


# Strategic Phase 8/9 acceptance: sealed transfer evaluation and multi-cycle expansion.
from mirror_crystal.phase8_cross_domain_intelligence import (
    CrossDomainBenchmarkCase, CrossDomainBenchmarkManifest, CrossDomainBenchmarkRunner,
)
from mirror_crystal.phase9_continuous_expansion import (
    ContinuousExpansionEngine, ExpansionCycleEvidence,
)


def phase8_manifest(name, offset=0):
    programming_source = source_proposal()
    mathematics_source = source_proposal()
    math_program = replace(
        mathematics_source.program, program_id="program:increment:mathematics",
        domain="mathematics",
    )
    math_capability = replace(
        mathematics_source.capability,
        capability_id="learned:increment:mathematics", domains=("mathematics",),
        provenance=RepresentationProvenance(
            source="mirror_verified_uol_program_learning", source_ref=math_program.fingerprint,
        ),
    )
    mathematics_source = replace(
        mathematics_source, program=math_program, capability=math_capability,
        handler=UOLProgramHandler(math_program, ()),
    )
    cases = []
    for index, start_value in enumerate((2, 5)):
        train = tuple(Example({"x": x + offset + index}, x + offset + index + 1)
                      for x in (start_value, start_value + 2, start_value + 4))
        heldout = tuple(Example({"x": x + offset + index}, x + offset + index + 1)
                        for x in (start_value + 20, start_value + 30))
        cases.append(CrossDomainBenchmarkCase(
            f"{name}-positive-math-{index}", programming_source, UOL(), "mathematics",
            train, heldout, target_math_postconditions(),
        ))
    train = tuple(Example({"x": x + offset}, x + offset + 1) for x in (3, 6, 10))
    heldout = tuple(Example({"x": x + offset}, x + offset + 1) for x in (24, 28))
    cases.append(CrossDomainBenchmarkCase(
        f"{name}-positive-programming", mathematics_source, UOL(), "programming",
        train, heldout, NativeDomainVerifierSuite.postconditions("programming"),
    ))
    cases.append(CrossDomainBenchmarkCase(
        f"{name}-negative-same-domain", programming_source, UOL(), "programming",
        (Example({"x": 1}, 2), Example({"x": 2}, 3), Example({"x": 3}, 4)),
        (Example({"x": 5}, 6), Example({"x": 6}, 7)),
        (), expect_transfer=False, expected_rejection="must differ",
    ))
    return CrossDomainBenchmarkManifest(name, tuple(cases))


def make_programming_observations(proposal):
    suite = NativeDomainVerifierSuite()
    engine = DomainVerificationEngine(suite.registry)
    observations = []
    for index, value in enumerate((3, 6, 10)):
        envelope = {
            "schema_version": 1, "domain": "programming",
            "payload": {"programming": {
                "program": program_dict(proposal.program), "output_name": "result",
            }},
        }
        task_id = f"target-program-task-{index}"
        step = StepExecutionResult(
            f"target-program-step-{index}", task_id, proposal.capability.capability_id, True,
            {"x": value, "__mirror_verification__": envelope},
            {"result": UOL().execute(proposal.program, {"x": value})},
        )
        execution = PlanExecutionReport(
            f"target-program-plan-{index}", task_id, ExecutionStatus.SUCCEEDED, (step,),
        )
        verification = engine.verify_candidate(
            execution, proposal.capability,
            requirements={proposal.capability.capability_id: suite.requirement("programming")},
            required_checks={proposal.capability.capability_id: suite.required_checks("programming")},
        )
        observations.append(VerifiedProgramObservation(
            task_id, proposal.capability, execution, verification,
        ))
    return tuple(observations)


def run_phase8_manifest(name, offset=0, with_verifiers=True):
    manifest = phase8_manifest(name, offset)
    verifiers = {}
    if with_verifiers:
        for case in manifest.cases:
            if case.expect_transfer and case.target_domain == "mathematics":
                verifiers[case.case_id] = lambda proposal: make_math_observations(
                    proposal.program, proposal.capability,
                )
            elif case.expect_transfer:
                verifiers[case.case_id] = make_programming_observations
    return CrossDomainBenchmarkRunner().run(manifest, target_verifiers=verifiers)

def test_phase8_benchmark_requires_target_verification_and_passes_negative_control():
    report = run_phase8_manifest("phase8-accepted")
    assert report.passed
    assert len(report.learned_proposals) == 3
    assert sum(item.passed and not item.transferred for item in report.results) == 1
    assert all(item.capability.executable for item in report.learned_proposals)


def test_phase8_benchmark_fails_closed_without_target_verifier():
    report = run_phase8_manifest("phase8-no-verifier", with_verifiers=False)
    assert not report.passed
    assert all(not item.passed for item in report.results if item.transferred)


def test_phase9_requires_three_distinct_benchmark_cycles_and_unseen_tasks():
    reports = tuple(
        run_phase8_manifest(f"phase9-cycle-{index}", offset=index * 50)
        for index in range(3)
    )
    assert all(item.passed for item in reports)
    cycles = tuple(
        ExpansionCycleEvidence(
            f"cycle-{index}", report,
            tuple(f"task-{index}-{part}" for part in range(3)),
            (f"unseen-{index}-a", f"unseen-{index}-b"),
            ("reviewer-a", "reviewer-b"), True, "f" * 64,
        )
        for index, report in enumerate(reports)
    )
    proposal = ContinuousExpansionEngine().propose(cycles)
    assert proposal.requires_explicit_promotion
    assert len(proposal.cycle_ids) == 3
    assert len(set(proposal.benchmark_fingerprints)) == 3
    assert proposal.target_domains == ("mathematics",)
    assert len(proposal.learned_proposals) >= 3


def test_phase9_rejects_reused_tasks_and_baseline_drift():
    import pytest
    reports = tuple(
        run_phase8_manifest(f"phase9-invalid-{index}", offset=index * 70)
        for index in range(3)
    )
    duplicate_tasks = tuple(
        ExpansionCycleEvidence(
            f"duplicate-{index}", report, ("same", "task-b", "task-c"),
            (f"unseen-{index}",), ("reviewer-a", "reviewer-b"), True, "f" * 64,
        )
        for index, report in enumerate(reports)
    )
    with pytest.raises(ValueError, match="disjoint"):
        ContinuousExpansionEngine().propose(duplicate_tasks)

    drifted = (
        ExpansionCycleEvidence("baseline-a", reports[0], ("a1", "a2", "a3"), ("ua",), ("reviewer-a", "reviewer-b"), True, "a" * 64),
        ExpansionCycleEvidence("baseline-b", reports[1], ("b1", "b2", "b3"), ("ub",), ("reviewer-a", "reviewer-b"), True, "b" * 64),
        ExpansionCycleEvidence("baseline-c", reports[2], ("c1", "c2", "c3"), ("uc",), ("reviewer-a", "reviewer-b"), True, "a" * 64),
    )
    with pytest.raises(ValueError, match="same frozen baseline"):
        ContinuousExpansionEngine().propose(drifted)
