from dataclasses import replace
import json

import pytest

from mirror_crystal import (
    Blueprint,
    BlueprintKind,
    BlueprintProvenance,
    CapabilityError,
    CapabilitySpec,
    EvidenceLevel,
    EvidenceRecord,
    LearningGateReport,
    ProvenanceRecord,
    SubtaskSpec,
    TaskSpec,
    UniversalIntelligenceEngine,
    VerificationEvidence,
    VerificationLevel,
)
from mirror_crystal.universal_representation import canonical_json, stable_id


def strict_gate(engine):
    base = [
        {"input": value, "expected": value * 2}
        for value in (2, 3, 5)
    ]
    held_out = [{"input": 11, "expected": 22}]
    adversarial = [{"input": -7, "expected": -14}]
    return engine.evaluate_learning_gate(
        candidate={"operation": "double"},
        cases=base,
        seeds=(17, 29, 43),
        held_out_cases=held_out,
        adversarial_cases=adversarial,
        independent_verifier="mirror-test-independent-replay-v2",
        candidate_runner=lambda _candidate, case, _seed: case["input"] * 2,
        independent_check=lambda _candidate, case, output, _suite: output == case["expected"],
    )


def test_universal_arithmetic_is_composed_and_independently_recomputed():
    engine = UniversalIntelligenceEngine()
    result = engine.solve_prompt("What is 6 times 7?")

    assert result["status"] == "verified"
    assert result["verified"] is True
    assert result["answer"] == "42.0"
    assert result["verification"]["level"] == "formally_verified"
    assert "independent_decimal_recompute:numeric.multiply" in result["verification"]["checks_passed"]
    assert result["plan"]["steps"][0]["capability_id"] == "numeric.multiply"


def test_division_by_zero_fails_without_a_verified_answer():
    result = UniversalIntelligenceEngine().solve_prompt("What is 10 divided by 0?")
    assert result["status"] == "execution_failed"
    assert result["verified"] is False
    assert "ZeroDivisionError" in result["errors"][0]


def test_unrecognized_domains_return_capability_gaps_not_invented_solutions():
    engine = UniversalIntelligenceEngine()
    result = engine.solve_prompt("Design a biological model of a cell membrane")

    assert result["status"] == "capability_gap"
    assert result["verified"] is False
    assert result["capability_gaps"]
    assert "capability" in result["answer"].lower()


def test_typed_multistep_composition_uses_outputs_from_prior_steps():
    engine = UniversalIntelligenceEngine()
    task = TaskSpec(
        task_id="task:compose",
        request="Add two numbers, then compare the sum to a threshold.",
        goal="Compute a sum and test whether it is below a threshold.",
        domain="general",
        inputs={"left": 2, "right": 3, "threshold": 6},
        input_schema={"left": "number", "right": "number", "threshold": "number"},
        required_outputs={"result": "boolean"},
        subtasks=(
            SubtaskSpec(
                subtask_id="add",
                goal="Add the operands",
                domain="mathematics",
                capability_tags=("numeric.add",),
                capability_id="numeric.add",
                input_bindings={"left": "input:left", "right": "input:right"},
                output_schema={"result": "number"},
            ),
            SubtaskSpec(
                subtask_id="compare",
                goal="Compare sum to threshold",
                domain="general",
                capability_tags=("compare.less_than",),
                capability_id="compare.less_than",
                input_bindings={"left": "step:add.result", "right": "input:threshold"},
                output_schema={"result": "boolean"},
                dependencies=("add",),
                acceptance_criteria=({
                    "field": "compare.result",
                    "op": "equals",
                    "expected": True,
                },),
            ),
        ),
    )
    result = engine.solve(task)

    assert result["status"] == "verified"
    assert result["candidate_outputs"] == {"result": True}
    assert [step["capability_id"] for step in result["executions"]] == [
        "numeric.add", "compare.less_than"
    ]
    assert result["verification"]["level"] == "execution_tested"


def test_task_dependency_cycle_is_rejected():
    engine = UniversalIntelligenceEngine()
    task = TaskSpec(
        task_id="cycle",
        request="cycle",
        goal="cycle",
        subtasks=(
            SubtaskSpec("a", "A", dependencies=("b",)),
            SubtaskSpec("b", "B", dependencies=("a",)),
        ),
    )
    result = engine.solve(task)
    assert result["status"] == "planning_failed"
    assert result["verified"] is False


def test_unverified_capability_never_becomes_executable():
    engine = UniversalIntelligenceEngine()
    spec = CapabilitySpec(
        capability_id="unsafe.fake",
        name="Unverified fake operator",
        domains=("general",),
        input_schema={"value": "number"},
        output_schema={"result": "number"},
        tags=("unsafe.fake",),
        executor_key="unsafe.fake",
        evidence=EvidenceRecord(level=VerificationLevel.UNVERIFIED),
    )
    engine.registry.register(spec, lambda values: {"result": values["value"] + 1})
    assert engine.registry.handler("unsafe.fake") is None


def test_blueprint_knowledge_is_indexed_without_becoming_an_executor():
    engine = UniversalIntelligenceEngine()
    prompt, response = "How do cells work?", "Cells are basic units of life."
    blueprint = Blueprint(
        blueprint_id="legacy:cell",
        kind=BlueprintKind.FACT,
        domain="biology",
        intent="legacy_fact",
        prompt_pattern=prompt,
        response_template=response,
        provenance=BlueprintProvenance(
            source="test",
            content_sha256=Blueprint._digest(prompt, response),
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    assert engine.ingest_blueprints([blueprint]) == 1
    nodes = engine.registry.search_knowledge("cells", domain="biology")
    assert nodes and nodes[0].node_id == "legacy:cell"
    assert nodes[0].evidence.level == VerificationLevel.UNVERIFIED
    assert engine.registry.handler("legacy:cell") is None


def test_learning_gate_blocks_weak_evidence_and_promotes_replayable_composition():
    engine = UniversalIntelligenceEngine()
    task = TaskSpec(
        task_id="learn:sum-compare",
        request="Add then compare.",
        goal="Add left and right, then compare to threshold.",
        domain="general",
        inputs={"left": 2, "right": 3, "threshold": 6},
        input_schema={"left": "number", "right": "number", "threshold": "number"},
        required_outputs={"result": "boolean"},
        subtasks=(
            SubtaskSpec(
                "add", "Add left and right", "mathematics", ("numeric.add",),
                "numeric.add",
                {"left": "input:left", "right": "input:right"},
                {"result": "number"},
            ),
            SubtaskSpec(
                "compare", "Compare to threshold", "general", ("compare.less_than",),
                "compare.less_than",
                {"left": "step:add.result", "right": "input:threshold"},
                {"result": "boolean"},
                ("add",),
            ),
        ),
        acceptance_criteria=({
            "field": "compare.result", "op": "equals", "expected": True
        },),
    )
    result = engine.solve(task)
    assert result["verified"] is True

    weak = LearningGateReport(
        cases_generated=3,
        cases_passed=2,
        seed_count=1,
        held_out_passed=False,
        adversarial_passed=False,
        independent_verifier="",
        failures=("one regression failed",),
    )
    with pytest.raises(CapabilityError):
        engine.promote_verified_composition(task, result, weak)

    persisted = []
    learned = engine.promote_verified_composition(
        task, result, strict_gate(engine), persist=persisted.append
    )
    assert learned.evidence.independently_reproduced is True
    assert learned.capability_id in [x.capability_id for x in engine.registry.list_capabilities()]
    assert persisted == [learned]

    # The learned graph is reusable with new values, not the original training example.
    handler = engine.registry.handler(learned.capability_id)
    assert handler({"left": 4, "right": 2, "threshold": 7}) == {"result": True}
    assert handler({"left": 7, "right": 3, "threshold": 8}) == {"result": False}


def test_persisted_learned_capability_can_be_reloaded_from_blueprint():
    engine = UniversalIntelligenceEngine()
    task = TaskSpec(
        task_id="persist:sum",
        request="Add inputs",
        goal="Add two numbers",
        domain="mathematics",
        inputs={"left": 4, "right": 8},
        input_schema={"left": "number", "right": "number"},
        required_outputs={"result": "number"},
        required_capability_tags=("numeric.add",),
        subtasks=(
            SubtaskSpec(
                "sum", "Add left and right", "mathematics",
                ("numeric.add",), "numeric.add",
                {"left": "input:left", "right": "input:right"},
                {"result": "number"},
            ),
        ),
    )
    result = engine.solve(task)
    learned = engine.promote_verified_composition(task, result, strict_gate(engine))
    prompt = learned.name
    response = canonical_json(learned.to_dict())
    blueprint = Blueprint(
        blueprint_id=f"universal:{learned.capability_id}",
        kind=BlueprintKind.PROCEDURE,
        domain=learned.domains[0],
        intent="universal_capability.v1",
        prompt_pattern=prompt,
        response_template=response,
        provenance=BlueprintProvenance(
            source="test",
            content_sha256=Blueprint._digest(prompt, response),
        ),
        verification=VerificationEvidence(
            level=EvidenceLevel.INDEPENDENTLY_REPRODUCED,
            checks_passed=learned.evidence.checks_passed,
            verifier=learned.evidence.verifier,
        ),
    )

    reloaded = UniversalIntelligenceEngine()
    assert reloaded.load_learned_blueprints([blueprint]) == 1
    loaded_spec = reloaded.registry.get(learned.capability_id)
    assert loaded_spec is not None and loaded_spec.trusted
    loaded_handler = reloaded.registry.handler(learned.capability_id)
    assert loaded_handler({"left": 10, "right": 5}) == {"result": 15.0}


def test_cross_domain_transfer_requires_fresh_target_domain_gate():
    engine = UniversalIntelligenceEngine()
    proposal = engine.propose_cross_domain_transfer("logic.and", "biology")
    assert proposal.evidence.level == VerificationLevel.UNVERIFIED
    assert engine.registry.handler(proposal.capability_id) is None

    with pytest.raises(CapabilityError):
        engine.promote_cross_domain_transfer(
            proposal,
            LearningGateReport(1, 1, 1, False, False, ""),
        )

    promoted = engine.promote_cross_domain_transfer(proposal, strict_gate(engine))
    assert promoted.domains == ("biology",)
    assert promoted.evidence.independently_reproduced is True
    assert engine.registry.handler(promoted.capability_id) is not None
    assert engine.registry.handler(promoted.capability_id)({"left": True, "right": False}) == {"result": False}


def test_task_and_capability_contracts_round_trip_without_losing_types():
    task = TaskSpec(
        task_id="roundtrip",
        request="Round trip",
        goal="Return a boolean",
        domain="general",
        inputs={"left": True, "right": False},
        input_schema={"left": "boolean", "right": "boolean"},
        required_outputs={"result": "boolean"},
        required_capability_tags=("logic.and",),
    )
    restored = TaskSpec.from_dict(task.to_dict())
    assert restored.fingerprint == task.fingerprint

    engine = UniversalIntelligenceEngine()
    spec = engine.registry.get("logic.and")
    restored_spec = CapabilitySpec.from_dict(spec.to_dict())
    assert restored_spec.capability_id == spec.capability_id
    assert restored_spec.input_schema == spec.input_schema
    assert restored_spec.evidence.trusted_for_execution


def test_new_domain_adapter_uses_same_representation_planner_and_verifier():
    engine = UniversalIntelligenceEngine()

    capability = CapabilitySpec(
        capability_id="biology.sequence_count",
        name="Count biological observations",
        domains=("biology",),
        input_schema={"observations": "sequence"},
        output_schema={"count": "integer"},
        tags=("biology.sequence_count",),
        executor_key="biology.sequence_count",
        evidence=EvidenceRecord(level=VerificationLevel.UNVERIFIED),
        provenance=ProvenanceRecord(source="test-domain-adapter"),
    )

    task = TaskSpec(
        task_id="biology-count",
        request="Count observations",
        goal="Count the observations",
        domain="biology",
        inputs={"observations": ["cell-a", "cell-b", "cell-c"]},
        input_schema={"observations": "sequence"},
        required_outputs={"count": "integer"},
        subtasks=(
            SubtaskSpec(
                subtask_id="count",
                goal="Count biological observations",
                domain="biology",
                capability_tags=("biology.sequence_count",),
                capability_id="biology.sequence_count",
                input_bindings={"observations": "input:observations"},
                output_schema={"count": "integer"},
            ),
        ),
        acceptance_criteria=({
            "field": "count.count",
            "op": "equals",
            "expected": 3,
        },),
    )

    def verify(_task, _plan, _executions, _outputs):
        return True, ("biology_adapter_contract_passed",), ()

    def interpret(prompt, inputs):
        return replace(
            task,
            request=prompt,
            task_id=stable_id("biology-task", prompt),
            inputs=inputs or task.inputs,
        )

    engine.register_domain_adapter(
        "biology",
        capabilities=(
            (
                capability,
                lambda values: {"count": len(values["observations"])},
                lambda: True,
            ),
        ),
        verifier=verify,
        interpreter=interpret,
    )
    result = engine.solve_prompt("Count these cell observations", domain_hint="biology")
    assert result["status"] == "verified"
    assert result["candidate_outputs"] == {"count": 3}
    assert result["verification"]["verifier"] == "Mirror UniversalVerifier/biology"


def test_natural_language_calculator_request_uses_mirror_native_ast_and_tdd():
    result = UniversalIntelligenceEngine().solve_prompt(
        "Build a Python calculator that supports addition, subtraction, multiplication and division."
    )
    assert result["status"] == "verified"
    assert result["verified"] is True
    assert result["task"]["domain"] == "programming"
    assert result["verification"]["level"] == "execution_tested"
    source = result["candidate_outputs"]["code"]
    assert "def calculate" in source
    assert "ZeroDivisionError" in source
    assert any("tdd_assertions_passed:" in check for check in result["verification"]["checks_passed"])
    assert "program.calculator.synthesize" in [
        step["capability_id"] for step in result["executions"]
    ]


def test_generic_domain_execution_without_acceptance_criteria_is_not_verified():
    engine = UniversalIntelligenceEngine()
    task = TaskSpec(
        task_id="unverified-general",
        request="Do something",
        goal="Use a general primitive",
        domain="general",
        inputs={"left": True, "right": False},
        subtasks=(
            SubtaskSpec(
                subtask_id="and",
                goal="Conjoin booleans",
                domain="general",
                capability_tags=("logic.and",),
                capability_id="logic.and",
                input_bindings={"left": "input:left", "right": "input:right"},
                output_schema={"result": "boolean"},
            ),
        ),
    )
    result = engine.solve(task)
    assert result["status"] == "verification_failed"
    assert result["verified"] is False
    assert "generic_verification_requires_acceptance_criteria" in result["verification"]["checks_failed"]


def test_generic_domain_execution_without_acceptance_criteria_is_not_verified():
    engine = UniversalIntelligenceEngine()
    task = TaskSpec(
        task_id="unverified-general",
        request="Do something",
        goal="Use a general primitive",
        domain="general",
        inputs={"left": True, "right": False},
        subtasks=(
            SubtaskSpec(
                subtask_id="and",
                goal="Conjoin booleans",
                domain="general",
                capability_tags=("logic.and",),
                capability_id="logic.and",
                input_bindings={"left": "input:left", "right": "input:right"},
                output_schema={"result": "boolean"},
            ),
        ),
    )
    result = engine.solve(task)
    assert result["status"] == "verification_failed"
    assert result["verified"] is False
    assert "generic_verification_requires_acceptance_criteria" in result["verification"]["checks_failed"]


def test_learning_gate_report_cannot_be_forged_or_reused_by_another_engine():
    engine = UniversalIntelligenceEngine()
    valid = strict_gate(engine)
    assert valid.passed is True
    assert engine._is_learning_gate_attested(valid) is True

    forged = replace(valid, cases_generated=999, cases_passed=999)
    assert forged.passed is True
    assert engine._is_learning_gate_attested(forged) is False

    other_engine = UniversalIntelligenceEngine()
    assert other_engine._is_learning_gate_attested(valid) is False

    unsigned = LearningGateReport(
        cases_generated=9,
        cases_passed=9,
        seed_count=3,
        held_out_passed=True,
        adversarial_passed=True,
        independent_verifier="fabricated",
    )
    assert unsigned.passed is False
    assert engine._is_learning_gate_attested(unsigned) is False


def test_learning_gate_runs_all_seeds_held_out_and_adversarial_cases():
    engine = UniversalIntelligenceEngine()
    seen = []
    report = engine.evaluate_learning_gate(
        candidate="candidate",
        cases=[1, 2, 3],
        seeds=[1, 2, 3],
        held_out_cases=[4],
        adversarial_cases=[-1],
        independent_verifier="independent-test",
        candidate_runner=lambda candidate, case, seed: (seen.append((case, seed)) or case * 2),
        independent_check=lambda candidate, case, output, suite: output == case * 2,
    )
    assert report.passed is True
    assert report.cases_generated == 9
    assert report.cases_passed == 9
    assert report.seed_count == 3
    assert len(seen) == 15  # 3x3 training, 1x3 held-out, 1x3 adversarial


def test_learning_gate_requires_three_distinct_base_cases_not_three_repeated_seeds():
    engine = UniversalIntelligenceEngine()
    report = engine.evaluate_learning_gate(
        candidate="candidate",
        cases=[{"value": 2}],
        seeds=[17, 29, 43],
        held_out_cases=[{"value": 5}],
        adversarial_cases=[{"value": -1}],
        independent_verifier="independent-test",
        candidate_runner=lambda _candidate, case, _seed: case["value"],
        independent_check=lambda _candidate, case, output, _suite: output == case["value"],
    )
    assert report.cases_generated == 3
    assert report.seed_count == 3
    assert report.base_case_count == 1
    assert report.passed is False
    assert "base_suite_requires_at_least_3_distinct_cases" in report.failures


def test_failed_capability_registration_self_test_never_adds_executor():
    from mirror_crystal.universal_intelligence import CapabilityRegistry

    registry = CapabilityRegistry()
    spec = CapabilitySpec(
        capability_id="general.always_wrong",
        name="Incorrect operation",
        domains=("general",),
        input_schema={"value": "integer"},
        output_schema={"result": "integer"},
        tags=("general.always_wrong",),
        evidence=EvidenceRecord(level=VerificationLevel.UNVERIFIED),
    )
    registry.register(
        spec,
        lambda values: {"result": values["value"] + 999},
        self_test=lambda: False,
    )
    assert registry.get("general.always_wrong") is not None
    assert registry.handler("general.always_wrong") is None
    assert registry.executable() == ()


def test_natural_language_sort_request_composes_native_program_ir_and_tdd():
    result = UniversalIntelligenceEngine().solve_prompt(
        "Write a Python function to sort a list of numbers using bubble sort."
    )
    assert result["status"] == "verified"
    assert result["verified"] is True
    assert result["task"]["domain"] == "programming"
    assert "def bubble_sort" in result["candidate_outputs"]["code"]
    assert "tdd_assertions_passed:4/4" in result["verification"]["checks_passed"]
    assert result["plan"]["steps"][0]["capability_id"] == "program.sort.bubble.synthesize"


def test_retrieved_legacy_knowledge_is_exposed_as_context_not_promoted_to_capability():
    engine = UniversalIntelligenceEngine()
    prompt = "How do cells work?"
    response = "Cells are the basic structural and functional units of living organisms."
    blueprint = Blueprint(
        blueprint_id="legacy:cell_context",
        kind=BlueprintKind.FACT,
        domain="biology",
        intent="legacy_fact",
        prompt_pattern=prompt,
        response_template=response,
        provenance=BlueprintProvenance(
            source="curated-test-source",
            content_sha256=Blueprint._digest(prompt, response),
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    engine.ingest_blueprints([blueprint])
    result = engine.solve_prompt(prompt)

    assert result["status"] == "capability_gap"
    assert result["verified"] is False
    assert result["retrieved_knowledge"]
    assert result["retrieved_knowledge"][0]["node_id"] == "legacy:cell_context"
    assert result["retrieved_knowledge"][0]["verification_level"] == "unverified"
    assert engine.registry.handler("legacy:cell_context") is None
