from __future__ import annotations

from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_composition import CapabilityCompositionPlanner
from mirror_crystal.capability_learning import (
    CapabilityLearningEngine, CapabilityLearningObservation,
)
from mirror_crystal.capability_matching import CapabilityMatcher
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.execution import CapabilityExecutionEngine, NativeOperationRegistry
from mirror_crystal.knowledge_foundation import KnowledgeRecord
from mirror_crystal.task_understanding import TaskUnderstandingEngine
from mirror_crystal.universal_representation import RepresentationProvenance
from mirror_crystal.verification import (
    IndependentCheck, IndependentVerificationEngine, IndependentVerifierRegistry,
    VerificationStatus,
)


def transform_value(inputs):
    return {"intermediate": inputs["input_value"] * 2}


def verify_value(inputs):
    return {"result": inputs["intermediate"] + 1}


def transform_positive(context):
    return context.step.outputs["intermediate"] > 0


def transform_exact(context):
    return context.step.outputs["intermediate"] == context.step.inputs["input_value"] * 2


def result_positive(context):
    return context.step.outputs["result"] > 0


def result_exact(context):
    return context.step.outputs["result"] == context.step.inputs["intermediate"] + 1


def procedure(knowledge_id, intent, inputs, outputs, postcondition):
    return KnowledgeRecord(
        knowledge_id=knowledge_id,
        kind="procedure",
        domain="mathematics",
        subject=intent,
        predicate="native_procedure",
        object_value={"operation": intent},
        statement=f"Native procedure {intent}.",
        provenance=RepresentationProvenance(
            source="phase1_native_procedure_catalog",
            source_ref=f"mirror://procedure/{knowledge_id}",
        ),
        verification=VerificationEvidence(
            level=EvidenceLevel.EXECUTION_TESTED,
            checks_passed=("native-handler-unit-test",),
            verifier="native-handler-test-suite",
        ),
        capability_contract={
            "domains": ["mathematics"],
            "intents": [intent],
            "inputs": [{"name": name, "value_type": kind} for name, kind in inputs],
            "outputs": [{"name": name, "value_type": kind} for name, kind in outputs],
            "postconditions": [postcondition],
        },
    )


def test_complete_phase1_to_phase9_pipeline_from_knowledge_to_learned_capability(tmp_path):
    vault = DiskCrystalVault(tmp_path / "universal-intelligence.dcrystal")
    try:
        # Phase 1: canonical, provenance-carrying knowledge items with explicit
        # typed procedure contracts. Assumptions/unverified records cannot
        # masquerade as assertable facts or executable procedures.
        builder_record = procedure(
            "double-input", "build",
            (("input_value", "integer"),),
            (("intermediate", "integer"),),
            "intermediate_created",
        )
        verifier_record = procedure(
            "increment-result", "verify",
            (("intermediate", "integer"),),
            (("result", "integer"),),
            "result_verified",
        )
        vault.store_knowledge_record(builder_record)
        vault.store_knowledge_record(verifier_record)
        builder = vault.get_knowledge_record("double-input").to_capability_spec()
        verifier = vault.get_knowledge_record("increment-result").to_capability_spec()
        capabilities = (builder, verifier)
        for item in capabilities:
            vault.store_capability(item)

        # Phase 2: knowledge and capabilities coexist in one universal graph.
        graph = vault.load_representation_graph(domain="mathematics", node_limit=100)
        assert len(graph.nodes) >= 4
        assert all(node.domain == "mathematics" for node in graph.nodes)

        # Phases 3-6: understand/decompose, match typed capabilities, and compose
        # a dependency-ordered plan with an intermediate output-to-input binding.
        task_engine = TaskUnderstandingEngine()
        matcher = CapabilityMatcher()
        composer = CapabilityCompositionPlanner()
        registry = NativeOperationRegistry()
        registry.register(builder.capability_id, transform_value)
        registry.register(verifier.capability_id, verify_value)
        check_registry = IndependentVerifierRegistry()
        check_registry.register(
            IndependentCheck("transform-positive", "checker-alpha", transform_positive,
                             ("intermediate_created",)),
            independent=True,
        )
        check_registry.register(
            IndependentCheck("transform-invariant", "checker-beta", transform_exact,
                             ("intermediate_created",)),
            independent=True,
        )
        check_registry.register(
            IndependentCheck("result-positive", "checker-alpha", result_positive,
                             ("result_verified",)),
            independent=True,
        )
        check_registry.register(
            IndependentCheck("result-invariant", "checker-beta", result_exact,
                             ("result_verified",)),
            independent=True,
        )
        verifier_engine = IndependentVerificationEngine(check_registry)
        observations = []

        # Three independent problem instances provide the replication gate
        # required before Phase 9 can generalize the composition.
        for value in (2, 5, 11):
            task = task_engine.interpret(
                "Build a result from input_value, then verify the result.",
                domain_hint="mathematics",
                known_inputs={"input_value": value},
            )
            assert len(task.subtasks) == 2
            match_report = matcher.match(task, capabilities)
            assert not match_report.unmatched_subtasks
            plan = composer.compose(task, capabilities, match_report=match_report)
            assert plan.ready and len(plan.steps) == 2
            assert plan.steps[1].dependency_step_ids == (plan.steps[0].step_id,)
            assert plan.steps[1].input_bindings[0].source == "step_output"

            # Phase 7: only the host-registered handlers execute.
            execution = CapabilityExecutionEngine(registry).execute(
                plan, capabilities, task_inputs=task.known_inputs,
                task_context=task.context,
            )
            assert execution.succeeded
            assert execution.steps[-1].outputs["result"] == value * 2 + 1

            # Phase 8: two separately registered checkers independently verify
            # every step and every declared postcondition.
            verification = verifier_engine.verify(
                execution, capabilities,
                required_checks={
                    builder.capability_id: ("transform-positive", "transform-invariant"),
                    verifier.capability_id: ("result-positive", "result-invariant"),
                },
            )
            assert verification.status is VerificationStatus.VERIFIED
            assert set(verification.verifier_ids) == {"checker-alpha", "checker-beta"}
            observations.append(CapabilityLearningObservation(
                task, plan, execution, verification,
            ))

        # Phase 9: repeated, structurally identical and independently verified
        # compositions become a proposal; promotion remains an explicit action.
        proposal = CapabilityLearningEngine().propose(observations, capabilities)
        assert proposal.requires_explicit_promotion
        learned = proposal.capability
        assert learned.executable
        assert learned.verification.level is EvidenceLevel.INDEPENDENTLY_REPRODUCED
        assert [port.name for port in learned.inputs] == ["input_value"]
        assert [port.name for port in learned.outputs] == ["result"]
        assert CapabilityLearningEngine.persist(vault, proposal) == learned.capability_id
        assert vault.get_capability(learned.capability_id) == learned
        assert vault.get_knowledge_record(builder_record.knowledge_id) == builder_record
    finally:
        vault.close()
