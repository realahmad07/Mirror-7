from __future__ import annotations

import pytest

from mirror7_backend.cognitive_orchestrator import UnifiedCognitiveOrchestrator
from mirror7_backend.universal_intelligence import UniversalIntelligencePipeline


class EmptyVault:
    def list_knowledge_records(self, *, domain=None, limit=500):
        return ()

    def list_capabilities(self, *, domain=None, limit=500):
        return ()


class MemoryStub:
    def resolve_memory_context(self, session_id, prompt):
        return None

    def auto_extract_and_remember(self, session_id, prompt):
        raise AssertionError("questions must not be auto-extracted as memory")


class ReasoningStub:
    def solve(self, prompt):
        return {"ok": False}


class CodeStub:
    def run_tdd_cycle(self, *args, **kwargs):
        raise AssertionError("non-code prompt must not invoke code verification")


def pipeline():
    return UniversalIntelligencePipeline(vault_provider=EmptyVault)


def test_analyze_returns_task_contract_and_blocked_composition_without_capabilities():
    result = pipeline().analyze(
        "Build a calculator",
        domain_hint="mathematics",
        known_inputs={"left": 4, "right": 8},
    )

    assert result["status"] == "blocked"
    assert result["task"]["intent"] == "build"
    assert result["task"]["domain"] == "mathematics"
    assert result["knowledge_to_uol"]["rules_imported"] == 0
    assert result["capability_matching"]["match_count"] == 0
    assert result["composition_plan"]["status"] == "blocked"
    assert result["execution"]["attempted"] is False


def test_synthesis_returns_candidate_but_never_executes_or_promotes_it():
    result = pipeline().synthesize({
        "name": "increment",
        "domain": "mathematics",
        "inputs": [{"name": "x", "value_type": "integer"}],
        "output_type": "integer",
        "training": [
            {"inputs": {"x": 1}, "expected": 2},
            {"inputs": {"x": 2}, "expected": 3},
        ],
        "validation": [
            {"inputs": {"x": 3}, "expected": 4},
        ],
        "max_depth": 1,
        "candidate_budget": 2000,
    })

    assert result["status"] == "candidate_unverified"
    assert result["program"]["domain"] == "mathematics"
    assert result["candidate_capability"]["verification"]["level"] == "unverified"
    assert result["verification"] == "UNVERIFIED"
    assert result["execution_allowed"] is False
    assert result["promotion_allowed"] is False
    assert result["validation_passed"] == 1


def test_synthesis_rejects_overlapping_training_and_heldout_examples():
    with pytest.raises(ValueError, match="disjoint"):
        pipeline().synthesize({
            "name": "increment",
            "domain": "mathematics",
            "inputs": [{"name": "x", "value_type": "integer"}],
            "output_type": "integer",
            "training": [
                {"inputs": {"x": 1}, "expected": 2},
                {"inputs": {"x": 2}, "expected": 3},
            ],
            "validation": [{"inputs": {"x": 2}, "expected": 3}],
        })


def test_default_cognitive_qa_includes_universal_intelligence_trace():
    universal = pipeline()
    orchestrator = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: None,
        memory_provider=MemoryStub,
        reasoning_provider=ReasoningStub,
        code_provider=CodeStub,
        universal_intelligence_provider=lambda: universal,
    )

    result = orchestrator.ask("Explain how Mirror's universal architecture works")

    assert result["universal_intelligence"]["task"]["intent"] == "explain"
    assert "universal_intelligence" in result["sources_consulted"]
    assert result["universal_intelligence"]["execution"]["attempted"] is False


def _double_handler(inputs):
    return {"result": inputs["value"] * 2}


def _independently_verify_double(context):
    return context.step.outputs["result"] == context.step.inputs["value"] * 2


class ExecutableVault(EmptyVault):
    def __init__(self, capability):
        self.capability = capability

    def list_capabilities(self, *, domain=None, limit=500):
        return (self.capability,)


def test_normal_qna_executes_only_with_independent_verification():
    from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
    from mirror_crystal.capability_model import CapabilitySpec, PortContract
    from mirror_crystal.execution import CapabilityExecutionEngine, NativeOperationRegistry
    from mirror_crystal.universal_representation import RepresentationProvenance
    from mirror_crystal.verification import (
        IndependentCheck,
        IndependentVerificationEngine,
        IndependentVerifierRegistry,
    )

    capability = CapabilitySpec(
        capability_id="double-value",
        name="Double a value",
        description="Build a doubled value from one integer input",
        domains=("general",),
        intents=("build",),
        inputs=(PortContract("value", "integer"),),
        outputs=(PortContract("result", "integer"),),
        postconditions=("result_is_double",),
        provenance=RepresentationProvenance(source="backend-qna-test"),
        verification=VerificationEvidence(
            level=EvidenceLevel.EXECUTION_TESTED,
            checks_passed=("handler-unit-test",),
            verifier="handler-tests",
        ),
    )
    operations = NativeOperationRegistry()
    operations.register("double-value", _double_handler)
    verifiers = IndependentVerifierRegistry()
    verifiers.register(
        IndependentCheck(
            "double-result",
            "independent-checker",
            _independently_verify_double,
            ("result_is_double",),
        ),
        independent=True,
    )
    pipeline = UniversalIntelligencePipeline(
        vault_provider=lambda: ExecutableVault(capability),
        execution_engine=CapabilityExecutionEngine(operations),
        verification_engine=IndependentVerificationEngine(verifiers),
        required_verification_checks={"double-value": ("double-result",)},
    )
    orchestrator = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: None,
        memory_provider=MemoryStub,
        reasoning_provider=ReasoningStub,
        code_provider=CodeStub,
        universal_intelligence_provider=lambda: pipeline,
    )

    result = orchestrator.ask(
        "Build a double from value",
        universal_options={
            "domain_hint": "general",
            "known_inputs": {"value": 6},
        },
    )

    assert result["task_execution_verified"] is True
    assert result["task_result_verified"] is True
    assert result["task_execution"]["status"] == "succeeded"
    assert result["task_execution"]["verification"]["verified"] is True
    assert result["task_execution"]["verified_outputs"] == {"result": 12}
    assert "12" in result["answer"]
