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
