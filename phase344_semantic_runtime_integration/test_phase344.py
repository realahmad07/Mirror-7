from phase140_unified_cognitive_runtime.mirror7_phase140 import UnifiedCognitiveRuntime
from phase343_semantic_state import SemanticState


def test_runtime_emits_semantic_state_for_raw_text():
    runtime = UnifiedCognitiveRuntime()
    report = runtime.step("Explain variance")
    assert isinstance(report.semantic_state, SemanticState)
    assert report.semantic_state.operations == ("explain",)
    assert "variance" in report.semantic_state.entities


def test_runtime_carries_semantic_context_across_turns():
    runtime = UnifiedCognitiveRuntime()
    first = runtime.step("Explain quantum mechanics")
    second = runtime.step("Explain this again")
    assert first.semantic_state is not None
    assert second.semantic_state is not None
    assert "anaphoric_reference" in second.semantic_state.context
    assert "carryover" in second.semantic_state.context
    assert "inherits_entity" in second.semantic_state.context


def test_runtime_preserves_existing_numeric_state_contract():
    runtime = UnifiedCognitiveRuntime()
    report = runtime.step({"x": 1})
    assert report.semantic_state is None
    assert report.state == {"x": 1}


def test_runtime_operation_change_updates_semantic_state():
    runtime = UnifiedCognitiveRuntime()
    explain = runtime.step("Explain variance").semantic_state
    debug = runtime.step("Debug variance").semantic_state
    assert explain is not None and debug is not None
    assert explain.operations != debug.operations
    assert explain.entities == debug.entities
    assert explain.desired_output != debug.desired_output
