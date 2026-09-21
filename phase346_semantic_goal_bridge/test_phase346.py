from phase140_unified_cognitive_runtime.mirror7_phase140 import UnifiedCognitiveRuntime


def test_raw_text_can_induce_a_goal_when_none_is_supplied():
    runtime = UnifiedCognitiveRuntime()
    report = runtime.step("Explain variance")
    assert report.semantic_state is not None
    assert report.goal == "explanation:variance"


def test_explicit_goal_remains_authoritative():
    runtime = UnifiedCognitiveRuntime()
    report = runtime.step("Explain variance", goal="custom-goal")
    assert report.goal == "custom-goal"


def test_semantic_goal_is_not_created_for_unrecognized_operation():
    runtime = UnifiedCognitiveRuntime()
    report = runtime.step("variance")
    assert report.semantic_state is not None
    assert report.semantic_state.desired_output is None
    assert report.goal is None
