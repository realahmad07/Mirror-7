from phase140_unified_cognitive_runtime.mirror7_phase140 import UnifiedCognitiveRuntime


def test_semantic_state_is_published_to_workspace():
    runtime = UnifiedCognitiveRuntime()
    report = runtime.step("Explain variance under 10 words")
    assert report.semantic_state is not None
    published = runtime.workspace.latest("semantic_state")
    assert published is not None
    payload = published.value
    assert payload["operations"] == ["explain"]
    assert "variance" in payload["entities"]
    assert "numeric" in payload["constraints"]
    assert "upper_bound" in payload["constraints"]


def test_numeric_runtime_path_does_not_receive_text_semantics():
    runtime = UnifiedCognitiveRuntime()
    runtime.step({"x": 1})
    assert runtime.semantic_state is None
    assert runtime.workspace.latest("semantic_state") is None
