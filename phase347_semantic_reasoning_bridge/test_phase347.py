from phase140_unified_cognitive_runtime.mirror7_phase140 import UnifiedCognitiveRuntime
from phase343_semantic_state import SemanticState
from phase347_semantic_reasoning_bridge import build_reasoning_context


def test_semantic_state_becomes_explicit_reasoning_context():
    runtime = UnifiedCognitiveRuntime()
    runtime.step("Explain variance under 10 words")
    item = runtime.workspace.latest("reasoning_context")
    assert item is not None
    context = item.value
    assert context["operations"] == ("explain",)
    assert "variance" in context["entities"]
    assert "upper_bound" in context["constraints"]
    assert context["desired_output"] == "explanation"


def test_reasoning_context_is_not_world_state():
    runtime = UnifiedCognitiveRuntime()
    runtime.step("Calculate 12 + 7")
    context = runtime.workspace.latest("reasoning_context").value
    world = runtime.workspace.latest("state")
    assert context["desired_output"] == "computed_result"
    assert world is not None
    assert world.value != context


def test_numeric_runtime_remains_unmodified():
    runtime = UnifiedCognitiveRuntime()
    runtime.step({"x": 1})
    assert runtime.workspace.latest("reasoning_context") is None
    assert runtime.semantic_state is None


def test_adapter_is_read_only_and_deterministic():
    state = SemanticState(
        entities=("variance",),
        operations=("explain",),
        constraints=("numeric",),
        context=(),
        desired_output="explanation",
        evidence=("operation:explain",),
    )
    a = build_reasoning_context(state)
    b = build_reasoning_context(state)
    assert a == b
    assert a["entities"] == ("variance",)
    assert a["operations"] == ("explain",)
