import compileall
import importlib


def test_backend_import_surface():
    mod = importlib.import_module("mirror7_backend")
    for name in ("BackendSession", "CheckpointStore", "ActionGateway"):
        assert hasattr(mod, name)


def test_backend_tree_compiles():
    assert compileall.compile_dir("mirror7_backend", quiet=1)


def test_research_runtime_still_imports():
    mod = importlib.import_module(
        "phase140_unified_cognitive_runtime.mirror7_phase140"
    )
    assert hasattr(mod, "UnifiedCognitiveRuntime")
