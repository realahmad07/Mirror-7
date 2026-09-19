from .mirror7_phase312 import build_default_adapters, build_default_frontier

def test_default_registry_has_four_capabilities():
    assert build_default_adapters().names==("compositional_language","planning","sequence_extrapolation","symbolic_reasoning")

def test_default_frontier_has_four_targets():
    assert {t.name for t in build_default_frontier().targets}=={"sequence_extrapolation","planning","compositional_language","symbolic_reasoning"}

def test_frontier_has_unfinished_targets():
    assert build_default_frontier().top_gap() is not None

def test_adapter_names_match_targets():
    assert set(build_default_adapters().names)=={t.name for t in build_default_frontier().targets}
