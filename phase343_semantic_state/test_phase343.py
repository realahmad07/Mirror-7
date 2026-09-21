from __future__ import annotations

from .mirror7_phase343 import SemanticStateInducer

def test_progressive_state_structure():
    s=SemanticStateInducer()
    a=s.discover("Explain World War 1")
    b=s.discover("Explain World War 2")
    assert a.operations==("explain",)
    assert b.operations==("explain",)
    assert a.desired_output==b.desired_output
    assert a.entities!=b.entities
    assert s.state_signature(a)==s.state_signature(b)

def test_entity_operation_separation():
    s=SemanticStateInducer()
    a=s.discover("Calculate 12 + 7")
    b=s.discover("Explain 12 + 7")
    assert a.operations!=b.operations
    assert "numeric" in a.constraints and "numeric" in b.constraints

def test_context_carryover():
    s=SemanticStateInducer()
    previous=s.discover("Explain quantum mechanics")
    current=s.discover("Explain this again",previous=previous)
    assert "anaphoric_reference" in current.context
    assert "carryover" in current.context
    assert "inherits_entity" in current.context

def test_constraints_are_structural():
    s=SemanticStateInducer()
    a=s.discover("Write Python code under 20 lines")
    assert "code" in a.constraints and "upper_bound" in a.constraints and "numeric" in a.constraints

def test_schema_induction_is_support_bounded():
    s=SemanticStateInducer(min_support=2)
    s.discover_batch(["explain A","explain B","debug C"])
    schema=s.induced_schema()
    assert "explain" in schema["operations"]
    assert "debug" not in schema["operations"]

def test_unknown_input_fails_closed():
    try: SemanticStateInducer().discover("")
    except ValueError: pass
    else: raise AssertionError("empty input must fail closed")

def test_deterministic_and_seed_independent():
    for _seed in (1,2,3):
        s=SemanticStateInducer()
        a=s.discover("Compare Python and Rust")
        b=s.discover("Compare Python and Rust")
        assert a==b

def test_adversarial_operation_change_changes_state():
    s=SemanticStateInducer()
    explain=s.discover("Explain variance")
    debug=s.discover("Debug variance")
    assert explain.operations!=debug.operations
    assert explain.entities==debug.entities
    assert explain.desired_output!=debug.desired_output
