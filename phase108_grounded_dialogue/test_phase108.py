from .mirror7_phase108 import GroundedDialogue

def test_requires_goal():
    d=GroundedDialogue(["deadline"]); assert d.clarification().startswith("What outcome")

def test_retains_explicit_context():
    d=GroundedDialogue(["deadline"]); s=d.ingest("x",goal="ship",slots={"deadline":"Friday"})
    assert s.goal=="ship" and s.slots["deadline"]=="Friday" and d.clarification() is None

def test_missing_slot_gets_targeted_question():
    d=GroundedDialogue(["platform","budget"]); d.ingest("x",goal="deploy",slots={"platform":"linux"})
    q=d.clarification(); assert "budget" in q and "platform" not in q

def test_empty_slot_is_not_treated_as_knowledge():
    d=GroundedDialogue(["budget"]); d.ingest("x",goal="deploy",slots={"budget":""}); assert d.state().unresolved==("budget",)
