import pytest
from .mirror7_phase251_260 import (
    SafetyViolationError,
    SafetyValidator,
    TrajectoryReproducer,
    ValidatedAgent
)

def test_adversarial_rejection():
    agent = ValidatedAgent(["move", "read"], seed=42)
    # Valid
    assert agent.act("move", {"x": 1}, "ok") is True
    # Invalid action
    assert agent.act("drop_tables", {}, "ok") is False
    # Invalid args (complex object representing a payload)
    assert agent.act("move", {"x": {"payload": "eval()"}}, "ok") is False

def test_strict_determinism():
    agent1 = ValidatedAgent(["move", "read"], seed=10)
    agent2 = ValidatedAgent(["move", "read"], seed=10)
    
    agent1.act("move", {"x": 1}, "obs1")
    agent1.act("read", {"file": "a"}, "obs2")
    
    agent2.act("move", {"x": 1}, "obs1")
    agent2.act("read", {"file": "a"}, "obs2")
    
    assert agent1.reproducer.get_fingerprint() == agent2.reproducer.get_fingerprint()

def test_memory_poisoning_defense():
    agent = ValidatedAgent(["move", "read"], seed=10)
    fp1 = agent.reproducer.get_fingerprint()
    
    # Attempting to act with a poisonous action fails closed and DOES NOT record transition
    agent.act("poison", {}, "bad")
    
    fp2 = agent.reproducer.get_fingerprint()
    assert fp1 == fp2 # State unchanged
