import pytest
from .mirror7_phase241_250 import (
    InvalidActionError,
    SafetyBoundary,
    ExternalEnvironmentAdapter,
    EmbodiedAgent
)

def test_action_dispatch():
    env = ExternalEnvironmentAdapter()
    agent = EmbodiedAgent(env, allowed_actions=["move", "look"])
    
    res = agent.act("move", {"dx": 1, "dy": 0})
    assert res["status"] == "success"
    assert env.state["position"] == (1, 0)
    assert len(env.history) == 1

def test_partial_observation():
    env = ExternalEnvironmentAdapter()
    agent = EmbodiedAgent(env, allowed_actions=["move", "look"])
    
    res = agent.act("look", {})
    assert res["status"] == "success"
    assert "apple" in agent.internal_state["seen_objects"]

def test_invalid_action_rejection():
    env = ExternalEnvironmentAdapter()
    agent = EmbodiedAgent(env, allowed_actions=["move", "look"])
    
    with pytest.raises(InvalidActionError):
        agent.act("destroy_world", {})
    
    # Environment should remain untouched
    assert len(env.history) == 0

def test_deterministic_embodiment():
    env1 = ExternalEnvironmentAdapter()
    agent1 = EmbodiedAgent(env1, allowed_actions=["move", "look"])
    
    env2 = ExternalEnvironmentAdapter()
    agent2 = EmbodiedAgent(env2, allowed_actions=["move", "look"])
    
    res1 = agent1.act("move", {"dx": 2, "dy": 2})
    res2 = agent2.act("move", {"dx": 2, "dy": 2})
    
    assert res1 == res2
    assert env1.state == env2.state
    assert env1.history == env2.history
