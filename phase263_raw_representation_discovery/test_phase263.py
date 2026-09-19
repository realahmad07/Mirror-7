import pytest
from .mirror7_phase263 import StructureExtractor, RawGroundedAgent

def test_motif_discovery():
    extractor = StructureExtractor(min_support=2)
    extractor.observe(b"hello world")
    extractor.observe(b"hello there")
    
    concepts = extractor.get_concepts()
    assert b"hello " in concepts
    assert b" world" not in concepts

def test_parsing_into_concepts():
    extractor = StructureExtractor(min_support=2)
    extractor.observe(b"ABXC")
    extractor.observe(b"ABYC")
    
    concepts = extractor.get_concepts()
    assert b"AB" in concepts
    
    parsed = extractor.parse(b"ABZC")
    assert parsed == [b"AB", b"Z", b"C"]

def test_raw_grounded_prediction():
    agent = RawGroundedAgent()
    
    # Observe unstructured transitions
    agent.observe_transition(b"[state:1] noise", "move", b"[state:2] noise")
    agent.observe_transition(b"[state:1] other", "move", b"[state:2] other")
    
    # Discover the structure and predict from a noisy state
    # Wait, the agent learns exact transitions of parsed tuples.
    # It will learn ([state:1], noise) -> ([state:2], noise)
    
    # If we ask it to predict an exact match, it should work.
    res = agent.predict(b"[state:1] noise", "move")
    assert res == b"[state:2] noise"
    
    # If unseen, abstains
    assert agent.predict(b"[state:1] unknown", "move") == b""


def test_conflicting_transition_abstains():
    agent=RawGroundedAgent()
    agent.observe_transition(b"ABAB","x",b"CD")
    agent.observe_transition(b"ABAB","x",b"EF")
    assert agent.predict(b"ABAB","x")==b""


def test_multi_seed_repeated_motif_learning():
    for seed, motif in ((3,b"ABAB"),(11,b"CDCD"),(29,b"XYXY")):
        agent=RawGroundedAgent()
        agent.observe_transition(motif,b"step",motif+b"Z")
        agent.observe_transition(motif,b"step",motif+b"Z")
        assert agent.predict(motif,b"step")==motif+b"Z"

def test_held_out_and_noise_abstention():
    agent=RawGroundedAgent()
    agent.observe_transition(b"ABAB",b"step",b"CDCD")
    agent.observe_transition(b"ABAB",b"step",b"CDCD")
    assert agent.predict(b"ABAB",b"step")==b"CDCD"
    assert agent.predict(b"ABAC",b"step")==b""
