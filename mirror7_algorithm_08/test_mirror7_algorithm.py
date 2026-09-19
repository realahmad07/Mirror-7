import time
from mirror7_algorithm_08.mirror7_algorithm import KnowledgeConsolidationEngine, Experience, Knowledge

def basic_test():
    print("--- basic_test ---")
    engine = KnowledgeConsolidationEngine(promotion_threshold=3)
    for i in range(10):
        engine.add_experience(Experience({"fact": "A"}, "source1"))
    engine.consolidate()
    assert len(engine.lts) == 1
    assert engine.lts[0].content == {"fact": "A"}
    print("basic_test passed")

def harder_test():
    print("--- harder_test ---")
    engine = KnowledgeConsolidationEngine(promotion_threshold=3)
    # Exact matches form a cluster
    engine.add_experience(Experience({"fact": "B", "val": 1}, "source2"))
    engine.add_experience(Experience({"fact": "B", "val": 1}, "source2"))
    engine.add_experience(Experience({"fact": "B", "val": 1}, "source2"))
    engine.consolidate()
    assert len(engine.lts) == 1
    # Check schema
    assert engine.lts[0].schema == {"fact": "B", "val": "1"}
    print("harder_test passed")

def unseen_test():
    print("--- unseen_test ---")
    engine = KnowledgeConsolidationEngine()
    engine.add_experience(Experience({"fact": "C"}, "s3"))
    # Query for something not there
    results = engine.query_knowledge({"fact": "D"})
    assert len(results) == 0
    print("unseen_test passed")

def adversarial_test():
    print("--- adversarial_test ---")
    engine = KnowledgeConsolidationEngine(promotion_threshold=3)
    # Establish strong knowledge
    for i in range(10):
        engine.add_experience(Experience({"fact": "X"}, "s", repetitions=1))
    engine.consolidate()
    
    # Flood with contradiction but individually weak (no clustering since we consolidate immediately or they overwrite? 
    # Actually add_experience checks contradictions on the fly)
    for i in range(5):
        engine.add_experience(Experience({"fact": "Y"}, "s", repetitions=1)) # Same keys, different value
        
    assert len(engine.lts) == 1
    assert engine.lts[0].content == {"fact": "X"}
    # Confidence should be reduced though
    print(f"Confidence after contradictions: {engine.lts[0].confidence}")
    print("adversarial_test passed")

def ambiguous_test():
    print("--- ambiguous_test ---")
    engine = KnowledgeConsolidationEngine(promotion_threshold=3)
    # Both A and B have 3 reps
    for i in range(3):
        engine.add_experience(Experience({"key": "A"}, "s"))
    for i in range(3):
        engine.add_experience(Experience({"key": "B"}, "s"))
        
    engine.consolidate()
    # The first one to be evaluated gets promoted, the second one contradicts it but has equal support
    # Our logic says "if rep_count > contradicted_k.support_count: Override". 
    # Since 3 is not > 3, the first one stays.
    assert len(engine.lts) == 1
    print(f"LTS holds: {engine.lts[0].content}")
    print("ambiguous_test passed")

def failure_recovery_test():
    print("--- failure_recovery_test ---")
    engine = KnowledgeConsolidationEngine(max_knowledge=2, promotion_threshold=1)
    engine.add_experience(Experience({"id": 1}, "s"))
    engine.add_experience(Experience({"id": 2}, "s"))
    engine.add_experience(Experience({"id": 3}, "s"))
    engine.consolidate()
    
    # LTS should enforce max_knowledge = 2 (may promote fewer)
    assert len(engine.lts) <= 2, f"LTS exceeded max_knowledge: {len(engine.lts)}"
    # At least one should be promoted (threshold=1, we have 3 experiences)
    assert len(engine.lts) >= 1, f"No knowledge promoted: {len(engine.lts)}"
    print("failure_recovery_test passed")

def resource_limit_test():
    print("--- resource_limit_test ---")
    engine = KnowledgeConsolidationEngine(max_buffer=10, max_knowledge=5, promotion_threshold=2)
    # Add 100 experiences
    for i in range(100):
        engine.add_experience(Experience({"id": i % 15}, "s"))
    assert len(engine.stb) <= 10
    
    engine.consolidate()
    assert len(engine.lts) <= 5
    print("resource_limit_test passed")

def main():
    basic_test()
    harder_test()
    unseen_test()
    adversarial_test()
    ambiguous_test()
    failure_recovery_test()
    resource_limit_test()

if __name__ == "__main__":
    main()
