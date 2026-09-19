"""
Test suite for ConceptTree
"""

import sys
from mirror7_algorithm_01.mirror7_algorithm import ConceptTree, Observation

def run_tests():
    passed = 0
    total = 0
    
    def basic_test():
        print("Running basic_test...")
        tree = ConceptTree()
        # Animals vs Vehicles
        tree.observe({"legs": 4, "fur": "yes", "wheels": 0})
        tree.observe({"legs": 4, "fur": "yes", "wheels": 0})
        tree.observe({"legs": 0, "fur": "no", "wheels": 4})
        tree.observe({"legs": 0, "fur": "no", "wheels": 4})
        
        c1 = tree.classify({"legs": 4, "fur": "yes", "wheels": 0})
        c2 = tree.classify({"legs": 0, "fur": "no", "wheels": 4})
        
        if c1 != c2:
            return True
        return False
        
    def harder_test():
        print("Running harder_test...")
        tree = ConceptTree()
        for i in range(10):
            tree.observe({"color": "red", "shape": "circle"})
            tree.observe({"color": "blue", "shape": "circle"})
            tree.observe({"color": "red", "shape": "square"})
            tree.observe({"color": "blue", "shape": "square"})
        
        return len(tree.get_concepts()) > 4
        
    def unseen_test():
        print("Running unseen_test...")
        tree = ConceptTree()
        tree.observe({"size": "large"})
        c = tree.classify({"weight": "heavy"})
        return c is not None

    def adversarial_test():
        print("Running adversarial_test...")
        tree = ConceptTree()
        tree.observe({"a": 1})
        tree.observe({"a": 2})
        c = tree.classify({"a": 3, "b": 4})
        return c is not None
        
    def ambiguous_test():
        print("Running ambiguous_test...")
        tree = ConceptTree()
        tree.observe({"x": 1, "y": 1})
        tree.observe({"x": 2, "y": 2})
        c = tree.classify({"x": 1, "y": 2})
        return c is not None

    def failure_recovery_test():
        print("Running failure_recovery_test...")
        tree = ConceptTree()
        try:
            tree.observe({})
            tree.classify({})
            return True
        except Exception as e:
            return False

    def resource_limit_test():
        print("Running resource_limit_test...")
        tree = ConceptTree(max_nodes=10)
        for i in range(100):
            tree.observe({f"feature_{i}": i})
        return len(tree.get_concepts()) <= 10
        
    tests = [
        basic_test,
        harder_test,
        unseen_test,
        adversarial_test,
        ambiguous_test,
        failure_recovery_test,
        resource_limit_test
    ]
    
    for test in tests:
        total += 1
        try:
            if test():
                passed += 1
                print(f"PASS")
            else:
                print(f"FAIL")
        except Exception as e:
            print(f"FAIL with exception: {e}")
            
    print(f"Total passed: {passed}/{total}")
    if passed != total:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
