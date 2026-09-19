"""
Test suite for ProgramSynthesis
"""
import sys
from mirror7_algorithm_02.mirror7_algorithm import ProgramSynthesis

def run_tests():
    passed = 0
    total = 0
    
    def basic_test():
        print("Running basic_test...")
        ps = ProgramSynthesis()
        prog = ps.synthesize([([1], 2), ([2], 3), ([5], 6)])
        return prog is not None and prog.evaluate([10]) == 11

    def harder_test():
        print("Running harder_test...")
        ps = ProgramSynthesis()
        prog = ps.synthesize([([1], 3), ([2], 5), ([5], 11)])
        return prog is not None and prog.evaluate([10]) == 21
        
    def unseen_test():
        print("Running unseen_test...")
        ps = ProgramSynthesis()
        prog = ps.synthesize([([1], 2)])
        return prog.evaluate([100]) is not None
        
    def adversarial_test():
        print("Running adversarial_test...")
        ps = ProgramSynthesis()
        prog = ps.synthesize([([1], 2), ([1], 3)])
        return prog is not None 

    def ambiguous_test():
        print("Running ambiguous_test...")
        ps = ProgramSynthesis()
        prog = ps.synthesize([([0], 0)]) 
        return prog is not None
        
    def failure_recovery_test():
        print("Running failure_recovery_test...")
        ps = ProgramSynthesis(timeout_steps=1)
        prog = ps.synthesize([([1], 9999)])
        return prog is not None 
        
    def resource_limit_test():
        print("Running resource_limit_test...")
        ps = ProgramSynthesis(max_depth=1, beam_width=1)
        prog = ps.synthesize([([1], 3)]) 
        score = 0
        if prog.evaluate([1]) == 3:
            score = 1
        return score == 0 

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
