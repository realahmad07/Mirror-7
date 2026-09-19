import sys
from mirror7_algorithm_09.mirror7_algorithm import MultimodalGrounding, ModalityToken

def basic_test():
    mg = MultimodalGrounding(alignment_threshold=0.4)
    for i in range(5):
        mg.observe([
            ModalityToken("language", "cat", {"text": 1.0}, i),
            ModalityToken("visual", "animal_shape", {"edges": 0.8}, i)
        ])
    
    assert "animal_shape" in mg.retrieve("language", "cat", "visual")
    print("basic_test: PASS")

def harder_test():
    mg = MultimodalGrounding(alignment_threshold=0.4)
    for i in range(5):
        mg.observe([
            ModalityToken("language", "dog", {}, i),
            ModalityToken("visual", "dog_shape", {}, i)
        ])
        mg.observe([
            ModalityToken("language", "bird", {}, i),
            ModalityToken("visual", "bird_shape", {}, i)
        ])
    assert "dog_shape" in mg.retrieve("language", "dog", "visual")
    assert "bird_shape" in mg.retrieve("language", "bird", "visual")
    assert "dog_shape" not in mg.retrieve("language", "bird", "visual")
    print("harder_test: PASS")

def unseen_test():
    mg = MultimodalGrounding(alignment_threshold=0.4)
    assert mg.retrieve("language", "unseen", "visual") == []
    assert mg.ground("language", "unseen") is None
    print("unseen_test: PASS")

def adversarial_test():
    mg = MultimodalGrounding(alignment_threshold=0.5)
    for i in range(10):
        mg.observe([
            ModalityToken("language", "cat", {}, i),
            ModalityToken("visual", "animal_shape", {}, i)
        ])
    # Spurious occurrences
    mg.observe([ModalityToken("language", "cat", {}, 11), ModalityToken("audio", "car_horn", {}, 11)])
    
    assert "animal_shape" in mg.retrieve("language", "cat", "visual")
    assert "car_horn" not in mg.retrieve("language", "cat", "audio")
    print("adversarial_test: PASS")

def ambiguous_test():
    mg = MultimodalGrounding(alignment_threshold=0.2)
    for i in range(10):
        mg.observe([
            ModalityToken("language", "apple", {}, i),
            ModalityToken("visual", "red_circle", {}, i),
            ModalityToken("visual", "green_circle", {}, i)
        ])
    visuals = mg.retrieve("language", "apple", "visual")
    assert "red_circle" in visuals
    assert "green_circle" in visuals
    print("ambiguous_test: PASS")

def failure_recovery_test():
    mg = MultimodalGrounding(max_concepts=2, alignment_threshold=0.1)
    for i in range(3):
        mg.observe([ModalityToken("mod1", f"t{i}", {}, 0), ModalityToken("mod2", f"v{i}", {}, 0)])
    assert len(mg.concepts) <= 2
    print("failure_recovery_test: PASS")

def resource_limit_test():
    mg = MultimodalGrounding(max_pairs=5)
    for i in range(10):
        mg.observe([ModalityToken("m1", f"x{i}", {}, 0), ModalityToken("m2", f"y{i}", {}, 0)])
    assert len(mg.co_occurrences) <= 5
    print("resource_limit_test: PASS")

if __name__ == "__main__":
    try:
        basic_test()
        harder_test()
        unseen_test()
        adversarial_test()
        ambiguous_test()
        failure_recovery_test()
        resource_limit_test()
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
