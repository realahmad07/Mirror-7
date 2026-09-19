import sys
from mirror7_algorithm_05.mirror7_algorithm import SemanticMemoryGraph, MemoryNode, MemoryEdge

def run_tests():
    passed = 0
    failed = 0
    
    def test(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"PASS: {name}")
            passed += 1
        else:
            print(f"FAIL: {name}")
            failed += 1

    # 1. basic_test
    graph = SemanticMemoryGraph()
    graph.add_node("A", "concept", "A", {}, 1.0, "source1")
    graph.add_node("B", "concept", "B", {}, 1.0, "source1")
    graph.add_edge("A", "B", "is_a", 0.9, "source1")
    res = graph.query("A")
    test("basic_test", res is not None and len(res['connections']) == 1)

    # 2. harder_test
    graph.add_node("C", "concept", "C", {}, 1.0, "source1")
    graph.add_edge("B", "C", "is_a", 0.8, "source1")
    path = graph.find_path("A", "C")
    test("harder_test", len(path) == 2)

    # 3. unseen_test
    test("unseen_test", graph.query("Z") is None and len(graph.find_path("A", "Z")) == 0)

    # 4. adversarial_test
    graph.add_edge("C", "A", "contradicts", 0.9, "source2")
    con = graph.detect_contradictions()
    test("adversarial_test", len(con) > 0)

    # 5. ambiguous_test
    graph.add_node("e1", "entity", "E", {"entity_id": "1", "color": "red"}, 0.8, "s1")
    graph.add_node("e2", "entity", "E", {"entity_id": "1", "color": "blue"}, 0.6, "s2")
    con2 = graph.detect_contradictions()
    test("ambiguous_test", len([c for c in con2 if "mismatch" in c[2]]) > 0)

    # 6. failure_recovery_test
    # Evict critical node manually (simulate failure)
    del graph.nodes["B"]
    graph._evict_nodes(0) # trigger cleanup if any, but let's test if query fails gracefully
    test("failure_recovery_test", len(graph.find_path("A", "C")) == 0)

    # 7. resource_limit_test
    small_graph = SemanticMemoryGraph(max_nodes=50, max_edges=100)
    for i in range(100):
        small_graph.add_node(f"n{i}", "concept", "lbl", {}, 0.5, "sys")
    test("resource_limit_test", len(small_graph.nodes) <= 50)

    print(f"Total passed: {passed}, failed: {failed}")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
