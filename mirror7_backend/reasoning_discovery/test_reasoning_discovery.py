"""
Unit and integration tests for Mirror 7 Reasoning Discovery Engine.
"""
import pytest
from mirror7_backend.reasoning_discovery import (
    ProblemRepresentation,
    ReasoningStrategy,
    TestCase,
    SafeSandbox,
    ProblemExplorer,
    StrategyGenerator,
    StrategyComparator,
    StrategyMemory,
    StrategyRefiner,
    NoveltyEvaluator,
    ReasoningDiscoveryEngine,
)


def test_sandbox_security_enforcement():
    sandbox = SafeSandbox()

    # 1. Reject forbidden import
    bad_import_strat = ReasoningStrategy(
        strategy_id="bad_1",
        name="Bad Import",
        procedure_code="import os\ndef solve(): return os.getcwd()",
        entry_point="solve",
        rationale="testing security",
    )
    res1 = sandbox.execute(bad_import_strat, [TestCase((), "test")])
    assert res1.is_correct is False
    assert "Forbidden import" in str(res1.error_message)

    # 2. Reject forbidden builtin (open)
    bad_open_strat = ReasoningStrategy(
        strategy_id="bad_2",
        name="Bad Open",
        procedure_code="def solve():\n    f = open('secret.txt', 'w')\n    return 0",
        entry_point="solve",
        rationale="testing security",
    )
    res2 = sandbox.execute(bad_open_strat, [TestCase((), 0)])
    assert res2.is_correct is False
    assert "Forbidden function call" in str(res2.error_message)

    # 3. Allow safe algorithmic code
    safe_strat = ReasoningStrategy(
        strategy_id="safe_1",
        name="Safe Add",
        procedure_code="def add(a, b): return a + b",
        entry_point="add",
        rationale="basic addition",
    )
    res3 = sandbox.execute(safe_strat, [TestCase((2, 3), 5), TestCase((-1, 1), 0)])
    assert res3.is_correct is True
    assert res3.passed_tests == 2


def test_problem_explorer():
    explorer = ProblemExplorer()
    prob = explorer.formalize_problem(
        problem_id="prime_01",
        objective="Determine if integer N is prime",
        raw_description="The input integer N must be positive. Assume standard decimal representation. If N <= 1, it cannot be prime.",
        domain="number_theory",
        sample_cases=[((2,), True), ((4,), False), ((17,), True)],
    )
    assert len(prob.test_cases) == 3
    assert len(prob.constraints) >= 1
    assert len(prob.assumptions) >= 1

    iter_rep, func_rep = explorer.formulate_dual_representations(prob)
    assert "iterative" in iter_rep.problem_id
    assert "functional" in func_rep.problem_id


def test_strategy_generation_and_execution():
    explorer = ProblemExplorer()
    prob = explorer.formalize_problem(
        problem_id="prob_prime",
        objective="Determine whether N is prime",
        raw_description="N must be positive integer.",
        sample_cases=[((2,), True), ((3,), True), ((4,), False), ((17,), True), ((20,), False)],
    )
    generator = StrategyGenerator()
    candidates = generator.generate_candidate_strategies(prob)
    assert len(candidates) >= 2

    sandbox = SafeSandbox()
    for cand in candidates:
        res = sandbox.execute(cand, prob.test_cases)
        assert res.is_correct is True
        assert res.passed_tests == 5


def test_strategy_comparator():
    explorer = ProblemExplorer()
    prob = explorer.formalize_problem(
        problem_id="search_prob",
        objective="Search target in sorted array",
        raw_description="Array is sorted in ascending order.",
        sample_cases=[
            (([1, 3, 5, 7, 9], 5), 2),
            (([1, 3, 5, 7, 9], 10), -1),
        ],
    )
    gen = StrategyGenerator()
    candidates = gen.generate_candidate_strategies(prob)
    linear = [c for c in candidates if "linear" in c.strategy_id][0]
    binary = [c for c in candidates if "bisection" in c.strategy_id][0]

    comparator = StrategyComparator()
    comp = comparator.compare(binary, linear, prob.test_cases)
    assert comp.accuracy_delta == 0.0
    assert "Candidate" in comp.summary


def test_strategy_memory(tmp_path):
    mem_file = tmp_path / "strategies.json"
    memory = StrategyMemory(storage_path=mem_file)

    strat = ReasoningStrategy(
        strategy_id="strat_test",
        name="Test Algorithm",
        procedure_code="def run(): return 42",
        entry_point="run",
        rationale="testing",
    )
    memory.store_strategy(strat, domain="test_domain")

    # Reload from disk
    loaded_mem = StrategyMemory(storage_path=mem_file)
    retrieved = loaded_mem.get_strategy("strat_test")
    assert retrieved is not None
    assert retrieved.name == "Test Algorithm"
    assert len(loaded_mem.find_by_domain("test_domain")) == 1


def test_strategy_refiner():
    refiner = StrategyRefiner()
    base_fib = ReasoningStrategy(
        strategy_id="fib_base",
        name="Recursive Fib",
        procedure_code="def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
        entry_point="fib",
        rationale="naive",
        version=1,
    )
    memoized = refiner.refine_with_memoization(base_fib)
    assert memoized.version == 2
    assert memoized.parent_strategy_id == "fib_base"
    assert "memo" in memoized.strategy_id


def test_novelty_evaluator():
    evaluator = NoveltyEvaluator()
    code_a = "def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0: return False\n    return True"
    code_b = "def is_prime(n):\n    # Just renamed variable\n    for divisor in range(2, n):\n        if n % divisor == 0: return False\n    return True"
    code_c = "import math\ndef is_prime(n):\n    i = 2\n    while i <= math.isqrt(n):\n        if n % i == 0: return False\n        i += 1\n    return True"

    strat_a = ReasoningStrategy("a", "A", code_a, "is_prime", "")
    strat_b = ReasoningStrategy("b", "B", code_b, "is_prime", "")
    strat_c = ReasoningStrategy("c", "C", code_c, "is_prime", "")

    test_cases = [TestCase((7,), True), TestCase((8,), False)]

    # B has identical AST structure to A -> Not novel
    rep_b = evaluator.evaluate_novelty(strat_b, [strat_a], test_cases)
    assert rep_b.structural_similarity > 0.90
    assert rep_b.is_novel is False

    # C has While loop and math.isqrt -> Different AST structure
    rep_c = evaluator.evaluate_novelty(strat_c, [strat_a], test_cases)
    assert rep_c.structural_similarity < 0.85


def test_discovery_engine_end_to_end(tmp_path):
    mem_file = tmp_path / "engine_mem.json"
    engine = ReasoningDiscoveryEngine(memory=StrategyMemory(storage_path=mem_file))

    explorer = ProblemExplorer()
    prob = explorer.formalize_problem(
        problem_id="prime_discovery",
        objective="Determine if integer N is prime",
        raw_description="N must be positive.",
        sample_cases=[((2,), True), ((3,), True), ((4,), False), ((13,), True), ((15,), False)],
    )

    cycle_result = engine.run_discovery_cycle(prob)
    assert cycle_result.accepted_strategy is not None
    assert cycle_result.best_execution.is_correct is True
    assert len(cycle_result.evaluated_strategies) >= 2
    assert cycle_result.accepted_strategy.strategy_id in cycle_result.novelty_reports
