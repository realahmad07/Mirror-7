from mirror_crystal import UniversalIntelligenceEngine


def test_arithmetic_expression_tree_is_decomposed_into_ordered_capability_steps():
    result = UniversalIntelligenceEngine().solve_prompt("What is (2 + 3) * 4?")
    assert result["status"] == "verified"
    assert result["verified"] is True
    assert result["answer"] == "20.0"
    assert [step["capability_id"] for step in result["executions"]] == [
        "numeric.add", "numeric.multiply"
    ]
    assert "independent_decimal_recompute:numeric.add" in result["verification"]["checks_passed"]
    assert "independent_decimal_recompute:numeric.multiply" in result["verification"]["checks_passed"]


def test_arithmetic_expression_parser_rejects_python_code_and_names():
    engine = UniversalIntelligenceEngine()
    task = engine.interpreter.interpret("Calculate __import__('os').system('echo unsafe')")
    assert task.domain == "mathematics"
    assert task.subtasks == ()
    result = engine.solve(task)
    assert result["status"] == "capability_gap"
    assert result["verified"] is False
