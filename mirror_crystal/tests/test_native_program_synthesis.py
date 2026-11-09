import ast

import pytest

from mirror_crystal.native_program_synthesis import (
    BinaryOperation,
    DispatchCase,
    FunctionProgram,
    NativeProgramSynthesizer,
    ProgramExpression,
    ProgramIR,
    ProgramStatement,
    PythonASTCompiler,
)


def test_calculator_is_built_from_typed_ir_and_valid_python_ast():
    result = NativeProgramSynthesizer().build_calculator()
    source = result["code"]
    parsed = ast.parse(source)

    assert result["program_ir"]["operations"] == [
        "add", "subtract", "multiply", "divide"
    ]
    assert len(parsed.body) == 1
    assert isinstance(parsed.body[0], ast.FunctionDef)
    assert parsed.body[0].name == "calculate"
    assert "def calculate" in source
    assert "ZeroDivisionError" in source


def test_generated_calculator_executes_all_declared_operations():
    source = NativeProgramSynthesizer().build_calculator()["code"]
    namespace = {}
    exec(compile(source, "<mirror-native-test>", "exec"), {"__builtins__": {}}, namespace)
    calculate = namespace["calculate"]

    assert calculate(8, 3, "add") == 11
    assert calculate(8, 3, "subtract") == 5
    assert calculate(8, 3, "multiply") == 24
    assert calculate(8, 2, "divide") == 4


def test_generated_calculator_rejects_division_by_zero_and_unknown_operator():
    source = NativeProgramSynthesizer().build_calculator()["code"]
    namespace = {}
    exec(compile(source, "<mirror-native-test>", "exec"), namespace, namespace)

    with pytest.raises(ZeroDivisionError):
        namespace["calculate"](1, 0, "divide")
    with pytest.raises(ValueError):
        namespace["calculate"](1, 2, "power")


def test_program_ir_rejects_unknown_binary_operators():
    program = ProgramIR(
        module_name="unsafe_operator",
        functions=(
            FunctionProgram(
                name="calculate",
                parameters=("left", "right", "operation"),
                dispatch_parameter="operation",
                cases=(DispatchCase("power", BinaryOperation("power", "left", "right")),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="unsupported arithmetic operation"):
        PythonASTCompiler().compile(program)


def test_bubble_sort_program_is_composed_from_loops_conditionals_and_sequence_nodes():
    result = NativeProgramSynthesizer().build_bubble_sort()
    source = result["code"]
    parsed = ast.parse(source)
    assert len(parsed.body) == 1
    assert isinstance(parsed.body[0], ast.FunctionDef)
    assert parsed.body[0].name == "bubble_sort"
    assert "for " in source
    assert "if " in source
    assert result["program_ir"]["operations"] == [
        "sequence.copy", "loop.for_range", "compare.greater_than", "sequence.swap"
    ]


def test_generated_bubble_sort_executes_empty_sorted_reverse_duplicate_and_negative_inputs():
    source = NativeProgramSynthesizer().build_bubble_sort()["code"]
    namespace = {}
    exec(compile(source, "<mirror-native-bubble-sort>", "exec"), {"__builtins__": {}}, namespace)
    bubble_sort = namespace["bubble_sort"]

    assert bubble_sort([]) == []
    assert bubble_sort([1, 2, 3]) == [1, 2, 3]
    assert bubble_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert bubble_sort([5, 5, -1, 0]) == [-1, 0, 5, 5]


def test_program_ir_rejects_forbidden_identifiers_and_non_allowlisted_calls():
    forbidden_name = ProgramIR(
        module_name="bad_name",
        functions=(
            FunctionProgram(
                name="calculate",
                parameters=("__import__",),
                cases=(),
                dispatch_parameter="",
            ),
        ),
    )
    with pytest.raises(ValueError, match="forbidden identifier"):
        PythonASTCompiler().compile(forbidden_name)

    forbidden_call = ProgramIR(
        module_name="bad_call",
        functions=(
            FunctionProgram(
                name="calculate",
                parameters=("value",),
                cases=(),
                dispatch_parameter="",
                statements=(
                    ProgramStatement(
                        operation="return",
                        expression=ProgramExpression(
                            "call",
                            "__import__",
                            (ProgramExpression("literal", "os"),),
                        ),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="allowlist"):
        PythonASTCompiler().compile(forbidden_call)
