import ast

import pytest

from mirror_crystal.native_program_synthesis import (
    BinaryOperation,
    DispatchCase,
    FunctionProgram,
    NativeProgramSynthesizer,
    ProgramIR,
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
