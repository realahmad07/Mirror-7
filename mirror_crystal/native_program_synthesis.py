"""Native Python program construction from a small typed program IR.

This is a domain adapter for the universal capability engine, not a language model.
New program forms are built by composing explicit syntax/operation nodes into Python
AST, then compiled and tested by Mirror's existing TDD sandbox.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BinaryOperation:
    operation: str
    left: str
    right: str


@dataclass(frozen=True)
class DispatchCase:
    label: str
    operation: BinaryOperation


@dataclass(frozen=True)
class FunctionProgram:
    name: str
    parameters: tuple[str, ...]
    cases: tuple[DispatchCase, ...]
    dispatch_parameter: str
    fallback_exception: str = "ValueError"


@dataclass(frozen=True)
class ProgramIR:
    module_name: str
    functions: tuple[FunctionProgram, ...]
    version: int = 1


class PythonASTCompiler:
    """Compile the restricted ProgramIR to Python AST; no source-template concatenation."""

    _BINARY_OPERATORS = {
        "add": ast.Add,
        "subtract": ast.Sub,
        "multiply": ast.Mult,
        "divide": ast.Div,
    }

    @staticmethod
    def _name(identifier: str) -> ast.Name:
        if not identifier.isidentifier():
            raise ValueError(f"invalid identifier: {identifier}")
        return ast.Name(id=identifier, ctx=ast.Load())

    def _binary(self, operation: BinaryOperation) -> ast.expr:
        operator = self._BINARY_OPERATORS.get(operation.operation)
        if operator is None:
            raise ValueError(f"unsupported arithmetic operation: {operation.operation}")
        return ast.BinOp(
            left=self._name(operation.left),
            op=operator(),
            right=self._name(operation.right),
        )

    def _dispatch_case(self, case: DispatchCase, dispatch_parameter: str) -> ast.If:
        test = ast.Compare(
            left=self._name(dispatch_parameter),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=case.label)],
        )
        if case.operation.operation == "divide":
            zero_test = ast.Compare(
                left=self._name(case.operation.right),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=0)],
            )
            zero_guard = ast.If(
                test=zero_test,
                body=[ast.Raise(
                    exc=ast.Call(
                        func=self._name("ZeroDivisionError"),
                        args=[ast.Constant(value="division by zero")],
                        keywords=[],
                    ),
                    cause=None,
                )],
                orelse=[],
            )
            body = [
                zero_guard,
                ast.Return(value=self._binary(case.operation)),
            ]
        else:
            body = [ast.Return(value=self._binary(case.operation))]
        return ast.If(test=test, body=body, orelse=[])

    def compile(self, program: ProgramIR) -> str:
        if program.version != 1:
            raise ValueError(f"unsupported ProgramIR version: {program.version}")
        if not program.functions:
            raise ValueError("ProgramIR must contain at least one function")

        definitions: list[ast.stmt] = []
        for function in program.functions:
            if not function.name.isidentifier():
                raise ValueError(f"invalid function name: {function.name}")
            if function.dispatch_parameter not in function.parameters:
                raise ValueError("dispatch_parameter must be a declared function parameter")
            if not function.cases:
                raise ValueError("function requires at least one dispatch case")

            body: list[ast.stmt] = []
            for case in function.cases:
                body.append(self._dispatch_case(case, function.dispatch_parameter))
            body.append(ast.Raise(
                exc=ast.Call(
                    func=self._name(function.fallback_exception),
                    args=[ast.Constant(value="unsupported operation")],
                    keywords=[],
                ),
                cause=None,
            ))
            definitions.append(ast.FunctionDef(
                name=function.name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=name) for name in function.parameters],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=body,
                decorator_list=[],
                returns=None,
                type_comment=None,
            ))

        module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
        # Syntax compilation catches malformed generated trees before source is returned.
        compile(module, f"<{program.module_name}>", "exec")
        return ast.unparse(module) + "\n"


class NativeProgramSynthesizer:
    """Build a calculator from typed arithmetic operators already present in Mirror."""

    @staticmethod
    def calculator_ir() -> ProgramIR:
        return ProgramIR(
            module_name="mirror_native_calculator",
            functions=(
                FunctionProgram(
                    name="calculate",
                    parameters=("left", "right", "operation"),
                    dispatch_parameter="operation",
                    cases=(
                        DispatchCase("add", BinaryOperation("add", "left", "right")),
                        DispatchCase("subtract", BinaryOperation("subtract", "left", "right")),
                        DispatchCase("multiply", BinaryOperation("multiply", "left", "right")),
                        DispatchCase("divide", BinaryOperation("divide", "left", "right")),
                    ),
                ),
            ),
        )

    def build_calculator(self) -> dict[str, Any]:
        ir = self.calculator_ir()
        source = PythonASTCompiler().compile(ir)
        return {
            "code": source,
            "program_ir": {
                "version": ir.version,
                "module_name": ir.module_name,
                "function_count": len(ir.functions),
                "operations": [
                    case.operation.operation
                    for function in ir.functions
                    for case in function.cases
                ],
            },
        }
