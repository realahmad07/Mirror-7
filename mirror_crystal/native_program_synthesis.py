"""Native program construction from Mirror's typed, restricted Program IR.

Programs are assembled as expression and statement nodes, compiled into Python
AST, syntax-checked, and passed to Mirror's isolated TDD verifier. This module
does not invoke a language model or build code by concatenating text templates.
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
class ProgramExpression:
    """Typed expression node: literal, variable, binary, compare, call, subscript, list, tuple, unary."""
    operation: str
    value: Any = None
    args: tuple["ProgramExpression", ...] = ()


@dataclass(frozen=True)
class ProgramStatement:
    """Typed statement node: assign, if, for_range, return, raise."""
    operation: str
    target: str = ""
    expression: ProgramExpression | None = None
    targets: tuple[ProgramExpression, ...] = ()
    start: ProgramExpression | None = None
    stop: ProgramExpression | None = None
    body: tuple["ProgramStatement", ...] = ()
    otherwise: tuple["ProgramStatement", ...] = ()
    exception: str = "ValueError"


@dataclass(frozen=True)
class FunctionProgram:
    name: str
    parameters: tuple[str, ...]
    cases: tuple[DispatchCase, ...]
    dispatch_parameter: str
    fallback_exception: str = "ValueError"
    statements: tuple[ProgramStatement, ...] = ()


@dataclass(frozen=True)
class ProgramIR:
    module_name: str
    functions: tuple[FunctionProgram, ...]
    version: int = 1


class PythonASTCompiler:
    """Compile a restricted typed ProgramIR to Python AST without arbitrary source injection."""

    _BINARY_OPERATORS = {
        "add": ast.Add,
        "subtract": ast.Sub,
        "multiply": ast.Mult,
        "divide": ast.Div,
        "floor_divide": ast.FloorDiv,
        "modulo": ast.Mod,
    }
    _COMPARE_OPERATORS = {
        "equal": ast.Eq,
        "not_equal": ast.NotEq,
        "less_than": ast.Lt,
        "less_or_equal": ast.LtE,
        "greater_than": ast.Gt,
        "greater_or_equal": ast.GtE,
    }
    _UNARY_OPERATORS = {"not": ast.Not, "negate": ast.USub}
    _SAFE_CALLS = {"len", "list", "range", "min", "max", "sum", "abs"}
    _FORBIDDEN_IDENTIFIERS = {
        "eval", "exec", "compile", "open", "__import__", "input",
        "globals", "locals", "vars", "getattr", "setattr", "delattr",
        "breakpoint", "exit", "quit", "_mc_step_tick",
    }
    _SAFE_EXCEPTIONS = {"ValueError", "TypeError", "ZeroDivisionError"}

    @staticmethod
    def _name(identifier: str, *, store: bool = False) -> ast.Name:
        if (
            not isinstance(identifier, str)
            or not identifier.isidentifier()
            or identifier in PythonASTCompiler._FORBIDDEN_IDENTIFIERS
            or (identifier.startswith("__") and identifier.endswith("__"))
        ):
            raise ValueError(f"invalid or forbidden identifier: {identifier}")
        return ast.Name(id=identifier, ctx=ast.Store() if store else ast.Load())

    def _expression(self, expr: ProgramExpression, *, store: bool = False) -> ast.expr:
        if not isinstance(expr, ProgramExpression):
            raise TypeError("Program IR expressions must be ProgramExpression objects")
        operation = expr.operation
        if operation == "literal":
            if not isinstance(expr.value, (str, int, float, bool, type(None))):
                raise ValueError("Program IR literal must be a scalar")
            if store:
                raise ValueError("literal cannot be an assignment target")
            return ast.Constant(value=expr.value)
        if operation == "variable":
            return self._name(str(expr.value), store=store)
        if operation == "binary":
            if store or len(expr.args) != 2:
                raise ValueError("binary expressions require exactly two operands")
            operator = self._BINARY_OPERATORS.get(str(expr.value))
            if operator is None:
                raise ValueError(f"unsupported binary operation: {expr.value}")
            return ast.BinOp(
                left=self._expression(expr.args[0]),
                op=operator(),
                right=self._expression(expr.args[1]),
            )
        if operation == "compare":
            if store or len(expr.args) != 2:
                raise ValueError("comparison expressions require exactly two operands")
            operator = self._COMPARE_OPERATORS.get(str(expr.value))
            if operator is None:
                raise ValueError(f"unsupported comparison: {expr.value}")
            return ast.Compare(
                left=self._expression(expr.args[0]),
                ops=[operator()],
                comparators=[self._expression(expr.args[1])],
            )
        if operation == "unary":
            if store or len(expr.args) != 1:
                raise ValueError("unary expressions require exactly one operand")
            operator = self._UNARY_OPERATORS.get(str(expr.value))
            if operator is None:
                raise ValueError(f"unsupported unary operation: {expr.value}")
            return ast.UnaryOp(op=operator(), operand=self._expression(expr.args[0]))
        if operation == "call":
            if store or not isinstance(expr.value, str) or expr.value not in self._SAFE_CALLS:
                raise ValueError(f"call is not in the Program IR allowlist: {expr.value}")
            return ast.Call(
                func=self._name(expr.value),
                args=[self._expression(item) for item in expr.args],
                keywords=[],
            )
        if operation == "subscript":
            if len(expr.args) != 2:
                raise ValueError("subscript expressions require a container and index")
            return ast.Subscript(
                value=self._expression(expr.args[0]),
                slice=self._expression(expr.args[1]),
                ctx=ast.Store() if store else ast.Load(),
            )
        if operation in {"list", "tuple"}:
            if store:
                elements = [self._expression(item, store=True) for item in expr.args]
            else:
                elements = [self._expression(item) for item in expr.args]
            if operation == "list":
                return ast.List(elts=elements, ctx=ast.Store() if store else ast.Load())
            return ast.Tuple(elts=elements, ctx=ast.Store() if store else ast.Load())
        raise ValueError(f"unsupported Program IR expression: {operation}")

    def _statement(self, statement: ProgramStatement) -> ast.stmt:
        if not isinstance(statement, ProgramStatement):
            raise TypeError("Program IR statements must be ProgramStatement objects")
        operation = statement.operation
        if operation == "assign":
            if statement.expression is None or not statement.targets:
                raise ValueError("assign statements require targets and an expression")
            return ast.Assign(
                targets=[self._expression(item, store=True) for item in statement.targets],
                value=self._expression(statement.expression),
            )
        if operation == "if":
            if statement.expression is None:
                raise ValueError("if statements require a condition")
            body = [self._statement(item) for item in statement.body]
            otherwise = [self._statement(item) for item in statement.otherwise]
            if not body:
                raise ValueError("if statement body must not be empty")
            return ast.If(
                test=self._expression(statement.expression),
                body=body,
                orelse=otherwise,
            )
        if operation == "for_range":
            if not statement.target.isidentifier() or statement.start is None or statement.stop is None:
                raise ValueError("for_range requires a valid target, start, and stop")
            body = [self._statement(item) for item in statement.body]
            if not body:
                raise ValueError("for_range body must not be empty")
            return ast.For(
                target=self._name(statement.target, store=True),
                iter=ast.Call(
                    func=self._name("range"),
                    args=[self._expression(statement.start), self._expression(statement.stop)],
                    keywords=[],
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
        if operation == "return":
            if statement.expression is None:
                return ast.Return(value=None)
            return ast.Return(value=self._expression(statement.expression))
        if operation == "raise":
            if statement.exception not in self._SAFE_EXCEPTIONS:
                raise ValueError(f"unsupported Program IR exception: {statement.exception}")
            message = statement.expression or ProgramExpression("literal", "Mirror program error")
            return ast.Raise(
                exc=ast.Call(
                    func=self._name(statement.exception),
                    args=[self._expression(message)],
                    keywords=[],
                ),
                cause=None,
            )
        raise ValueError(f"unsupported Program IR statement: {operation}")

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
            body = [zero_guard, ast.Return(value=self._binary(case.operation))]
        else:
            body = [ast.Return(value=self._binary(case.operation))]
        return ast.If(test=test, body=body, orelse=[])

    def _binary(self, operation: BinaryOperation) -> ast.expr:
        operator = self._BINARY_OPERATORS.get(operation.operation)
        if operator is None:
            raise ValueError(f"unsupported arithmetic operation: {operation.operation}")
        return ast.BinOp(
            left=self._name(operation.left),
            op=operator(),
            right=self._name(operation.right),
        )

    def compile(self, program: ProgramIR) -> str:
        if program.version != 1:
            raise ValueError(f"unsupported ProgramIR version: {program.version}")
        if not program.functions:
            raise ValueError("ProgramIR must contain at least one function")

        definitions: list[ast.stmt] = []
        for function in program.functions:
            self._name(function.name)
            if len(set(function.parameters)) != len(function.parameters):
                raise ValueError("function parameters must be unique")
            for parameter in function.parameters:
                self._name(parameter)
            if function.statements:
                body = [self._statement(item) for item in function.statements]
            else:
                if function.dispatch_parameter not in function.parameters:
                    raise ValueError("dispatch_parameter must be a declared function parameter")
                if not function.cases:
                    raise ValueError("function requires dispatch cases or typed statements")
                labels = [case.label for case in function.cases]
                if len(set(labels)) != len(labels):
                    raise ValueError("dispatch labels must be unique")
                body = [
                    self._dispatch_case(case, function.dispatch_parameter)
                    for case in function.cases
                ]
                if function.fallback_exception not in self._SAFE_EXCEPTIONS:
                    raise ValueError("unsupported dispatch fallback exception")
                body.append(ast.Raise(
                    exc=ast.Call(
                        func=self._name(function.fallback_exception),
                        args=[ast.Constant(value="unsupported operation")],
                        keywords=[],
                    ),
                    cause=None,
                ))
            if not body:
                raise ValueError("function body must not be empty")
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
        compile(module, f"<{program.module_name}>", "exec")
        return ast.unparse(module) + "\n"


class NativeProgramSynthesizer:
    """Construct programs by composing typed operations and control-flow nodes."""

    @staticmethod
    def _var(name: str) -> ProgramExpression:
        return ProgramExpression("variable", name)

    @staticmethod
    def _lit(value: Any) -> ProgramExpression:
        return ProgramExpression("literal", value)

    @staticmethod
    def _call(name: str, *args: ProgramExpression) -> ProgramExpression:
        return ProgramExpression("call", name, tuple(args))

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

    @staticmethod
    def bubble_sort_ir() -> ProgramIR:
        values = NativeProgramSynthesizer._var("result")
        left_index = NativeProgramSynthesizer._var("j")
        right_index = ProgramExpression(
            "binary", "add", (NativeProgramSynthesizer._var("j"), NativeProgramSynthesizer._lit(1))
        )
        left_item = ProgramExpression("subscript", None, (values, left_index))
        right_item = ProgramExpression("subscript", None, (values, right_index))
        swap_targets = ProgramExpression("tuple", None, (left_item, right_item))
        swap_values = ProgramExpression("tuple", None, (right_item, left_item))

        swap = ProgramStatement(
            operation="if",
            expression=ProgramExpression("compare", "greater_than", (left_item, right_item)),
            body=(
                ProgramStatement(
                    operation="assign",
                    targets=(swap_targets,),
                    expression=swap_values,
                ),
            ),
        )
        inner_loop = ProgramStatement(
            operation="for_range",
            target="j",
            start=NativeProgramSynthesizer._lit(0),
            stop=ProgramExpression(
                "binary",
                "subtract",
                (
                    NativeProgramSynthesizer._call("len", values),
                    ProgramExpression(
                        "binary",
                        "add",
                        (
                            NativeProgramSynthesizer._var("i"),
                            NativeProgramSynthesizer._lit(1),
                        ),
                    ),
                ),
            ),
            body=(swap,),
        )
        outer_loop = ProgramStatement(
            operation="for_range",
            target="i",
            start=NativeProgramSynthesizer._lit(0),
            stop=NativeProgramSynthesizer._call("len", values),
            body=(inner_loop,),
        )
        copy_values = ProgramStatement(
            operation="assign",
            targets=(values,),
            expression=NativeProgramSynthesizer._call("list", NativeProgramSynthesizer._var("items")),
        )
        return ProgramIR(
            module_name="mirror_native_bubble_sort",
            functions=(
                FunctionProgram(
                    name="bubble_sort",
                    parameters=("items",),
                    cases=(),
                    dispatch_parameter="",
                    statements=(
                        copy_values,
                        outer_loop,
                        ProgramStatement(operation="return", expression=values),
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

    def build_bubble_sort(self) -> dict[str, Any]:
        ir = self.bubble_sort_ir()
        source = PythonASTCompiler().compile(ir)
        return {
            "code": source,
            "program_ir": {
                "version": ir.version,
                "module_name": ir.module_name,
                "function_count": len(ir.functions),
                "operations": ["sequence.copy", "loop.for_range", "compare.greater_than", "sequence.swap"],
            },
        }
