from dataclasses import dataclass
import ast
from phase291_patch_model import PatchPlan, PatchPlanValidator

@dataclass(frozen=True)
class ValidationReport:
    accepted: bool
    reasons: tuple[str,...]

class PatchValidator:
    """Validates bounded patch plans and a pure, non-importing candidate source subset."""
    _ALLOWED_NODES = {
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
        ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Subscript, ast.Slice,
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.If,
        ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
        ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    }

    def validate_plan(self, plan: PatchPlan) -> ValidationReport:
        if not PatchPlanValidator().validate(plan):
            return ValidationReport(False, ("invalid patch plan",))
        reasons=[]
        for op in plan.operations:
            if len(op.old)>200 or len(op.new)>200:
                reasons.append("replacement too large")
        return ValidationReport(not reasons, tuple(reasons))

    def validate_source(self, source: str) -> ValidationReport:
        try:
            tree=ast.parse(source)
        except SyntaxError as exc:
            return ValidationReport(False,(f"syntax error: {exc.msg}",))
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        if len(funcs)!=1 or funcs[0].name!="solve":
            return ValidationReport(False,("source must define exactly one solve function",))
        if any(not isinstance(n,(ast.FunctionDef,ast.Assign)) for n in tree.body):
            return ValidationReport(False,("top-level nodes are restricted",))
        for node in ast.walk(tree):
            if type(node) not in self._ALLOWED_NODES:
                return ValidationReport(False,(f"forbidden AST node: {type(node).__name__}",))
            if isinstance(node,ast.Call):
                return ValidationReport(False,("function calls are not allowed",))
            if isinstance(node,ast.Attribute):
                return ValidationReport(False,("attribute access is not allowed",))
        return ValidationReport(True,())
