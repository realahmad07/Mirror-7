from dataclasses import dataclass
import ast
from phase291_patch_model import PatchPlan, PatchPlanValidator

@dataclass(frozen=True)
class ValidationReport:
    accepted: bool
    reasons: tuple[str,...]

class PatchValidator:
    """Validates patch plans and resulting Python syntax/structure before evaluation."""
    def validate_plan(self,plan:PatchPlan)->ValidationReport:
        if not PatchPlanValidator().validate(plan):
            return ValidationReport(False,("invalid patch plan",))
        reasons=[]
        for op in plan.operations:
            if op.kind=="replace_literal" and len(op.old)>200:
                reasons.append("literal replacement too large")
            if op.kind=="replace_expression" and len(op.new)>200:
                reasons.append("expression replacement too large")
        return ValidationReport(not reasons,tuple(reasons))

    def validate_source(self,source:str)->ValidationReport:
        try:
            tree=ast.parse(source)
        except SyntaxError as exc:
            return ValidationReport(False,(f"syntax error: {exc.msg}",))
        # Python 3 has no ast.Exec node; imports are blocked explicitly below
        banned=(ast.Import,ast.ImportFrom)
        for node in ast.walk(tree):
            if isinstance(node,banned):
                return ValidationReport(False,("imports are not allowed in candidate source",))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {"eval","exec","compile","__import__"}:
                return ValidationReport(False,(f"forbidden call: {node.func.id}",))
        return ValidationReport(True,())
