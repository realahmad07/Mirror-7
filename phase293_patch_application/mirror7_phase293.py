from dataclasses import dataclass
from phase291_patch_model import PatchPlan
from phase292_patch_validation import PatchValidator

@dataclass(frozen=True)
class PatchResult:
    applied: bool
    source: str
    reason: str

class PatchApplier:
    """Applies only exact bounded replacements after validation."""
    def apply(self,source:str,plan:PatchPlan)->PatchResult:
        plan_report=PatchValidator().validate_plan(plan)
        if not plan_report.accepted:
            return PatchResult(False,source,"plan rejected")
        candidate=source
        for op in plan.operations:
            if candidate.count(op.old)!=1:
                return PatchResult(False,source,f"expected exactly one match for {op.path}")
            candidate=candidate.replace(op.old,op.new,1)
        source_report=PatchValidator().validate_source(candidate)
        if not source_report.accepted:
            return PatchResult(False,source,"candidate source rejected")
        return PatchResult(True,candidate,"applied")
