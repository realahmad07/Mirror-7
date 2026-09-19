from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class PatchOp:
    kind: str
    path: str
    old: str
    new: str

@dataclass(frozen=True)
class PatchPlan:
    target: str
    operations: Tuple[PatchOp,...]
    rationale: str

class PatchPlanValidator:
    ALLOWED_KINDS=frozenset({"replace_literal","replace_expression"})
    def validate(self,plan:PatchPlan)->bool:
        if not plan.target or not plan.rationale or not plan.operations:
            return False
        for op in plan.operations:
            if op.kind not in self.ALLOWED_KINDS or not op.path or op.old==op.new:
                return False
        return True
