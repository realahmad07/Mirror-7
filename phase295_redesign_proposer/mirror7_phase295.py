from typing import List, Sequence
from phase291_patch_model import PatchOp, PatchPlan

class RedesignProposer:
    """Turns a bounded observed failure into finite source-level redesign hypotheses."""
    def propose(self,target:str,failures:Sequence[str])->List[PatchPlan]:
        if not target or not failures:
            return []
        joined=" ".join(str(x).lower() for x in failures)
        plans=[]
        if "sequence" in joined or "delta" in joined or "extrapol" in joined:
            plans.extend([
                PatchPlan(target,(PatchOp("replace_expression","solve.return","values[-1]","values[-1] + (values[-1]-values[-2])"),),"add learned delta"),
                PatchPlan(target,(PatchOp("replace_expression","solve.return","values[-1]","values[-1] + 1"),),"bounded constant increment"),
                PatchPlan(target,(PatchOp("replace_expression","solve.return","values[-1]","values[-1] - (values[-1]-values[-2])"),),"reverse delta hypothesis"),
            ])
        if "constant" in joined:
            plans.append(PatchPlan(target,(PatchOp("replace_expression","solve.return","values[-1]","values[-1] + 0"),),"constant-preserving control"))
        # Stable deduplication.
        unique=[]
        seen=set()
        for plan in plans:
            key=tuple((o.kind,o.path,o.old,o.new) for o in plan.operations)
            if key not in seen:
                seen.add(key); unique.append(plan)
        return unique
