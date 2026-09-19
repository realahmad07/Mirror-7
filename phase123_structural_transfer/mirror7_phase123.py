from dataclasses import dataclass
from itertools import permutations
from typing import Mapping

@dataclass(frozen=True)
class TransferResult:
    transferred: bool
    mapping: dict[str,str]
    confidence: float
    reason: str

class StructuralTransfer:
    """Transfers structure when a graph isomorphism exists; chooses a deterministic mapping."""
    def transfer(self, source: Mapping[str, tuple[str,...]], target: Mapping[str, tuple[str,...]]) -> TransferResult:
        if len(source) != len(target):
            return TransferResult(False,{},0.0,"structure mismatch")
        sn=tuple(sorted(source)); tn=tuple(sorted(target))
        if sorted(len(v) for v in source.values()) != sorted(len(v) for v in target.values()):
            return TransferResult(False,{},0.0,"degree-profile mismatch")
        candidates=[]
        for perm in permutations(tn):
            mapping=dict(zip(sn,perm))
            ok=True
            for s in sn:
                mapped_neighbors={mapping[x] for x in source[s] if x in mapping}
                if mapped_neighbors != set(target[mapping[s]]):
                    ok=False
                    break
            if ok:
                candidates.append(mapping)
        if not candidates:
            return TransferResult(False,{},0.0,"no structural isomorphism")
        chosen=min(candidates,key=lambda m:tuple(m[k] for k in sn))
        confidence=1.0/len(candidates)
        reason="structurally transferred" if len(candidates)==1 else f"structurally transferred with {len(candidates)} valid symmetries"
        return TransferResult(True,chosen,confidence,reason)