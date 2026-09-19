from dataclasses import dataclass
@dataclass(frozen=True)
class GroundedConcept:
    fingerprint:str; views:tuple[str,...]; support:int
class MultimodalGrounder:
    def fingerprint(self,view):
        if isinstance(view,str): return "text:"+" ".join(view.lower().split())
        if isinstance(view,tuple): return "seq:"+",".join(map(str,view))
        if isinstance(view,list): return "grid:"+";".join(",".join(map(str,row)) if isinstance(row,list) else str(row) for row in view)
        return "other:"+repr(view)
    def ground(self,views):
        fs=tuple(self.fingerprint(v) for v in views)
        return GroundedConcept("|".join(sorted(fs)),fs,len(fs))
