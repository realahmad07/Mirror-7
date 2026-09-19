from dataclasses import dataclass
from math import sqrt

@dataclass(frozen=True)
class NoveltyReport:
    novel: bool
    distance: float
    threshold: float
    reason: str

class NoveltyDetector:
    """Detects observations outside a bounded learned neighborhood."""
    def __init__(self, threshold=1.0, max_points=256):
        self.threshold=threshold; self.max_points=max_points; self.points=[]
    def observe(self, point):
        p=tuple(float(x) for x in point)
        if not self.points:
            self.points.append(p); return NoveltyReport(True,float("inf"),self.threshold,"no prior evidence")
        d=min(sqrt(sum((a-b)**2 for a,b in zip(p,q))) if len(p)==len(q) else float("inf") for q in self.points)
        novel=d>self.threshold
        self.points.append(p)
        self.points=self.points[-self.max_points:]
        return NoveltyReport(novel,d,self.threshold,"novel" if novel else "familiar")