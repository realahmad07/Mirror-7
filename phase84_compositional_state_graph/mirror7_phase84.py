"""Phase 84: bounded compositional state graph."""
from collections import defaultdict
class CompositionalStateGraph:
    def __init__(self,max_nodes=512):
        if max_nodes<1: raise ValueError("invalid max_nodes")
        self.max_nodes=max_nodes; self.edges=defaultdict(dict)
    def add(self,state,action,result):
        s=tuple(float(x) for x in state); r=tuple(float(x) for x in result)
        if not s or not r: raise ValueError("empty state")
        if s not in self.edges and len(self.edges)>=self.max_nodes: raise RuntimeError("node budget exhausted")
        self.edges[s][action]=r
    def predict(self,state,action): return self.edges.get(tuple(float(x) for x in state),{}).get(action)
    def successors(self,state): return dict(self.edges.get(tuple(float(x) for x in state),{}))
