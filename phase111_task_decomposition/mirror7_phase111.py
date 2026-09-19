from dataclasses import dataclass

@dataclass(frozen=True)
class TaskNode:
    name: str
    deps: tuple[str,...]=()
    cost: int=1

class TaskDecomposer:
    """Produces a dependency-safe execution order under a hard step budget."""
    def order(self,nodes:list[TaskNode],budget:int):
        by={n.name:n for n in nodes}
        if len(by)!=len(nodes): raise ValueError("duplicate task")
        for n in nodes:
            if n.cost<1 or any(d not in by for d in n.deps): raise ValueError("invalid dependency")
        done=[]; used=0
        remaining=set(by)
        while remaining:
            ready=sorted(n for n in remaining if all(d in done for d in by[n].deps))
            if not ready: raise ValueError("dependency cycle")
            chosen=next((x for x in ready if used+by[x].cost<=budget),None)
            if chosen is None: return tuple(done)
            done.append(chosen); remaining.remove(chosen); used+=by[chosen].cost
        return tuple(done)
