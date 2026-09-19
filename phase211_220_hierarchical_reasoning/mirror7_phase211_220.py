"""Phases 211-220: bounded hierarchical reasoning.

Reasoning is represented as explicit goal trees with dependencies, reusable
subplans, verification predicates and resource budgets. The system never
pretends an unverified subgoal is complete.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Iterable


@dataclass
class GoalNode:
    name: str
    children: list["GoalNode"]=field(default_factory=list)
    depends_on: tuple[str,...]=()
    done: bool=False
    verified: bool=False
    evidence: str|None=None

    def add(self,*nodes:"GoalNode")->None: self.children.extend(nodes)


class HierarchicalReasoner:
    def __init__(self,budget:int=64):
        if budget<1: raise ValueError("budget must be positive")
        self.budget=budget
        self.steps=0
        self.library:dict[str,tuple[str,...]]={}
        self.completed:set[str]=set()

    def register_skill(self,name:str,steps:Iterable[str])->None:
        s=tuple(steps)
        if not s: raise ValueError("empty skill")
        self.library[name]=s

    def expand(self,goal:GoalNode)->GoalNode:
        if self.steps>=self.budget: raise RuntimeError("reasoning budget exhausted")
        self.steps+=1
        if goal.name in self.library and not goal.children:
            goal.children=[GoalNode(x) for x in self.library[goal.name]]
        for child in goal.children:
            if child.name in goal.depends_on:
                raise ValueError("self dependency")
        return goal

    def topological(self,root:GoalNode)->tuple[str,...]:
        out=[]; seen=set()
        def visit(n:GoalNode):
            if n.name in seen: return
            for dep in n.depends_on:
                if dep==n.name: raise ValueError("cycle")
                out.append(dep) if dep not in seen else None
            seen.add(n.name); out.append(n.name)
            for c in n.children: visit(c)
        visit(root)
        return tuple(dict.fromkeys(out))

    def verify(self,node:GoalNode,predicate:Callable[[GoalNode],bool])->bool:
        if any(not c.verified for c in node.children): return False
        ok=bool(predicate(node))
        node.verified=ok
        node.done=ok
        node.evidence=sha256(repr((node.name,ok,tuple(c.name for c in node.children))).encode()).hexdigest()[:16]
        if ok: self.completed.add(node.name)
        return ok

    def plan(self,root:GoalNode)->tuple[str,...]:
        self.expand(root)
        order=self.topological(root)
        return tuple(x for x in order if x not in self.completed)

    def execute(self,root:GoalNode,verifier:Callable[[GoalNode],bool])->bool:
        self.plan(root)
        nodes={}
        def collect(n):
            nodes[n.name]=n
            for c in n.children: collect(c)
        collect(root)
        pending=set(nodes)
        while pending:
            progressed=False
            for name in tuple(sorted(pending)):
                if self.steps>=self.budget: return False
                n=nodes[name]
                if any(dep not in self.completed for dep in n.depends_on): continue
                if any(not c.verified for c in n.children): continue
                self.steps+=1
                if not self.verify(n,verifier): return False
                pending.remove(name)
                progressed=True
            if not progressed: return False
        return root.verified

    def compose(self,names:Iterable[str])->GoalNode:
        children=[]
        for name in names:
            if name not in self.library: raise KeyError(name)
            children.append(GoalNode(name))
        return GoalNode("composed",children=children)

    def transfer(self,name:str,rename:dict[str,str])->GoalNode:
        if name not in self.library: raise KeyError(name)
        return GoalNode(rename.get(name,name),[GoalNode(rename.get(x,x)) for x in self.library[name]])

    def reset_budget(self,budget:int|None=None)->None:
        self.steps=0
        if budget is not None: self.budget=budget
