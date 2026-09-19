from dataclasses import dataclass
@dataclass(frozen=True)
class Goal:
    name:str; priority:float; parent:str|None; progress:float
class GoalManager:
    def __init__(self,max_goals=64): self.max_goals=max_goals; self.goals={}
    def add(self,name,priority=1.0,parent=None):
        if len(self.goals)>=self.max_goals and name not in self.goals: return False
        self.goals[name]=Goal(name,float(priority),parent,0.0); return True
    def update(self,name,progress):
        if name not in self.goals: return False
        g=self.goals[name]; self.goals[name]=Goal(g.name,g.priority,g.parent,max(0.0,min(1.0,float(progress)))); return True
    def next(self):
        active=[g for g in self.goals.values() if g.progress<1.0]
        return max(active,key=lambda g:(g.priority,-g.progress),default=None)
