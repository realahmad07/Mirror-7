
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class Observation:
    visible: Tuple[Tuple[int,int],...]
    pending: int
    step: int

class StochasticGrid:
    """Small reproducible partially observed environment with delayed feedback."""

    ACTIONS=("R","L","U","D","WAIT")
    def __init__(self, size:int=4, slip:float=0.15, delay:int=1, seed:int=0):
        if size<2 or not 0<=slip<1 or delay<0: raise ValueError("invalid environment bounds")
        self.size=size; self.slip=slip; self.delay=delay; self.rng=random.Random(seed)
        self.pos=(0,0); self.goal=(size-1,size-1); self.step_count=0
        self.pending: List[int]=[]

    def reset(self):
        self.pos=(0,0); self.step_count=0; self.pending=[]
        return self.observe()

    def _move(self, action:str)->Tuple[int,int]:
        x,y=self.pos
        dx,dy={"R":(1,0),"L":(-1,0),"U":(0,1),"D":(0,-1),"WAIT":(0,0)}[action]
        return (min(self.size-1,max(0,x+dx)),min(self.size-1,max(0,y+dy)))

    def observe(self)->Observation:
        x,y=self.pos
        visible=[]
        for dx,dy in ((0,0),(1,0),(-1,0),(0,1),(0,-1)):
            p=(x+dx,y+dy)
            if 0<=p[0]<self.size and 0<=p[1]<self.size:
                visible.append(p)
        return Observation(tuple(sorted(visible)),len(self.pending),self.step_count)

    def step(self, action:str):
        if action not in self.ACTIONS: raise ValueError("invalid action")
        actual=action
        if action!="WAIT" and self.rng.random()<self.slip:
            actual=self.rng.choice(self.ACTIONS[:-1])
        self.pos=self._move(actual)
        reward=1 if self.pos==self.goal else 0
        self.step_count+=1
        self.pending.append(reward)
        if len(self.pending)>self.delay:
            delivered=self.pending.pop(0)
        else:
            delivered=None
        return self.observe(), delivered

    def done(self)->bool:
        return self.pos==self.goal

class SafeEmbodiedController:
    def __init__(self, env:StochasticGrid):
        self.env=env
        self.failures: Dict[Tuple[Tuple[Tuple[int,int],...],str],int]={}

    def choose(self, obs:Observation)->str:
        # Never emits unknown actions; prefers actions that move toward the known goal.
        x,y=self.env.pos; gx,gy=self.env.goal
        candidates=[]
        if x<gx: candidates.append("R")
        if y<gy: candidates.append("U")
        candidates += ["WAIT","L","D"]
        for a in candidates:
            if self.failures.get((obs.visible,a),0)==0:
                return a
        return "WAIT"

    def run(self, max_steps:int=100)->bool:
        obs=self.env.reset()
        for _ in range(max_steps):
            if self.env.done(): return True
            action=self.choose(obs)
            obs,_=self.env.step(action)
        return self.env.done()
