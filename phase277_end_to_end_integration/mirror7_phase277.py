
from typing import Callable, Dict, List, Optional, Tuple
from phase267_general_representation.mirror7_phase267 import GeneralRepresentation
from phase268_compositional_semantics.mirror7_phase268 import CompositionalSemantics
from phase269_open_affordance_discovery.mirror7_phase269 import AffordanceModel
from phase270_long_horizon_planning.mirror7_phase270 import LongHorizonPlanner

class Mirror7EndToEnd:
    """Small end-to-end bridge: raw input -> language -> action evidence -> plan."""

    def __init__(self):
        self.representation=GeneralRepresentation()
        self.language=CompositionalSemantics()
        self.affordance=AffordanceModel()
        self.transitions=[]

    def learn(self,before_raw:bytes,instruction:str,action:str,before_state:Tuple[int,...],after_state:Tuple[int,...]):
        self.representation.discover(before_raw)
        self.language.learn_primitive(instruction,action)
        self.affordance.observe(before_state,(action,()),after_state)
        self.transitions.append((before_raw,action,before_state,after_state))

    def plan(self,start:Tuple[int,...],instruction:str,goal:Callable[[Tuple[int,...]],bool],
             max_depth:int=12)->Optional[List[str]]:
        try:
            expr=self.language.parse(instruction)
        except ValueError:
            return None
        allowed=self.language.execute_plan(expr)
        actions=list(dict.fromkeys(allowed))
        if not actions:
            return None
        def model(state,action):
            return self.affordance.effect(tuple(state),(action,()))
        p=LongHorizonPlanner(model,actions,max_expansions=5000)
        return p.plan(start,goal,max_depth=max_depth)
