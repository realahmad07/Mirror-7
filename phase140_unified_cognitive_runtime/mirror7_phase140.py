from dataclasses import dataclass
from phase129_cognitive_workspace import CognitiveWorkspace
from phase130_state_fusion import StateFusion
from phase134_goal_manager import GoalManager
from phase135_research_scheduler import ResearchScheduler
from phase136_self_debugger import SelfDebugger
from phase137_consolidation_engine import ConsolidationEngine
from phase139_multimodal_grounder import MultimodalGrounder

@dataclass(frozen=True)
class RuntimeReport:
    state:object
    goal:str|None
    research:str|None
    grounded:str
    rules:int

class UnifiedCognitiveRuntime:
    """Bounded coordination layer for generalization mechanisms."""
    def __init__(self):
        self.workspace=CognitiveWorkspace()
        self.fusion=StateFusion()
        self.goals=GoalManager()
        self.research=ResearchScheduler()
        self.debugger=SelfDebugger()
        self.consolidator=ConsolidationEngine()
        self.grounder=MultimodalGrounder()
    def step(self,observations,goal=None,research_tasks=(),views=()):
        fused=self.fusion.fuse(observations)
        if fused:
            self.workspace.publish("state",fused.value,fused.confidence,"fusion")
            self.consolidator.observe("state",fused.value)
        if goal and goal not in self.goals.goals:
            self.goals.add(goal,1.0)
        chosen=self.research.select(list(research_tasks),10) if research_tasks else None
        grounded=self.grounder.ground(list(views)) if views else self.grounder.ground([])
        self.workspace.publish("grounding",grounded.fingerprint,1.0,"grounder")
        g=self.goals.next()
        return RuntimeReport(fused.value if fused else None,g.name if g else None,chosen.name if chosen else None,grounded.fingerprint,len(self.consolidator.rules()))
