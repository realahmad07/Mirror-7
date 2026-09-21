from dataclasses import dataclass
from collections.abc import Mapping
from phase129_cognitive_workspace import CognitiveWorkspace
from phase130_state_fusion import StateFusion
from phase134_goal_manager import GoalManager
from phase135_research_scheduler import ResearchScheduler
from phase136_self_debugger import SelfDebugger
from phase137_consolidation_engine import ConsolidationEngine
from phase139_multimodal_grounder import MultimodalGrounder
from phase343_semantic_state import SemanticStateInducer, SemanticState

@dataclass(frozen=True)
class RuntimeReport:
    state:object
    goal:str|None
    research:str|None
    grounded:str
    rules:int
    semantic_state:SemanticState|None=None

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
        self.semantic_inducer=SemanticStateInducer()
        self.semantic_state:SemanticState|None=None
    def step(self,observations,goal=None,research_tasks=(),views=()):
        if isinstance(observations, str) and observations.strip():
            self.semantic_state = self.semantic_inducer.discover(
                observations, previous=self.semantic_state
            )
            semantic_payload = self.semantic_state.as_dict()
            self.workspace.publish("semantic_state", semantic_payload, 1.0, "semantic_inducer")
        if isinstance(observations, Mapping):
            fusion_input=[(observations, 1.0)]
        elif isinstance(observations, tuple) and len(observations) == 2:
            fusion_input=[observations]
        elif isinstance(observations, list):
            fusion_input=observations
        else:
            fusion_input=[(observations, 1.0)]
        fused=self.fusion.fuse(fusion_input)
        if fused:
            self.workspace.publish("state",fused.value,fused.confidence,"fusion")
            self.consolidator.observe("state",fused.value)
        if goal and goal not in self.goals.goals:
            self.goals.add(goal,1.0)
        chosen=self.research.select(list(research_tasks),10) if research_tasks else None
        ground_views = list(views) if views else [observations]
        grounded=self.grounder.ground(ground_views)
        self.workspace.publish("grounding",grounded.fingerprint,1.0,"grounder")
        g=self.goals.next()
        return RuntimeReport(fused.value if fused else None,g.name if g else None,chosen.name if chosen else None,grounded.fingerprint,len(self.consolidator.rules()),self.semantic_state)
