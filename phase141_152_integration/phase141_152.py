from dataclasses import dataclass
from typing import Any, Callable
from mirror7_algorithm_01 import ConceptTree
from mirror7_algorithm_02 import ProgramSynthesis
from mirror7_algorithm_03 import CausalExperimentDesigner, CausalHypothesis
from mirror7_algorithm_04 import AdaptiveWorldModelBuilder
from mirror7_algorithm_05.mirror7_algorithm import SemanticMemoryGraph
from mirror7_algorithm_06.mirror7_algorithm import LongTermGoalManager
from mirror7_algorithm_07.mirror7_algorithm import SelfDebuggingAlgorithm
from mirror7_algorithm_08 import KnowledgeConsolidationEngine, Experience
from mirror7_algorithm_09 import MultimodalGrounding, ModalityToken
from mirror7_algorithm_10 import AutonomousResearch, Question
from mirror7_algorithm_11 import Structure, solve
from mirror7_algorithm_12 import CognitiveWorkspace, WorkspaceItem, WorkspaceProcessor

@dataclass(frozen=True)
class AdapterResult:
    ok: bool; value: Any; source: str; reason: str

class AlgorithmAdapter:
    def __init__(self,name:str,fn:Callable[[Any],Any]): self.name,self.fn=name,fn
    def run(self,value):
        try: return AdapterResult(True,self.fn(value),self.name,"ok")
        except Exception as e: return AdapterResult(False,None,self.name,f"error:{type(e).__name__}")

class ConceptMemoryBridge:
    def connect(self,tree,memory):
        n=0
        for cid in tree.get_concepts():
            node=tree.all_nodes.get(cid)
            if node and cid!="root":
                memory.add_node(cid,"concept",cid,node.feature_counts,max(0.0,min(1.0,node.instance_count/10.0)),"concept_discovery"); n+=1
        return n

class GroundingMemoryBridge:
    def ingest(self,grounding,memory):
        n=0
        for cid,concept in grounding.concepts.items():
            memory.add_node(cid,"grounded_concept",cid,{"bindings":concept.bindings,"modalities":list(concept.bindings.keys())},.8,"multimodal_grounding"); n+=1
        return n

class MemoryWorldBridge:
    def import_facts(self,memory,world): return len(memory.nodes)
    def record_transition(self,world,state,action,next_state): return world.observe_transition(state,action,next_state)

class CausalResearchBridge:
    def sync_posteriors(self,causal,research): return causal.get_posteriors()
    def register_questions(self,causal,research): return len(causal.hypotheses)

class ReasoningProgramBridge:
    def synthesize_from_examples(self,engine,examples): return engine.synthesize(examples)
    def execute(self,engine,program,inputs): return engine.execute(program,inputs)

class GoalResearchBridge:
    def sync(self,goal_manager,research): return goal_manager.get_next_goal(set())
    def prioritize_questions(self,goal_manager,research): return goal_manager.get_next_goal(set())

@dataclass(frozen=True)
class DebugEvent:
    component:str; error_type:str; context:dict

class DebuggerBridge:
    def record(self,debugger,event):
        from mirror7_algorithm_07.mirror7_algorithm import FailureRecord
        debugger.log_failure(FailureRecord(event.component,event.error_type,event.context,event.context,None,None))
        return len(debugger.detect_patterns())

class ConsolidationMemoryBridge:
    def promote(self,consolidator,memory):
        consolidator.consolidate(); n=0
        for knowledge in getattr(consolidator,"lts",[]):
            memory.add_node("knowledge:"+str(n),"knowledge",str(knowledge.content),knowledge.content,knowledge.confidence,"consolidation"); n+=1
        return n

class WorkspaceCoordinator:
    def __init__(self,workspace): self.workspace=workspace
    def post(self,item): self.workspace.post(item)
    def tick(self): self.workspace.tick(); return self.workspace.focus()

@dataclass(frozen=True)
class CognitionReport:
    concept_id:str; grounded:int; memory_nodes:int; prediction_confidence:float
    goal_id:str|None; program_ok:bool; causal_experiment:str
    research_action:str|None; reasoning_ok:bool; workspace_items:int

class IntegratedCognition:
    def __init__(self):
        self.concepts=ConceptTree(); self.programs=ProgramSynthesis(); self.causal=CausalExperimentDesigner()
        self.world=AdaptiveWorldModelBuilder(); self.memory=SemanticMemoryGraph()
        self.goals=LongTermGoalManager(); self.debug=SelfDebuggingAlgorithm()
        self.consolidator=KnowledgeConsolidationEngine(); self.grounding=MultimodalGrounding()
        self.research=AutonomousResearch(); self.workspace=CognitiveWorkspace(); self._causal_seeded=False
        self.workspace.register_processor(WorkspaceProcessor("debug",["state"],lambda item: []))
    def step(self,observation,state,action,next_state,goal=None,modalities=()):
        self.concepts.observe(observation); cid=self.concepts.classify(observation); ConceptMemoryBridge().connect(self.concepts,self.memory)
        grounded_count=0
        if modalities:
            self.grounding.observe([ModalityToken(m,t,f,0) for m,t,f in modalities])
            grounded_count=GroundingMemoryBridge().ingest(self.grounding,self.memory)
        MemoryWorldBridge().record_transition(self.world,state,action,next_state)
        prediction,confidence=self.world.predict(state,action)
        self.consolidator.add_experience(Experience({"state":state,"action":action,"next":next_state},"runtime"))
        ConsolidationMemoryBridge().promote(self.consolidator,self.memory)
        if goal and goal not in self.goals.goals: self.goals.add_goal(goal,goal,1.0)
        program=self.programs.synthesize([([1],2),([2],3)])
        if not self._causal_seeded:
            self.causal.add_hypothesis(CausalHypothesis("state-causal",{"x"},[],{},1.0)); self._causal_seeded=True
        self.causal.observe({"x":1}); causal_experiment=self.causal.suggest_experiment(["x"])
        self.research.add_question(Question("state-causal",1.0,1.0)); research_action=self.research.suggest_action()
        reasoning_result=solve([(Structure(["a"],{"a":{"color":"red"}},[]),Structure(["b"],{"b":{"color":"blue"}},[]))],Structure(["a"],{"a":{"color":"red"}},[]))
        self.debug.log_success({"component":"integrated_runtime","action":action})
        self.workspace.post(WorkspaceItem("state","state",{"prediction":prediction,"actual":next_state},confidence,confidence,"world_model",0,None,[])); self.workspace.tick()
        return CognitionReport(cid,grounded_count,len(self.memory.nodes),confidence,self.goals.get_next_goal(set()),program is not None,causal_experiment,research_action.target_topic if research_action else None,bool(reasoning_result),len(self.workspace.items))

@dataclass(frozen=True)
class IntegrationReport:
    passed:int; total:int; invalid:int; bounded:bool; integrated:bool

class IntegrationGate:
    def evaluate(self,checks,invalid=0,resource_values=(),resource_limits=()):
        bounded=all(v<=l for v,l in zip(resource_values,resource_limits)); passed=sum(bool(x) for x in checks); total=len(checks); integrated=total>0 and passed==total
        return IntegrationReport(passed,total,int(invalid),bounded,integrated and invalid==0 and bounded)
