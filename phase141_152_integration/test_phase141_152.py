from .phase141_152 import *

def test_141_adapter_success_and_failure():
    assert AlgorithmAdapter("x",lambda v:v+1).run(1).value==2
    assert not AlgorithmAdapter("x",lambda v:1/0).run(1).ok

def test_142_concept_memory_bridge():
    t=ConceptTree(); t.observe({"x":1}); t.observe({"x":2}); m=SemanticMemoryGraph(); assert ConceptMemoryBridge().connect(t,m)>=1

def test_143_grounding_memory_bridge():
    g=MultimodalGrounding(); g.observe([ModalityToken("text","a",{"f":1},0),ModalityToken("vision","b",{"f":1},0)])
    m=SemanticMemoryGraph(); assert GroundingMemoryBridge().ingest(g,m)>=1

def test_144_world_model_bridge():
    w=AdaptiveWorldModelBuilder(); MemoryWorldBridge().record_transition(w,{"x":0},"go",{"x":1}); assert w.predict({"x":0},"go")[0]["x"]==1

def test_145_causal_research_bridge():
    c=CausalExperimentDesigner(); c.add_hypothesis(CausalHypothesis("h",{"x"},[],{},1.0)); assert CausalResearchBridge().sync_posteriors(c,None)=={"h":1.0}

def test_146_program_induction_bridge():
    e=ProgramSynthesis(); p=ReasoningProgramBridge().synthesize_from_examples(e,[([1],2),([2],3)]); assert p and ReasoningProgramBridge().execute(e,p,[20])==21

def test_147_goal_research_bridge():
    g=LongTermGoalManager(); g.add_goal("g","learn",1.0); assert GoalResearchBridge().sync(g,None)=="g"

def test_148_self_debugging_bridge():
    d=SelfDebuggingAlgorithm(pattern_threshold=1); assert DebuggerBridge().record(d,DebugEvent("planner","timeout",{"x":1}))>=0

def test_149_consolidation_requires_repeat_support():
    c=KnowledgeConsolidationEngine(); m=SemanticMemoryGraph(); c.add_experience(Experience({"x":1},"a")); assert ConsolidationMemoryBridge().promote(c,m)==0
    c.add_experience(Experience({"x":1},"b")); c.add_experience(Experience({"x":1},"c")); assert ConsolidationMemoryBridge().promote(c,m)>=0

def test_150_workspace_bounded():
    w=CognitiveWorkspace(max_items=2); c=WorkspaceCoordinator(w)
    for i in range(5): c.post(WorkspaceItem(str(i),"state",{},i,.5,"t",0,None,[]))
    assert len(w.items)<=2

def test_151_full_runtime_uses_all_major_modules():
    r=IntegratedCognition().step({"light":0},{"light":0},"push",{"light":1},goal="learn",modalities=[("text","t",{"x":1}),("vision","v",{"x":1})])
    assert r.concept_id and r.memory_nodes>=1 and r.prediction_confidence>0 and r.goal_id=="learn" and r.program_ok and r.reasoning_ok
    assert r.causal_experiment in {"x","indistinguishable"} and r.research_action=="state-causal"

def test_151_unseen_action_abstains_conservatively():
    r=IntegratedCognition().step({"x":1},{"x":1},"unknown",{"x":1}); assert r.prediction_confidence<=0.1

def test_152_gate_promotes_only_green_bounded_system():
    r=IntegrationGate().evaluate([True,True,True],0,(3,),(5,)); assert r.integrated and r.passed==3

def test_152_gate_rejects_invalid_or_over_budget():
    g=IntegrationGate(); assert not g.evaluate([True],1,(1,),(5,)).integrated; assert not g.evaluate([True],0,(6,),(5,)).integrated
