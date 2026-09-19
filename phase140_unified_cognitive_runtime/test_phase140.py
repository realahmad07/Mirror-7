from .mirror7_phase140 import UnifiedCognitiveRuntime
from phase135_research_scheduler import ResearchTask
def test_integrated_runtime():
 r=UnifiedCognitiveRuntime().step([("state_a",.8),("state_a",.7),("state_b",.2)],goal="learn",research_tasks=[ResearchTask("probe",.9,.9,.9,1)],views=["Hello",(1,2)])
 assert r.state=="state_a" and r.goal=="learn" and r.research=="probe" and r.rules==0
def test_runtime_handles_unknown_state():
 assert UnifiedCognitiveRuntime().step([]).state is None
def test_runtime_grounding_is_present():
 assert "text:" in UnifiedCognitiveRuntime().step([],views=["X"]).grounded
