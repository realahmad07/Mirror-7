from phase299_capability_frontier import CapabilityFrontier, CapabilityTarget
from phase306_sequence_frontier_adapter import SequenceFrontierAdapter
from phase308_planning_frontier_adapter import PlanningFrontierAdapter
from phase309_language_frontier_adapter import LanguageFrontierAdapter
from phase310_reasoning_frontier_adapter import ReasoningFrontierAdapter
from phase311_adapter_registry import AdapterRegistry

def build_default_adapters():
    return AdapterRegistry([SequenceFrontierAdapter(),PlanningFrontierAdapter(),LanguageFrontierAdapter(),ReasoningFrontierAdapter()])

def build_default_frontier():
    return CapabilityFrontier([
        CapabilityTarget("sequence_extrapolation",0.0,1.0,1.0),
        CapabilityTarget("planning",0.0,1.0,1.1),
        CapabilityTarget("compositional_language",0.5,1.0,1.2),
        CapabilityTarget("symbolic_reasoning",0.4,1.0,1.3),
    ])
