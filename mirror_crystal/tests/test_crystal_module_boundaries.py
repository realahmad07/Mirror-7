from __future__ import annotations
from mirror_crystal.engine import BinarySessionMemory as EngineSessionMemory
from mirror_crystal.engine import CrystalResponse as EngineCrystalResponse
from mirror_crystal.engine import split_compound_intents as engine_split
from mirror_crystal.intent import split_compound_intents
from mirror_crystal.response_types import CrystalResponse
from mirror_crystal.session_memory import BinarySessionMemory

def test_shared_primitives_remain_backward_compatible():
    assert EngineSessionMemory is BinarySessionMemory
    assert EngineCrystalResponse is CrystalResponse
    assert engine_split is split_compound_intents

def test_compound_intent_helper_preserves_single_question():
    parts = split_compound_intents("Explain database indexing and also explain transaction isolation.")
    assert len(parts) == 2
    assert "database indexing" in parts[0]
    assert "transaction isolation" in parts[1]

def test_session_memory_computes_declared_limit_exactly():
    memory = BinarySessionMemory()
    memory.update_from_prompt("session-a", "Our maximum capacity is 20.")
    result = memory.evaluate_constraint_check("session-a", "3 batches of 7 workers")
    assert result is not None
    answer, trace = result
    assert "21" in answer and "exceeds your 20 limit by 1" in answer
    assert any("3 * 7 = 21" in item for item in trace)
