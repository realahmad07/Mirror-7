import pytest
from .mirror7_phase261_plus import ExternalGeneralizationHarness, AGIEvaluationGate, LeakageError

def mock_agent(task, context):
    if task.get("id") == "task1":
        return {"success": True}
    return {"success": False}

def test_unseen_task_acceptance():
    harness = ExternalGeneralizationHarness(mock_agent)
    res = harness.evaluate_task({"id": "task1"}, {})
    assert res["success"] is True

def test_no_data_leakage():
    harness = ExternalGeneralizationHarness(mock_agent)
    with pytest.raises(LeakageError):
        harness.evaluate_task({"id": "task2"}, {"knowledge": "training_corpus"})

def test_agi_gate_conservative_rejection():
    harness = ExternalGeneralizationHarness(mock_agent)
    gate = AGIEvaluationGate(threshold=1.0)
    
    # Mix of pass and fail tasks
    tasks = [{"id": "task1"}, {"id": "unknown_task"}]
    
    # Should not claim AGI if it doesn't pass all tests perfectly (and in reality, ever)
    assert gate.evaluate(harness, tasks) is False
