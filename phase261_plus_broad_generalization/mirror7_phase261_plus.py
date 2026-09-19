from typing import Any, Callable, Dict, List

class LeakageError(Exception):
    pass

class ExternalGeneralizationHarness:
    def __init__(self, agent_logic: Callable):
        self.agent_logic = agent_logic
        self.training_context = {"data": "training_corpus"}

    def evaluate_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # Strict isolation check
        if "training_corpus" in str(context) or "training_corpus" in str(task):
            raise LeakageError("Training data leaked into evaluation context.")
            
        return self.agent_logic(task, context)

class AGIEvaluationGate:
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def evaluate(self, harness: ExternalGeneralizationHarness, tasks: List[Dict[str, Any]]) -> bool:
        successes = 0
        for task in tasks:
            result = harness.evaluate_task(task, {"eval_mode": True})
            if result.get("success"):
                successes += 1
        
        score = successes / len(tasks) if tasks else 0.0
        # Will not return True (AGI) unless score >= threshold (1.0), which is essentially impossible for broad unseen domains.
        return score >= self.threshold
