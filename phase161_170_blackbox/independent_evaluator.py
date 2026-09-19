from __future__ import annotations
import json
import sys

def evaluate(payload):
    required = ("raw_representation_accuracy","opaque_action_accuracy","goal_success","contamination_detected","max_memory")
    if any(k not in payload for k in required):
        return {"passed": False, "reason": "missing_metric"}
    checks = {
        "raw_representation": payload["raw_representation_accuracy"] >= 0.75,
        "opaque_actions": payload["opaque_action_accuracy"] >= 0.75,
        "hidden_goal": payload["goal_success"] >= 0.75,
        "contamination_control": bool(payload["contamination_detected"]),
        "bounded_memory": payload["max_memory"] <= 1000,
    }
    return {"passed": all(checks.values()), "checks": checks}

def main():
    result = evaluate(json.load(sys.stdin))
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__":
    main()
