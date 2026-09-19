"""Independent-style evaluator entry point for Phase 158/160.
It accepts only a summary JSON object and applies fixed acceptance predicates.
"""
from __future__ import annotations
import json, sys

def evaluate(payload):
    required = ["prediction_accuracy", "unknown_confidence", "max_memory", "max_workspace"]
    if any(k not in payload for k in required):
        return {"passed": False, "reason": "missing_metric"}
    checks = {
        "prediction_accuracy": payload["prediction_accuracy"] >= 0.75,
        "unknown_confidence": payload["unknown_confidence"] <= 0.10,
        "max_memory": payload["max_memory"] <= 1000,
        "max_workspace": payload["max_workspace"] <= 100,
    }
    return {"passed": all(checks.values()), "checks": checks}

def main():
    payload = json.load(sys.stdin)
    result = evaluate(payload)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__":
    main()
