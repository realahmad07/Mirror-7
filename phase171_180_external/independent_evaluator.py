"""Independent evaluator; intentionally imports no Mirror 7 runtime."""
from __future__ import annotations
import json, sys
from typing import Any, Mapping

def evaluate_pack_result(payload: Mapping[str, Any]) -> dict:
    required = {"total_episodes","completed_episodes","successes","contamination","pack_fingerprint"}
    if not required.issubset(payload):
        return {"passed": False, "reason": "missing_metric"}
    total = int(payload["total_episodes"])
    completed = int(payload["completed_episodes"])
    successes = int(payload["successes"])
    contamination = bool(payload["contamination"])
    fingerprint = str(payload["pack_fingerprint"])
    checks = {
        "nonempty_pack": total > 0,
        "all_completed": completed == total,
        "all_successful": successes == total,
        "no_contamination": not contamination,
        "fingerprint_present": len(fingerprint) == 64,
    }
    return {"passed": all(checks.values()), "checks": checks}

def main() -> None:
    result = evaluate_pack_result(json.load(sys.stdin))
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__":
    main()
