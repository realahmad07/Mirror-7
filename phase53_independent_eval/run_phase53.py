from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import build_blind_suite, leakage_audit, run_suite, summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", nargs="+", required=True)
    parser.add_argument("--report", default="phase53_result.json")
    args = parser.parse_args()

    tasks = build_blind_suite()
    leakage = leakage_audit(tasks)
    print(f"PHASE 53 BLIND SUITE: {len(tasks)} episodes")
    print(f"TRAIN EPISODES: {sum(t.split == 'train' for t in tasks)}")
    print(f"HELD-OUT EPISODES: {sum(t.split == 'heldout' for t in tasks)}")
    print(f"LEAKAGE AUDIT: {'PASS' if not leakage else 'FAIL'}")
    if leakage:
        for item in leakage:
            print("  ", item)
        return 2

    scores = run_suite(args.agent)
    for score in scores:
        print(f"[{'PASS' if score.solved else 'FAIL'}] {score.split:7} {score.task_id}: family={score.family} steps={score.steps} invalid={score.invalid_actions}")

    summary = summarize(scores)
    print(json.dumps(summary, indent=2, sort_keys=True))
    Path(args.report).write_text(
        json.dumps({"summary": summary, "episodes": [s.__dict__ for s in scores]},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("PHASE 53 GENERALIZATION GATE:", "PASS" if summary["passed"] else "FAIL")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
