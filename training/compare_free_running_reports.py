"""Compare two frozen native free-running diagnostic reports without loading a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.free_running_eval import compare_evaluation_reports


def compare_report_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    baseline_file = Path(baseline_path)
    candidate_file = Path(candidate_path)
    report = compare_evaluation_reports(
        json.loads(baseline_file.read_text(encoding="utf-8")),
        json.loads(candidate_file.read_text(encoding="utf-8")),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = compare_report_files(args.baseline, args.candidate, args.output)
    print(json.dumps({
        "paired_samples": report["paired_samples"],
        "deltas": report["deltas"],
        "interpretation_boundary": report["interpretation_boundary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
