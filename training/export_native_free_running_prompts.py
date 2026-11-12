"""Export the repository's legacy held-out response cases for a V5 smoke run.

This is a convenience bridge to the existing 10-dimension evaluator, not a
source-disjoint or promotion-grade benchmark. See the manifest emitted beside
the JSONL for explicit limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training.evaluation_suite.benchmark_dataset import load_evaluation_benchmark


def build_prompt_cases(partition: str = "held_out") -> list[dict[str, Any]]:
    if partition not in {"dev", "held_out"}:
        raise ValueError("partition must be 'dev' or 'held_out'")
    examples = load_evaluation_benchmark(partition=partition)
    if not examples:
        raise ValueError(f"no evaluation examples found for partition {partition!r}")

    cases = []
    for example in examples:
        target = example.reference_targets[0] if example.reference_targets else None
        cases.append({
            "case_id": example.example_id,
            "prompt": example.prompt,
            "target": target,
            "dimension": example.dimension.value,
            "source_partition": example.partition,
        })
    return cases


def export_prompt_bundle(
    output: str | Path,
    *,
    partition: str = "held_out",
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = build_prompt_cases(partition)
    jsonl = "".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
        for case in cases
    )
    encoded = jsonl.encode("utf-8")
    output_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()

    manifest = {
        "format": "mirror7-native-free-running-prompt-bundle-v1",
        "source": "training.evaluation_suite.benchmark_dataset",
        "source_partition": partition,
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "prompt_file": str(output_path),
        "prompt_file_sha256": digest,
        "benchmark_class": "legacy_repository_smoke",
        "promotion_eligible": False,
        "limitations": [
            "This repository-held-out partition has not been audited for disjointness from every external/Drive training corpus.",
            "Keyword, lexical and exact-match metrics are diagnostics, not semantic or factual correctness.",
            "Use a separately sourced, rights-reviewed, frozen benchmark with independent labels for promotion.",
        ],
    }
    resolved_manifest = Path(manifest_path) if manifest_path is not None else Path(str(output_path) + ".manifest.json")
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(resolved_manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("held_out", "dev"), default="held_out")
    parser.add_argument("--output", required=True, help="Destination JSONL prompt bundle")
    parser.add_argument("--manifest", default=None, help="Optional manifest output path")
    args = parser.parse_args(argv)

    report = export_prompt_bundle(
        args.output,
        partition=args.partition,
        manifest_path=args.manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
