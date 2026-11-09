"""Create blinded reviewer packets without fabricating review decisions.

The packet is an administrative aid only. It does not approve cases or change
the frozen benchmark manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .factual_benchmark import load_factual_dataset


def build_review_packet(
    dataset_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cases, digest = load_factual_dataset(
        dataset_path,
        expected_sha256=manifest.get("dataset_sha256"),
        require_independent_review=False,
    )
    packet_cases = []
    for case in cases:
        packet_cases.append({
            "case_id": case["case_id"],
            "prompt": case["prompt"],
            "answerable": bool(case["answerable"]),
            "claims": [
                {
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["claim_text"],
                    "source_publisher": claim["source_publisher"],
                    "source_url": claim["source_url"],
                    "source_locator": claim["source_locator"],
                    "source_quote": claim["source_quote"],
                    "source_excerpt_sha256": claim["source_excerpt_sha256"],
                    "license_spdx": claim["license_spdx"],
                    "required_fragments": claim.get("required_fragments", []),
                    "reviews": [],
                }
                for claim in case.get("claims", [])
            ],
            "case_reviews": [],
            "required_independent_reviewers": 2,
        })
    return {
        "packet": "mirror7-independent-factual-review-v1",
        "dataset_sha256": digest,
        "source_review_status": manifest.get("review_status", "unknown"),
        "case_count": len(packet_cases),
        "notice": (
            "No reviews have been created. Independent reviewers must inspect "
            "the primary source and record their own decision, method, identity, "
            "and timezone-aware timestamp. Never fill these fields on another "
            "reviewer's behalf."
        ),
        "cases": packet_cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    packet = build_review_packet(args.dataset, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "case_count": packet["case_count"],
        "review_status": "pending",
        "dataset_sha256": packet["dataset_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
