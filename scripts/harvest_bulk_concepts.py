"""CLI entrypoint for Mirror 7's Automated Multi-Source Zero-Bloat Bulk Concept Harvester.

Usage:
    python scripts/harvest_bulk_concepts.py
    python scripts/harvest_bulk_concepts.py --max-vital 1050
    python scripts/harvest_bulk_concepts.py --offline
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirror_crystal.bulk_concept_harvester import (
    run_bulk_concept_harvest,
    save_harvest_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror 7 Automated Multi-Source Bulk Concept Harvester")
    parser.add_argument(
        "--vault",
        type=Path,
        default=ROOT / "artifacts" / "mirror_uol_skills.dcrystal",
        help="Target .dcrystal SQLite vault path",
    )
    parser.add_argument(
        "--max-vital",
        type=int,
        default=2200,
        help="Max Wikipedia Vital Articles Level 3+4 titles to harvest",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live Wikipedia/arXiv HTTP calls and harvest only local/browser packs",
    )
    args = parser.parse_args()

    report = run_bulk_concept_harvest(
        vault_path=args.vault,
        include_live_wikipedia=not args.offline,
        include_live_arxiv=not args.offline,
        include_python_docs=True,
        max_vital_wikipedia=args.max_vital,
    )
    jpath, mpath = save_harvest_artifacts(report, ROOT / "artifacts")
    print("=" * 72)
    print("MIRROR 7 AUTOMATED BULK CONCEPT HARVEST COMPLETE")
    print("=" * 72)
    print(f"Total Unique 5-Facet ConceptNodes : {report.total_unique_concepts:,}")
    print(f"Domains Covered                   : {report.total_domains}")
    print(f"Cross-Concept Combinatorial Pairs : {report.combinatorial_cross_pairs:,}")
    print(f"Open-Ended Discourse Combinations : {report.discourse_operator_combinations:,}")
    print(f"Raw Payload Distilled             : {report.raw_bytes_fetched / 1_000_000:.2f} MB")
    print(f"Distilled Vault Footprint         : {report.distilled_node_bytes / 1_000_000:.2f} MB")
    print(f"Zero-Bloat Raw-Text Eviction      : {report.compression_eviction_pct:.2f}%")
    print(f"O(1) Median Lookup Latency (CPU)  : {report.median_lookup_ms:.3f} ms")
    print(f"Elapsed Time                      : {report.elapsed_seconds:.2f} s")
    print(f"Source Counts                     : {report.source_counts}")
    print(f"Artifacts Saved                   : {jpath.name}, {mpath.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
