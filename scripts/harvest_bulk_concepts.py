"""CLI entrypoint for Mirror 7's Automated Multi-Source Zero-Bloat Bulk Concept Harvester.

Usage:
    python scripts/harvest_bulk_concepts.py
    python scripts/harvest_bulk_concepts.py --offline
    python scripts/harvest_bulk_concepts.py --phase1-wiki-jsonl path/to/wiki_concepts.jsonl --target-wiki 42000
    python scripts/harvest_bulk_concepts.py --phase2-admit --browser-jsonl ... --plugin-jsonl ... --wiki-jsonl ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirror_crystal.bulk_concept_harvester import (
    harvest_wikipedia_phase2_to_jsonl,
    run_bulk_concept_harvest,
    run_phase2_unified_admission,
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
        default=12000,
        help="Max Wikipedia Vital Articles Level 3+4+5 titles to harvest in legacy mode",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live Wikipedia/arXiv HTTP calls and harvest only local/browser packs",
    )
    parser.add_argument(
        "--phase1-wiki-jsonl",
        type=Path,
        default=None,
        help="Phase 1: Harvest Wikipedia Vital Level 5 + STEM Category Trees into staged JSONL (read-only on vault)",
    )
    parser.add_argument(
        "--target-wiki",
        type=int,
        default=42000,
        help="Target count of quality-gated unique Wikipedia concepts for --phase1-wiki-jsonl",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=9,
        help="Concurrent MediaWiki API worker threads (8-10 respectful concurrency)",
    )
    parser.add_argument(
        "--phase2-admit",
        action="store_true",
        help="Phase 2: Run audit_and_sort_vault + monotonic tier zero-overwrite admission from staged JSONL files",
    )
    parser.add_argument(
        "--browser-jsonl",
        type=Path,
        default=None,
        help="Staged Tier-3 browser-verified JSONL path",
    )
    parser.add_argument(
        "--plugin-jsonl",
        type=Path,
        default=None,
        help="Staged Tier-2.5 verified science plugin JSONL path",
    )
    parser.add_argument(
        "--wiki-jsonl",
        type=Path,
        default=None,
        help="Staged Tier-2 Wikipedia Level 5 + STEM JSONL path",
    )
    args = parser.parse_args()

    if args.phase1_wiki_jsonl is not None:
        meta = harvest_wikipedia_phase2_to_jsonl(
            output_jsonl=args.phase1_wiki_jsonl,
            vault_path=args.vault,
            target_concepts=args.target_wiki,
            max_workers=args.workers,
            progress_callback=lambda msg: print(f"[wiki-harvester] {msg}", flush=True),
        )
        print("=" * 72)
        print("MIRROR 7 PHASE-1 WIKIPEDIA STAGING HARVEST COMPLETE")
        print("=" * 72)
        print(json.dumps(meta, indent=2))
        return 0

    if args.phase2_admit:
        report = run_phase2_unified_admission(
            vault_path=args.vault,
            browser_jsonl=args.browser_jsonl,
            plugin_jsonl=args.plugin_jsonl,
            wiki_jsonl=args.wiki_jsonl,
            run_audit_first=True,
            vacuum=True,
        )
    else:
        report = run_bulk_concept_harvest(
            vault_path=args.vault,
            include_live_wikipedia=not args.offline,
            include_live_arxiv=not args.offline,
            include_python_docs=True,
            max_vital_wikipedia=args.max_vital,
            vacuum=True,
        )
    jpath, mpath = save_harvest_artifacts(report, ROOT / "artifacts")
    print("=" * 72)
    print("MIRROR 7 BULK HARVEST & UNIFIED ADMISSION COMPLETE")
    print("=" * 72)
    print(f"Existing Protected Nodes Before   : {report.existing_concepts_before:,}")
    print(f"Low-Quality Stubs Audited Out     : {report.audit_removed_stubs:,}")
    print(f"Net New Unique Nodes Inserted     : {report.inserted_new_concepts:,}")
    print(f"  - Science Plugins (Tier 2.5)    : {report.plugin_concepts_inserted:,}")
    print(f"  - Wikipedia L5+STEM (Tier 2)    : {report.wiki_concepts_inserted:,}")
    print(f"Lower/Equal Overwrites Prevented  : {report.overwrites_prevented:,}")
    print(f"Canonical Duplicates Blocked      : {report.canonical_duplicates_blocked:,}")
    print(f"Alias Collisions Blocked          : {report.alias_duplicates_blocked:,}")
    print(f"Content-Hash Duplicates Blocked   : {report.content_duplicates_blocked:,}")
    print("-" * 72)
    print(f"Total Unique 5-Facet ConceptNodes : {report.total_unique_concepts:,}")
    print(f"Domains Covered                   : {report.total_domains}")
    print(f"Cross-Concept Combinatorial Pairs : {report.combinatorial_cross_pairs:,}")
    print(f"Open-Ended Discourse Combinations : {report.discourse_operator_combinations:,}")
    print(f"Cumulative Raw Payload Distilled  : {report.raw_bytes_fetched / 1_000_000:.2f} MB")
    print(f"Distilled Node Payload Footprint  : {report.distilled_node_bytes / 1_000_000:.2f} MB")
    print(f"Zero-Bloat Raw-Text Eviction      : {report.compression_eviction_pct:.2f}%")
    print(f"Compacted .dcrystal Vault on Disk : {report.vault_disk_mb:.2f} MB")
    print(f"Cold Load / Warm Init (ms)        : {report.cold_load_ms:.2f} ms / {report.warm_load_ms:.3f} ms")
    print(f"Peak Python Heap RAM (tracemalloc): {report.peak_ram_mb:.2f} MB")
    print(f"O(1) Median Lookup Latency (CPU)  : {report.median_lookup_ms:.3f} ms")
    print(f"Elapsed Time                      : {report.elapsed_seconds:.2f} s")
    print(f"Source Counts                     : {report.source_counts}")
    print(f"Artifacts Saved                   : {jpath.name}, {mpath.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
