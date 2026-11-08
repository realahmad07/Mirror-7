"""CLI entry point: `python -m mirror_eval`."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .runner import format_eval_report, run_full_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Mirror 7 honest independent CPU evaluation.")
    parser.add_argument(
        "--no-prod-vault",
        action="store_true",
        help="Skip probing artifacts/mirror_crystal_all_datasets.dcrystal.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/mirror_eval_report.json",
        help="Path to write JSON evaluation report.",
    )
    args = parser.parse_args(argv)

    report = run_full_eval(
        check_production_vault=not args.no_prod_vault,
        save_json_path=Path(args.out),
    )
    print(format_eval_report(report))
    return 0 if report["overall_passed"] == report["overall_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
