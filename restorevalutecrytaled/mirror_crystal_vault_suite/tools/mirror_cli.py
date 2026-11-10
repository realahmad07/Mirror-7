#!/usr/bin/env python3
"""Compatibility launcher for the canonical repository-root Mirror 7 CLI."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "mirror_cli.py").is_file()
        and (parent / "mirror7_backend" / "service.py").is_file()
    ),
    None,
)
if REPO_ROOT is None:
    raise SystemExit("Mirror 7 repository root was not found above this launcher.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirror_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
