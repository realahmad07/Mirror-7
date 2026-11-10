# Restored Mirror Crystal bundle

This folder preserves the two uploaded ZIP contents in one organized layout.

## Canonical layout

```text
restorevalutecrytaled/
├── README.md
├── mirror_crystal/                    # One canonical copy of the 36-file engine/test package
└── mirror_crystal_vault_suite/
    ├── artifacts/
    │   ├── mirror_crystal_v1.mcrystal # 4,146,587-byte serialized engine artifact
    │   └── mirror_eval_report.json    # Archived evaluation report
    ├── tools/
    │   ├── expand_vault_frontier.py
    │   ├── mirror_cli.py
    │   └── teach_dataset.py
    ├── run_cli.bat
    └── run_studio.bat
```

## Organization results

- Both ZIPs contained the **same `mirror_crystal` package byte-for-byte** (identical Git tree SHA). The canonical copy is `restorevalutecrytaled/mirror_crystal/`; the duplicate under `mirror_crystal_vault_suite/mirror_crystal/` has been removed. No unique source or test file was removed.
- The restored package contains 36 tracked Python files: 21 package modules and 15 test modules.
- The suite retains its two original artifacts, two Windows launchers, and three helper tools. The helper tools use the single canonical package.
- The CLI launcher resolves the repository's `mirror7_backend` from the repository root. The Studio launcher starts that backend from the correct working directory.

## Vault artifact status

The production knowledge vault is now **present and verified** in this repository via Git LFS:
- **Vault File:** `restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal`
- **File Size:** `422,084,608 bytes` (~402.5 MB uncompressed; 344.86 MB compressed ZIP)
- **Git LFS Object:** `sha256:87e02dd075c4df9f31df73b943358b75b513ffaaf215123fc735ff13973b9988`
- **SQLite Status:** `PRAGMA integrity_check` verified (`ok`)
- **Knowledge Scale:** **368,270 rules** across 17 domains (Olympiad math ~124k, General reasoning ~116k, Algorithms ~60k, etc.)

In addition:
- `mirror_crystal_vault_suite/artifacts/mirror_crystal_v1.mcrystal` (4,146,587 bytes) remains available as the compact in-RAM binary crystal brain.
- `teach_dataset.py` and `expand_vault_frontier.py` can now point directly to `restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal`.

## Run and test

From the repository root:

```bash
python -m compileall -q restorevalutecrytaled/mirror_crystal restorevalutecrytaled/mirror_crystal_vault_suite/tools
cd restorevalutecrytaled
python -m pytest -q mirror_crystal/tests
```

The archived `mirror_eval_report.json` records 55/55 checks as passed. That is the report stored in the ZIP; it is not a fresh run performed by this organization change. The workflow `.github/workflows/restored-crystal-bundle.yml` now runs the restored package tests when this folder changes.

### Teach a dataset to an existing production vault

```bash
python mirror_crystal_vault_suite/tools/teach_dataset.py --vault "PATH_TO_EXISTING_SQLITE_DCRYSTAL" dataset.jsonl
```

### Expand an existing production vault

```bash
python mirror_crystal_vault_suite/tools/expand_vault_frontier.py --vault "PATH_TO_EXISTING_SQLITE_DCRYSTAL"
```

Both tools fail closed when the specified database is missing or is not SQLite. Never point them at the `.mcrystal` artifact.
