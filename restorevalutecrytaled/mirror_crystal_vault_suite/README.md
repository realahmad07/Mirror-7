# Mirror Crystal vault suite

This directory contains the serialized engine artifact, archived evaluation report, helper tools, and Windows launchers.

- `artifacts/mirror_crystal_v1.mcrystal`: serialized Mirror Crystal engine snapshot.
- `artifacts/mirror_eval_report.json`: archived evaluation report.
- `tools/`: frontier expansion, dataset teaching, and interactive backend CLI.
- `run_cli.bat` / `run_studio.bat`: Windows launchers.

The shared `mirror_crystal` package lives one directory above this suite at `../mirror_crystal/`; there is no duplicate package here.

The MCRYSTAL artifact is not a SQLite `.dcrystal` file. The teaching and frontier tools require an existing SQLite vault path via `--vault` and will not create one implicitly.
