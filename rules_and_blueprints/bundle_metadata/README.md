# Mirror Crystal Publication Bundle v2

This bundle contains the original four Crystal ZIP archives and the General Reasoning candidate pack for publication to the public Mirror-7 repository.

## Contents

- `raw_archives/crystal_1.zip` through `crystal_4.zip` — unchanged user-provided archives.
- `raw_archives/Mirror_General_Reasoning_10K_Candidate_Pack_v1.zip` — 10,266 General Reasoning candidate records.
- `manifest.json` — sizes, SHA-256 checksums, record counts, evidence and licence metadata.
- `LICENSES.md` — source-specific licence and attribution register.
- `PUBLISH_TO_GITHUB.md` — exact Codex/Antigravity publication steps.
- `SHA256SUMS.txt` — bundle-file checksums.

## Publish to GitHub

The GitHub branch already contains the ZIP-loading runtime, response test, publisher script, data card, and source-specific licence documentation. Use the branch named in `manifest.json`.

From the Mirror-7 checkout:

```powershell
git fetch origin datasets/crystal-archive-integration-2026-11-10
git checkout datasets/crystal-archive-integration-2026-11-10
python scripts/publish_crystal_bundle.py --bundle "C:\path\to\Mirror_Crystal_Publication_Bundle_v2.zip" --replace-legacy-mcrystal --commit --push
```

The publisher verifies bundle checksums and each ZIP, copies all five ZIPs into `datasets/crystal_archive/raw/`, writes a raw manifest, tests ZIP loading and Mirror responses, then removes `artifacts/mirror_crystal_v1.mcrystal` only after checking its bytes are identical to the MCRYSTAL payload embedded in `crystal_4.zip`. It commits and pushes the changes to the selected branch.

## Runtime access after publication

Any AI or developer can access the public archives by cloning the repository or downloading the files under `datasets/crystal_archive/raw/`. Mirror itself can load its CPU brain directly from the archive:

```bash
python -m mirror_crystal query --brain datasets/crystal_archive/raw/crystal_4.zip --prompt "What is protocol code 1000?"
```

Run a wider response probe:

```bash
python scripts/test_crystal_archive_responses.py --brain datasets/crystal_archive/raw/crystal_4.zip
```

## Data and licence status

This is a mixed-source dataset. The root Apache-2.0 licence does not grant a blanket licence over every source record. The OASST2-derived subset has an Apache-2.0 source declaration; several other inputs have incomplete or absent per-record provenance and are marked `NOASSERTION`. The licence register preserves those distinctions. Confirm rights before any broader downstream redistribution.

The 368,270 figure is a taxonomy claim for a separate full database; that database is not included in these five ZIPs. The embedded MCRYSTAL has 6,798 rules and reports zero sandbox-verified rules. The 500-row sample does not include per-rule verification artefacts. The General Reasoning pack's 10,240 synthetic entries are 320 methods specialised across 32 contexts, not 10,240 independently verified discoveries.
