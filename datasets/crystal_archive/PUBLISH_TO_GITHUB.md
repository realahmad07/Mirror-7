# Publish the Crystal archives to this public repository

The source ZIPs are intentionally kept in the owner-facing publication bundle until the repository owner performs the publication step. The ZIPs are user-provided, mixed-source datasets; see [LICENSES.md](LICENSES.md) before redistribution.

## One-command publication

1. Download and extract `Mirror_Crystal_Publication_Bundle_v2.zip` outside the Mirror-7 checkout.
2. From the Mirror-7 checkout, run:

```powershell
git fetch origin datasets/crystal-archive-integration-2026-11-10
git checkout datasets/crystal-archive-integration-2026-11-10
python scripts/publish_crystal_bundle.py --bundle "C:\path\to\Mirror_Crystal_Publication_Bundle_v2.zip" --replace-legacy-mcrystal --commit --push
```

The publisher:

- verifies the bundle's SHA-256 manifest and all five source ZIPs;
- copies the four Crystal ZIPs and the General Reasoning 10K pack to `datasets/crystal_archive/raw/`;
- writes a fresh `raw_manifest.json` with checksums;
- loads the MCRYSTAL brain directly from `crystal_4.zip`;
- runs the ZIP-load regression test and the Mirror response probes;
- removes `artifacts/mirror_crystal_v1.mcrystal` only when its bytes exactly match the brain embedded in `crystal_4.zip` and both tests pass;
- commits and pushes the archive files and duplicate-file removal.

If any checksum or response test fails, the script stops before deleting the standalone brain file.

## Verify after publication

The public raw files will be browsable/downloadable from:

`https://github.com/realahmad07/Mirror-7/tree/datasets/crystal-archive-integration-2026-11-10/datasets/crystal_archive/raw`

Any AI or developer can clone the repository or download the individual ZIPs. No special API key or model provider is needed to read the JSONL/JSON data.

Run Mirror directly against the embedded CPU brain:

```bash
python -m mirror_crystal query --brain datasets/crystal_archive/raw/crystal_4.zip --prompt "What is protocol code 1000?"
python scripts/test_crystal_archive_responses.py --brain datasets/crystal_archive/raw/crystal_4.zip
```

The four archives do **not** contain the full external 368,270-rule SQLite/WAL vault; they contain a taxonomy snapshot and the separate 6,798-rule MCRYSTAL brain. Evidence and licensing limitations are documented in [DATA_CARD.md](DATA_CARD.md) and [LICENSES.md](LICENSES.md).
