# GitHub publication (Codex / Antigravity)

1. Extract this bundle somewhere outside the Mirror-7 checkout.
2. Checkout the integration branch:

```powershell
git fetch origin datasets/crystal-archive-integration-2026-11-10
git checkout datasets/crystal-archive-integration-2026-11-10
```

3. Run the branch's guarded publisher:

```powershell
python scripts/publish_crystal_bundle.py --bundle "C:\path\to\Mirror_Crystal_Publication_Bundle_v2.zip" --replace-legacy-mcrystal --commit --push
```

The script verifies all checksums, copies the five ZIPs to the repo, runs the archive-load regression test and response probes, and removes `artifacts/mirror_crystal_v1.mcrystal` only if it exactly matches the embedded payload and all tests pass. If a test fails, it aborts before deleting the old asset.

4. Check the new GitHub Actions run. When it is green, the raw data is directly downloadable by any AI/developer from `datasets/crystal_archive/raw/`.

Do not assign a blanket Apache-2.0 licence to all raw files; see LICENSES.md.
