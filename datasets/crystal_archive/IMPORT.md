# Local import workflow

The owner-facing integration ZIP includes the importer script. Its path is used directly in the commands below; it is not a committed data file in this metadata-only PR.

1. Extract the bundle. Keep the original four ZIPs unchanged.
2. From the Mirror-7 checkout, run the importer without apply mode to prepare:
   - staged/legacy_pairs_to_ingest.jsonl
   - canonical General Reasoning JSONL and provenance sidecars
   - collision quarantine JSONL files
   - dry_run_report.json
3. Inspect every quarantine file. In particular, do not send conflicting prompt groups through ingest_jsonl because the legacy importer can skip later answers after retaining the first.
4. Ensure the checkout includes the dependency-closure runtime fix from this branch.
5. Rerun with apply mode only against the intended local vault. The importer backs up the SQLite database before writing and checks SQLite integrity afterward.

Example:

```bash
python /path/to/bundle/tools/import_crystal_archive_bundle.py \
  --archives-dir /path/to/bundle/raw_archives \
  --general-pack /path/to/bundle/raw_archives/Mirror_General_Reasoning_10K_Candidate_Pack_v1.zip \
  --vault artifacts/mirror_crystal_all_datasets.dcrystal \
  --output-dir artifacts/crystal_archive_import
```

Add the apply flag only after reviewing the dry-run report and quarantine outputs.

No evidence status is promoted by import. The uploaded 500-rule sample has no per-rule verification objects, and MCRYSTAL reports zero sandbox-verified rules. All canonical imports stay unverified.
