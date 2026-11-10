# Mirror 7 rules and blueprints

This directory holds the rule, blueprint, and triplet data extracted from the Crystal Publication Bundle v2. A guarded rebuild workflow uses these files to reconstruct the production SQLite `.dcrystal` vault. The source files remain immutable; the rebuild creates a separate database and only publishes it after validation and regression tests pass.

## Directory map

| Path | Contents | Admission status |
|---|---|---|
| `blueprints/` | 1,014 legacy roadmap prompt/response records | Original source preserved |
| `blueprints/staging/` | 997 records with internally unique normalized prompts | Candidate slice; unverified |
| `blueprints/review_required/` | 17 records across 7 prompt-collision groups | Held for review |
| `general_reasoning/rules/` | 10,240 synthetic candidates + 26 source-indexed methods | Unverified; source metadata preserved |
| `general_reasoning/metadata/` | Family/context index, manifest, source register | Preserved during import |
| `verified_sample_rules/` | 500 legacy sample-rule records and taxonomy metadata | Historical folder name; not independently verified |
| `triplets/` | Four JSONL parts, 39,428 reported examples | Converted to retrieval pairs where a usable positive response exists |
| `bundle_metadata/` | Source manifest, checksums, and licensing notes | Source-specific rights retained |

## Rebuild and trust policy

Run the guarded local builder with:

```bash
python scripts/rebuild_publication_vault.py
```

The GitHub Actions workflow `.github/workflows/rebuild-publication-vault.yml` runs the same builder when these inputs or the builder change. On success it commits:

- `artifacts/mirror_crystal_all_datasets.dcrystal` — rebuilt production vault
- `artifacts/mirror_crystal_rebuild_report.json` — counts, deduplication/quarantine details, hashes, runtime smoke tests, and test outcomes

The builder reconstructs the active rule store from the tracked `mirror_crystal_v1.mcrystal` artifact, the four triplet parts, and the 500-row sample (excluding canary fixtures). It imports roadmap and General Reasoning material into Mirror's canonical blueprint table. It uses exact normalized-prompt deduplication across the source files and Mirror's existing same-response similarity folding. Conflicting answers are quarantined rather than silently overwritten. Test/evaluation splits are not activated.

**No legacy trust label is treated as proof.** The rebuilt rule store uses trust tier 3 for imported records. Roadmap blueprints, synthesized General Reasoning candidates, and source-supported candidates remain `EvidenceLevel.UNVERIFIED` in the canonical contract; original evidence status, source metadata, family/context IDs, and unrun tests remain in the sidecar table. A source-supported label is not independent expert review, execution testing, or formal proof.

The source licensing is mixed. OASST2-derived records are declared Apache-2.0 by their upstream dataset card; several other sources, the roadmap blueprints, sample rules, and embedded MCRYSTAL are marked `NOASSERTION`. The repository's Apache-2.0 code licence does not automatically relicense these datasets. See `bundle_metadata/LICENSES.md`.

The `368,270` value in `verified_sample_rules/mirror_368k_vault_taxonomy.json` is a separate taxonomy claim. The full 368,270-rule database is not included in this publication bundle, and the rebuilt vault report must not claim that count.
