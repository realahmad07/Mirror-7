# Mirror 7 rules and blueprints

This directory is the published staging area for the rule, blueprint, and triplet material extracted from the Crystal Publication Bundle. **Publishing these files does not automatically load them into the live Crystal vault.** They must pass the admission process below before they become active knowledge.

## Directory map

| Path | Contents | Current handling |
|---|---|---|
| `blueprints/` | 1,014 legacy roadmap prompt/response records in JSON and JSONL | Staged for canonical conversion and review |
| `blueprints/staging/` | Records with internally unique normalized prompts | Candidate slice only; not verified |
| `blueprints/review_required/` | Prompt-collision clusters | Hold until reviewed |
| `general_reasoning/rules/` | 10,240 synthetic candidates + 26 source-indexed methods | Unverified candidates; use the pack validator and converter |
| `general_reasoning/metadata/` | Family/context index, manifest, source register | Preserve with every conversion |
| `verified_sample_rules/` | 500 legacy sample-rule records and taxonomy metadata | Historical folder name; records are **not verified** |
| `triplets/` | Four JSONL parts, 39,428 reported examples | Dataset/training material, not executable rules |
| `bundle_metadata/` | Source bundle manifest, checksums, and licensing notes | Preserve source-specific provenance and rights |

## Admission findings

### Roadmap blueprints

The source JSONL contains 1,014 valid JSON records and 1,014 unique IDs. A normalized-prompt audit found **7 collision groups containing 17 records**. Those 17 records are copied to `blueprints/review_required/prompt_collision_clusters_17.jsonl`; the other **997 records** are copied to `blueprints/staging/mirror_997_unambiguous_prompt_candidates.jsonl`.

This filtering detects repeated normalized prompts only. It does not establish semantic uniqueness or correctness. The raw 1,014-record source files remain unchanged. The 1,014 source records have no per-record evidence field; legacy `trust_tier` values (502 tier 1, 512 tier 2) do not prove verification. Only 8 records contain test code, and those tests have not been executed as part of this publication audit.

### General reasoning pack

The pack manifest defines 10,240 model-synthesized candidates as 320 procedure families × 32 contexts, plus 26 source-indexed method candidates. The 10,240 candidates are marked `unverified`; their tests are `not_run`. The 26 source-indexed items are marked `source_supported`, which is not expert review, execution testing, or formal verification. Preserve `family_id`, `context_id`, test status, and source metadata. Run `general_reasoning/tools/validate_pack.py`; do not send this schema to the legacy 110-record Q&A importer.

### Sample rules and taxonomy

The 500 sample records have `trust_tier=2` but no per-record verification artifact or evidence status. They must not be promoted to verified facts based on the filename or directory name. The 368,270 value in the taxonomy JSON is an external taxonomy claim; the full 368,270-rule database is not included in this publication.

### Triplets and licensing

The 39,428 triplets are conversational/training examples and should remain separate from executable blueprint rules. The licence register is mixed-source: OASST2 is declared Apache-2.0 by its upstream dataset card; several other source groups and the 1,014 roadmap blueprints are `NOASSERTION`. The repository's Apache-2.0 code licence does not automatically relicense these datasets.

## Safe path into the live vault

1. Verify source checksums and parse every complete dataset file locally.
2. Review the prompt-collision file; do not silently choose conflicting responses.
3. Take a read-only snapshot of the actual production `.dcrystal` vault and deduplicate across all relevant domains.
4. Convert candidates through Mirror's canonical `Blueprint` contract. Keep all new records `unverified`; preserve original source status and family/context data in sidecars.
5. Produce a dry-run report, create and verify a SQLite backup, then import a small batch (at most 100).
6. Run retrieval, domain, integrity, and regression tests. Stop on any failure and preserve the backup.

The live production vault was not available in this repository publication context, so no vault-wide deduplication or production import is claimed here.
