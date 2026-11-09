# Crystal archive comparison — 2026-11-10

## Inventory

| Dataset | Total | Dry-run disposition |
|---|---:|---|
| Triplets | 39,428 | 30,648 pair inputs queued after duplicate/conflict filtering |
| Roadmap blueprints | 1,014 | 997 canonical candidates retained; 17 conflict rows quarantined |
| MCRYSTAL | 6,798 | 6,648 pair inputs queued; 2 internal conflicts and 148 roadmap collisions excluded |
| Sample rule file | 500 | 498 non-canary pairs queued; 2 canaries excluded |
| General Reasoning pack | 10,266 | 10,266 canonical candidates and legacy pairs retained |

Prepared totals: **48,060 legacy-pair inputs** and **17,911 canonical blueprint candidates**. These are staging counts—not net-new records in the user's private production vault.

## Important collision findings

- Triplets have 34,188 unique normalized prompts from 39,428 records.
- 3,406 triplet prompt groups have conflicting answers (7,240 records); the entire groups are quarantined.
- 1,406 same-answer duplicate excess rows are collapsed.
- 166 triplet prompt groups overlap roadmap prompts without an identical answer. 231 raw triplet records are involved; after internal triplet conflict quarantine, 134 safe representatives are additionally withheld.
- Roadmap blueprints contain 7 conflicting prompt groups totaling 17 rows. All 17 are quarantined.
- MCRYSTAL has one conflicting duplicate prompt group (2 rows), and 148 prompts that conflict with roadmap prompt/answer pairs. These are excluded from primary ingestion.
- The 500-rule sample is named "verified", but records contain only trust tier 2 and no per-record verification artifact. Two canaries are excluded; canonical evidence stays unverified.
- MCRYSTAL's embedded stats explicitly report 0 sandbox-verified and 0 sandbox-rejected rules.
- The 368,270 count is a taxonomy snapshot for a separate external vault. That full database was not uploaded.
- The bundled 55/55 evaluation is historical evidence; it was not rerun.
- The 10,266 General Reasoning pack consists of 10,240 context-specialized candidates from 320 authored families × 32 contexts, plus 26 source-indexed methodology candidates. It is not 10,266 independently verified discoveries.

## Import safeguards

Mirror's legacy ingest_jsonl path skips repeated prompts and can keep the first answer without reconciling later conflicting answers. The local importer therefore quarantines conflicting groups before ingestion, keeps original archive ZIPs unchanged, and creates a SQLite online backup before apply mode.

This branch fixes two runtime scale issues: blueprint composition loads only the requested blueprint's dependency closure instead of the first 10,000 records, and prompt lookup uses informative multiword prefixes so large generic first-word buckets do not hide later matches. Versioned startup migration rebuilds the stored prompt index. Regression tests cover a target after 10,000 unrelated blueprints and a rule beyond 512 broadly prefixed candidates.

## Limits

Exact normalized prompt collisions were checked. Semantic near-duplicate review is not complete. Raw source ZIPs are not committed publicly until per-source redistribution/license rights are established. No production database was written during this audit.
