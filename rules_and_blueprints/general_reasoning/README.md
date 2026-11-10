# Mirror-7 General Reasoning — 10K Candidate Pack v1

## Contents

- `rules/general_reasoning_candidates_10240.jsonl`: 10,240 model-synthesized, context-specialized candidate procedures.
- `rules/general_reasoning_source_supported_26.jsonl`: source-indexed methodology rules paraphrased from official public guidance.
- `metadata/manifest.json`: exact counts, evidence states, and admission gates.
- `metadata/family_index.json`: 320 base procedure families and 32 scenario contexts.
- `metadata/source_register.json`: source metadata and license notes.
- `tools/validate_pack.py`: structural and exact-duplicate checks.
- `tools/convert_to_canonical_blueprints.py`: converts detailed records to Mirror's `Blueprint.from_dict` schema.
- `tools/ANTIGRAVITY_CODEX_IMPORT_PROMPT.md`: local production-vault dedupe and safe-ingestion instructions.

## Important count interpretation

The 10,240 generated candidates are **320 authored reasoning procedure families × 32 operating-context specializations**. They are not 10,240 independent scientific facts or 10,240 distinct cognitive primitives. The specialization changes when and how a procedure applies (for example, under sparse evidence, incomplete inputs, time pressure, delayed feedback, or distribution shift). Mirror should retain `family_id` and `context_id` to avoid treating siblings as unrelated discoveries.

The 26 source-indexed records are additional methodology candidates tied to specific sections of official guidance. `source_supported` means the cited passage appears to support the paraphrase; it does not mean an independent reviewer has approved it.

## Evidence and trust

- 10,240 records: `evidence.status = unverified`; `authorship.origin = model_synthesized`; every test is `not_run`.
- 26 records: `evidence.status = source_supported`; source locators and excerpt hashes are recorded; semantic review is still required.
- No record is execution-tested, formally verified, or independently reproduced.
- Do not increase trust tier or evidence state merely because JSON parsing or ingestion succeeds.

## Local deduplication is mandatory

The repository contains 10 existing `general_reasoning` authored seed records, plus adjacent `general`, `deductive_logic`, `relational_logic`, and `algorithms` records. The private production vault (`artifacts/mirror_crystal_all_datasets.dcrystal`) is local-only and was not available here. This pack therefore cannot claim to be net-new against the live vault. Compare the candidate records against the entire local vault and all adjacent domains before ingestion.

Do **not** feed these records into `scripts/ingest_authored_rule_pack.py` as-is. That script validates the legacy 110-record question/answer seed format, not this detailed blueprint schema. Use the converter for canonical schema output, then integrate through the actual current blueprint/vault API after local code inspection and dry-run.

## Source and licensing notes

The source-supported subset uses the UK Government's Magenta Book Annex A and Green Book (both identified on GOV.UK as Open Government Licence v3.0 except where otherwise stated), plus NASA handbook material marked `NOASSERTION` pending rights review. Third-party content exclusions apply. The candidate synthesis records do not copy source passages and have no asserted source-derived factual content.

OpenStax Psychology 2e was deliberately excluded: its page explicitly says the material may not be ingested into LLMs or generative AI offerings without OpenStax permission.

## Validation

Run from the extracted directory:

```bash
python tools/validate_pack.py
python tools/validate_pack.py --existing-jsonl path/to/other/rules.jsonl
python tools/convert_to_canonical_blueprints.py --input rules/general_reasoning_candidates_10240.jsonl --output /tmp/general_reasoning_canonical.jsonl
```

These checks validate file structure, ID uniqueness, and exact prompt duplication within supplied JSONL files. They do **not** replace semantic deduplication or verification.
