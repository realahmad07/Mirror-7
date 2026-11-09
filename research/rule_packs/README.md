# Mirror Domain Rule Seed v1

This pack contributes **110 original, domain-oriented reasoning rules**: 10 each for the 11 domain labels used in the current Mirror rule inventory.

## Important status

- These are authored heuristic/procedure seeds, not 110 independently sourced factual records.
- Every row is labelled `trust_tier=3` and `evidence_status=unverified`.
- They are not canonical blueprints and do not receive independent factual verification from being imported.
- The pack is deliberately small and reviewable. It is **not** the requested 10,000-per-domain expansion and must not be reported as 110,000 rules.
- Do not copy the records into a benchmark or claim that rule count alone improves answer quality.

## Contents

| Domain | Seed rules |
|---|---:|
| olympiad_math | 10 |
| general_reasoning | 10 |
| algorithms | 10 |
| general | 10 |
| relational_logic | 10 |
| physics_science | 10 |
| provide a helpful response | 10 |
| mathematics | 10 |
| quantum_computing | 10 |
| deductive_logic | 10 |
| Specialized Cognitive Pillars | 10 |
| **Total** | **110** |

## Validate and ingest into the local production vault

The large production vault `artifacts/mirror_crystal_all_datasets.dcrystal` is local-only and is intentionally excluded from Git. The commit of this pack therefore updates the source-controlled repository; it does not mutate a developer's local SQLite vault remotely.

From the repository root, stop the Mirror backend and run:

```bash
python scripts/ingest_authored_rule_pack.py
```

This validates the pack and performs a dry run without changing the vault. To apply:

```bash
python scripts/ingest_authored_rule_pack.py --apply
```

The importer creates a SQLite online backup at `<vault>.pre-authored-seed-v1.bak` before writing. It reports actual new rows separately from duplicate prompts that already exist. If the backup path already exists, choose another with `--backup`; it will not overwrite a backup.

## Route to 10,000 per domain

Do not scale this file by templating the same answer thousands of times. Build each future tranche from licensed, domain-specific source material; retain source and license metadata; deduplicate against the existing vault; execute code and math checks where applicable; and keep factual records unverified until the repository's independent claim-review gate is satisfied. Measure held-out accuracy, unsupported-answer rate, abstention, coverage, and latency after each tranche.
