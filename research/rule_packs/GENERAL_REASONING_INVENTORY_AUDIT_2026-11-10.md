# General Reasoning Domain — Inventory Gate (2026-11-10)

## Decision

**Do not ingest another General Reasoning batch yet.** The repository already declares a large General Reasoning and Instruction corpus, but its published inventory figures conflict and the production vault is local-only. Adding more records before inspecting that vault would violate the no-duplication/no-noise requirement.

This is an inventory hold, not a claim that the domain is complete or that the existing records are all high-quality reusable blueprints.

## What the tracked repository shows

1. \`research/rule_packs/mirror_domain_seed_v1.jsonl\` contains **10** \`general_reasoning\` seed pairs. All are \`trust_tier=3\` and \`evidence_status=unverified\`; the pack README describes them as authored heuristic/procedure seeds, not independently sourced factual records.
2. The main README describes **146,388 General Reasoning & Instruction Rules**, naming Stanford Alpaca, Databricks Dolly 15k, and SQuAD 2.0.
3. A separate README milestone reports **51,720 Alpaca + 14,771 Dolly + 11,862 SQuAD 2.0 = 78,353** ingested records. That does not reconcile with the 146,388 General Reasoning subtotal (difference: **68,035**).
4. The six domain subtotals displayed in the README add to **359,405**, not the stated **368,270+** total (difference: **8,865**).
5. Related existing categories include algorithms, general, relational logic, and deductive logic. The 110-record seed pack also has 10 records in each of its 11 labels, so a new General Reasoning tranche must be checked against those related categories—not only against the 10 exact-domain seeds.
6. \`mirror_eval/data/source_backed_factual_v1.jsonl\` is an evaluation-only set. Its documentation explicitly says it must not enter training or normal vault ingestion.

The README figures are repository declarations, **not a fresh count from the production database**. They must not be represented as independently audited totals.

## Why live ingestion is blocked

The production vault, \`artifacts/mirror_crystal_all_datasets.dcrystal\`, is described by the repository as local-only and excluded from Git. It is not available through this repository connection. Therefore the actual prompts, answers, canonical blueprints, domain labels, and duplicate rate in the live vault cannot be inspected from this session.

No rules were added to the live vault, and no existing records were modified.

## New read-only audit utility

Run this on the machine that has the actual production vault:

\`\`\`bash
python scripts/audit_domain_rule_inventory.py \\
  --vault artifacts/mirror_crystal_all_datasets.dcrystal \\
  --domain general_reasoning \\
  --domain general \\
  --domain relational_logic \\
  --domain deductive_logic \\
  --output "%TEMP%/mirror_general_reasoning_inventory.json" \\
  --export-jsonl "%TEMP%/mirror_general_reasoning_existing_records.jsonl"
\`\`\`

The command:

- opens SQLite in read-only mode and does not initialize or modify the vault;
- reports the actual total and per-domain rule counts;
- reports legacy trust-tier counts separately from evidence levels;
- counts canonical blueprints by domain and evidence level;
- detects exact duplicate prompts after Unicode-aware case/punctuation normalization;
- optionally exports the selected prompt/response records for manual semantic-duplicate review.

The export can contain private or user-provided text. Keep it local and **do not commit it to GitHub**. Exact prompt normalization is not semantic deduplication; inspect near-duplicates and capability overlap before admission.

## Admission gate for the first domain

Proceed only after the live inventory has been run and the exported records have been compared against the proposed candidates. Any future tranche should contain atomic, reusable reasoning procedures—not a paraphrase farm of generic instruction answers—and should retain source, license, scope, limitations, tests, and its true evidence status.

The next report must state: candidates reviewed, exact duplicates, semantic duplicates, new candidates, rejected/quarantined candidates, evidence status, test commands/results, and the before/after live-vault counts. Do not promote an evidence label merely because ingestion or a test command succeeds.

## Files added in this audit

- \`scripts/audit_domain_rule_inventory.py\`
- \`mirror_crystal/tests/test_domain_rule_inventory_audit.py\`

The test is included in the standard \`python -m pytest -q mirror_crystal\` CI suite. Passing that test validates the audit utility against a temporary synthetic SQLite vault; it does **not** prove the inaccessible production vault was audited or that any rule was fed.
