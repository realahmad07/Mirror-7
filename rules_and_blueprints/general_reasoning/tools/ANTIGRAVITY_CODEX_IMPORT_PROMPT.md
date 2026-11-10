# Antigravity / Codex prompt — local General Reasoning pack admission

You are working inside the user's local Mirror-7 checkout. Process `general_reasoning_10k_candidate_v1.zip` only after inspecting the current branch and the actual production vault. Do not assume this pack is net-new.

## Goal
Safely assess and, only after an explicit local dry-run passes, admit genuinely new General Reasoning candidate blueprints. Preserve evidence labels and provenance. The pack contains 10,240 unverified context-specialized candidates from 320 families × 32 contexts, plus 26 source-supported methodology candidates. It is not 10,266 independent verified facts.

## Mandatory sequence
1. Unzip to a staging directory outside the production vault. Run `python tools/validate_pack.py` from the extracted directory. Confirm exact counts and schema. Do not alter source files.
2. Inspect the current repository's canonical `Blueprint` contract, blueprint storage and import methods, tests, and all domain labels. Do not assume the older Q&A `scripts/ingest_authored_rule_pack.py` accepts this detailed JSONL format; it does not.
3. Inventory the actual production `.dcrystal` vault read-only. Use a consistent SQLite snapshot or backup for the audit. Enumerate rules and canonical blueprints, preserving IDs, domains, trust/evidence metadata, prompt patterns, and aliases. Never write during inventory.
4. Deduplicate across the entire vault, not only `general_reasoning`: especially `general`, `deductive_logic`, `relational_logic`, `algorithms`, and any reasoning/cognitive domains. Use exact normalized text first, then semantic similarity using the repository's existing local mechanisms. If no dependable semantic method exists, produce candidate clusters for review instead of pretending lexical similarity is semantic deduplication.
5. Treat records sharing `family_id` as context specializations. Do not mistake the 32 siblings as 32 independent core principles. Preserve family/context metadata in a sidecar if the canonical schema cannot store them. Do not import a sibling when its only difference is cosmetic wording or a context that does not change preconditions, steps, limits, or tests.
6. Review the 26 source-supported records individually. Confirm that source URL, locator, excerpt, hash, source version, and license apply to the paraphrase. `source_supported` is not expert review, execution testing, or formal proof. For canonical `Blueprint` conversion, keep verification `unverified` because the current `EvidenceLevel` enum has no `source_supported` value; preserve original evidence state in the sidecar. Do not upgrade trust.
7. Exclude evaluation-only datasets and any private or unapproved material. Do not use OpenStax Psychology 2e content unless the user has separately obtained permission; the pack excludes it because its terms prohibit AI ingestion without permission.
8. Produce a dry-run report with: total input, valid records, duplicate exact, duplicate semantic, conflict, license-review hold, source-supported, unverified, proposed new, rejected, and manual-review counts. Include representative collision pairs and reasons. No database writes yet.
9. Build or use a transactional importer that writes canonical blueprints through the repository's real API. Before any write, create a verified SQLite backup and record its hash. Keep all synthesized records unverified / trust tier 3. Keep source-supported candidates unverified in the canonical enum unless the code is explicitly upgraded with regression tests to represent that state safely.
10. Import in a small batch (maximum 100) first. Run unit tests, retrieval tests, integrity checks, and regression tests. Confirm that inserted rows and reported counts match. Only then consider later batches. Stop immediately on any failed check; preserve the backup and report the exact failure.
11. Keep the tests in this pack as tests, not training facts. They are all `not_run` until actually executed. Do not mark a rule passed merely because it parses or imports.

## Hard prohibitions
- Do not weaken validation to force records through.
- Do not convert `unverified` or `source_supported` into trusted facts.
- Do not count duplicates as new rules.
- Do not overwrite or purge existing rules.
- Do not write to the vault before backup, dry-run, and user-approved import plan.
- Do not claim 10,000 net-new rules unless the local vault comparison proves it.

## Deliverables
Return the audit report, deduplication mapping, rejected/manual-review list, proposed import manifest, backup path and SHA-256 (if import was approved), exact tests run, integrity result, and exact count of newly inserted records. If any required local API or schema is unclear, stop at read-only audit and explain the blocker; do not invent an import path.
