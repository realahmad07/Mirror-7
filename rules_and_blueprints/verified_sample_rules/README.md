# Sample rule records — verification status

The directory name `verified_sample_rules` is retained for compatibility with the supplied publication layout. It is **not a verification claim**.

The 500 rows in `sample_verified_rules_pillars.jsonl` all carry legacy `trust_tier=2`, but the records do not contain per-rule verification evidence, verification timestamps, verifier identities, or source/licence provenance. Mirror's canonical adapter must therefore keep them at `EvidenceLevel.UNVERIFIED` unless independent evidence is added and accepted by the normal verification process.

The file is a sample/reference set, not an approved batch for automatic trust promotion. The `mirror_368k_vault_taxonomy.json` file is metadata describing an external taxonomy claim; it does not include the 368,270-rule database.
