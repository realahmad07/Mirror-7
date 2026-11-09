# Crystal archive data index

This directory publishes the inventory, integrity hashes, comparison report, and import plan for the four Crystal archives uploaded for Mirror-7.

## Dataset and licence

- [DATA_CARD.md](DATA_CARD.md) — contents, counts, usage, and important caveats.
- [LICENSES.md](LICENSES.md) — source-specific licence and attribution register. This is a mixed-source dataset; the root software licence does not apply to every upstream record.

## Files

- [manifest.json](manifest.json) — archive SHA-256 values, observed record counts, dry-run counts, evidence and licensing status.
- [COMPARISON.md](COMPARISON.md) — duplicate, collision, quarantine, and source-evidence findings.
- [IMPORT.md](IMPORT.md) — local import workflow and safety rules.

## Raw data publication status

The raw ZIP payloads are **not committed to this public repository yet**. The archives attribute records to OASST2, pre-Quantum, curated internet, and master datasets, and include an MCRYSTAL snapshot; complete per-file redistribution rights were not supplied for all of them. The 368,270-rule full production vault is also not present in the uploads. The manifest records SHA-256 values so the uploaded source archives can be checked locally.

A combined owner-facing download bundle was prepared, containing all four original ZIPs, the General Reasoning 10K candidate pack, the audit, and the local importer. The public repository contains the provenance-safe report and integration metadata until source rights are confirmed.

## Important status

The import report is a **dry run**, not a claim that the user's production .dcrystal database has been modified. Run the local importer from the actual Mirror checkout against the actual vault, inspect the quarantine output, and only then use its explicit apply mode. Canonical evidence remains unverified; a trust tier or a filename such as "verified sample" is not independent verification.


## Test Mirror against the stored brain

After the raw archives are published, run `python scripts/test_crystal_archive_responses.py --brain datasets/crystal_archive/raw/crystal_4.zip`. It records a known-rule retrieval, an unseen exact modular-arithmetic query, and an out-of-scope abstention probe.
