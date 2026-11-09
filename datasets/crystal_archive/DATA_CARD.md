# Mirror Crystal Archive — dataset card

## Purpose

A portable distribution of the Crystal triplets, curated blueprints, sample rules, MCRYSTAL CPU brain, taxonomy snapshot, historical evaluation report, and the 10K General Reasoning candidate pack. The archive is intended for Mirror-7 and external tools that can read ZIP, JSONL, JSON, or the embedded MCRYSTAL payload.

## Files

- crystal_1.zip: triplets 1–9,857; 1,014 curated blueprints in JSON/JSONL; 15-stage roadmap summary.
- crystal_2.zip: triplets 9,858–19,714; 368,270-rule taxonomy snapshot; 500 sample rule records.
- crystal_3.zip: triplets 19,715–29,571; domain distribution; historical 55/55 evaluation report.
- crystal_4.zip: triplets 29,572–39,428; mirror_crystal_v1.mcrystal CPU brain (6,798 stored rules).
- Mirror_General_Reasoning_10K_Candidate_Pack_v1.zip: 10,240 synthesized context-specialized candidates plus 26 source-indexed methodology candidates.

## Counts and distinctions

- Triplet sources: pre_Quantum_dataset.jsonl (17,573), mirror_oasst2_curated_v1.jsonl (14,399), mirror_complete_master_v2.jsonl (2,767), and mirror_internet_curated_v1.jsonl (4,689).
- Roadmap blueprints: 1,014.
- Sample rules: 500; the filename says verified, but records expose only trust tier 2 and no independent verification artefact.
- MCRYSTAL brain: 6,798 rules; embedded stats report 0 sandbox-verified and 0 sandbox-rejected.
- The 368,270 value is a taxonomy snapshot for an external SQLite/WAL vault; the full vault is not present in these ZIPs.
- The 55/55 report is historical bundled evidence, not a benchmark rerun for this release.
- The General Reasoning pack contains 320 authored families × 32 context specializations. It is not 10,240 independent verified discoveries.

## Use

Any AI or application can clone the public repository, download the ZIP archives, parse the JSONL/JSON files, or load the MCRYSTAL brain directly from crystal_4.zip through MirrorCrystalEngine.load(). No external LLM API is required for the MCRYSTAL response path.

From the Mirror-7 repository root:
python -m mirror_crystal query --brain datasets/crystal_archive/raw/crystal_4.zip --prompt "What is protocol code 1000?"

For a fuller response probe:
python scripts/test_crystal_archive_responses.py --brain datasets/crystal_archive/raw/crystal_4.zip

See LICENSES.md for source-specific licence and attribution status. Do not treat the mixed-source collection as uniformly Apache-2.0.
