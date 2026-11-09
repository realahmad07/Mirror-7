# Crystal archive — data licensing and attribution

**This is a mixed-source collection. There is no single licence that automatically covers every byte in every archive.** The repository root Apache-2.0 licence covers Mirror-7 software and original code contributions that the project owner can license; it does not relicense third-party datasets by implication.

## Source inventory and current status

| Archive content | Records | Declared upstream/source | Licence status |
|---|---:|---|---|
| Triplets, all four parts | 39,428 | Four source labels recorded per row | Mixed; see below |
| mirror_oasst2_curated_v1.jsonl | 14,399 | OpenAssistant OASST2 | Apache-2.0 is declared on the current official Hugging Face dataset card. Preserve OpenAssistant attribution and upstream licence. |
| pre_Quantum_dataset.jsonl | 17,573 | Mirror-provided/pre-Quantum source label | NOASSERTION in archive; no per-record source URLs or licence field. |
| mirror_complete_master_v2.jsonl | 2,767 | Mirror complete-master label | NOASSERTION; no separate per-record provenance/licence declaration. |
| mirror_internet_curated_v1.jsonl | 4,689 | Internet-curated source label | NOASSERTION; individual sources and terms are not included in these rows. |
| mirror_all_1014_blueprints.jsonl | 1,014 | Mirror roadmap and industry-vertical collection | NOASSERTION in the records; trust tier 2 is not a copyright licence or verification certificate. |
| sample_verified_rules_pillars.jsonl | 500 | Sampled rule records | NOASSERTION; rows show trust tier 2 but no per-record verification artefact or source licence. Two canaries are excluded from normal ingestion. |
| mirror_crystal_v1.mcrystal inside crystal_4.zip | 6,798 | MCRYSTAL snapshot | No embedded licence. Embedded stats report 0 sandbox-verified and 0 sandbox-rejected rules. |
| mirror_368k_vault_taxonomy.json | Taxonomy only | Reported 368,270-rule external vault | The full SQLite/WAL database is not part of the supplied archives. |
| mirror_eval_report_55_of_55.json | Evaluation report | Historical Mirror evaluation | Historical project artefact; provenance of any embedded dataset-derived material should be reviewed separately. |
| General Reasoning candidate pack | 10,240 synthesized + 26 source-indexed methods | Mirror-authored synthesis plus public methods | The 10,240 authored candidates may be licensed by the Mirror repository owner under Apache-2.0 as an original project contribution. The 26 source-indexed methods retain per-record license_spdx: OGL-UK-3.0 for identified UK Government methods; NOASSERTION for NASA-derived methods pending rights review. |

## Upstream licence references

### OpenAssistant OASST2

The current Hugging Face OpenAssistant/oasst2 dataset card declares apache-2.0. This applies only to records actually derived from OASST2, not to other triplet source labels.

- Dataset card: https://huggingface.co/datasets/OpenAssistant/oasst2
- Project: https://github.com/LAION-AI/Open-Assistant
- Attribution: OpenAssistant contributors / LAION-AI, OpenAssistant Conversations Release 2.

### UK Government Green Book and Magenta Book

The source-indexed General Reasoning methods cite these HM Treasury publications. GOV.UK identifies the publications; public-sector information is generally offered under the Open Government Licence v3.0 except where stated otherwise. Preserve source attribution and check for any third-party material excluded from the OGL.

- The Green Book: https://www.gov.uk/government/publications/the-green-book--2
- The Magenta Book: https://www.gov.uk/government/publications/the-magenta-book
- OGL v3.0 terms: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
- Suggested attribution: Contains public sector information licensed under the Open Government Licence v3.0.

### NASA handbook material

The 26-record source-indexed General Reasoning subset includes some references to NASA Systems Engineering Handbook material. The source register marks those rows NOASSERTION; no separate licence notice accompanied the extracted text. Keep those rows in provenance review until the applicable NASA and third-party terms are checked. Do not infer a blanket public-domain licence.

## Licence handling rules

1. Keep source labels and original archive files unchanged.
2. Do not replace NOASSERTION with Apache-2.0 merely because the Mirror code is Apache-2.0.
3. Preserve record-level provenance.license_spdx and source URLs when converting to canonical blueprints.
4. A trust tier or a filename containing "verified" is not a copyright licence or a reproducible verification artefact.
5. Where upstream rights are unknown, label the status clearly rather than inventing a licence.
6. Corrections should add evidence, not silently rewrite upstream licence history.
