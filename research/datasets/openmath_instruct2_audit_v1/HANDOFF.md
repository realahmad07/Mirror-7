# Mirror 7 — OpenMathInstruct-2 Audit Handoff (v1)

**Status: audit only. Training is NOT approved.**

## Source and immutable input

- Dataset: `nvidia/OpenMathInstruct-2`
- Configuration: `default`
- Split: `train`
- Pinned revision: `469216e3f46f4dacf476b382e192485ea51a143e`
- License: CC-BY-4.0 — retain attribution to NVIDIA and the source dataset card when redistributing source-derived records.
- Records: 750 (100 previously preserved + 650 newly sampled)
- Shuffle seed: 2026; streaming buffer: 10,000
- Input JSONL SHA256: `2c6f0c50202f0c43f52ed690fed1a1d9352afeab32508740e78447740e4bde30`

## Screening result

- ACCEPT_CANDIDATE: 617
- MANUAL_REVIEW: 133
- REJECT_CANDIDATE: 0
- Duplicate prompt groups: 4 (record pairs: 213/419, 222/397, 307/360, 398/452)
- Exact duplicate groups: 0

Flags (heuristic review signals, not proven defects):
- Solution inconsistency / underspecification language: 104
- Assumption / approximation language: 35
- Prompt contamination / compound task: 4
- Boxed expected answer needs domain review: 5
- Possible concatenated problem: 1
- Unsupported assertion language: 1
- Unbalanced LaTeX braces: 2
- Multiple question markers: 1

## Critical interpretation

The audit is a heuristic screening pass, not a proof checker. The 617 ACCEPT_CANDIDATE records are **not verified training examples**. The 133 MANUAL_REVIEW records need deeper review. No record is approved for training by this report. Do not change `training_approved` to true without explicit user approval after correctness, provenance, contamination, duplicate, and split checks.

## Files expected in this folder

- `manifest.json`
- `comprehensive_audit_report.json`
- `screening_audit_750.json`
- `manual_review_queue.jsonl`
- `openmath_750_review_pool.jsonl`
- `comprehensive_audit_records.jsonl`

The six files were restored from a Drive ZIP and verified in Colab:
- 750 input records
- SHA256 exactly matches the immutable input fingerprint above
- All six expected files present

## Continuation plan

1. Preserve this audit snapshot unchanged.
2. Review the 133-record queue with stronger domain-aware checks; heuristic flags are not automatic rejections.
3. Improve mathematical verification and record explicit evidence for each accept/reject/manual decision.
4. Deduplicate and check split leakage / benchmark contamination.
5. Produce a new versioned candidate dataset and manifest; never overwrite this audit snapshot.
6. Keep training approval false until the user explicitly approves the final manifest.

## Mirror 7 project constraints

- Preserve the existing Mirror 7 cognitive architecture; this work enriches the native language model and must not replace the architecture.
- Do not introduce permanent teacher-model inference dependencies.
- Do not use paid APIs.
- Do not train on hidden chain-of-thought traces automatically.
- Keep SFT, preference, pretraining, and evaluation datasets separate.
- Do not start training without explicit user approval.

