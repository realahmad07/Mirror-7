# OpenMathInstruct-2 V4 Independent Archive Review

**Review outcome: FAIL — V4 is not a usable dataset handoff and is not approved for training.**

Review performed against the uploaded file `Mirror7_OpenMath_Final_Handoff_v4.zip`.

## Archive identity and reproducibility

- Archive size: **237,930 bytes**
- Archive SHA-256: `9d2cfb68ceac28cd71e74efb9417d70cdb603cb40c52c6f887b64e0ed2a9038e`
- ZIP integrity test: **PASS** (`zipfile.testzip()` returned no corrupt member)
- Archive entries: **9**
- Original pool: **750 JSONL records**, all parse successfully
- Original pool SHA-256: `2c6f0c50202f0c43f52ed690fed1a1d9352afeab32508740e78447740e4bde30` — exact expected fingerprint

The 750 original rows contain the fields `problem`, `generated_solution`, `expected_answer`, and `problem_source`. They do **not** contain an embedded record ID field. For this archive, records can only be addressed by their 1-based JSONL line position unless an external, verifiable ID map is supplied.

## Blocking findings

### 1. All five V4 partition files are empty

The following files each have a size of **0 bytes**:

- `openmath_accepted_v4.jsonl`
- `openmath_answer_only_v4.jsonl`
- `openmath_rejected_v4.jsonl`
- `openmath_manual_review_v4.jsonl`
- `openmath_verification_pending_v4.jsonl`

Therefore, the ZIP contains **zero actual partitioned records**. It does not contain the claimed 95 accepted records or any of the other claimed dispositions.

### 2. Summary counts contradict the output files

`summary_v4.json` claims:

| Disposition | Claimed count |
|---|---:|
| Accepted | 95 |
| Answer only | 15 |
| Rejected | 19 |
| Manual review | 6 |
| Verification pending | 615 |
| **Total claimed** | **750** |

These numbers sum to 750, but none of the five corresponding JSONL files contains any records. The summary therefore does not describe the delivered partition files.

### 3. Global readiness report contradicts the summary and artifacts

`MIRROR7_GLOBAL_TRAINING_READINESS_V4.md` claims 95 OpenMath records approved and 616 verification-pending, while the summary claims 615 pending. It also says global training is approved for conversational SFT combined with the OpenMath subset.

That approval is not supported by the archive: the accepted partition is empty, and no train/validation/test files or combined training manifest are present. The report's claim that the OASST2 source is approved is also not independently substantiated by this ZIP; the OASST2 data and its validation evidence are not included.

### 4. Required deliverables are absent

The archive does not include the promised:

- Full V4 mathematical audit report
- Review decisions and record-level verification evidence
- Regression test source
- Regression test execution logs
- Train, validation, and test JSONL files
- Training manifest or training configuration
- CPU data-pipeline test results
- README and reproducibility instructions
- Independent manifest-checker script

### 5. Manifest scope

All **7 entries declared** in `manifest_v4.json` have byte sizes and SHA-256 values that match their corresponding archive members. This confirms only those declared entries. The original 750-record source pool is not covered by the manifest, and the manifest does not hash itself.

## Actual archive inventory

| File | Bytes | Result |
|---|---:|---|
| `MIRROR7_GLOBAL_TRAINING_READINESS_V4.md` | 507 | Present; claims unsupported by delivered partitions |
| `manifest_v4.json` | 1,008 | Declared entries match |
| `openmath_750_review_pool.jsonl` | 906,870 | 750 valid rows; fingerprint matches |
| `openmath_accepted_v4.jsonl` | 0 | **Empty** |
| `openmath_answer_only_v4.jsonl` | 0 | **Empty** |
| `openmath_manual_review_v4.jsonl` | 0 | **Empty** |
| `openmath_rejected_v4.jsonl` | 0 | **Empty** |
| `openmath_verification_pending_v4.jsonl` | 0 | **Empty** |
| `summary_v4.json` | 264 | Counts do not match output files |

## Safe disposition

- Preserve the 750-record source pool as an immutable input.
- Treat the V4 disposition counts as unsubstantiated metadata, not as record-level audit results.
- Do not use V4's empty partitions as training data.
- Do not treat OpenMath or global Mirror 7 training as approved on the basis of this archive.
- Rebuild the partition files and record-level evidence from the original pool; retain stable line-based source indices or provide a reproducible ID map.
- Generate reports, summaries, splits, and manifests from the actual materialized files, then test the extracted ZIP independently.

## Final decision

**V4 archive integrity: PASS.**

**Original source-pool fingerprint: PASS.**

**Partition materialization: FAIL.**

**Mathematical verification evidence: NOT DELIVERED / NOT VERIFIABLE FROM THIS ZIP.**

**OpenMath training approval: NO.**

**Global Mirror 7 training approval from this handoff: NO.**

This review note records the delivered archive's actual contents. It does not claim to have re-audited the mathematics of all 750 source problems.
