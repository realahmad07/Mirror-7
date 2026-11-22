# Mirror 7 CPU-Native Learning Research — Stage S1 State-Frame, Ledger & Parser-Only Report

- **Stage:** `S1` (Typed State Frames + Partial `ObservationMask` + Append-Only `EpisodeLedger` + Separate Parser-Only Scoring Harness)
- **S1 Acceptance Gate:** `PASS` (16 / 16 checks passed)
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s1_experiments`
- **Ledger Schema / Parser Bench Version:** `s1-ledger-v1` / `s1-parser-bench-v1.0`
- **Fixed Seeds:** `[101, 202, 303, 404, 505]` (530 parser cases/seed = 2,650 total parser evaluations)
- **AST Import Isolation:** `PASS` (14 non-evaluator files, 5 S1 learner files scanned, 0 violations)
- **Environment:** Python `3.14.4` | `Windows-11-10.0.26200-SP0` | CPUs `8` | RAM `15.77 GB` | Git `323161c7dd84` (`research/cpu-native-learning-s0`)

> **Scope & Integrity Notice:** Stage S1 implements and verifies L0/L2 Typed State Frames, Partial `ObservationMask` ingestion, L3 Append-Only `EpisodeLedger`, and the separate Parser-Only Scoring Harness. **No ITL operator induction (S2), contrast learning (S3), SHARD search guidance (S4), or RDC consolidation (S5) is implemented or claimed in this stage.**

## 1. Stage S1 Acceptance Gate Verification

| Check | Status |
|---|:---:|
| `five_distinct_fixed_seeds_present` | PASS |
| `deterministic_s1_manifest_replay` | PASS |
| `ast_strict_learner_isolation` | PASS |
| `s0_committed_manifests_intact` | PASS |
| `ledger_disk_reload_and_hash_chain_verified` | PASS |
| `ledger_append_only_triggers_and_freeze_verified` | PASS |
| `partial_observation_mask_unobserved_not_false_verified` | PASS |
| `test_time_memory_audit_hooks_verified` | PASS |
| `s1_parser_s0_unseen_ci95_low_ge_99pct` | PASS |
| `s1_parser_beats_existing_ccg_by_ge_70pp` | PASS |
| `s1_parser_exact_byte_spans_ci95_low_ge_99pct` | PASS |
| `s1_parser_unseen_paraphrase_ci95_low_ge_95pct` | PASS |
| `s1_parser_state_typing_ablation_gain_ge_50pp` | PASS |
| `s1_parser_ambiguity_preservation_ci95_low_ge_95pct` | PASS |
| `s1_parser_ood_abstention_100pct` | PASS |
| `cpu_budget_within_limit` | PASS |

## 2. Separate Parser-Only Scoring Harness — 5-Seed Summary Matrix

| Parser | S0 Unseen Exact Role (Mean ± 95% CI) | Temporal Frozen Exact | Unseen Paraphrases Exact | State-Typed Disambig. (Unambig. Exact) | Partial-Mask Occluded Exact | Overall Unseen Byte-Span Exact | Genuinely Ambiguous (Flag Acc / Top-K Recall) | OOD Unparseable Abstain | Parses/s (1-core) | Peak Heap (KB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `existing_mirror_bilingual_ccg` | 20.83% ± 0.00% | 26.67% | 0.00% | 0.00% | 0.00% | 18.96% | 0.00% / 0.00% | 60.00% | 2,390.3 | 16.7 |
| `s1_typed_frame_parser_no_types` | 100.00% ± 0.00% | 100.00% | 100.00% | 0.00% | 100.00% | 98.33% | 75.00% / 75.00% | 100.00% | 7,321.8 | 5.5 |
| `s1_typed_frame_parser_full` | 100.00% ± 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% / 100.00% | 100.00% | 7,068.9 | 5.5 |

## 3. Per-Seed Unseen Unambiguous Exact Role-Alignment Accuracy (5 Splits, 480 Cases/Seed = 2,400 Total)

| Parser | Seed 101 | Seed 202 | Seed 303 | Seed 404 | Seed 505 | Mean | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| `existing_mirror_bilingual_ccg` | 18.96% | 18.96% | 18.96% | 18.96% | 18.96% | 18.96% | [18.96%, 18.96%] |
| `s1_typed_frame_parser_no_types` | 98.33% | 98.33% | 98.33% | 98.33% | 98.33% | 98.33% | [98.33%, 98.33%] |
| `s1_typed_frame_parser_full` | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | [100.00%, 100.00%] |

## 4. State-Typing Ablation (`s1_state_typed_disambiguation`, 16 Cases/Seed = 80 Total)

| Parser | Top-1 Exact Role | Unambiguous Exact Role | Ambiguity Rate | Top-K Candidate Recall | Role Slot F1 |
|---|---:|---:|---:|---:|---:|
| `existing_mirror_bilingual_ccg` | 0.00% | 0.00% | 0.00% | 0.00% | 38.10% |
| `s1_typed_frame_parser_no_types` | 50.00% | 0.00% | 100.00% | 75.00% | 72.00% |
| `s1_typed_frame_parser_full` | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% |

## 5. Append-Only SQLite `EpisodeLedger`, Partial `ObservationMask` & Memory-Access Audit (5 Seeds)

| Seed | Records | Ingest (ep/s) | Reload Exact & Hash Chain | UPDATE/DELETE/Freeze Blocked | Partial/Asymmetric Mask Checks | Occluded Preserved / Revealed Not Added | Canary Hit Audit | Unseen Exact Hits | Raw Payload (bits) | Compressed (bits) | SQLite File (B) |
|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| `101` | 33 | 73.8 | PASS | PASS | 8/8 | 8 / 6 | 100.00% | 0.00% | 1,780,584 | 145,216 | 282,624 |
| `202` | 33 | 71.8 | PASS | PASS | 8/8 | 8 / 2 | 100.00% | 0.00% | 1,780,952 | 143,040 | 282,624 |
| `303` | 33 | 57.2 | PASS | PASS | 8/8 | 4 / 8 | 100.00% | 0.00% | 1,779,864 | 143,832 | 282,624 |
| `404` | 33 | 57.1 | PASS | PASS | 8/8 | 2 / 6 | 100.00% | 0.00% | 1,773,488 | 142,896 | 282,624 |
| `505` | 33 | 84.0 | PASS | PASS | 8/8 | 2 / 4 | 100.00% | 0.00% | 1,775,744 | 143,680 | 282,624 |

## 6. Representative Parser Failure Samples (Seed 101)

### `existing_mirror_bilingual_ccg`
- `s0_train_train_s101_move_00` (`s0_train_instructions`): inst=`clara_1 move lamp_1 from table_1 to shelf_1` | gold=`move {'actor': 'clara_1', 'destination': 'shelf_1', 'object': 'lamp_1', 'source': 'table_1'}` | pred_top=`move {'actor': 'clara_1', 'target': 'lamp_1 from table_1 to shelf_1'}` (ambig=False, cands=1, abstain=False)
- `s0_train_train_s101_move_01` (`s0_train_instructions`): inst=`bob_5 move book_5 from table_5 to shelf_5` | gold=`move {'actor': 'bob_5', 'destination': 'shelf_5', 'object': 'book_5', 'source': 'table_5'}` | pred_top=`move {'actor': 'bob_5', 'target': 'book_5 from table_5 to shelf_5'}` (ambig=False, cands=1, abstain=False)
- `s0_train_train_s101_move_02` (`s0_train_instructions`): inst=`alice_6 move cup_6 from counter_6 to crate_6` | gold=`move {'actor': 'alice_6', 'destination': 'crate_6', 'object': 'cup_6', 'source': 'counter_6'}` | pred_top=`move {'actor': 'alice_6', 'target': 'cup_6 from counter_6 to crate_6'}` (ambig=False, cands=1, abstain=False)
- `s0_train_train_s101_move_03` (`s0_train_instructions`): inst=`bob_8 move bell_8 from drawer_8 to table_8` | gold=`move {'actor': 'bob_8', 'destination': 'table_8', 'object': 'bell_8', 'source': 'drawer_8'}` | pred_top=`move {'actor': 'bob_8', 'target': 'bell_8 from drawer_8 to table_8'}` (ambig=False, cands=1, abstain=False)

### `s1_typed_frame_parser_no_types`
- `s1_typed_s101_01_ditransitive_do_io` (`s1_state_typed_disambiguation`): inst=`amara_6 gives tablet_6 lyra_6` | gold=`give {'actor': 'amara_6', 'object': 'tablet_6', 'recipient': 'lyra_6'}` | pred_top=`give {'actor': 'amara_6', 'object': 'lyra_6', 'recipient': 'tablet_6'}` (ambig=True, cands=2, abstain=False)
- `s1_typed_s101_03_polysemous_transfer_surface` (`s1_state_typed_disambiguation`): inst=`marcus_3 transfers amulet_3 to mantel_3` | gold=`move {'actor': 'marcus_3', 'destination': 'mantel_3', 'object': 'amulet_3', 'source': 'plinth_3'}` | pred_top=`give {'actor': 'marcus_3', 'object': 'amulet_3', 'recipient': 'mantel_3'}` (ambig=True, cands=2, abstain=False)
- `s1_typed_s101_05_ditransitive_do_io` (`s1_state_typed_disambiguation`): inst=`amara_4 gives tablet_4 devon_4` | gold=`give {'actor': 'amara_4', 'object': 'tablet_4', 'recipient': 'devon_4'}` | pred_top=`give {'actor': 'amara_4', 'object': 'devon_4', 'recipient': 'tablet_4'}` (ambig=True, cands=2, abstain=False)
- `s1_typed_s101_07_polysemous_transfer_surface` (`s1_state_typed_disambiguation`): inst=`farah_5 transfers tablet_5 to mantel_5` | gold=`move {'actor': 'farah_5', 'destination': 'mantel_5', 'object': 'tablet_5', 'source': 'hearth_5'}` | pred_top=`give {'actor': 'farah_5', 'object': 'tablet_5', 'recipient': 'mantel_5'}` (ambig=True, cands=2, abstain=False)

### `s1_typed_frame_parser_full`
- *0 failures across all 520 seed-101 parser cases.*

## 7. Next Branch Decision

- **S1 Gate:** PASS — Independent S1 gate review ready. Proceed to **Stage S2 (ITL Operator Induction)** only after gate approval.
- **S0 Regression Status:** Committed S0 manifests verified bit-exact (`verify_committed_manifests = PASS`).
