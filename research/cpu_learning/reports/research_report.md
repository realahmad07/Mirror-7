# Mirror 7 CPU-Native Learning Research — Stage S0 Baseline Report

- **Stage:** `S0` (Benchmark Harness, Hidden Relational Simulator & Baselines)
- **S0 Acceptance Gate:** `PASS`
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_experiments`
- **Generator Version:** `s0-relworld-v1.0` / Temporal `s0-temporal-v2.0`
- **Fixed Seeds:** `[101, 202, 303, 404, 505]` (6 demos/action = 24 train episodes/seed)
- **Environment:** Python `3.14.4` | `Windows-11-10.0.26200-SP0` | CPUs `8` | RAM `15.77 GB` | Git `6b486b7c4ec7` (`research/cpu-native-learning-s0`)

## 1. S0 Acceptance Gate Verification

| Check | Status |
|---|:---:|
| `five_distinct_fixed_seeds_present` | PASS |
| `deterministic_seed_replay_all_seeds` | PASS |
| `pool_disjointness_verified` | PASS |
| `all_solvers_frozen_before_temporal` | PASS |
| `hidden_oracle_exact_accuracy_100pct` | PASS |
| `hidden_oracle_collateral_zero` | PASS |
| `hidden_oracle_operator_recovery_f1_100pct` | PASS |
| `episode_lookup_unseen_zero_leakage` | PASS |
| `episode_lookup_canary_retrieval_audited_all_seeds` | PASS |
| `cpu_budget_within_limit` | PASS |
| `shard_modular_addition_canary_confirmed` | PASS |

## 2. Five-Seed Baseline Matrix on Unseen Generator Splits (9 Splits, 168 Unseen Tasks/Seed = 840 Total)

| Solver | Category | Unseen Exact Acc (Mean ± 95% CI) | Depth-4 Acc | Depth-6 Acc | Temporal Frozen | Abstain Rate | Collateral Rate | Preds/s (1-core, uninstrumented) | Peak Heap (KB) | Model MDL (bits) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hand_coded_hidden_oracle` | `evaluator_ceiling` | 100.00% ± 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 1609.3 | 23.8 | 2,888 |
| `noop_identity_floor` | `sanity_floor` | 5.95% ± 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 6267.2 | 18.3 | 0 |
| `episode_lookup` | `memory_baseline` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 4267.6 | 19.6 | 11,581 |
| `structural_knn` | `memory_baseline` | 2.98% ± 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 4.57% | 719.6 | 168.2 | 30,357 |
| `shard_hdc_retrieval_no_itl` | `shard_baseline` | 3.57% ± 0.74% | 0.00% | 0.00% | 0.00% | 0.00% | 4.56% | 426.5 | 75.3 | 249,262 |
| `current_mirror_solver` | `existing_mirror` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 185.0 | 33.1 | 0 |
| `small_cpu_neural_reference` | `comparison_only_neural` | 2.74% ± 2.00% | 0.00% | 0.00% | 0.00% | 0.00% | 2.64% | 658.8 | 56.5 | 23,019 |

## 3. Per-Seed Unseen Learning Exact Accuracy Table

| Solver | Seed 101 | Seed 202 | Seed 303 | Seed 404 | Seed 505 | Mean | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| `hand_coded_hidden_oracle` | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | [100.00%, 100.00%] |
| `noop_identity_floor` | 5.95% | 5.95% | 5.95% | 5.95% | 5.95% | 5.95% | [5.95%, 5.95%] |
| `episode_lookup` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | [0.00%, 0.00%] |
| `structural_knn` | 2.98% | 2.98% | 2.98% | 2.98% | 2.98% | 2.98% | [2.98%, 2.98%] |
| `shard_hdc_retrieval_no_itl` | 3.57% | 2.98% | 2.98% | 4.17% | 4.17% | 3.57% | [2.83%, 4.31%] |
| `current_mirror_solver` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | [0.00%, 0.00%] |
| `small_cpu_neural_reference` | 1.19% | 2.98% | 1.79% | 2.38% | 5.36% | 2.74% | [0.74%, 4.74%] |

## 4. Retrieval-vs-Learning Canary Audit (`canary_exact_memorization`, 8 Exact Train Copies/Seed)

| Solver | Raw Exact Match | Retrieval Hit Rate | Max Learning-Scored Tasks/Seed |
|---|---:|---:|---:|
| `hand_coded_hidden_oracle` | 100.00% | 0.00% | 0 |
| `noop_identity_floor` | 0.00% | 0.00% | 0 |
| `episode_lookup` | 100.00% | 100.00% | 0 |
| `structural_knn` | 100.00% | 100.00% | 0 |
| `shard_hdc_retrieval_no_itl` | 100.00% | 100.00% | 0 |
| `current_mirror_solver` | 0.00% | 0.00% | 0 |
| `small_cpu_neural_reference` | 100.00% | 0.00% | 0 |

## 5. Bit-Budget & MDL Memory Accounting (PDF p. 15 Mean Across 5 Seeds)

| Solver | Raw Train Data (bits) | Compressed Train (bits) | Episode Store (bits) | Active Bundle (bits) | Operator Library (bits) | Compressed Model MDL (bits) | MDL / Raw Train |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hand_coded_hidden_oracle` | 704,301 | 52,427 | 0 | 0 | 14,288 | 2,888 | 0.0041 |
| `noop_identity_floor` | 704,301 | 52,427 | 0 | 0 | 0 | 0 | 0.0000 |
| `episode_lookup` | 704,301 | 52,427 | 44,656 | 0 | 0 | 11,581 | 0.0164 |
| `structural_knn` | 704,301 | 52,427 | 243,437 | 0 | 0 | 30,357 | 0.0431 |
| `shard_hdc_retrieval_no_itl` | 704,301 | 52,427 | 277,552 | 245,760 | 0 | 249,262 | 0.3539 |
| `current_mirror_solver` | 704,301 | 52,427 | 0 | 0 | 0 | 0 | 0.0000 |
| `small_cpu_neural_reference` | 704,301 | 52,427 | 25,814 | 19,661 | 0 | 23,019 | 0.0327 |

## 6. Independent 5-Seed SHARD HDC Arithmetic Canary (PDF p. 3 Reproduction)

| Task | Classes | Chance | Pure HDC Similarity (Mean ± 95% CI) | SHARD Bit-Perceptron (Mean ± 95% CI) | Clipped 95% CI (Bit-Perceptron) | Learned Readout Bits |
|---|---:|---:|---:|---:|---|---:|
| `comparison` | 2 | 50.0% | 71.00% ± 18.18% | 89.00% ± 9.96% | [79.04%, 98.96%] | 8,192 |
| `carry` | 2 | 50.0% | 95.00% ± 5.81% | 98.00% ± 3.40% | [94.60%, 100.00%] | 8,192 |
| `modular_addition` | 10 | 10.0% | 4.00% ± 1.70% | 7.00% ± 2.60% | [4.40%, 9.60%] | 40,960 |

### SHARD Canary Per-Seed Bit-Perceptron Accuracy

| Task | Seed 101 | Seed 202 | Seed 303 | Seed 404 | Seed 505 |
|---|---:|---:|---:|---:|---:|
| `comparison` | 75.00% | 95.00% | 90.00% | 92.50% | 92.50% |
| `carry` | 100.00% | 95.00% | 100.00% | 95.00% | 100.00% |
| `modular_addition` | 10.00% | 5.00% | 7.50% | 7.50% | 5.00% |

## 7. Representative Baseline Failure Samples (Seed 101)

### `noop_identity_floor`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['Have kieran_6 give vial_6 over to orion_6']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['nadia_5 open casket_5']`

### `episode_lookup`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['Have kieran_6 give vial_6 over to orion_6']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['nadia_5 open casket_5']`

### `structural_knn`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['Have kieran_6 give vial_6 over to orion_6']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['nadia_5 open casket_5']`

### `shard_hdc_retrieval_no_itl`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['Have kieran_6 give vial_6 over to orion_6']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['nadia_5 open casket_5']`

### `current_mirror_solver`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['Have kieran_6 give vial_6 over to orion_6']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['nadia_5 open casket_5']`

### `small_cpu_neural_reference`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['Have kieran_6 give vial_6 over to orion_6']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['nadia_5 open casket_5']`

## 8. Next Branch Decision

- **S0 Gate:** PASS — Ready to advance to **S1 (Typed State-Frame + Append-Only Episode Ledger)** after review.
- **Preserved External Context (PDF p. 3):** Prior external five-organ evaluation (`scripts/run_five_organ_external_eval.py`) found useful constrained parsing and symbolic probes, weak long-memory retrieval, 0/20 MBPP passes, and below-chance StoryCloze ranking. S0 preserves those findings as baseline context without replacing them with internal test counts.
