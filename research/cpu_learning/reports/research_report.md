# Mirror 7 CPU-Native Learning Research — Stage S0 Baseline Report

- **Stage:** `S0` (Benchmark Harness, Hidden Relational Simulator & Baselines)
- **S0 Acceptance Gate:** `PASS`
- **Reproduce Command:** `python -m research.cpu_learning.evaluation.run_experiments`
- **Generator Version:** `s0-relworld-v1.0` / Temporal `s0-temporal-v2.0`
- **Fixed Seeds:** `[101, 202, 303, 404, 505]` (6 demos/action = 24 train episodes/seed)
- **Environment:** Python `3.14.4` | `Windows-11-10.0.26200-SP0` | CPUs `8` | RAM `15.77 GB` | Git `f41599bff4ac` (`research/cpu-native-learning-s0`)

## 1. S0 Acceptance Gate Verification

| Check | Status |
|---|:---:|
| `deterministic_seed_replay` | PASS |
| `pool_disjointness_verified` | PASS |
| `hidden_oracle_exact_accuracy_100pct` | PASS |
| `hidden_oracle_collateral_zero` | PASS |
| `hidden_oracle_operator_recovery_f1_100pct` | PASS |
| `episode_lookup_canary_retrieval_audited` | PASS |
| `five_fixed_seeds_present` | PASS |

## 2. Five-Seed Baseline Matrix on Unseen Generator Splits (9 Splits, 168 Unseen Tasks/Seed = 840 Total)

| Solver | Category | Unseen Exact Acc (Mean ± 95% CI) | Depth-4 Acc | Depth-6 Acc | Temporal Frozen | Abstain Rate | Collateral Rate | Preds/s (1-core) | Peak Heap (KB) | Model MDL (bits) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hand_coded_hidden_oracle` | `evaluator_ceiling` | 100.00% ± 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 742.6 | 98.2 | 2,256 |
| `noop_identity_floor` | `sanity_floor` | 5.95% ± 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 1479.0 | 89.9 | 0 |
| `episode_lookup` | `memory_baseline` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 985.3 | 61.1 | 11,483 |
| `structural_knn` | `memory_baseline` | 2.98% ± 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.23% | 408.0 | 233.1 | 29,768 |
| `shard_hdc_retrieval_no_itl` | `shard_baseline` | 3.21% ± 0.40% | 0.00% | 0.00% | 0.00% | 0.00% | 10.22% | 299.6 | 355.6 | 249,176 |
| `current_mirror_solver` | `existing_mirror` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 141.8 | 641.8 | 0 |
| `small_cpu_neural_reference` | `comparison_only_neural` | 3.57% ± 1.05% | 0.00% | 0.00% | 0.00% | 0.00% | 6.39% | 383.1 | 122.1 | 22,749 |

## 3. Per-Seed Unseen Learning Exact Accuracy Table

| Solver | Seed 101 | Seed 202 | Seed 303 | Seed 404 | Seed 505 | Mean | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| `hand_coded_hidden_oracle` | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | [100.00%, 100.00%] |
| `noop_identity_floor` | 5.95% | 5.95% | 5.95% | 5.95% | 5.95% | 5.95% | [5.95%, 5.95%] |
| `episode_lookup` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | [0.00%, 0.00%] |
| `structural_knn` | 2.98% | 2.98% | 2.98% | 2.98% | 2.98% | 2.98% | [2.98%, 2.98%] |
| `shard_hdc_retrieval_no_itl` | 2.98% | 2.98% | 3.57% | 3.57% | 2.98% | 3.21% | [2.81%, 3.62%] |
| `current_mirror_solver` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | [0.00%, 0.00%] |
| `small_cpu_neural_reference` | 4.17% | 2.98% | 4.17% | 2.38% | 4.17% | 3.57% | [2.53%, 4.62%] |

## 4. Retrieval-vs-Learning Canary Audit (`canary_exact_memorization`, 8 Exact Train Copies/Seed)

| Solver | Raw Exact Match | Retrieval Hit Rate | Learning-Scored Tasks/Seed |
|---|---:|---:|---:|
| `hand_coded_hidden_oracle` | 100.00% | 0.00% | 0 |
| `noop_identity_floor` | 0.00% | 0.00% | 0 |
| `episode_lookup` | 100.00% | 100.00% | 0 |
| `structural_knn` | 100.00% | 100.00% | 0 |
| `shard_hdc_retrieval_no_itl` | 100.00% | 100.00% | 0 |
| `current_mirror_solver` | 0.00% | 0.00% | 0 |
| `small_cpu_neural_reference` | 100.00% | 0.00% | 0 |

## 5. Independent 5-Seed SHARD HDC Arithmetic Canary (PDF p. 3 Reproduction)

| Task | Classes | Chance | Pure HDC Similarity (Mean ± 95% CI) | SHARD Bit-Perceptron (Mean ± 95% CI) | Learned Readout Bits |
|---|---:|---:|---:|---:|---:|
| `comparison` | 2 | 50.0% | 71.00% ± 18.18% | 89.00% ± 9.96% | 8,192 |
| `carry` | 2 | 50.0% | 95.00% ± 5.81% | 98.00% ± 3.40% | 8,192 |
| `modular_addition` | 10 | 10.0% | 4.00% ± 1.70% | 7.00% ± 2.60% | 40,960 |

## 6. Representative Baseline Failure Samples (Seed 101)

### `noop_identity_floor`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['Have zara_3 give lantern_3 over to orion_3']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['nadia_8 open locker_8']`

### `episode_lookup`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['Have zara_3 give lantern_3 over to orion_3']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['nadia_8 open locker_8']`

### `structural_knn`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['Have zara_3 give lantern_3 over to orion_3']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['nadia_8 open locker_8']`

### `shard_hdc_retrieval_no_itl`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['Have zara_3 give lantern_3 over to orion_3']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['nadia_8 open locker_8']`

### `current_mirror_solver`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['Have zara_3 give lantern_3 over to orion_3']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`True`, collateral_changes=`0`, instructions=`['nadia_8 open locker_8']`

### `small_cpu_neural_reference`
- `unseen_names_s101_00` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['zara_2 move scroll_2 from ledge_2 to altar_2']`
- `unseen_names_s101_01` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`1`, instructions=`['Have zara_3 give lantern_3 over to orion_3']`
- `unseen_names_s101_02` (`heldout_unseen_names`, depth=1): abstained=`False`, collateral_changes=`0`, instructions=`['nadia_8 open locker_8']`

## 7. Next Branch Decision

- **S0 Gate:** PASS — Ready to advance to **S1 (Typed State-Frame + Append-Only Episode Ledger)** after review.
- **Preserved External Context (PDF p. 3):** Prior external five-organ evaluation (`scripts/run_five_organ_external_eval.py`) found useful constrained parsing and symbolic probes, weak long-memory retrieval, 0/20 MBPP passes, and below-chance StoryCloze ranking. S0 preserves those findings as baseline context without replacing them with internal test counts.
