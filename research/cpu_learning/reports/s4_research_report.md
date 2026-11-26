# Mirror 7 CPU-Native Learning Research — Stage S4 SHARD-Guided Proposal Generation Report

- **Stage:** `S4` (10,240-bit BSC Encoders, Multi-Table LSH Index, Warm-Started Proposal & Goal Planning + Bind-Depth Stress Probe)
- **S4 Acceptance Gate:** `PASS` (20 / 20 checks passed)
- **SHARD Keep/Remove Decision (PDF p. 9):** `KEEP_SHARD_AS_PROPOSER`
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s4_experiments`
- **Schema Versions:** `Encoder=s4-shard-bsc10240-v1`, `Index=s4-shard-lsh-index-v1`, `Proposer=s4-shard-proposer-v1`
- **Scaled Operator Library:** `48` verified operators (`4` S3-induced + `44` deterministic distractors)
- **Fixed Seeds:** `[101, 202, 303, 404, 505]` (6 demos/action)
- **AST Isolation & Zero-Hardcoding Audit:** `PASS` (31 non-evaluator, 16 learner, 7 ITL, 4 SHARD files; 0 import violations, 0 hardcoded literals)
- **Environment:** Python `3.14.4` | `Windows-11-10.0.26200-SP0` | CPUs `8` | RAM `15.77 GB` | Git `a82d2c4f87e3` (`research/cpu-native-learning-s0`)

> **Core Principle (PDF p. 15):** Similarity is NOT proof. SHARD 10,240-bit BSC hypervectors and multi-table LSH indexes strictly rank candidate operators and entity-role slots; ITL exact verification (`OperatorCard.apply()` + exact state/delta equality) ALWAYS decides acceptance.

## 1. Stage S4 Acceptance Gate Verification (Lower 95% CI Bounds Across 5 Fixed Seeds)

| Check | Status |
|---|:---:|
| `five_distinct_fixed_seeds_present` | PASS |
| `demos_per_action_le_8` | PASS |
| `ast_strict_learner_isolation_and_zero_hardcoded_literals` | PASS |
| `s0_s1_s2_s3_committed_manifests_intact` | PASS |
| `shard_indexes_frozen_block_insertions_all_seeds` | PASS |
| `scaled_operator_library_size_ge_48` | PASS |
| `task_a_shard_exact_accuracy_ci95_low_ge_99pct` | PASS |
| `task_a_shard_beats_retrieval_only_no_itl_by_ge_50pp` | PASS |
| `task_a_shard_proposal_reduction_ge_75pct_vs_unguided_and_random` | PASS |
| `task_a_shard_cpu_speedup_ge_2x_vs_unguided_and_random` | PASS |
| `task_a_shard_collateral_rate_ci95_high_lt_2pct` | PASS |
| `task_b_shard_exact_goal_accuracy_ci95_low_ge_95pct` | PASS |
| `task_b_shard_proposal_reduction_ge_50pct_vs_bfs_and_random` | PASS |
| `task_b_shard_cpu_speedup_ge_2x_vs_bfs_and_random` | PASS |
| `task_b_shard_collateral_rate_ci95_high_lt_2pct` | PASS |
| `task_c_bind_depth_permuted_accuracy_ci95_low_ge_95pct_through_depth6` | PASS |
| `task_c_permuted_strictly_beats_naive_at_depth6` | PASS |
| `shard_keep_decision_confirmed` | PASS |
| `bit_budget_accounted_and_nonzero` | PASS |
| `cpu_budget_within_limit` | PASS |

## 2. Task A — Unlabelled Transition Operator & Role-Slot Recovery (48 Operators, 280 Unseen Tasks)

| Method | Exact Acc (Mean ± 95% CI) | Mean Proposals (Mean ± 95% CI) | CPU Time/Seed (ms) | Collateral |
|---|---:|---:|---:|---:|
| `shard_guided_proposer` | 100.00% ± 0.00% | 1.00 ± 0.00 | 35.14 | 0.000% |
| `unguided_exhaustive_proposer` | 100.00% ± 0.00% | 214.66 ± 0.12 | 1085.62 | 0.000% |
| `random_proposer` | 100.00% ± 0.00% | 248.24 ± 24.86 | 1246.31 | 0.000% |
| `shard_retrieval_only_no_itl` | 0.00% ± 0.00% | 1.00 ± 0.00 | 12.07 | 2.565% |

- **SHARD vs Unguided Exhaustive:** `99.53%` proposal reduction (`214.7×` fewer, 95% CI low `214.5×`) and **`31.91×` real CPU speedup** (95% CI low `26.64×`).
- **SHARD vs Random Proposer:** `99.59%` proposal reduction (`248.2×` fewer, 95% CI low `223.4×`) and **`37.81×` real CPU speedup** (95% CI low `28.54×`).
- **Similarity-Only Ablation (`shard_retrieval_only_no_itl`):** `0.00%` exact recovery on disjoint unseen vocabularies (confirms similarity alone cannot substitute for ITL verification).

## 3. Task B — Multi-Step Goal-Directed Plan Search (Depths 2 & 4, 48 Operators, 120 Unseen Tasks)

| Planner | Overall Goal Acc (Mean ± 95% CI) | Depth-2 Acc | Depth-4 Acc | Temporal Acc | Mean Proposals | CPU Time/Seed (ms) | Collateral |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shard_guided_planner` | 100.00% ± 0.00% | 100.00% | 100.00% | 100.00% | 22.00 | 154.59 | 0.000% |
| `unguided_bfs_planner` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 600.00 | 2567.70 | 0.000% |
| `random_walk_planner` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 600.00 | 2961.76 | 0.000% |

- **SHARD vs Unguided BFS:** `96.33%` proposal reduction (`27.3×` fewer, 95% CI low `25.6×`) and **`16.74×` real CPU speedup** (95% CI low `15.60×`).
- **SHARD vs Random Walk:** `96.33%` proposal reduction (`27.3×` fewer, 95% CI low `25.6×`) and **`19.56×` real CPU speedup** (95% CI low `15.81×`).

## 4. Task C — Bind-Depth Geometry Stress Probe (Depths 1–6, Codebook=64, 5 Seeds)

| Depth | Role-Permuted Clean-Up Acc (Mean ± 95% CI) | Permuted Mean Sim (/10,240) | Naive Superposition Acc | Naive Mean Sim (/10,240) |
|---:|---:|---:|---:|---:|
| 1 | 100.00% ± 0.00% | 10240.0 | 100.00% | 10240.0 |
| 2 | 100.00% ± 0.00% | 5127.4 | 50.80% | 5252.1 |
| 3 | 100.00% ± 0.00% | 5116.4 | 34.67% | 5415.5 |
| 4 | 100.00% ± 0.00% | 3840.9 | 26.40% | 4150.7 |
| 5 | 100.00% ± 0.00% | 3842.8 | 22.40% | 4396.9 |
| 6 | 100.00% ± 0.00% | 3199.7 | 20.27% | 3955.5 |

## 5. Bit-Budget Accounting (5-Seed Mean, 48-Operator Scaled Library)

- `raw_training_data_bits`: `704,301`
- `compressed_training_data_bits`: `52,427`
- `operator_library_bits` (48 cards raw): `199,528` (compressed MDL: `12,104`)
- `active_bundle_bits` (48 × 10,240-bit BSC + 12-table LSH index): `509,952`
- `episode_store_bits` (SHARD episodic LSH index): `333,824`
- `compressed_model_mdl_bits` (operators + BSC index): `503,624` (`MDL / Raw Train = 0.7151`)

## 6. Stage S4 Keep/Remove Decision & Next Stage

- **Decision:** `KEEP_SHARD_AS_PROPOSER` — SHARD 10,240-bit BSC + LSH proposal guidance reduces BOTH search proposals AND real single-core CPU runtime by over an order of magnitude at equal or strictly higher exact ITL-verified accuracy (`0.000%` collateral).
- **S4 Gate:** PASS — Ready for `@auditor` independent verification before advancing to Stage S5.
