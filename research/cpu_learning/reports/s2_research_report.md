# Mirror 7 CPU-Native Learning Research — Stage S2 ITL Operator Induction Report

- **Stage:** `S2` (Domain-Agnostic ITL Operator Induction from S1 `EpisodeLedger`)
- **S2 Acceptance Gate:** `PASS` (19 / 19 checks passed)
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s2_experiments`
- **OperatorCard Schema / S0 Generator Version:** `s2-operator-card-v1` / `s0-relworld-v1.0` (Temporal `s0-temporal-v2.0`)
- **Fixed Seeds:** `[101, 202, 303, 404, 505]` (6 demos/action = 24 train episodes/seed; 168 unseen tasks/seed = 840 total)
- **AST Isolation & Zero-Hardcoding Audit:** `PASS` (22 non-evaluator, 10 learner, 5 ITL files; 0 import violations, 0 hardcoded domain literals)
- **Environment:** Python `3.14.4` | `Windows-11-10.0.26200-SP0` | CPUs `8` | RAM `15.77 GB` | Git `5a247de7b03b` (`research/cpu-native-learning-s0`)

> **Research Integrity Notice:** Every `OperatorCard` in Stage S2 is induced strictly from the 6 public before/action/after demonstrations per action ingested into S1's append-only `EpisodeLedger` (0 imports of `hidden_actions.py`/`oracle.py`, 0 hardcoded action/predicate names, 0 test-time episode lookups). Stage S3 active counterexample/RDC guard refinement, S4 SHARD search guidance, and S5 library compression are NOT claimed here.

## 1. Stage S2 Acceptance Gate Verification (Lower 95% CI Bounds Across 5 Fixed Seeds)

| Check | Status |
|---|:---:|
| `five_distinct_fixed_seeds_present` | PASS |
| `demos_per_action_le_8` | PASS |
| `ast_strict_learner_isolation_and_zero_hardcoded_literals` | PASS |
| `s0_and_s1_committed_manifests_intact` | PASS |
| `all_solvers_frozen_before_temporal` | PASS |
| `itl_depth4_ci95_low_ge_95pct` | PASS |
| `itl_depth6_ci95_low_ge_90pct` | PASS |
| `itl_overall_unseen_ci95_low_ge_95pct` | PASS |
| `itl_temporal_frozen_ci95_low_ge_95pct` | PASS |
| `itl_operator_recovery_precision_ci95_low_ge_90pct` | PASS |
| `itl_operator_recovery_recall_ci95_low_ge_90pct` | PASS |
| `itl_collateral_rate_ci95_high_lt_2pct` | PASS |
| `itl_beats_episode_lookup_by_ge_20pp` | PASS |
| `itl_beats_shard_hdc_no_itl_by_ge_50pp` | PASS |
| `itl_zero_test_time_retrieval_and_canary_excluded` | PASS |
| `itl_mdl_strictly_compresses_training_data` | PASS |
| `itl_contrast_ablation_improves_guard_recall` | PASS |
| `s1_s2_raw_utf8_e2e_ci95_low_ge_95pct` | PASS |
| `cpu_budget_within_limit` | PASS |

## 2. Five-Seed Head-to-Head Matrix on 9 Unseen Splits (840 Unseen Tasks Total)

| Solver | Category | Unseen Exact Acc (Mean ± 95% CI) | Guard Contrasts | Depth-4 Acc | Depth-6 Acc | Temporal Frozen | Op-Rec Prec / Rec / F1 | Collateral Rate | Preds/s (1-core) | Peak Heap (KB) | Model MDL (bits) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `itl_operator_learner` | `itl_candidate_learner` | 100.00% ± 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 97.6% / 100.0% / 98.8% | 0.000% | 1773.9 | 907.3 | 4,440 |
| `itl_operator_learner_no_contrast` | `itl_ablation` | 99.40% ± 0.00% | 95.00% | 100.00% | 100.00% | 100.00% | 97.5% / 95.1% / 96.3% | 0.030% | 1665.6 | 907.0 | 4,176 |
| `itl_operator_learner_raw_utf8_e2e` | `s1_s2_end_to_end_learner` | 100.00% ± 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 97.6% / 100.0% / 98.8% | 0.000% | 569.0 | 905.4 | 4,440 |
| `noop_identity_floor` | `sanity_floor` | 5.95% ± 0.00% | 50.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 0.000% | 7862.2 | 18.3 | 0 |
| `episode_lookup` | `memory_baseline` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 0.000% | 4264.0 | 19.6 | 11,581 |
| `structural_knn` | `memory_baseline` | 2.98% ± 0.00% | 25.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 4.567% | 863.4 | 168.2 | 30,357 |
| `shard_hdc_retrieval_no_itl` | `shard_baseline` | 3.57% ± 0.74% | 30.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 4.563% | 620.7 | 75.3 | 249,262 |
| `small_cpu_neural_reference` | `comparison_only_neural` | 2.74% ± 2.00% | 23.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 2.644% | 920.5 | 56.5 | 23,019 |

## 3. Per-Split Exact Learning Accuracy — `itl_operator_learner` vs Ablation & End-to-End

| Unseen Split | `itl_operator_learner` | `itl_operator_learner_no_contrast` | `itl_operator_learner_raw_utf8_e2e` | `shard_hdc_retrieval_no_itl` | `episode_lookup` |
|---|---:|---:|---:|---:|---:|
| `heldout_unseen_names` | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_role_swaps` | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_guard_contrasts` | 100.00% | 95.00% | 100.00% | 30.00% | 0.00% |
| `heldout_repeated_idempotent` | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_order_reversal` | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_comp_depth2` | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_comp_depth4` | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_comp_depth6` | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_temporal_frozen` | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |

## 4. Bit-Budget & MDL Compression Audit (5-Seed Mean)

| Solver | Raw Train (bits) | Zlib Train (bits) | Episode Store (bits) | Operator Library Raw (bits) | Model MDL (bits) | MDL / Raw Train |
|---|---:|---:|---:|---:|---:|---:|
| `itl_operator_learner` | 704,301 | 52,427 | 0 | 22,720 | 4,440 | 0.0063 |
| `itl_operator_learner_no_contrast` | 704,301 | 52,427 | 0 | 20,832 | 4,176 | 0.0059 |
| `itl_operator_learner_raw_utf8_e2e` | 704,301 | 52,427 | 0 | 22,720 | 4,440 | 0.0063 |
| `episode_lookup` | 704,301 | 52,427 | 44,656 | 0 | 11,581 | 0.0164 |
| `shard_hdc_retrieval_no_itl` | 704,301 | 52,427 | 277,552 | 0 | 249,262 | 0.3539 |
| `small_cpu_neural_reference` | 704,301 | 52,427 | 25,814 | 0 | 23,019 | 0.0327 |

## 5. Honest Scientific Analysis of Induced Operator Schemas (Seed 101)

- **Induced schemas across all 4 actions:** `41 / 41` gold clauses recovered (`100.00%` recall) with `1` conservative extra clause (`97.62%` precision, `98.80%` F1): `(?source, accessible, true)` on `move`.
- **Why that 1 conservative clause is scientifically honest:** In all 6 training demonstrations of `move`, `?source` always has `accessible=true` (only `?destination` is ever made inaccessible in the training negatives). Without Stage S3 active counterexample probing, an honest learner cannot distinguish whether `source.accessible=true` is necessary or incidental. Stage S3 RDC counterfactual experiments will directly test and prune uncontrasted preconditions.
- **Causal ablation proof:** Disabling negative-contrast 1-hop induction (`itl_operator_learner_no_contrast`) causes recall to drop from `100.00%` -> `95.12%` (`give` misses `(?object_container_if_any, is_open, true)` and `is_locked=false`) and drops `heldout_guard_contrasts` from `100.00%` -> `95.00%` with nonzero collateral corruption.

## 6. Next Stage Decision

- **S2 Gate:** PASS — All 19 Stage S2 ITL gates passed across 5 fixed seeds. Ready for **Stage S3 (RDC Verification, Active Counterexample Generation & Exception/Precondition Refinement)**.
