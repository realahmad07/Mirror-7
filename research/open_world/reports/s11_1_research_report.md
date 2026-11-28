# Stage S11.1 Raw-Observation Open-World Learning — Reproducible Results & Gate Ledger

- **Stage / Protocol:** `S11.1` / `s11.1-raw-observation-learning-v1`
- **Honest Verdict:** `S11_1_PARTIAL_PROGRESS__PREREGISTERED_GATES_NOT_ALL_PASSED`
- **S11.1 Complete (All 4 Preregistered Gates Passed):** `False` (`1/4` gates passed)
- **Reproduce:** `PYTHONHASHSEED=0 python -m research.open_world.evaluation.run_s11_1_experiment`

## 1. Full S11.1 Learner vs. Six Controlled Ablations (Hidden Evaluation Splits)

| Benchmark | Variant | Primary Score (95% CI) | Val Mirr | Evid F1 | Unans Rec | False Abs | ECE | SHARD Cands/Q | CPU ms/item | Peak RAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced_copa` | `s11_1_full` | `47.61% [43.2,51.8]` | `14.77%` | `73.80%` | `—` | `0.00%` | `0.0612` | `46.70` | `6.256` | `104.346` |
| `balanced_copa` | `A1_no_evidence_grounding_verifier` | `47.61% [43.2,51.8]` | `14.77%` | `73.80%` | `—` | `0.00%` | `0.0612` | `46.70` | `4.824` | `0.428` |
| `balanced_copa` | `A2_no_bitemporal_supersession` | `47.61% [43.2,51.8]` | `14.77%` | `73.80%` | `—` | `0.00%` | `0.0612` | `46.70` | `4.793` | `0.428` |
| `balanced_copa` | `A3_no_asymmetric_causal_lift` | `48.61% [43.8,52.8]` | `15.44%` | `74.30%` | `—` | `0.00%` | `0.0272` | `46.70` | `4.669` | `0.421` |
| `balanced_copa` | `A4_no_capability_anti_unification` | `47.61% [43.2,51.8]` | `14.77%` | `73.80%` | `—` | `0.00%` | `0.0612` | `112.71` | `5.105` | `0.473` |
| `balanced_copa` | `A5_no_shard_lsh_indexing` | `47.61% [43.2,51.8]` | `14.77%` | `73.80%` | `—` | `0.00%` | `0.0612` | `361.00` | `6.630` | `0.432` |
| `balanced_copa` | `A6_no_abstention_gate` | `47.61% [43.2,51.8]` | `14.77%` | `73.80%` | `—` | `0.00%` | `0.0612` | `46.70` | `5.011` | `0.428` |
| `squad_v2` | `s11_1_full` | `30.67% [26.2,34.8]` | `—` | `75.40%` | `25.24%` | `13.33%` | `0.0381` | `9.88` | `3.757` | `1.181` |
| `squad_v2` | `A1_no_evidence_grounding_verifier` | `28.11% [23.9,32.3]` | `—` | `75.40%` | `20.95%` | `13.33%` | `0.0552` | `9.88` | `7.664` | `0.094` |
| `squad_v2` | `A2_no_bitemporal_supersession` | `30.67% [26.2,34.8]` | `—` | `75.40%` | `25.24%` | `13.33%` | `0.0381` | `9.88` | `6.957` | `0.119` |
| `squad_v2` | `A3_no_asymmetric_causal_lift` | `30.67% [26.2,34.8]` | `—` | `75.40%` | `25.24%` | `13.33%` | `0.0381` | `9.88` | `4.873` | `0.119` |
| `squad_v2` | `A4_no_capability_anti_unification` | `30.93% [26.6,34.2]` | `—` | `75.40%` | `27.14%` | `13.33%` | `0.0762` | `193.04` | `5.097` | `0.177` |
| `squad_v2` | `A5_no_shard_lsh_indexing` | `30.67% [26.2,34.8]` | `—` | `75.40%` | `25.24%` | `13.33%` | `0.0381` | `10.00` | `6.027` | `0.117` |
| `squad_v2` | `A6_no_abstention_gate` | `18.89% [15.5,22.2]` | `—` | `83.05%` | `0.00%` | `0.00%` | `0.0438` | `9.88` | `4.576` | `0.119` |
| `longmemeval` | `s11_1_full` | `22.16% [16.8,27.2]` | `—` | `74.95%` | `35.71%` | `10.64%` | `0.0616` | `16.36` | `127.939` | `7.327` |
| `longmemeval` | `A1_no_evidence_grounding_verifier` | `19.62% [13.5,24.8]` | `—` | `74.37%` | `7.14%` | `11.70%` | `0.0505` | `16.36` | `130.183` | `1.796` |
| `longmemeval` | `A2_no_bitemporal_supersession` | `16.27% [11.7,21.4]` | `—` | `64.65%` | `35.71%` | `12.77%` | `0.0434` | `16.36` | `117.961` | `1.839` |
| `longmemeval` | `A3_no_asymmetric_causal_lift` | `22.16% [16.8,27.2]` | `—` | `74.95%` | `35.71%` | `10.64%` | `0.0616` | `16.36` | `92.590` | `1.845` |
| `longmemeval` | `A4_no_capability_anti_unification` | `20.68% [14.9,25.9]` | `—` | `74.95%` | `35.71%` | `10.64%` | `0.0800` | `141.51` | `107.905` | `1.845` |
| `longmemeval` | `A5_no_shard_lsh_indexing` | `22.16% [16.8,27.2]` | `—` | `74.95%` | `35.71%` | `10.64%` | `0.0616` | `18.00` | `94.678` | `1.836` |
| `longmemeval` | `A6_no_abstention_gate` | `20.35% [14.3,25.4]` | `—` | `79.81%` | `0.00%` | `0.00%` | `0.0303` | `16.36` | `84.700` | `1.838` |

## 2. Preregistered S11.1 Gate Evaluation (Zero Gate Weakening)

### `S11_1_G1` — `FAIL`
- [ ] `beats_best_non_oracle_by_5pp`: `FAIL`
- [ ] `non_overlapping_bootstrap_ci95`: `FAIL`
- [ ] `val_mirrored_pair_consistency_ge_50pct`: `FAIL`
- [x] `zero_manual_schemas`: `PASS`

### `S11_1_G2` — `FAIL`
- [x] `answerable_f1_beats_best_baseline_by_5pp`: `PASS`
- [ ] `overall_f1_beats_best_non_blanket_baseline_by_5pp`: `FAIL`
- [ ] `overall_f1_beats_all_non_oracle_baselines_by_5pp`: `FAIL`
- [x] `evidence_f1_ge_best_baseline`: `PASS`
- [ ] `unanswerable_abstention_recall_ge_70pct`: `FAIL`
- [x] `answerable_false_abstention_le_30pct`: `PASS`
- [x] `ece_le_0_15`: `PASS`

### `S11_1_G3` — `FAIL`
- [x] `primary_score_beats_best_non_oracle_by_5pp`: `PASS`
- [x] `cross_category_transfer_improved`: `PASS`
- [x] `session_evidence_f1_ge_best_baseline`: `PASS`
- [ ] `unanswerable_abs_recall_ge_60pct`: `FAIL`

### `S11_1_G4` — `PASS`
- [x] `zero_gpu_llm_api_neural`: `PASS`
- [x] `zero_manual_domain_schemas`: `PASS`
- [x] `ast_isolation_passed`: `PASS`
- [x] `all_freeze_locks_enforced`: `PASS`
- [x] `zero_post_freeze_memory_writes`: `PASS`
- [x] `peak_tracemalloc_under_256mb`: `PASS`
- [x] `production_dcrystal_vault_untouched`: `PASS`

## 3. Unseen-Combination & Stage 9 Rollback Probe Suite

- [x] `unseen_bitemporal_supersession_latest_and_historical`: `PASS`
- [x] `unseen_contradictory_observation_marked_contested`: `PASS`
- [x] `unseen_executable_date_arithmetic`: `PASS`
- [x] `unseen_adversarial_antonym_and_entity_swap_abstention`: `PASS`
- [x] `stage9_poison_quarantine_and_non_destructive_rollback`: `PASS`
