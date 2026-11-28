# Stage S11.1 Raw-Observation Open-World Learning — Reproducible Results & Gate Ledger

- **Stage / Protocol:** `S11.1` / `s11.1-raw-observation-learning-v1`
- **Honest Verdict:** `S11_1_PARTIAL_PROGRESS__PREREGISTERED_GATES_NOT_ALL_PASSED`
- **S11.1 Complete (All 4 Preregistered Gates Passed):** `False` (`1/4` gates passed)
- **Reproduce:** `PYTHONHASHSEED=0 python -m research.open_world.evaluation.run_s11_1_experiment`

## 1. Full S11.1 Learner vs. Six Controlled Ablations (Hidden Evaluation Splits)

| Benchmark | Variant | Primary Score (95% CI) | Val Mirr | Evid F1 | Unans Rec | False Abs | ECE | SHARD Cands/Q | CPU ms/item | Peak RAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced_copa` | `s11_1_full` | `49.20% [44.8,53.8]` | `17.45%` | `74.60%` | `—` | `0.00%` | `0.0346` | `50.25` | `0.591` | `60.962` |
| `balanced_copa` | `A1_no_evidence_grounding_verifier` | `49.20% [44.8,53.8]` | `17.45%` | `74.60%` | `—` | `0.00%` | `0.0346` | `50.25` | `0.560` | `60.943` |
| `balanced_copa` | `A2_no_bitemporal_supersession` | `49.20% [44.8,53.8]` | `17.45%` | `74.60%` | `—` | `0.00%` | `0.0346` | `50.25` | `0.498` | `60.943` |
| `balanced_copa` | `A3_no_asymmetric_causal_lift` | `47.61% [43.2,52.2]` | `16.11%` | `73.80%` | `—` | `0.00%` | `0.0287` | `50.25` | `0.467` | `60.942` |
| `balanced_copa` | `A4_no_capability_anti_unification` | `49.20% [44.8,53.8]` | `17.45%` | `74.60%` | `—` | `0.00%` | `0.0346` | `110.95` | `0.840` | `60.954` |
| `balanced_copa` | `A5_no_shard_lsh_indexing` | `49.20% [44.8,53.8]` | `17.45%` | `74.60%` | `—` | `0.00%` | `0.0346` | `362.00` | `1.401` | `60.942` |
| `balanced_copa` | `A6_no_abstention_gate` | `49.20% [44.8,53.8]` | `17.45%` | `74.60%` | `—` | `0.00%` | `0.0346` | `50.25` | `0.498` | `60.943` |
| `squad_v2` | `s11_1_full` | `28.89% [24.0,32.5]` | `—` | `81.75%` | `20.00%` | `3.33%` | `0.0505` | `9.87` | `3.943` | `2.406` |
| `squad_v2` | `A1_no_evidence_grounding_verifier` | `26.69% [22.0,30.4]` | `—` | `79.52%` | `16.67%` | `8.57%` | `0.0534` | `9.87` | `3.125` | `2.430` |
| `squad_v2` | `A2_no_bitemporal_supersession` | `28.89% [24.0,32.5]` | `—` | `81.75%` | `20.00%` | `3.33%` | `0.0505` | `9.87` | `4.055` | `2.400` |
| `squad_v2` | `A3_no_asymmetric_causal_lift` | `28.89% [24.0,32.5]` | `—` | `81.75%` | `20.00%` | `3.33%` | `0.0505` | `9.87` | `4.204` | `2.401` |
| `squad_v2` | `A4_no_capability_anti_unification` | `29.13% [24.6,32.6]` | `—` | `81.75%` | `22.86%` | `3.33%` | `0.0671` | `193.54` | `4.688` | `2.753` |
| `squad_v2` | `A5_no_shard_lsh_indexing` | `28.89% [24.0,32.5]` | `—` | `81.75%` | `20.00%` | `3.33%` | `0.0505` | `10.00` | `3.869` | `2.398` |
| `squad_v2` | `A6_no_abstention_gate` | `19.13% [15.7,22.3]` | `—` | `83.37%` | `0.00%` | `0.00%` | `0.0349` | `9.87` | `4.092` | `2.401` |
| `longmemeval` | `s11_1_full` | `24.37% [17.9,30.1]` | `—` | `60.16%` | `57.14%` | `27.66%` | `0.0296` | `16.46` | `85.009` | `11.643` |
| `longmemeval` | `A1_no_evidence_grounding_verifier` | `22.79% [16.4,28.3]` | `—` | `72.66%` | `28.57%` | `14.89%` | `0.0122` | `16.46` | `79.904` | `11.674` |
| `longmemeval` | `A2_no_bitemporal_supersession` | `17.25% [12.5,22.2]` | `—` | `50.74%` | `42.86%` | `32.45%` | `0.0390` | `16.46` | `83.540` | `11.644` |
| `longmemeval` | `A3_no_asymmetric_causal_lift` | `24.37% [17.9,30.1]` | `—` | `60.16%` | `57.14%` | `27.66%` | `0.0296` | `16.46` | `94.833` | `11.644` |
| `longmemeval` | `A4_no_capability_anti_unification` | `22.96% [16.5,29.1]` | `—` | `60.16%` | `57.14%` | `27.66%` | `0.0077` | `141.56` | `86.015` | `11.872` |
| `longmemeval` | `A5_no_shard_lsh_indexing` | `24.37% [17.9,30.1]` | `—` | `60.16%` | `57.14%` | `27.66%` | `0.0296` | `18.00` | `249.304` | `11.643` |
| `longmemeval` | `A6_no_abstention_gate` | `22.34% [15.5,27.1]` | `—` | `80.72%` | `0.00%` | `0.00%` | `0.0185` | `16.46` | `178.063` | `11.643` |

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
- [ ] `session_evidence_f1_ge_best_baseline`: `FAIL`
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
- [x] `unseen_contradictory_observation_marked_contested_and_abstained`: `PASS`
- [x] `unseen_executable_date_arithmetic`: `PASS`
- [x] `unseen_adversarial_antonym_and_entity_swap_abstention`: `PASS`
- [x] `stage9_poison_quarantine_and_non_destructive_rollback`: `PASS`
- [x] `unseen_utf8_nonlatin_and_multichoice_causal_robustness`: `PASS`
