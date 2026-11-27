# Stage S11.0 Open-World Generalization Foundation — Reproducible Baseline Report

- **Stage / Protocol:** `S11.0` / `s11.0-open-world-eval-v1`
- **Foundation Gate:** `PASS` (`11/11` checks passed)
- **Open-World Verdict:** `S11_0_EVALUATION_FOUNDATION_ESTABLISHED__OPEN_WORLD_GENERALIZATION_NOT_YET_ACHIEVED`
- **Reproduce:** `PYTHONHASHSEED=0 python -m research.open_world.evaluation.run_s11_0_foundation`
- **Zero Raw Dataset Redistribution:** Pinned external datasets are SHA-256 verified at runtime into gitignored `.external_eval_cache/`.

## 1. Source-Disjoint Benchmark Matrix (Hidden Evaluation Splits)

| Benchmark | Solver | Category | Primary Acc | Selective Acc | Evid F1 | Unans Abstain Rec | False Abstain | Brier | ECE | MDL Bits | CPU ms/item | Peak RAM (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced_copa` | `existing_mirror_system` | `existing_mirror` | `35.46%` | `50.57%` | `72.71%` | `—` | `29.88%` | `0.2588` | `0.1915` | `744` | `1.401` | `0.347` |
| `balanced_copa` | `retrieval_only_bm25` | `retrieval_only` | `48.21%` | `48.21%` | `74.10%` | `—` | `0.00%` | `0.2915` | `0.0890` | `74,856` | `3.299` | `0.458` |
| `balanced_copa` | `existing_rule_blueprint` | `existing_rule_blueprint` | `0.00%` | `—` | `0.00%` | `—` | `100.00%` | `0.4225` | `0.6500` | `21,784` | `0.156` | `0.015` |
| `balanced_copa` | `simple_symbolic_extractor` | `simple_symbolic` | `47.21%` | `47.21%` | `73.61%` | `—` | `0.00%` | `0.2909` | `0.0913` | `545,216` | `0.311` | `1.671` |
| `balanced_copa` | `evaluator_only_oracle` | `evaluator_ceiling` | `100.00%` | `100.00%` | `100.00%` | `—` | `0.00%` | `0.0000` | `0.0000` | `0` | `0.023` | `0.011` |
| `squad_v2` | `existing_mirror_system` | `existing_mirror` | `49.97%` | `16.97%` | `3.97%` | `99.05%` | `95.71%` | `0.2511` | `0.0037` | `744` | `2.902` | `0.553` |
| `squad_v2` | `retrieval_only_bm25` | `retrieval_only` | `22.31%` | `13.55%` | `68.44%` | `23.33%` | `19.52%` | `0.3867` | `0.4565` | `276,912` | `0.632` | `0.969` |
| `squad_v2` | `existing_rule_blueprint` | `existing_rule_blueprint` | `49.76%` | `0.00%` | `0.00%` | `99.52%` | `100.00%` | `0.2726` | `0.1518` | `21,784` | `0.186` | `0.015` |
| `squad_v2` | `simple_symbolic_extractor` | `simple_symbolic` | `30.95%` | `18.03%` | `66.03%` | `36.67%` | `23.33%` | `0.3864` | `0.3882` | `276,848` | `2.530` | `0.961` |
| `squad_v2` | `evaluator_only_oracle` | `evaluator_ceiling` | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `0.00%` | `0.0000` | `0.0000` | `0` | `0.037` | `0.011` |
| `longmemeval` | `existing_mirror_system` | `existing_mirror` | `2.28%` | `2.76%` | `11.77%` | `0.00%` | `18.62%` | `0.2691` | `0.5074` | `744` | `849.783` | `1.090` |
| `longmemeval` | `retrieval_only_bm25` | `retrieval_only` | `4.27%` | `4.57%` | `73.33%` | `0.00%` | `6.91%` | `0.6021` | `0.7568` | `1,120,784` | `66.445` | `4.608` |
| `longmemeval` | `existing_rule_blueprint` | `existing_rule_blueprint` | `6.93%` | `—` | `0.00%` | `100.00%` | `100.00%` | `0.4017` | `0.5807` | `21,784` | `0.309` | `0.018` |
| `longmemeval` | `simple_symbolic_extractor` | `simple_symbolic` | `4.11%` | `3.19%` | `66.98%` | `21.43%` | `17.55%` | `0.4595` | `0.6477` | `1,120,760` | `105.507` | `3.828` |
| `longmemeval` | `evaluator_only_oracle` | `evaluator_ceiling` | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `0.00%` | `0.0000` | `0.0000` | `0` | `0.009` | `0.011` |

## 2. Stage S11.0 Foundation Gate Checklist

- [x] `ast_isolation_zero_llm_gpu_and_py311`: `PASS`
- [x] `s0_s10_audit_completed_and_gates_preserved`: `PASS`
- [x] `pr43_audited_not_merged_and_not_passing`: `PASS`
- [x] `external_dataset_licenses_verified_zero_git_redistribution`: `PASS`
- [x] `all_source_disjoint_splits_zero_leakage`: `PASS`
- [x] `all_5_required_baselines_evaluated`: `PASS`
- [x] `all_freeze_locks_enforced_zero_post_freeze_writes`: `PASS`
- [x] `evaluator_only_oracle_verified_ceiling_100pct`: `PASS`
- [x] `language_commonsense_and_long_memory_tasks_evaluated`: `PASS`
- [x] `production_dcrystal_vault_untouched`: `PASS`
- [x] `preregistration_protocol_written`: `PASS`
