# Mirror 7 CPU-Native Learning — Stage S5 Research Report (RDC Refutation & Consolidation)

- **Stage S5 Gate:** **PASS** (`24/24` checks passed across 5 fixed seeds)
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s5_experiments`
- **MDL Objective:** `J = L(library) + L(examples | library) + 10000 * verified_failures + 20000 * collateral_changes`
- **AST Isolation Audit:** `PASS` (`21` learner modules, `7` ITL, `4` SHARD, `5` RDC; `0` forbidden imports, `0` hardcoded domain literals)
- **CPU Budget:** Mean `6.54s/seed` (max `10.56s/seed` <= `30s`)

## 1. Stage S5 Acceptance Gates & All 4 RDC Kill Tests (95% CI Bounds, 5 Fixed Seeds)

| Check | Status |
|---|:---:|
| `ast_isolation_and_zero_hardcoded_literals_itl_shard_rdc` | PASS |
| `prior_s0_s1_s2_s3_s4_manifests_intact` | PASS |
| `kill_test_1_dream_zero_self_certified_all_seeds` | PASS |
| `kill_test_1_dream_j_strictly_reduced_all_seeds` | PASS |
| `kill_test_1_dream_ood_err_ci_strictly_below_plain_mdl` | PASS |
| `kill_test_1_dream_gain_vs_plain_mdl_ci_low_ge_0.15` | PASS |
| `kill_test_1_dream_ood_acc_ci_low_ge_0.95` | PASS |
| `kill_test_1_dream_beats_random_probes_ci_low_gt_0` | PASS |
| `kill_test_1_dream_collateral_ci_high_lt_0.02` | PASS |
| `kill_test_2_blame_top1_ci_low_ge_0.90` | PASS |
| `kill_test_2_blame_offline_top1_ci_low_ge_0.90` | PASS |
| `kill_test_2_blame_beats_random_by_40pp` | PASS |
| `kill_test_2_blame_beats_uniform_by_40pp` | PASS |
| `kill_test_2_ambiguity_multi_fault_honesty_all_seeds` | PASS |
| `kill_test_3_active_final_acc_ci_low_ge_0.98` | PASS |
| `kill_test_3_active_acc_ge_random` | PASS |
| `kill_test_3_active_query_reduction_ci_low_ge_0.35` | PASS |
| `kill_test_4_intra_safe_merge_reduces_j_and_bits_zero_regressions` | PASS |
| `kill_test_4_intra_unsafe_merge_rejected_and_rollback_verified` | PASS |
| `kill_test_4_cross_domain_reversibility_and_bits_saved_both_domains` | PASS |
| `kill_test_4_domain3_few_shot_transfer_acc_ci_low_ge_0.90` | PASS |
| `kill_test_4_domain3_few_shot_gain_ci_low_ge_0.20` | PASS |
| `kill_test_4_domain3_description_length_reduced_all_seeds` | PASS |
| `cpu_budget_within_30s_per_seed` | PASS |

## 2. Ablation 1 — Mechanism 1: Dream-Refutation Kill Test (`probe_budget=260`)

| Method | Unseen OOD Acc (Mean ± 95% CI) | OOD Error Rate | Collateral | Self-Certified Dreams |
|---|---:|---:|---:|---:|
| `plain_mdl_no_dreams` | 76.09% ± 0.00% | 23.91% | 0.341% | 0 |
| `rdc_random_probes` | 87.17% ± 12.27% | 12.83% | 0.124% | 0 |
| `rdc_dream_refutation` | 100.00% ± 0.00% | 0.00% | 0.000% | 0 |

- **Dream-Refutation vs Plain MDL (no dreams):** `+23.91pp` (95% CI low `+23.91pp`).
- **Dream-Refutation vs Random Probes (same budget):** `+12.83pp` (95% CI low `+0.55pp`).

## 3. Ablation 2 — Mechanism 3: Interventional Leave-One-Operator-Out Blame Kill Test (Depths 2–6)

| Method | Top-1 Exact Acc (Mean ± 95% CI) | MRR | Brier Score |
|---|---:|---:|---:|
| `interventional_blame` | 100.00% ± 0.00% | 1.0000 | 0.0000 |
| `interventional_blame_offline` | 100.00% ± 0.00% | 1.0000 | 0.0000 |
| `random_blame` | 34.82% ± 4.02% | 0.6043 | 0.4399 |
| `uniform_blame` | 0.00% ± 0.00% | 0.5155 | 0.2158 |

- **Interventional Blame vs Random Blame:** `+65.18pp` (95% CI low `+61.16pp`).
- **Interventional Blame vs Uniform Blame:** `+100.00pp` (95% CI low `+100.00pp`).
- **Ambiguity Multi-Fault Honesty Check:** `100%` PASS across all 5 seeds (`is_ambiguous=True`, `unique_culprit=None`, all faulty operators preserved in `suspected_operators`).

## 4. Ablation 3 — Mechanism 2: Active Max-Disagreement Query Selection Kill Test

| Strategy | Queries to >=98% OOD Acc (Mean ± 95% CI) | Total Queries Used | Final Unseen OOD Acc |
|---|---:|---:|---:|
| `max_disagreement_query` | 14.00 ± 0.00 | 16.00 | 100.00% |
| `random_query` | 93.20 ± 50.34 | 150.40 | 100.00% |

- **Simulator Query Reduction (`max_disagreement_query` vs `random_query`):** **`82.37%` fewer queries** (95% CI low `72.48%`).

## 5. Ablation 4 — Mechanism 4: Intra-Domain Safe Consolidation & Cross-Domain Concept Transfer (Kill Test 4)

### 5A. Intra-Domain Behavioral-Equivalence Consolidation & Rollback
- **Library Bits:** `4,720` -> `4,616` bits (**`-104` bits saved**).
- **MDL Objective `J`:** `4720.0` -> `4616.0` (`100.00%` regression retention, `0` regressions).
- **Safety & Rollback:** Unsafe merges (`open` vs `lock`, `move` vs `give`) rejected `100%` with `0` regressions; retired duplicate versions retained in `OperatorVersionStore` and deterministically restored via `rollback_consolidation()` across all 5 seeds.

### 5B. Cross-Domain Concept Abstraction (Domain 1 S0 + Domain 2 Arcane -> Held-Out Domain 3 Logistics Few-Shot)
- **Domain 1 (S0 `move`, `give`):** `3,464` -> `1,704` bits (**`-1,760` bits saved**).
- **Domain 2 (Arcane `channel`, `bestow`):** `3,448` -> `1,544` bits (**`-1,904` bits saved**).
- **Unamortized Joint MDL (Domain 1 + Domain 2 + Concepts):** `5,424` -> `4,336` bits (**`-1,088` bits saved**, `100%` lossless reversible reconstruction).

| Domain 3 Few-Shot Method (2 Positives/Action, 0 Negatives) | Domain 3 Unseen OOD Acc (Mean ± 95% CI) | Collateral | Description Bits |
|---|---:|---:|---:|
| `rdc_cross_domain_transfer` | 100.00% ± 0.00% | 0.000% | 1,680 |
| `cold_start_itl_no_concepts` | 63.64% ± 0.00% | 0.826% | 3,520 |

- **Held-Out 3rd-Domain Few-Shot Transfer Gain:** **`+36.36pp`** (95% CI low `+36.36pp`) and **`-1,840` description bits saved**.

## 6. Stage S5 Verdict

- **Verdict:** ALL FOUR RDC mechanisms independently pass their pre-registered kill tests on 95% confidence bounds across 5 fixed seeds with zero self-certified dreams, zero hardcoded domain literals, and zero regressions.
