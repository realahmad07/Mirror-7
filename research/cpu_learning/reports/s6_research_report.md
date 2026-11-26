# Mirror 7 CPU-Native Learning — Stage S6 Research Report (Mathematics, List & Code Transfer)

- **Stage S6 Gate:** **PASS** (`21/21` checks passed across 5 fixed seeds)
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s6_experiments`
- **AST & Lexical Leakage Audit:** `PASS` (`26` learner modules, `5` transfer modules; `0` forbidden imports, `0` hardcoded S0 literals, `0/20` HumanEval-Micro task names in `transfer_inducer.py`)
- **CPU Budget:** Mean `0.36s/seed` (max `0.44s/seed` <= `30s`)

## 1. Stage S6 Acceptance Gates (Lower 95% CI Bounds Across 5 Fixed Seeds)

| Check | Status |
|---|:---:|
| `ast_isolation_and_zero_hardcoded_s0_literals` | PASS |
| `zero_humaneval_task_names_in_transfer_inducer` | PASS |
| `prior_s0_s1_s2_s3_s4_s5_manifests_intact` | PASS |
| `math_s0_modular_addition_canary_ci95_low_ge_0.99` | PASS |
| `math_sparse_16_modular_addition_ci95_low_ge_0.99` | PASS |
| `math_beats_s0_shard_canary_by_90pp` | PASS |
| `math_unseen_multidigit_3_4_6_8_and_cascades_ci95_low_ge_0.98` | PASS |
| `math_novel_base7_5digit_transfer_ci95_low_ge_0.98` | PASS |
| `math_novel_base16_5digit_transfer_ci95_low_ge_0.98` | PASS |
| `list_unseen_len8_20_atomic_ci95_low_ge_0.95` | PASS |
| `list_depth2_compositions_ci95_low_ge_0.95` | PASS |
| `list_depth3_compositions_ci95_low_ge_0.95` | PASS |
| `code_ast_refactoring_compile_and_exec_ci95_low_ge_0.95` | PASS |
| `humaneval_micro_pdf_50pct_gate_passed` | PASS |
| `humaneval_micro_shard_itl_rdc_ci95_low_ge_0.65` | PASS |
| `humaneval_micro_beats_io_memorization_lookup` | PASS |
| `humaneval_micro_shard_beats_unguided_pass_rate_under_budget` | PASS |
| `humaneval_micro_shard_reduces_candidate_evals_ge_35pct` | PASS |
| `zero_collateral_slot_corruptions_all_seeds` | PASS |
| `mdl_compression_verified_all_seeds` | PASS |
| `cpu_budget_within_30s_per_seed` | PASS |

## 2. Domain 1 — Modular Addition Canary & Unseen Multi-Digit Column Arithmetic

| Benchmark / Split | Method | Unseen Accuracy (Mean ± 95% CI) | Notes |
|---|---|---:|---|
| S0 Modular Addition `(a + b) % 10` (60/40 split) | `s0_shard_pure_hdc_similarity` | 4.00% ± 1.70% | S0 HDC similarity baseline |
| S0 Modular Addition `(a + b) % 10` (60/40 split) | `s0_shard_bit_perceptron` | 7.00% ± 2.60% | S0 local bit-perceptron baseline |
| S0 Modular Addition `(a + b) % 10` (60/40 split) | **`s6_shard_itl_rdc_transfer`** | **100.00% ± 0.00%** | **`+93.00pp` over S0 SHARD** |
| Sparse Modular Addition `(a + b) % 10` (16/84 split) | `s6_shard_itl_rdc_transfer` | 100.00% ± 0.00% | 16 training pairs, 84 unseen pairs |
| Unseen Multi-Digit Base-10 (`3, 4, 6, 8` digits + carry cascades) | `s6_shard_itl_rdc_transfer` | 100.00% ± 0.00% | 16 single-column demos, **0 multi-digit demos** |
| Unseen Novel Radix `base=7` (5-digit addition + cascades) | `s6_shard_itl_rdc_transfer` | 100.00% ± 0.00% | 16 base-7 column demos, learner untouched |
| Unseen Novel Radix `base=16` (5-digit hex addition + cascades) | `s6_shard_itl_rdc_transfer` | 100.00% ± 0.00% | 16 base-16 column demos, learner untouched |

## 3. Domain 2 — Typed List Transformations (`len <= 5` Train -> Unseen `len 8..20` & Compositions)

| Evaluation Suite | Unseen Accuracy (Mean ± 95% CI) | Collateral Corruptions |
|---|---:|---:|
| Atomic List Ops (`len 8..20`, `[-500, 500]`, `[]` & `[x]`) | 100.00% ± 0.00% | 0 |
| Held-Out Depth-2 List Compositions (`len 8..20`) | 100.00% ± 0.00% | 0 |
| Held-Out Depth-3 List Compositions (`len 8..20`) | 100.00% ± 0.00% | 0 |

## 4. Domain 3A — Python AST Structural Refactoring Induction

- **Compile-Valid Rate:** `100.00%`
- **Exact Execution-Verified Unseen Function Accuracy:** **`100.00% ± 0.00%`** (across `rename_parameter`, `swap_if_negated_guard`, `fold_identity_arithmetic`, `wrap_nonnegative_clamp`).
- **Naive String-Replace Baseline on `rename_parameter`:** `0.00%` (corrupts string literals containing the old identifier).

## 5. Domain 3B — 20-Task Execution-Only HumanEval-Micro Benchmark (PDF p. 13 Gate >= 50%)

| Method | Hidden Test Pass Rate (Mean ± 95% CI) | Solved / 20 | Mean Candidate Evals / Task |
|---|---:|---:|---:|
| `shard_itl_rdc_code_inducer` | 95.00% ± 0.00% | 19.0 / 20 | 12.70 |
| `unguided_enumerative_code_inducer` | 85.00% ± 0.00% | 17.0 / 20 | 27.55 |
| `io_memorization_lookup` | 0.00% ± 0.00% | 0.0 / 20 | 0.00 |
| `prior_external_mbpp_reference` | 0.00% | 0.0 / 20 | — |

- **SHARD + ITL + RDC vs Unguided Enumerative (identical `48`-candidate budget):** `+10.00pp` higher pass rate AND **`53.90%` fewer candidate evaluations** (95% CI low `53.90%`).
- **Honest Underdetermined Task Note:** `count_vowels` (`1/20`) is intentionally unsolved because `transfer_inducer.py` contains zero hardcoded English alphabet/vowel constants and receives only 4 short words without docstrings (`19/20 = 95.00%` solved purely from generic combinators).

## 6. Bit-Budget Accounting (5-Seed Mean, Unified All-Domain Single-Archive MDL)

- `raw_training_bits` (math + list + AST + HumanEval): `377,306`
- `compressed_training_bits` (single-stream zlib): `33,448`
- `compressed_induced_library_bits` (single-stream zlib): `12,944` (`ratio_vs_raw = 0.0343`, `ratio_vs_compressed_train = 0.3870`)
- `sum_per_card_compressed_bits` (unshared per-card zlib sum): `35,104`

## 7. Stage S6 Verdict

- **Verdict:** PASS — The exact same before/action/after SHARD + ITL + RDC learner solves the S0 modular addition canary (100% vs 4%/7% HDC), unseen 8-digit & novel base-7/16 addition, unseen length-20 depth-3 list pipelines, unseen Python AST refactorings, and 95.0% (19/20, >= 50% PDF gate) on execution-only HumanEval-Micro.
