# Stage S7 Research Report — Raw UTF-8 Language Grounding, Compositional Realization & Byte-LM Comparison

- **Stage Gate Verdict:** `PASS` (`24/24` checks passed)
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s7_experiments`
- **Seeds:** `[101, 202, 303, 404, 505]`
- **Mean Per-Seed CPU Runtime:** `20.05s` (total `109.99s`)

## 1. Raw-UTF-8 Construction Induction & Compositional Execution (Unseen Held-Out Entities)

| Split | Tasks/Seed | Mirror Exact (`mean [95% CI]`) | Mirror Inter (`mean`) | Collateral | Lexical Baseline (`mean`) |
|---|---:|---:|---:|---:|---:|
| `s7_unseen_entity_substitution` | 20 | `100.00% [100.00%, 100.00%]` | `100.00%` | `0.000%` | `100.00%` |
| `s7_unseen_paraphrase_novel_verbs` | 20 | `100.00% [100.00%, 100.00%]` | `100.00%` | `0.000%` | `100.00%` |
| `s7_unseen_word_orders_passive_sov` | 24 | `100.00% [100.00%, 100.00%]` | `100.00%` | `0.000%` | `100.00%` |
| `s7_negated_instructions` | 20 | `100.00% [100.00%, 100.00%]` | `100.00%` | `0.000%` | `0.00%` |
| `s7_conditional_clauses` | 24 | `100.00% [100.00%, 100.00%]` | `100.00%` | `0.000%` | `0.00%` |
| `s7_chained_multistep_depth2` | 16 | `100.00% [100.00%, 100.00%]` | `100.00%` | `0.000%` | `0.00%` |
| `s7_chained_multistep_depth4` | 16 | `100.00% [100.00%, 100.00%]` | `100.00%` | `0.000%` | `0.00%` |
| `s7_chained_multistep_depth6` | 16 | `100.00% [100.00%, 100.00%]` | `100.00%` | `0.000%` | `0.00%` |

## 2. Ambiguity Preservation, Minimal-Contrast Disambiguation & Epistemic Abstention

- **Pre-Contrast Ambiguity Preserved (`observe_utterance_in_scene`):** `100.00%` (honest abstention `100.00%`)
- **Post-Contrast Disambiguated Exact Accuracy (2 minimal demos):** `100.00%` vs No-Contrast First-Guess Ablation `50.00%`
- **OOD, Malformed UTF-8 & Partial-Mask Epistemic Abstention:** `100.00%` (`0.000%` hallucinated mutations)

## 3. Plan-Conditioned Truth-Locked Realizer (Strictly Separated Dual Metrics)

| Realizer Method | Semantic Correctness | Hallucination Rate | Surface Fluency Score | Reference Perplexity (`bpb`) |
|---|---:|---:|---:|---:|
| `mirror_truth_locked` | `100.00%` | `0.00%` | `0.9576` | `2.7659` |
| `telegraphic_tuple_dump` | `100.00%` | `0.00%` | `0.3073` | `9.5297` |
| `ungated_fluent_assumer` | `37.50%` | `48.83%` | `0.9158` | `4.2368` |

## 4. Zero-Leakage Multi-Domain UTF-8 Byte-LM Bits-Per-Byte Benchmark (PDF p. 13)

| Model ID | Comparison Only | Uses Backprop | Held-Out `bits_per_byte` (`mean [95% CI]`) | Model Bits | Train Sec | Eval Sec |
|---|:---:|:---:|---:|---:|---:|---:|
| `mirror_ctw_construction_byte_lm` | `False` | `False` | `2.5649 [2.5457, 2.5841]` | `1,919,059` | `3.087s` | `0.602s` |
| `ppm_c_order5_byte_lm` | `False` | `False` | `2.5910 [2.5747, 2.6074]` | `307,614` | `0.261s` | `0.183s` |
| `byte_3gram_kneser_ney` | `False` | `False` | `3.0243 [3.0021, 3.0466]` | `40,790` | `0.086s` | `0.063s` |
| `small_cpu_byte_transformer_ref` | `True` | `True` | `4.3633 [4.2305, 4.4961]` | `111,680` | `6.242s` | `6.948s` |
| `uniform_byte_floor` | `False` | `False` | `8.0000 [8.0000, 8.0000]` | `8` | `0.000s` | `0.015s` |

## 5. Unified Single-Stream Grounding & Operator Archive Bit Accounting

- **Raw Grounding Training Corpus Bits:** `198,693`
- **Compressed Grounding Training Corpus Bits:** `11,442`
- **Compressed Unified Induced Archive Bits (Operators + Constructions + Guards):** `9,616`
- **Compression Ratio vs Compressed Training Corpus:** `0.8408`

## 6. Stage S7 Gate Checklist

- [x] `ast_import_and_domain_literal_isolation`: `PASS`
- [x] `comparison_only_transformer_strictly_isolated`: `PASS`
- [x] `prior_s0_through_s6_manifests_intact`: `PASS`
- [x] `pool_disjointness_verified_all_seeds`: `PASS`
- [x] `unseen_entity_substitution_ci95_low_ge_095`: `PASS`
- [x] `unseen_paraphrase_novel_verbs_ci95_low_ge_095`: `PASS`
- [x] `unseen_word_orders_passive_sov_ci95_low_ge_095`: `PASS`
- [x] `negated_instructions_exact_ci95_low_ge_095`: `PASS`
- [x] `conditional_clauses_exact_ci95_low_ge_095`: `PASS`
- [x] `chained_depth2_exact_and_inter_ci95_low_ge_095`: `PASS`
- [x] `chained_depth4_exact_and_inter_ci95_low_ge_095`: `PASS`
- [x] `chained_depth6_exact_and_inter_ci95_low_ge_090`: `PASS`
- [x] `grounding_collateral_ci95_high_lt_002`: `PASS`
- [x] `mirror_strictly_beats_lexical_baseline`: `PASS`
- [x] `ambiguity_preserved_before_contrast_100pct`: `PASS`
- [x] `ambiguity_resolved_after_minimal_contrast_ci95_low_ge_095`: `PASS`
- [x] `ood_and_partial_mask_abstention_100pct_zero_hallucination`: `PASS`
- [x] `realizer_semantic_ci95_low_ge_098_and_zero_hallucination`: `PASS`
- [x] `realizer_dual_metrics_strictly_independent`: `PASS`
- [x] `byte_lm_mirror_strictly_beats_3gram_kneser_ney_ci95`: `PASS`
- [x] `byte_lm_mirror_strictly_beats_small_cpu_transformer_ci95`: `PASS`
- [x] `byte_lm_mirror_matches_or_beats_ppmc_order5`: `PASS`
- [x] `byte_lm_probability_normalization_and_transformer_grad_verified`: `PASS`
- [x] `cpu_budget_within_30s_per_seed`: `PASS`
