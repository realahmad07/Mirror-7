# Stage S8 Research Report — Continual Learning, Regression Replay & Cross-Domain Transfer

- **Stage Gate Verdict:** `PASS` (`19/19` checks passed)
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s8_experiments`
- **Seeds:** `[101, 202, 303, 404, 505]`
- **Mean Per-Seed CPU Runtime:** `5.49s` (total `31.81s`)

## 1. Final Phase-5 Continual Retention Across Chronological Domains (Phases 1–4)

| Domain | Mirror `ContinualKnowledgeBase` (`mean [95% CI]`) | `NaiveOverwriteNoReplay` | `SmallCPUNeuralContinualRef` (`comparison_only`) |
|---|---:|---:|---:|
| `phase1_relational` | `100.00% [100.00%, 100.00%]` | `21.43%` | `11.88%` |
| `phase2_math_and_lists` | `100.00% [100.00%, 100.00%]` | `0.00%` | `3.33%` |
| `phase3_code_and_ast` | `97.73% [97.73%, 97.73%]` | `0.00%` | `0.00%` |
| `phase4_utf8_grounding` | `100.00% [100.00%, 100.00%]` | `31.67%` | `21.00%` |
| **Mean Retention (Phases 1–4)** | **`99.43% [99.43%, 99.43%]`** | `13.27%` | `9.05%` |

## 2. Mandatory Regression Replay, Poison Quarantine & Non-Destructive Hash Audit

- **Prior Verified Skill Hashes Rewritten:** `0` across all 5 seeds (`non_destructive_verified = True`)
- **Injected Poisoned Updates Caught, Rolled Back & Quarantined (Relational + List + UTF-8):** `100.00%` (`3/3` per seed, `0.000%` post-rollback regression)

## 3. Cross-Domain Positive Transfer (D3 Logistics & D4 Biolab) & Scoped Negative-Transfer Rejection

- **Source Abstraction Bit Savings (S0 + Arcane):** Domain 1 saved `1080` bits, Domain 2 saved `1904` bits (`2` universal schemas promoted; `3` non-isomorphic schemas scoped to `domain_local`)

| Target Domain | Cold-Start ITL | Mirror Scoped Transfer | Lift (`mean [95% CI]`) | Negative Transfer | Ungated Forced Ablation | Marginal Bit Savings |
|---|---:|---:|---:|---:|---:|---:|
| `domain3_logistics` | `63.64%` | `100.00%` | `+36.36pp [+36.36pp, +36.36pp]` | `0.000%` | `100.00%` | `+1872` bits |
| `domain4_biolab` | `63.64%` | `100.00%` | `+36.36pp [+36.36pp, +36.36pp]` | `0.000%` | `100.00%` | `+1872` bits |
| `domain5_incompatible_reactor` | `100.00%` | `100.00%` | `+0.00pp [+0.00pp, +0.00pp]` | `0.000%` | `50.00%` | `+0` bits |

## 4. Sub-Linear Library Growth Across Phases 1–5

| Phase | Active Skills | Cumulative Raw Train Bits | Cumulative Compressed Train Bits | Active Library Bits | Replay Ledger Bits | Library / Raw Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `Phase 1: Relational World` | `4` | `704,317` | `52,446` | `4,992` | `4,579` | `0.0071` |
| `Phase 2: Mathematics & Lists` | `12` | `1,045,526` | `75,042` | `8,344` | `4,886` | `0.0080` |
| `Phase 3: Python AST & Code` | `35` | `1,085,718` | `87,179` | `18,360` | `5,030` | `0.0169` |
| `Phase 4: Raw UTF-8 Grounding` | `58` | `1,284,419` | `98,853` | `24,872` | `5,240` | `0.0194` |
| `Phase 5: Cross-Domain & Poison Rollbacks` | `64` | `1,657,507` | `129,570` | `29,960` | `5,619` | `0.0181` |

## 5. Stage S8 Gate Checklist

- [x] `ast_isolation_and_zero_hardcoded_literals`: `PASS`
- [x] `comparison_only_neural_continual_ref_isolated`: `PASS`
- [x] `prior_s0_through_s7_manifests_intact`: `PASS`
- [x] `s8_manifest_verified`: `PASS`
- [x] `phase1_relational_retention_ci95_low_ge_095`: `PASS`
- [x] `phase2_math_and_lists_retention_ci95_low_ge_095`: `PASS`
- [x] `phase3_code_and_ast_retention_ci95_low_ge_095`: `PASS`
- [x] `phase4_utf8_grounding_retention_ci95_low_ge_095`: `PASS`
- [x] `zero_forgetting_and_zero_rewritten_prior_skills`: `PASS`
- [x] `baselines_genuinely_learn_each_phase_before_forgetting`: `PASS`
- [x] `mirror_strictly_beats_naive_overwrite_ci95_by_25pp`: `PASS`
- [x] `mirror_strictly_beats_neural_continual_ref_ci95_by_25pp`: `PASS`
- [x] `poisoned_updates_100pct_caught_quarantined_and_rolled_back`: `PASS`
- [x] `cross_domain_source_bit_savings_positive`: `PASS`
- [x] `domain3_logistics_2shot_transfer_ci95_low_ge_095_and_lift_ge_20pp`: `PASS`
- [x] `domain4_biolab_2shot_transfer_ci95_low_ge_095_and_lift_ge_20pp`: `PASS`
- [x] `scoped_domain_local_rejection_100pct_and_zero_negative_transfer`: `PASS`
- [x] `sublinear_library_growth_verified`: `PASS`
- [x] `cpu_budget_within_15s_per_seed`: `PASS`
