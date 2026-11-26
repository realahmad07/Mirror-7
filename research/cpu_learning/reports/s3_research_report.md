# Mirror 7 CPU-Native Learning Research — Stage S3 Controlled Contrasts, Guard Refinement & Versioned Rollback Report

- **Stage:** `S3` (Single-Factor Precondition Ablation, Repeated-Evidence Guard Acquisition, Commutator/Depth Audits & Append-Only `OperatorVersionStore`)
- **S3 Acceptance Gate:** `PASS` (25 / 25 checks passed)
- **Reproduce Command:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s3_experiments`
- **Schema Versions:** `OperatorCard=s2-operator-card-v1`, `VersionStore=s3-operator-version-store-v1`, `ContrastAudit=s3-contrast-audit-v1`
- **Fixed Seeds:** `[101, 202, 303, 404, 505]` (6 demos/action = 24 train episodes/seed; 184 unseen tasks/seed = 920 total across 10 unseen splits)
- **AST Isolation & Zero-Hardcoding Audit:** `PASS` (25 non-evaluator, 12 learner, 7 ITL files; 0 import violations, 0 hardcoded domain literals)
- **Environment:** Python `3.14.4` | `Windows-11-10.0.26200-SP0` | CPUs `8` | RAM `15.77 GB` | Git `17d1ae169164` (`research/cpu-native-learning-s0`)

> **Research Integrity Notice:** All Stage S3 contrast probes interact with the simulator strictly through the public black-box `TransitionEnvironment.step(state, action, mask) -> (after_state, sparse_delta)` protocol (0 imports of `hidden_actions.py`/`oracle.py`, 0 hardcoded domain literals in `itl/`, 0 test-time episode reads, 0 post-freeze environment queries).

## 1. Stage S3 Acceptance Gate Verification (Lower 95% CI Bounds Across 5 Fixed Seeds)

| Check | Status |
|---|:---:|
| `five_distinct_fixed_seeds_present` | PASS |
| `demos_per_action_le_8` | PASS |
| `ast_strict_learner_isolation_and_zero_hardcoded_literals` | PASS |
| `s0_s1_s2_committed_manifests_intact` | PASS |
| `all_solvers_frozen_before_temporal` | PASS |
| `s3_zero_post_freeze_env_queries_and_zero_test_retrieval` | PASS |
| `s3_operator_recovery_precision_ci95_low_ge_99pct` | PASS |
| `s3_operator_recovery_recall_ci95_low_ge_99pct` | PASS |
| `s3_precision_strictly_beats_s2_passive` | PASS |
| `s3_beats_s2_passive_on_inaccessible_source_by_ge_80pp` | PASS |
| `s3_depth1_and_depth2_exact_and_intermediate_ci95_low_ge_95pct` | PASS |
| `s3_depth4_exact_and_intermediate_ci95_low_ge_95pct` | PASS |
| `s3_depth6_exact_and_intermediate_ci95_low_ge_90pct` | PASS |
| `s3_role_swaps_exact_and_intermediate_ci95_low_ge_95pct` | PASS |
| `s3_guard_contrasts_exact_and_intermediate_ci95_low_ge_95pct` | PASS |
| `s3_repeated_idempotent_exact_and_intermediate_ci95_low_ge_95pct` | PASS |
| `s3_order_reversal_exact_and_intermediate_ci95_low_ge_95pct` | PASS |
| `s3_temporal_frozen_exact_and_intermediate_ci95_low_ge_95pct` | PASS |
| `s3_collateral_rate_ci95_high_lt_2pct` | PASS |
| `s3_underconstrained_repair_ci95_low_ge_95pct` | PASS |
| `s3_one_shot_guard_stays_shadow_all_seeds` | PASS |
| `s3_version_store_promote_rollback_non_deletion_all_seeds` | PASS |
| `s3_contrast_audit_probes_all_passed_all_seeds` | PASS |
| `s3_beats_baselines_and_mdl_compresses` | PASS |
| `cpu_budget_within_limit` | PASS |

## 2. Five-Seed Head-to-Head Matrix (10 Unseen Splits, 920 Unseen Tasks Total)

| Solver | Category | 10-Split Unseen Exact (Mean ± 95% CI) | Inacc-Source Contrast | Depth-4 Exact / Inter | Depth-6 Exact / Inter | Temporal Frozen | Op-Rec Prec / Rec / F1 | Collateral | Preds/s (1-core) | Model MDL (bits) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `itl_s3_contrast_refined` | `s3_contrast_refined_learner` | 100.00% ± 0.00% | 100.00% | 100.0% / 100.0% | 100.0% / 100.0% | 100.00% | 100.00% / 100.00% / 100.00% | 0.000% | 4380.5 | 4,360 |
| `itl_s2_passive_no_contrast_probes` | `s2_passive_baseline` | 91.30% ± 0.00% | 0.00% | 100.0% / 100.0% | 100.0% / 100.0% | 100.00% | 97.62% / 100.00% / 98.80% | 0.000% | 4399.9 | 4,440 |
| `itl_s3_underconstrained_repaired` | `s3_guard_repair_learner` | 100.00% ± 0.00% | 100.00% | 100.0% / 100.0% | 100.0% / 100.0% | 100.00% | 100.00% / 100.00% / 100.00% | 0.000% | 4408.3 | 4,360 |
| `episode_lookup` | `memory_baseline` | 0.00% ± 0.00% | 0.00% | 0.0% / 0.0% | 0.0% / 0.0% | 0.00% | 0.00% / 0.00% / 0.00% | 0.000% | 10722.5 | 11,581 |
| `shard_hdc_retrieval_no_itl` | `shard_baseline` | 3.26% ± 0.67% | 0.00% | 0.0% / 0.0% | 0.0% / 0.0% | 0.00% | 0.00% / 0.00% / 0.00% | 4.395% | 1454.2 | 249,262 |

## 3. Per-Split Exact & Every-Step Intermediate Accuracy (`itl_s3_contrast_refined` vs `itl_s2_passive_no_contrast_probes` & `itl_s3_underconstrained_repaired`)

| Unseen Split | S3 Refined Exact | S3 Refined Inter | S2 Passive Exact | S3 Guard-Repaired Exact | `shard_hdc_retrieval_no_itl` | `episode_lookup` |
|---|---:|---:|---:|---:|---:|---:|
| `heldout_unseen_names` | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_role_swaps` | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_guard_contrasts` | 100.00% | 100.00% | 100.00% | 100.00% | 30.00% | 0.00% |
| `heldout_repeated_idempotent` | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_order_reversal` | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_comp_depth2` | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_comp_depth4` | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_comp_depth6` | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_temporal_frozen` | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| `heldout_inaccessible_source_contrast` | 100.00% | 100.00% | 0.00% | 100.00% | 0.00% | 0.00% |

## 4. Stage S3 Mechanistic Audits Across All 5 Seeds

- **Single-Factor Precondition Ablation (Refuting Incidental Invariants):** Across all 5 seeds, `itl_s3_contrast_refined` runs `32.0` black-box probes during `fit()` (`0` probes during test or after `.freeze()`), refutes the 1 uncontrasted incidental precondition `(?source, accessible, true)` on `move` across `4` distinct base worlds (`0` contradictions), promotes `op_move_v2` (`parent_version=1`) while retiring `op_move_v1` without deletion, and raises operator recovery precision from **`97.62%` -> `100.00%`** (`41/41` gold clauses, `0` FP, `0` FN) and `heldout_inaccessible_source_contrast` accuracy from **`0.00%` -> `100.00%` (+100.00pp)**.
- **Repeated-Evidence Guard Acquisition (Stripped -> Repaired):** Intentionally stripped under-constrained cards start at `63.41%` clause recall (`77.61%` F1) and `60.00%` accuracy on `heldout_guard_contrasts`; after Stage S3 repeated-contrast guard acquisition (`min_guard_support=2`), `itl_s3_underconstrained_repaired` recovers **`100.00%` recall (`100.00%` F1)** and **`100.00%`** 10-split unseen exact accuracy (`0.000%` collateral).
- **1-Shot Anomaly Shadow Gate:** Verified on all `5/5` seeds that a single over-generalization failure (and even duplicating the identical failure twice) registers strictly as `shadow` and NEVER promotes (`active_version` remains `1`), whereas `>= 2` independent failures promote cleanly while preserving prior versions intact.
- **Append-Only `OperatorVersionStore` & Rollback Audit:** Verified on all `5/5` seeds: `v1` retired without deletion on `v2` promotion, bit-exact JSON round-trip, frozen store mutation blocking, non-destructive `rollback(move, 1)` reverting inaccessible-source behavior, and non-destructive `rollback(move, 2)` restoring refined behavior.
- **Independent Contrast Probe Audit (`ContrastAuditReport`):** Passed on `5/5` seeds (`100.00%` exact & every-step intermediate accuracy across role-swap probes, `6.0` non-commuting `(a;b) != (b;a)` commutator pairs/seed, repeated/idempotent `(a, a)` probes, and depth-1/2/4/6 stratified postcondition verification).

## 5. Next Stage Decision

- **S3 Gate:** PASS — All 25 Stage S3 gates passed across 5 fixed seeds. Ready for `@auditor` independent verification before advancing to Stage S4.
