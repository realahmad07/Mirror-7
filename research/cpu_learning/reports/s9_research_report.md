# Stage S9 Research Report — Autonomous Learning Controller & Self-Curriculum

- **Stage Gate:** `PASS` (`21/21` checks passed)
- **Reproduce:** `PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s9_experiments`
- **Seeds:** `[101, 202, 303, 404, 505]` (`mean = 4.155s/seed`, `max = 4.712s/seed`)
- **Shared Active Query Cap:** `42` queries (`Autonomous` spent `31.0` & stopped early on verified frontiers; `RandomCurriculum` spent `42.0`; `PassiveStream` spent `0.0`)

## 1. Held-Out Multi-Frontier Accuracy Under Identical Query & CPU Budgets

| Metric / Frontier | Cold-Start Seed | `PassiveStreamController` | `RandomCurriculumController` | Mirror `AutonomousMetaController` (`mean [95% CI]`) |
|---|---:|---:|---:|---:|
| **Relational World (All Unseen Splits)** | `66.46%` | `73.29%` | `77.14%` | **`98.76% [98.76%, 98.76%]`** |
| — Compositional Depth 4 | `63.64%` | `81.82%` | `81.82%` | `100.00%` |
| — Compositional Depth 6 | `45.45%` | `45.45%` | `52.73%` | `100.00%` |
| **Raw UTF-8 Ambiguous Grounding** | `0.00%` | `0.00%` | `0.00%` | **`100.00% [100.00%, 100.00%]`** |
| **Functional Math & List Synthesis** | `35.42%` | `35.42%` | `38.75%` | **`100.00% [100.00%, 100.00%]`** |
| **Overall 3-Frontier Accuracy** | `33.96%` | `36.24%` | `38.63%` | **`99.59% [99.59%, 99.59%]`** |
| **Anchor Old-Skill Retention (Arcane + AST)** | `100.00%` | `100.00%` | `100.00%` | **`100.00%` (`0` hashes rewritten)** |

### Paired Lifts (`AutonomousMetaController` vs Baselines)
- **Lift over Cold-Start:** `+65.63pp [+64.41pp, +66.85pp]`
- **Lift over `PassiveStreamController`:** `+63.35pp [+62.13pp, +64.57pp]`
- **Lift over `RandomCurriculumController` (same `42`-query cap):** `+60.96pp [+55.83pp, +66.07pp]`

## 2. Interactive Non-Starvation SLA & Per-Subsystem CPU Accounting

- **Live Interactive Queries Served Concurrently:** `120` per seed (`600` total)
- **Starved / Dropped Interactive Queries:** `0` starved, `0` dropped (SLA `<= 5.0 ms`)
- **Interactive p95 Latency:** `1.3903 ms` (max `1.9008 ms`) vs Idle Baseline `1.1159 ms`
- **Mean Per-Seed Learning Subsystem CPU Breakdown:** `query_selection = 345.83 ms`, `verification = 6.17 ms`, `induction = 72.75 ms`, `replay = 199.89 ms`, `consolidation = 7.94 ms`, `retrieval = 11.82 ms`
- **MDL Library Compression:** `17251` library bits vs `481189` raw stream bits (`ratio = 0.0359`)

## 3. Stage S9 Gate Checklist

- [x] `ast_isolation_and_zero_hardcoded_literals`: `PASS`
- [x] `prior_s0_through_s8_manifests_intact`: `PASS`
- [x] `s9_manifest_verified`: `PASS`
- [x] `relational_frontier_final_acc_ge_095`: `PASS`
- [x] `relational_depth4_and_depth6_acc_ge_090`: `PASS`
- [x] `utf8_grounding_frontier_final_acc_ge_095`: `PASS`
- [x] `functional_math_list_frontier_final_acc_ge_095`: `PASS`
- [x] `overall_frontier_lift_over_cold_start_ge_035`: `PASS`
- [x] `beats_passive_stream_by_ge_15pp_non_overlapping_ci`: `PASS`
- [x] `beats_random_curriculum_by_ge_15pp_non_overlapping_ci`: `PASS`
- [x] `strictly_lower_queries_to_target_than_random`: `PASS`
- [x] `stops_querying_verified_frontiers_early`: `PASS`
- [x] `anchor_skill_retention_ge_095_zero_forgetting`: `PASS`
- [x] `faulty_candidate_caught_and_quarantined`: `PASS`
- [x] `zero_self_certified_promotions`: `PASS`
- [x] `freeze_enforced_zero_post_freeze_queries`: `PASS`
- [x] `collateral_damage_below_002`: `PASS`
- [x] `interactive_non_starvation_sla_p95_le_5ms`: `PASS`
- [x] `learning_slices_within_tick_budget`: `PASS`
- [x] `closed_loop_11_steps_and_mdl_consolidation_verified`: `PASS`
- [x] `cpu_budget_within_15s_per_seed`: `PASS`
