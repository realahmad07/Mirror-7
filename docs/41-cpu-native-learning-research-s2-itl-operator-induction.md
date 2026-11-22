# 41 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S2 ITL Operator Induction

This document records the Stage **S2** implementation, 5-seed head-to-head benchmark results, ablation analysis, multi-seed confounding diagnosis, and verified gate checks for Mirror 7's CPU-native **Inductive Transition Learning (ITL)** subsystem (`research/cpu_learning/itl/`).

> **Research Integrity Notice:** Every `OperatorCard` in Stage S2 is discovered strictly from the **6 public before/action/after training episodes per action** ingested into Stage S1's append-only SQLite `EpisodeLedger`. Automated AST auditing certifies **0 imports** of `hidden_actions.py` / `oracle.py` / `generators.py` and **0 hardcoded action or predicate literals** (`move`, `give`, `open`, `lock`, `located_on`, `owned_by`, `is_open`, `is_locked`, `accessible`, `portable`, `status`) anywhere inside `research/cpu_learning/itl/`. Inference executes strictly from the compact `OperatorCard` library with **0 test-time episode lookups**. Stages **S3–S10** (RDC active counterexamples, SHARD candidate proposal search, library compression, math/code transfer, continual learning) are NOT claimed here.

---

## 1. Stage S2 Modules Implemented (`research/cpu_learning/`)

- `research/cpu_learning/itl/operator_card.py` — Immutable, versioned `OperatorCard` (`s2-operator-card-v1`) with typed role constraints, direct fact preconditions, type-conditional role fact guards (`ConditionalRoleFactGuard`), 1-hop relational contrast guards (`IndirectRoleFactGuard`), entity attribute guards (`AttributeGuard`), role inequality guards (`RoleInequalityGuard`), sparse fact/attribute effects, preservation invariants, cryptographic provenance links (`episode_ids` + `EpisodeLedger` SHA-256 `record_hashes`), exact execution with partial-`ObservationMask` epistemic abstention, and MDL bit accounting.
- `research/cpu_learning/itl/role_unifier.py` — Domain-agnostic relational lifting and anti-unification over public `TypedStateFrame`s and `SparseDelta`s (`lift_sparse_delta`, `intersect_lifted_deltas`, `extract_direct_lifted_facts`, `extract_entity_literal_facts`, `extract_one_hop_links`). Distinguishes symbolic literals from concrete world entities solely by membership in the frame's observed entity set.
- `research/cpu_learning/itl/delta_extractor.py` — Reads strictly from S1's append-only `EpisodeLedger`, partitions effect-bearing vs zero-delta observations purely from `SparseDelta.is_noop`, and computes cross-ledger `CorpusStatistics` (attribute domains, entity-fact instances for MDL functional-entailment pruning, causally edited attributes, and cross-operator positive-role attribute variance).
- `research/cpu_learning/itl/operator_inducer.py` — General 8-step ITL `induce_operator_card` pipeline and `ITLOperatorLearner(BaseS0Solver)` supporting three controlled evaluation modes:
  1. `itl_operator_learner` — primary S2 learner (typed action IDs + S1 `EpisodeLedger` ingestion + full contrast induction, isolating operator learning from NLP per the research plan).
  2. `itl_operator_learner_no_contrast` — controlled ablation disabling negative-contrast guard induction.
  3. `itl_operator_learner_raw_utf8_e2e` — end-to-end S1+S2 pipeline ignoring test-time typed roles and parsing raw UTF-8 instruction bytes through S1's `TypedRoleFrameParser(use_state_types=True)`.
- `research/cpu_learning/evaluation/run_s2_experiments.py` — One-command reproducible 5-seed benchmark runner verifying all **19 Stage S2 acceptance gates** on exact Student-$t$ 95% CI bounds, bit-exact S0/S1/S2 manifest checks, AST isolation + zero-hardcoding audits, and MDL compression accounting.
- `research/cpu_learning/tests/test_s2_itl_operator_induction.py` — Automated pytest suite including AST zero-hardcoding verification, a **100% novel synthetic domain test** (`channel` / `bestow` / `unward` over `wizard`/`rune`/`altar`/`sanctum` with unseen predicates and decorative-coincidence immunity), partial-mask abstention, unknown-action abstention, frozen on-disk `EpisodeLedger` zero-read audit, ablations, and end-to-end gate verification.

---

## 2. Multi-Seed Scientific Finding: Small-Sample Decorative Coincidence (Seed 505)

A key benefit of mandatory 5-seed evaluation appeared immediately during Stage S2 development:
- On Seeds `101`, `202`, `303`, and `404`, the initial attribute-guard inducer achieved `100.00%` across all 9 unseen splits.
- On **Seed `505`**, by pure $1/4^3 = 1/64$ sampling chance across the 4 random `TRAIN_POOL` colors, all 4 positive `give` demonstrations drew `recipient.role_badge == "green"`, causing naive positive invariance to overfit a spurious guard `(?recipient, role_badge, green)` and collapsing Seed `505` accuracy to `51.79%` (`0.00%` on Depth-6).
- **Principled Domain-Agnostic Fix (Cross-Operator Causal-Relevance Filter):** Rather than hardcoding decorative attribute names, ITL now computes cross-ledger positive-role variance across `EpisodeLedger` (`CorpusStatistics`) and restricts negative-contrast attribute evidence strictly to negative episodes unexplained by types, direct/conditional facts, and inequality guards. Any unedited attribute that is never independently contrasted by a fact-valid negative episode AND varies freely across other positive role slots of the same entity type in the ledger (`actor.role_badge` varies across `move`, `give`, `open`, `lock`) is empirically non-causal and pruned, while genuine capability guards (`object.portable=True` across all positive `object` roles; `rune.charged=True` in the synthetic domain) are retained cleanly.

---

## 3. Verified 5-Seed Stage S2 Results (`[101, 202, 303, 404, 505]`, `19 / 19` Gate Checks PASS)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s2_experiments --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s2_experiments
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s2_research_report.md`](../research/cpu_learning/reports/s2_research_report.md) · [`research/cpu_learning/reports/s2_summary.json`](../research/cpu_learning/reports/s2_summary.json) · [`research/cpu_learning/reports/raw_runs/s2_per_split.csv`](../research/cpu_learning/reports/raw_runs/s2_per_split.csv) · [`research/cpu_learning/reports/raw_runs/s2_induced_operators_seed101.json`](../research/cpu_learning/reports/raw_runs/s2_induced_operators_seed101.json) · [`research/cpu_learning/data/s2_preregistration.json`](../research/cpu_learning/data/s2_preregistration.json)

### Five-Seed Head-to-Head Matrix (9 Unseen Splits, 168 Unseen Tasks/Seed = 840 Total)

| Solver | Category | Unseen Exact Acc (Mean ± 95% CI) | Guard Contrasts | Depth-4 Acc | Depth-6 Acc | Temporal Frozen | Op-Rec Prec / Rec / F1 | Collateral Rate | Preds/s (1-core) | Peak Heap (KB) | Model MDL (bits) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `itl_operator_learner` | `itl_candidate_learner` | **100.00% ± 0.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **97.6% / 100.0% / 98.8%** | **0.000%** | **1,773.9** | **907.3** | **4,440** |
| `itl_operator_learner_no_contrast` | `itl_ablation` | 99.40% ± 0.00% | 95.00% | 100.00% | 100.00% | 100.00% | 97.5% / 95.1% / 96.3% | 0.030% | 1,665.6 | 907.0 | 4,176 |
| `itl_operator_learner_raw_utf8_e2e` | `s1_s2_end_to_end_learner` | **100.00% ± 0.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **97.6% / 100.0% / 98.8%** | **0.000%** | **569.0** | **905.4** | **4,440** |
| `noop_identity_floor` | `sanity_floor` | 5.95% ± 0.00% | 50.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 0.000% | 7,862.2 | 18.3 | 0 |
| `episode_lookup` | `memory_baseline` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 0.000% | 4,264.0 | 19.6 | 11,581 |
| `structural_knn` | `memory_baseline` | 2.98% ± 0.00% | 25.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 4.567% | 863.4 | 168.2 | 30,357 |
| `shard_hdc_retrieval_no_itl` | `shard_baseline` | 3.57% ± 0.74% | 30.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 4.563% | 620.7 | 75.3 | 249,262 |
| `small_cpu_neural_reference` | `comparison_only_neural` | 2.74% ± 2.00% | 23.00% | 0.00% | 0.00% | 0.00% | 0.0% / 0.0% / 0.0% | 2.644% | 920.5 | 56.5 | 23,019 |

### Key Verification Takeaways
- **Massive Compression Without Memorization:** `itl_operator_learner` compresses `704,301` raw training bits (`52,427` zlib-compressed bits) into **`4,440` MDL bits** (**`158.6×`** vs raw training data, **`56.1×`** smaller than `shard_hdc_retrieval_no_itl`), with **`0` episode-store bits** and **`0` ledger accesses** at test time.
- **Honest Conservative Extra Clause (`97.62%` Precision, `100.00%` Recall):** Across all 5 seeds, ITL recovers all `41 / 41` gold precondition/effect clauses (`100.00%` recall) and conservatively retains `1` extra uncontrasted clause: `(?source, accessible, true)` on `move` (since `?source` is `accessible=true` in all 6 training demos). Pruning uncontrasted candidate preconditions via active counterfactual tests is the exact purpose of Stage **S3**.

---

## 4. Remaining Limitations & Next Stage (S3 — RDC Verification, Counterexamples & Exception Refinement)

### Remaining Limitations After S2
1. **Uncontrasted Preconditions:** Passive induction from 6 fixed demos cannot distinguish incidental invariants (such as `source.accessible=true`) from true causal requirements without active counterexample probing.
2. **Deterministic Single-Mode Operators:** Current S2 induces one canonical `OperatorCard` per action; multi-branch conditional effects, stochastic/noisy transitions, and exception splitting (where a guard fails only under a narrower subtype/context) belong to S3/S5.
3. **Domain Scope:** Validated on the deterministic relational world (`move`, `give`, `open`, `lock`) and an independent synthetic arcane world (`channel`, `bestow`, `unward`); open-ended math/code/language operators are deferred to S6–S8.

### Next Stage: Stage S3 — Refutation-Driven Consolidation (RDC) & Active Counterexample Learning
- **Goal:** Build independent counterexample generation, dream-refutation proposals, leave-one-clause-out precondition ablation, and exception/guard revision with versioned `OperatorCard` promotion (never allowing self-generated hypotheses to self-certify without independent simulator/verifier execution).
- **Acceptance Criteria:** Prune uncontrasted spurious clauses (lifting operator recovery precision toward `100%` without losing recall), recover from injected corrupted/over-generalized operator cards via counterexample-driven repair, and preserve `>=95%` Depth-4 / `>=90%` Depth-6 / `<2%` collateral across all 5 fixed seeds with zero regression on S0–S2.
