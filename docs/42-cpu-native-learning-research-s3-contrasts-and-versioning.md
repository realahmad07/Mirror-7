# 42 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S3 Controlled Contrasts, Guard Refinement & Versioned Rollback

This document records the Stage **S3** implementation, 5-seed head-to-head benchmark results, single-factor precondition ablation, repeated-evidence guard acquisition, commutator/depth postcondition audits, append-only `OperatorVersionStore` lifecycle verification, and adversarial audit findings for Mirror 7's CPU-native **Controlled Contrasts & Versioned Refinement** subsystem (`research/cpu_learning/itl/`).

> **Research Integrity Notice:** All Stage S3 contrast probes interact with the environment strictly through the public black-box `TransitionEnvironment.step(state, action, mask) -> (after_state, sparse_delta)` protocol (exposed via `BlackBoxSimulatorEnvironment` / `make_blackbox_environment()` in `research/cpu_learning/evaluation/run_experiments.py`). Automated AST auditing certifies **0 import violations** across `25` non-evaluator files (`12` learner files, `7` ITL files) and **0 hardcoded action or predicate literals** anywhere inside `research/cpu_learning/itl/`. Inference executes strictly from promoted `OperatorCard`s with **0 test-time episode reads** and **0 test-time or post-freeze environment queries**. Stages **S4–S10** (SHARD search guidance, RDC dream-refutation/consolidation, math/code transfer, language grounding, continual learning, autonomy, and broad external comparisons) are NOT claimed here.

---

## 1. Stage S3 Modules Implemented (`research/cpu_learning/`)

- `research/cpu_learning/itl/version_store.py` — Append-only `OperatorVersionStore` (`s3-operator-version-store-v1`) and immutable `OperatorVersionRecord` tracking `version`, `parent_version`, lifecycle states (`shadow`, `promoted`, `retired`), provenance/refutation evidence, contrast pass rates, non-destructive `promote()` (retiring prior active versions without deletion), non-destructive `rollback()` (strictly blocking unpromoted `shadow` versions), `.freeze()` mutation lock, and bit-exact `to_dict()` / `from_dict()` round-trip serialization.
- `research/cpu_learning/itl/contrast_harness.py` — Domain-agnostic active contrast engine (`s3-contrast-audit-v1`):
  - Public `TransitionEnvironment` protocol and black-box single-factor precondition ablation (`refine_operator_by_single_factor_ablation`), testing individual candidate clauses on minimally perturbed worlds and running joint verification across both `required_fact` and `attribute_guard` candidates plus joint negative-observation consistency checks.
  - Repeated-evidence guard acquisition requiring `>= min_guard_support` (`default=2`) distinct independent over-generalization failures before promotion; 1-shot anomalies and duplicated identical failures are strictly quarantined as `shadow`.
  - Independent `ContrastAuditReport` probes covering role swaps, non-commuting action-order `(a;b)` vs `(b;a)` commutator pairs, repeated/idempotent `(a, a)` sequences, and depth-stratified (`1, 2, 4, 6`) every-step intermediate postcondition verification.
  - `ContrastRefinedITLLearner(ITLOperatorLearner)` (S2 ITL + S3 `OperatorVersionStore` + S3 single-factor ablation with `.freeze()` blocking post-freeze environment queries) and `UnderconstrainedContrastRepairLearner` (evaluating recovery from intentionally stripped candidate cards missing guards).
- `research/cpu_learning/evaluation/run_experiments.py` — Added `BlackBoxSimulatorEnvironment` and `make_blackbox_environment()` so Stage S3 runner and tests interact with the simulator purely through the black-box step interface without expanding the evaluator-only import allowlist or altering committed S0 manifests.
- `research/cpu_learning/evaluation/run_s3_experiments.py` — Reproducible 5-seed benchmark runner (`0` direct imports of `hidden_actions.py` / `oracle.py`), verifying all **25 Stage S3 acceptance gates** on Student-$t$ 95% CI bounds, bit-exact S0/S1/S2/S3 manifest checks (`--verify-manifests`), AST isolation + zero-hardcoding audits, and evaluating `10` unseen splits (`9` S0 splits + `heldout_inaccessible_source_contrast` using `HELDOUT_POOL` = `184` unseen tasks/seed = `920` total) with explicit every-step intermediate postcondition accuracy.
- `research/cpu_learning/tests/test_s3_contrasts_and_versioning.py` — Independent adversarial pytest suite (`8` test suites; `29 / 29` total tests passing across `test_s0`..`test_s3`) covering AST isolation, expanded domain-literal scanning, black-box environment freeze isolation, `OperatorVersionStore` lifecycle/shadow-rollback rejection/non-deletion/corruption guards, disjunctive joint-ablation consistency, 1-shot and duplicate shadow gating, 100% novel synthetic arcane domain ablation/repair/`ContrastAuditReport`, on-disk `EpisodeLedger` zero test-time reads & zero post-freeze environment queries, `heldout_inaccessible_source_contrast` disjointness/byte-spans, and committed + fresh isolated gate verification.

---

## 2. Independent Adversarial Audit Findings Resolved Before Sign-Off

Two genuine lifecycle and ablation edge cases surfaced during independent adversarial testing and were fixed and regression-tested before Stage S3 sign-off:
1. **Disjunctive Joint-Ablation Report & Evidence Consistency (`contrast_harness.py`):** On a synthetic domain with a disjunctive precondition (e.g., `(?orb, glow, on) OR (?orb, hum, on)` or mixed fact + attribute disjunction), each clause passed single-factor ablation individually while joint ablation correctly detected contradiction and retained the dependent clause. Previously, `AblationClauseReport` records were not reconciled after joint ablation and joint verification omitted `attribute_guard` combinations and combined `neg_obs` checks. Joint ablation now unifies both `required_fact` and `attribute_guard` candidates with combined `neg_obs` verification and updates rejected `AblationClauseReport` entries in-place (`refuted=False`, incremented `contradiction_count`, explicit retention reason), ensuring `version_store` `evidence["refuted_clauses"]` records strictly surviving refuted clauses.
2. **Shadow-Gate Bypass Prevention in `OperatorVersionStore.rollback()` (`version_store.py`):** Previously, `rollback(action_id, target_version)` allowed rolling back directly onto an unpromoted `shadow` record, bypassing `promote()`. `rollback()` now raises `VersionStoreError` whenever the target version has status `"shadow"`, permitting rollback strictly to previously promoted (`"retired"`) or already `"promoted"` versions.

---

## 3. Verified 5-Seed Stage S3 Results (`[101, 202, 303, 404, 505]`, `25 / 25` Gate Checks PASS)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s3_experiments --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s3_experiments
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s3_research_report.md`](../research/cpu_learning/reports/s3_research_report.md) · [`research/cpu_learning/reports/s3_summary.json`](../research/cpu_learning/reports/s3_summary.json) · [`research/cpu_learning/reports/raw_runs/s3_per_split.csv`](../research/cpu_learning/reports/raw_runs/s3_per_split.csv) · [`research/cpu_learning/reports/raw_runs/s3_seed_101.json`](../research/cpu_learning/reports/raw_runs/s3_seed_101.json) · [`research/cpu_learning/data/s3_preregistration.json`](../research/cpu_learning/data/s3_preregistration.json)

### Five-Seed Head-to-Head Matrix (10 Unseen Splits, 184 Unseen Tasks/Seed = 920 Total)

| Solver | Category | 10-Split Unseen Exact (Mean ± 95% CI) | Inacc-Source Contrast | Depth-4 Exact / Inter | Depth-6 Exact / Inter | Temporal Frozen | Op-Rec Prec / Rec / F1 | Collateral | Preds/s (1-core) | Peak Heap (KB) | Model MDL (bits) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `itl_s3_contrast_refined` | `s3_contrast_refined_learner` | **100.00% ± 0.00%** | **100.00%** | **100.0% / 100.0%** | **100.0% / 100.0%** | **100.00%** | **100.00% / 100.00% / 100.00%** | **0.000%** | **4,380.5** | **931.1** | **4,360** |
| `itl_s2_passive_no_contrast_probes` | `s2_passive_baseline` | 91.30% ± 0.00% | 0.00% | 100.0% / 100.0% | 100.0% / 100.0% | 100.00% | 97.62% / 100.00% / 98.80% | 0.000% | 4,399.9 | 907.1 | 4,440 |
| `itl_s3_underconstrained_repaired` | `s3_guard_repair_learner` | **100.00% ± 0.00%** | **100.00%** | **100.0% / 100.0%** | **100.0% / 100.0%** | **100.00%** | **100.00% / 100.00% / 100.00%** | **0.000%** | **4,408.3** | **1,385.6** | **4,360** |
| `episode_lookup` | `memory_baseline` | 0.00% ± 0.00% | 0.00% | 0.0% / 0.0% | 0.0% / 0.0% | 0.00% | 0.00% / 0.00% / 0.00% | 0.000% | 10,722.5 | 19.6 | 11,581 |
| `shard_hdc_retrieval_no_itl` | `shard_baseline` | 3.26% ± 0.67% | 0.00% | 0.0% / 0.0% | 0.0% / 0.0% | 0.00% | 0.00% / 0.00% / 0.00% | 4.395% | 1,454.2 | 75.3 | 249,262 |

### Key Verification Takeaways
- **Single-Factor Ablation Eliminates S2's Incidental Precondition (`+100.00pp` on Inaccessible-Source Contrast):** Using `32.0` black-box environment probes during `fit()` (`0` probes during test or after `.freeze()`, `0.143s` fit / `0.042s` predict per seed), `itl_s3_contrast_refined` refutes the single uncontrasted incidental clause `(?source, accessible, true)` on `move` across `4` distinct base worlds (`0` contradictions), promoting `op_move_v2` (`parent_version=1`) while retiring `op_move_v1` without deletion. Operator recovery precision rises from **`97.62%` -> `100.00%`** (`41 / 41` gold clauses, `0` FP, `0` FN) and `heldout_inaccessible_source_contrast` accuracy jumps from **`0.00%` -> `100.00%` (`+100.00pp`)**, while MDL model size tightens from `4,440` -> **`4,360` bits** (**`161.5×`** compression vs `704,301` raw training bits; **`57.2×`** smaller than `shard_hdc_retrieval_no_itl`; `0` inference episode-store bits).
- **Repeated-Evidence Guard Acquisition & 1-Shot Shadow Gate (`5 / 5` Seeds):** Intentionally stripped under-constrained cards start at `63.41%` clause recall (`77.61%` F1) and `60.00%` accuracy on `heldout_guard_contrasts`; after Stage S3 repeated-contrast guard acquisition (`min_guard_support=2`), `itl_s3_underconstrained_repaired` recovers **`100.00%` precision / `100.00%` recall (`100.00%` F1)** and **`100.00%`** 10-split unseen exact accuracy (`0.000%` collateral). Across all `5 / 5` seeds, both a single failure and duplicating the identical failure twice remain strictly `shadow` (`active_version=1`) until `>= 2` independent failures trigger promotion.
- **Append-Only `OperatorVersionStore` & `ContrastAuditReport` (`5 / 5` Seeds):** Non-destructive promotion/retirement, bit-exact JSON round-trip, frozen mutation blocking, non-destructive rollback (`v2 -> v1 -> v2`), and independent role-swap, commutator (`6.0` non-commuting `(a;b) != (b;a)` pairs/seed), repeated/idempotent `(a, a)`, and Depth `1, 2, 4, 6` every-step intermediate postcondition audits (`100.00%` exact & intermediate across all 10 unseen splits) pass on all `5 / 5` seeds.

---

## 4. Remaining Limitations & Next Stage (S4 — SHARD Search Guidance & Candidate Proposals)

### Remaining Limitations After S3
1. **No SHARD Candidate Proposal Acceleration Yet:** Stage S3 refines exact relational operators directly over small action/role spaces; scaling role binding and primitive/clause candidate proposals over larger hypothesis spaces via CPU-native SHARD HDC/LSH structures (while preserving the hard rule that similarity proposes and never certifies) belongs to Stage **S4**.
2. **No Full RDC Dream-Refutation / Blame / MDL Consolidation Controller Yet:** Leave-one-operator-out multi-step plan blame, maximum-disagreement query selection, exception splitting across multi-branch operators, and unified description-length consolidation (`J = L(library) + L(examples|library) + beta*verified_failures + gamma*collateral_changes`) belong to Stage **S5**.
3. **Domain Scope:** Validated on the deterministic relational world (`move`, `give`, `open`, `lock`) and an independent synthetic arcane world (`channel`, `bestow`, `unward`, plus disjunctive synthetic preconditions); math/code transfer (including the discontinuous `(a + b) % 10` canary), language grounding, and continual learning remain deferred to **S6–S10**.

### Next Stage: Stage S4 — SHARD Search Guidance & Candidate Proposals
- **Goal:** Use CPU-native SHARD bit/HDC indexing strictly as a fast candidate proposer for episodic retrieval, role-binding hypotheses, and candidate operator/clause search inside ITL — proving wall-clock / search-branching speedup via ablation while preserving `100%` verification discipline (similarity proposes, independent transition verification decides).
- **Acceptance Criteria:** Demonstrate measurable search/proposal speedup or larger-space scaling over brute-force unguided ITL without regressing any S0–S3 accuracy, precision/recall, collateral, isolation, or manifest gates across all 5 fixed seeds.
