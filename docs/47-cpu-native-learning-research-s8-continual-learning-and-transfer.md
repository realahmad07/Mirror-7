# 47 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S8 Continual Learning, Regression Replay & Cross-Domain Transfer

This document records the Stage **S8** implementation, 5-seed chronological 5-phase continual retention benchmarks, cross-domain regression replay & poisoned-update quarantine/rollback verification, positive 2-shot transfer onto **two held-out unseen target domains** (`Logistics` & `Biolab`) alongside scoped `domain_local` negative-transfer rejection (`Reactor`), sub-linear library growth accounting, and adversarial audit findings for Mirror 7's CPU-native **Continual Learning & Cross-Domain Transfer** subsystem (`research/cpu_learning/continual/`).

> **Research Integrity Notice:** Stage S8 runs sequentially through 5 heterogeneous chronological phases (Relational World -> Mathematics & Lists -> Python AST & HumanEval-Micro -> Raw UTF-8 Grounding -> Novel Domains & Poison Rollbacks) inside one unified `ContinualKnowledgeBase` — **ZERO GPU, ZERO pretrained LLMs, ZERO backpropagation in Mirror's learners, ZERO rewritten prior verified skill hashes (`rewritten_prior_skills == 0`), and 100% Python 3.11 stdlib compatibility**. Automated AST & lexical auditing across `57` non-evaluator modules (`34` learner, `3` continual) certifies **0 import violations**, **0 hardcoded S0/D2/D3/D4/D5 literals**, and strict isolation of `SmallCPUNeuralContinualRef` (`comparison_only == True`). Both continual baselines (`NaiveOverwriteNoReplayBaseline` and `SmallCPUNeuralContinualRef`) are verified by gate check to genuinely learn each phase first (`>= 95%` and `>= 75%` immediate post-phase accuracy) before subsequent phases induce forgetting. Stages **S9–S10** (bounded autonomy and broad external comparisons) are NOT claimed here.

---

## 1. Stage S8 Modules Implemented (`research/cpu_learning/`)

- `research/cpu_learning/continual/__init__.py` — Public Stage S8 continual learning and cross-domain transfer exports.
- `research/cpu_learning/continual/replay_manager.py` — `ContinualKnowledgeBase` and `RegressionReplayManager` (`s8-replay-manager-v1`):
  - Manages heterogeneous versioned skill registries across chronological domains (relational `OperatorVersionStore`, S6 math/list/code `FunctionalOperatorCard` skills, and S7 `RawByteConstructionLearner` constructions) with immutable canonical SHA-256 skill hashes (`rewritten_prior_skills == 0`) and `.freeze()` (`ContinualLearnerFrozenError`).
  - `apply_candidate_update_with_replay(...)` executes archived regression suites across **all** prior domains before promoting any candidate; on any single-domain or cross-domain regression, it automatically rejects the candidate, rolls back to the verified parent version, and logs a `quarantined` record with offending test cases (`rollback_skill` strictly blocks `quarantined` and `shadow` targets).
  - Includes two mechanistic continual forgetting baselines: `NaiveOverwriteNoReplayBaseline` (`max_active_slots=28`, un-namespaced per-phase slot keys) and `SmallCPUNeuralContinualRef` (`comparison_only=True`, sequential SGD MLP without replay, analytical gradient error `< 1e-5`).
- `research/cpu_learning/continual/cross_domain.py` — `CrossDomainAbstractionRegistry` and `CrossDomainTransferEvaluator` (`s8-cross-domain-v1`) mining shared abstract guarded schemas across unrelated source domains (`S0 Relational` + `Arcane`), enforcing positive per-domain and unamortized joint MDL bit savings before universal promotion, scoping incompatible arity/guard schemas to `domain_local` (`promoted_universal=False`), and evaluating 2-shot transfer vs cold-start ITL and mechanistic ungated forced-transfer ablations.
- `research/cpu_learning/evaluation/run_s8_experiments.py` — Reproducible 5-seed chronological runner verifying all **19 Stage S8 acceptance gates**, bit-exact S0–S8 manifests (`--verify-manifests`), AST/lexical & Python 3.11 compatibility checks, and sub-linear library growth (`mean = 5.49s/seed <= 15s`, total `31.81s`).
- `research/cpu_learning/tests/test_s8_continual_and_transfer.py` — Independent adversarial pytest suite (`8` test suites; **`68 / 68`** total tests passing across `test_s0`..`test_s8`) covering AST/lexical & Python 3.11 stdlib scans, quarantine/shadow `rollback_skill` rejection, cross-domain UTF-8 breakage replay detection from upstream relational edits, immediate post-phase baseline learning + analytical backprop gradient verification, a **100%-alien 4-domain positive + incompatible scoped transfer test**, and committed + 2-seed isolated subset gate verification.

---

## 2. Independent Adversarial Audit Findings Resolved Before Sign-Off

Three concrete defects uncovered during independent adversarial review were fixed and regression-tested before Stage S8 sign-off:
1. **Quarantined/Shadow Rollback Protection (`replay_manager.py`):** `ContinualKnowledgeBase.rollback_skill(domain_id, skill_id, target_version)` now raises `ValueError` whenever the target version has `lifecycle_state` `"quarantined"` or `"shadow"`, permitting rollback strictly to previously verified `"promoted"` or `"retired"` versions.
2. **"Learn High First, Then Forget" Baseline Integrity (`run_s8_experiments.py`):** Configured `NaiveOverwriteNoReplayBaseline(max_active_slots=28)` so no intra-phase self-eviction occurs (`>= 95%` immediate post-phase accuracy across all Phases `1–4` before subsequent phases overwrite un-namespaced slots down to `13.27%`), upgraded `SmallCPUNeuralContinualRef` to learnable public structural/behavioral feature probes (`>= 75%` immediate post-phase accuracy across Phases `1–4` before sequential weight overwrite degrades retention to `9.05%`), and added the mandatory `baselines_genuinely_learn_each_phase_before_forgetting` gate.
3. **Mechanistic `domain_local` Forced-Transfer Ablation & Joint Bit Gate (`cross_domain.py`):** Replaced artificial contradictory guards in `_build_ungated_forced_ablation_cards` with `_project_concept_mechanistically_onto_card` (projecting actual `domain_local` schemas onto incompatible Domain 5 operators, exposing genuine `50.00%` negative transfer when `domain_local` scoping is disabled) and enforced `joint_bits_saved > 0` (`+752` unamortized joint bits saved) in `register_source_domain_pair`.

---

## 3. Verified 5-Seed Stage S8 Results (`[101, 202, 303, 404, 505]`, `19 / 19` Gate Checks PASS)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s8_experiments --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s8_experiments
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s8_research_report.md`](../research/cpu_learning/reports/s8_research_report.md) · [`research/cpu_learning/reports/s8_summary.json`](../research/cpu_learning/reports/s8_summary.json) · [`research/cpu_learning/reports/raw_runs/s8_continual_matrix.csv`](../research/cpu_learning/reports/raw_runs/s8_continual_matrix.csv) · [`research/cpu_learning/reports/raw_runs/s8_seed_101.json`](../research/cpu_learning/reports/raw_runs/s8_seed_101.json) · [`research/cpu_learning/data/s8_preregistration.json`](../research/cpu_learning/data/s8_preregistration.json)

### A. Final Phase-5 Continual Retention Across Chronological Domains (Phases `1–4`)

| Chronological Domain | Mirror `ContinualKnowledgeBase` (`Mean [95% CI]`) | `NaiveOverwriteNoReplay` | `SmallCPUNeuralContinualRef` (`comparison_only`) |
|---|---:|---:|---:|
| `phase1_relational` (Depths `1–6`) | **100.00% [100.00%, 100.00%]** | 21.43% | 11.88% |
| `phase2_math_and_lists` (Mod-10, Multi-Digit, Lists) | **100.00% [100.00%, 100.00%]** | 0.00% | 3.33% |
| `phase3_code_and_ast` (AST + HumanEval-Micro) | **97.73% [97.73%, 97.73%]** (`0.000%` drop) | 0.00% | 0.00% |
| `phase4_utf8_grounding` (Compositional UTF-8) | **100.00% [100.00%, 100.00%]** | 31.67% | 21.00% |
| **Mean Retention (Phases 1–4)** | **99.43% [99.43%, 99.43%]** | **13.27%** (`+86.16pp`) | **9.05%** (`+90.38pp`) |

- **Zero Forgetting & Poisoned-Update Quarantine (`5 / 5` Seeds):** `0.000%` retention drop across all 4 phases (`phase3_code_and_ast` retains its exact `97.73%` post-Phase-3 score from the `19/20` HumanEval-Micro + `4/4` AST suite), **`0`** prior verified skill hashes rewritten, and **`100.00%` (`3 / 3` per seed)** of injected poisoned updates (relational `move` guard drop, list `reverse` corruption, UTF-8 passive `shifted` role swap) caught, quarantined, and rolled back with **`0.000%`** post-rollback regression.

### B. Cross-Domain Positive Transfer (D3 Logistics & D4 Biolab) & Scoped Negative-Transfer Rejection (D5 Reactor)

- **Source Abstraction Bit Savings (`S0 Relational + Arcane`):** Saves `1,080` bits on D1, `1,904` bits on D2, and `752` unamortized joint bits (`2` universal schemas promoted; `3` non-isomorphic schemas scoped to `domain_local`).

| Unseen Target Domain | Cold-Start ITL | Mirror Scoped Transfer | Lift (`Mean [95% CI]`) | Negative Transfer | Ungated Forced Ablation | Marginal Bit Savings |
|---|---:|---:|---:|---:|---:|---:|
| `domain3_logistics` (2-shot, `44` OOD tasks/seed) | 63.64% | **100.00%** | **`+36.36pp [+36.36pp, +36.36pp]`** | **0.000%** | 100.00% | **+1,872 bits** |
| `domain4_biolab` (brand-new 4th domain, 2-shot, `44` OOD tasks/seed) | 63.64% | **100.00%** | **`+36.36pp [+36.36pp, +36.36pp]`** | **0.000%** | 100.00% | **+1,872 bits** |
| `domain5_incompatible_reactor` (`24` OOD tasks/seed) | 100.00% | **100.00%** | **`+0.00pp [+0.00pp, +0.00pp]`** | **0.000%** | 50.00% (`50.00%` neg transfer) | +0 bits |

### C. Sub-Linear Library Growth Across Phases `1–5` (`5.49s` mean/seed <= `15s`)

| Chronological Phase | Active Skills | Cumulative Raw Train Bits | Cumulative Compressed Train Bits | Active Library Bits | Replay Ledger Bits | Library / Raw Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `Phase 1: Relational World` | 4 | 704,317 | 52,446 | 4,992 | 4,579 | `0.0071` |
| `Phase 2: Mathematics & Lists` | 12 | 1,045,526 | 75,042 | 8,344 | 4,886 | `0.0080` |
| `Phase 3: Python AST & Code` | 35 | 1,085,718 | 87,179 | 18,360 | 5,030 | `0.0169` |
| `Phase 4: Raw UTF-8 Grounding` | 58 | 1,284,419 | 98,853 | 24,872 | 5,240 | `0.0194` |
| `Phase 5: Cross-Domain & Poison Rollbacks` | **64** | **1,657,507** | **129,570** | **29,960** | **5,619** | **`0.0181`** (`55.3×` vs raw) |

---

## 4. Remaining Limitations & Next Stage (S9 — Bounded Autonomous Exploration, Curiosity & Governance)

### Remaining Limitations After S8
1. **Externally Scheduled Phase Curricula So Far:** Stage S8 feeds chronological phases and candidate updates from a pre-registered phase schedule; autonomous closed-loop exploration — choosing what states/actions to probe next under strict safety budgets, epistemic uncertainty, and reversible governance — belongs to Stage **S9**.
2. **Full External Benchmark Re-Evaluation Deferred:** Direct re-evaluation on open external suites (`MBPP`, `LongMemEval`, `StoryCloze`) and end-to-end head-to-head system synthesis remain deferred to Stage **S10**.

### Next Stage: Stage S9 — Bounded Autonomous Exploration, Curiosity & Operational Governance
- **Goal:** Build a bounded, safety-gated CPU-native autonomous exploration controller that selects high-value epistemic/disagreement interventions in unfamiliar environments, discovers complete operator libraries within strict step/risk budgets, abstains or requests approval on irreversible/unsafe actions, and logs tamper-evident provenance.
- **Acceptance Criteria:** Outperform random and greedy exploration in operator-discovery sample efficiency while maintaining `0` safety/irreversible violations, `100%` governance audit integrity, and zero regression across all S0–S8 manifests and test suites across all 5 fixed seeds.
