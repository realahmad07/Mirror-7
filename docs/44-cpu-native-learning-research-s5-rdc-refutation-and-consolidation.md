# 44 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S5 RDC Refutation & Consolidation

This document records the Stage **S5** implementation, 5-seed evaluation across all **four pre-registered RDC kill tests**, intra-domain behavioral-equivalence compaction, cross-domain concept abstraction and 3rd-domain few-shot transfer (`Domain 1 S0 + Domain 2 Arcane -> Held-Out Domain 3 Logistics`), adversarial audit findings, and verified gate checks for Mirror 7's CPU-native **Refutation-Driven Consolidation (RDC)** subsystem (`research/cpu_learning/rdc/`).

> **Research Integrity Notice:** Stage S5 enforces four non-negotiable epistemic safeguards: (1) every system-generated "dream" is strictly a proposed test judged by an independent black-box `TransitionEnvironment` (**`0` self-certified dreams** across all 5 seeds); (2) the description-length objective `J = L(library) + L(examples | library) + 10000 * verified_failures + 20000 * collateral_changes` NEVER overrides ITL correctness gates — operators consolidate only under behavioral equivalence, strict reduction in `J`, and `100%` regression retention; (3) multi-fault blame attribution reports explicit ambiguity (`is_ambiguous=True`, `unique_culprit=None`) rather than forcing a single guess; and (4) retired operator versions are never deleted from `OperatorVersionStore`. Automated AST auditing certifies **0 import violations** across `38` non-evaluator modules (`21` learner, `7` ITL, `4` SHARD, `5` RDC) and **0 hardcoded domain literals** across an expanded 33-word S0/D2/D3 scan. Stages **S6–S10** (math/code transfer, language grounding, continual learning, autonomy, and broad external comparisons) are NOT claimed here.

---

## 1. Stage S5 Modules Implemented (`research/cpu_learning/`)

- `research/cpu_learning/rdc/__init__.py` — Public Stage S5 RDC exports.
- `research/cpu_learning/rdc/mdl_judge.py` — Exact description-length consolidation judge (`s5-rdc-mdl-v1`) evaluating `J = L(library) + L(examples | library) + beta * verified_failures + gamma * collateral_changes` (`DEFAULT_BETA = 10_000.0`, `DEFAULT_GAMMA = 20_000.0`), where `L(library)` measures compressed executable `OperatorCard` + shared concept template bits and `L(examples | library)` measures compressed residual bits of unpredicted deltas (`0` bits on exact match).
- `research/cpu_learning/rdc/refuter.py` — Active refutation and query engines (`s5-rdc-refuter-v1`):
  - **Mechanism 1 (`DreamRefuter`):** Synthesizes boundary and counterfactual dream probes directly from candidate `OperatorCard` structure and `CorpusStatistics` (`0` hardcoded domain names), executes every probe against the independent black-box `TransitionEnvironment` (`0` self-certified dreams), acquires repeated-evidence guards (`min_guard_support=2`) or prunes over-constrained clauses (`min_ablation_support=2`), and promotes revised cards through `OperatorVersionStore` strictly when verified and `J` improves.
  - **Mechanism 2 (`MaxDisagreementQueryEngine`):** Maintains a version-space committee of competing `OperatorCard` hypotheses and selects unlabelled probes maximizing committee vote-split / delta disagreement against the black-box environment.
- `research/cpu_learning/rdc/blame.py` — **Mechanism 3 (`InterventionalBlameEngine`, `s5-rdc-blame-v1`):** Performs leave-one-operator-out / replace-one-operator counterfactual trajectory replay across multi-step failures (Depths `2–6`, both online and offline replay modes), isolating exact culprit operators when uniquely identifiable and honestly reporting `is_ambiguous=True`, `unique_culprit=None` with full `suspected_operators` retention when multiple operators are simultaneously faulty.
- `research/cpu_learning/rdc/consolidation.py` — **Mechanism 4 (`RDCConsolidator`, `s5-rdc-consolidation-v1`):** Intra-domain behavioral-equivalence merging (rejecting unsafe merges, retiring duplicates non-destructively via additive `OperatorVersionStore.retire_active()`, and supporting deterministic `rollback_consolidation()`) plus lossless, reversible cross-domain relational concept abstraction (`ConceptTemplate`) transferring structural operator schemas to unseen domains from positives-only few-shot demos.
- `research/cpu_learning/evaluation/run_s5_experiments.py` — Reproducible 5-seed runner verifying all **24 Stage S5 acceptance & kill-test gates** on exact Student-$t$ 95% CI bounds, bit-exact S0–S5 manifest verification (`--verify-manifests`), and AST isolation audits (`mean = 6.54s/seed`, `max = 10.56s/seed <= 30s`).
- `research/cpu_learning/tests/test_s5_rdc_refutation_and_consolidation.py` — Independent adversarial pytest suite (`8` test suites; **`44 / 44`** total tests passing across `test_s0`..`test_s5`) covering AST isolation & 33-word domain-literal scanning, MDL judge invariants, fair probe-source-agnostic over-constraint refutation, exact env-query & joint-ablation MDL accounting, empty hypothesis-space guards, multi-fault blame ambiguity honesty, unsafe merge rejection & rollback, reversed-order predicate concept reversibility, D3 few-shot transfer, and committed 5-seed gate verification.

---

## 2. Independent Adversarial Audit Findings Resolved Before Sign-Off

Three genuine fairness, accounting, and input-validation defects in `research/cpu_learning/rdc/refuter.py` were identified during independent adversarial review and fixed before Stage S5 sign-off:
1. **Fair Probe-Source-Agnostic Over-Constraint Ablation (`DreamRefuter.refine_library`):** Initially, over-constraint ablation support was recorded only when `dream.dream_kind == "overconstraint_ablation"`, inadvertently preventing the `rdc_random_probes` control (`dream_kind == "random_perturbation"`) from pruning `(?source, accessible, true)` even when random probes hit `>= 2` valid single-clause violation counterexamples on Seeds `404` and `505`. Adding `_find_sole_violated_direct_literal(card, state, roles)` removed the `dream_kind` gate so both `rdc_dream_refutation` and `rdc_random_probes` run the **identical** refutation engine differing strictly in probe generation. Under this fair control, `rdc_random_probes` rose from `80.22%` to **`87.17% ± 12.27%`**, while `rdc_dream_refutation` still achieved **`100.00% ± 0.00%`** (**`+12.83pp`**, 95% CI low `+0.55pp > 0`).
2. **Exact `env_queries_used` & Joint-Ablation `verified_examples` Accounting:** `env_queries_used` now increments strictly on each actual `env.step()` call, and joint-ablation probe outcomes are appended to `verified_examples` so `before_mdl` and `after_mdl` compare over the exact same verified transition set.
3. **Empty Hypothesis-Space Guard (`MaxDisagreementQueryEngine.run_active_selection`):** Added immediate `ValueError("initial_hypotheses must be non-empty")` validation on empty hypothesis committees.

---

## 3. Verified 5-Seed Stage S5 Results (`[101, 202, 303, 404, 505]`, `24 / 24` Gate Checks PASS)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s5_experiments --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s5_experiments
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s5_research_report.md`](../research/cpu_learning/reports/s5_research_report.md) · [`research/cpu_learning/reports/s5_summary.json`](../research/cpu_learning/reports/s5_summary.json) · [`research/cpu_learning/reports/raw_runs/s5_ablations.csv`](../research/cpu_learning/reports/raw_runs/s5_ablations.csv) · [`research/cpu_learning/reports/raw_runs/s5_seed_101.json`](../research/cpu_learning/reports/raw_runs/s5_seed_101.json) · [`research/cpu_learning/data/s5_preregistration.json`](../research/cpu_learning/data/s5_preregistration.json)

### A. Ablation 1 — Kill Test 1: Dream-Refutation (`probe_budget = 260`)

| Method | Unseen OOD Acc (Mean ± 95% CI) | OOD Error Rate | Collateral | Self-Certified Dreams |
|---|---:|---:|---:|---:|
| `plain_mdl_no_dreams` | 76.09% ± 0.00% | 23.91% | 0.341% | 0 |
| `rdc_random_probes` (fair control) | 87.17% ± 12.27% | 12.83% | 0.124% | 0 |
| `rdc_dream_refutation` | **100.00% ± 0.00%** | **0.00%** | **0.000%** | **0** |

- **vs `plain_mdl_no_dreams`:** **`+23.91pp`** OOD accuracy gain (95% CI low `+23.91pp >= +15.00pp` gate), strict `J` reduction on all `5 / 5` seeds, and **`0`** self-certified dreams.
- **vs Fair `rdc_random_probes` (identical refutation engine, same budget):** **`+12.83pp`** OOD accuracy gain (95% CI low `+0.55pp > 0`), eliminating `0.124%` residual collateral corruption.

### B. Ablation 2 — Kill Test 2: Interventional Leave-One-Operator-Out Blame (Depths `2–6`)

| Method | Top-1 Exact Acc (Mean ± 95% CI) | MRR | Brier Score |
|---|---:|---:|---:|
| `interventional_blame` | **100.00% ± 0.00%** | **1.0000** | **0.0000** |
| `interventional_blame_offline` | **100.00% ± 0.00%** | **1.0000** | **0.0000** |
| `random_blame` | 34.82% ± 4.02% | 0.6043 | 0.4399 |
| `uniform_blame` | 0.00% ± 0.00% | 0.5155 | 0.2158 |

- **Blame Localization & Multi-Fault Honesty:** Both online and offline interventional blame beat `random_blame` by **`+65.18pp`** (95% CI low `+61.16pp`) and `uniform_blame` by **`+100.00pp`**, while passing dual-fault ambiguity honesty on **`5 / 5` seeds** (`is_ambiguous=True`, `unique_culprit=None`, all faulty operators retained in `suspected_operators`).

### C. Ablation 3 — Kill Test 3: Active Max-Disagreement Query Selection

| Strategy | Queries to `>= 98%` OOD Acc (Mean ± 95% CI) | Total Queries Used | Final Unseen OOD Acc |
|---|---:|---:|---:|
| `max_disagreement_query` | **14.00 ± 0.00** | **16.00** | **100.00%** |
| `random_query` | 93.20 ± 50.34 | 150.40 | 100.00% |

- **Query Sample Efficiency:** `max_disagreement_query` reaches `>= 98%` unseen OOD accuracy in **`82.37%` fewer simulator queries** than `random_query` (95% CI low `72.48% >= 35.00%` gate).

### D. Ablation 4 — Kill Test 4: Intra-Domain Safe Consolidation & Cross-Domain Concept Transfer

- **Intra-Domain Safe Consolidation & Rollback (`5 / 5` Seeds):** Compacts duplicate verified operators from `4,720` -> **`4,616` bits** (**`-104` bits**, `J: 4720.0 -> 4616.0`) with **`100.00%`** regression retention (`0` regressions), rejects **`100%`** of unsafe merges (`open` vs `lock`, `move` vs `give`), preserves retired versions in `OperatorVersionStore`, and passes deterministic `rollback_consolidation()` across all 5 seeds.
- **Cross-Domain Concept Abstraction (`Domain 1 S0 + Domain 2 Arcane`):** Compresses Domain 1 (`3,464` -> `1,704` bits, **`-1,760` bits**), Domain 2 (`3,448` -> `1,544` bits, **`-1,904` bits**), and **unamortized Joint MDL** (`Domain 1 + Domain 2 + Concepts`: `5,424` -> **`4,336` bits**, **`-1,088` bits saved**) with **`100%` lossless reversible** operator reconstruction.

| Held-Out Domain 3 Logistics Few-Shot (`2` Positives/Action, `0` Negatives, `44` OOD Tasks/Seed) | Domain 3 Unseen OOD Acc (Mean ± 95% CI) | Collateral | Description Bits |
|---|---:|---:|---:|
| `rdc_cross_domain_transfer` | **100.00% ± 0.00%** | **0.000%** | **1,680** |
| `cold_start_itl_no_concepts` | 63.64% ± 0.00% | 0.826% | 3,520 |

- **3rd-Domain Few-Shot Transfer Gain:** Transferring consolidated relational concepts lifts held-out Domain 3 Logistics accuracy (from 2 positive demos/action and **0 negative contrast demos**) by **`+36.36pp`** (95% CI low `+36.36pp >= +20.00pp` gate), eliminates `0.826%` collateral corruption, and saves **`-1,840` description bits** (`1,680` vs `3,520`).

---

## 4. Remaining Limitations & Next Stage (S6 — Symbolic/Discrete Math & Code Operator Induction)

### Remaining Limitations After S5
1. **Relational World Scope Only So Far:** Stages S0–S5 induce, guide, refute, and consolidate state-transition `OperatorCard`s over symbolic entity-relation graphs (S0 relational world, D2 Arcane, D3 Logistics). Discontinuous arithmetic rules — notably Stage S0's `(a + b) % 10` modular-addition canary where pure HDC (`4.00%`) and bit-perceptrons (`7.00%`) failed below chance — and executable program/code synthesis remain unaddressed until Stage **S6**.
2. **Raw Language Grounding & Continual Non-Stationary Streams Deferred:** Open-vocabulary raw text grounding without pre-tokenized role schemas, continual multi-task drift without catastrophic forgetting, bounded autonomous exploration, and broad external benchmarks (`MBPP`, `LongMemEval`, `StoryCloze`) remain deferred to Stages **S7–S10**.

### Next Stage: Stage S6 — Symbolic/Discrete Math & Code Operator Induction
- **Goal:** Extend the unified SHARD + ITL + RDC loop from relational graph transitions to discrete arithmetic/algebraic rules (explicitly solving the S0 discontinuous `(a + b) % 10` modular-addition canary from examples without hand-coding) and sandbox-verified code/program operator synthesis (`ZeroTrustSandboxJudge` / independent test harnesses), while assessing whether specialist `QuotientEGraph` equivalence earns keep/skip inclusion by ablation.
- **Acceptance Criteria:** Solve the S0 modular-addition canary (`>= 95%` unseen pair accuracy from training examples without hardcoded `mod 10`) and pass held-out compositional math/code synthesis gates on independent verifiers with zero regression across all S0–S5 manifests and test suites across all 5 fixed seeds.
