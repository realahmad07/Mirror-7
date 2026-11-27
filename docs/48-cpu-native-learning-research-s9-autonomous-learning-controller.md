# 48 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S9 Autonomous Learning Controller & Self-Curriculum

This document records the Stage **S9** implementation, 5-seed multi-frontier closed-loop autonomous learning benchmarks against `PassiveStreamController` and `RandomCurriculumController` under an identical **`42`-query** and CPU budget, interactive non-starvation SLA measurements (`120` concurrent live requests/seed = `600` total), per-subsystem CPU accounting, anchor-skill retention & poisoned-update rollback, MDL duplicate retirement, and resolved adversarial audit findings for Mirror 7's CPU-native **Autonomous Learning Controller & Self-Curriculum** subsystem (`research/cpu_learning/controller/`).

> **Research Integrity Notice:** Stage S9 orchestrates the full 11-step `L0–L7` closed loop (`Observe -> Frame -> Difference -> SHARD Propose -> ITL Generalise -> Contrast -> Independent Verify -> Correct -> Compose -> RDC MDL Consolidate -> ContinualKB Regression Replay`) strictly on CPU — **ZERO GPU, ZERO pretrained LLMs, ZERO backpropagation, ZERO self-certified promotions (`self_certified_promotions == 0`), ZERO post-freeze environment queries (`post_freeze_env_queries == 0`), ZERO rewritten anchor hashes, and 100% Python 3.11 stdlib compatibility**. Automated AST & lexical auditing across `62` non-evaluator modules (`37` learner, `3` controller) certifies **0 import violations**, **0 hardcoded S0..D5 domain literals**, **0 comparison-only baseline leaks**, and **0 Python 3.12+ constructs**. Control fairness is independently verified: given the full `900`-query pool, `RandomCurriculumController` achieves `>= 99.5%` overall (`98.76%` relational, `100%` UTF-8 grounding, `100%` math/lists), proving zero artificial handicaps. Stage **S10** (broad external benchmark evaluation & system synthesis) is NOT claimed here.

---

## 1. Stage S9 Modules Implemented (`research/cpu_learning/`)

- `research/cpu_learning/controller/__init__.py` — Public Stage S9 controller and curriculum exports.
- `research/cpu_learning/controller/curriculum.py` — `SelfCurriculumScheduler`, `AnomalySignal`, `CurriculumAction`, `CurriculumBudget`, `PassiveStreamController`, and `RandomCurriculumController` (`s9-self-curriculum-v1`):
  - Implements anomaly-triggered budgeting across multi-domain streams (`prediction_mismatch`, `epistemic_abstention`, `hypothesis_ambiguity`, `unresolved_guard_uncertainty`, `library_bloat_high`), ranking candidate next actions (`max_disagreement_query`, `contrast_probe`, `dream_refutation`, `induce_or_repair`, `consolidate_mdl`, `regression_replay`, `idle_noop`) by expected information gain per millisecond and halting active queries (`idle_noop`) once frontiers are verified.
  - Enforces per-frontier deadline slices without hardcoded floor overruns and implements two fair budget-matched controls (`PassiveStreamController` and `RandomCurriculumController`).
- `research/cpu_learning/controller/meta_controller.py` — `AutonomousMetaController`, `FreezeGatedVerifierGate`, `IndependentVerificationReceipt`, `InteractiveLatencyMonitor`, `ControllerTickTrace`, and `ControllerFrozenError` (`s9-autonomous-meta-controller-v1`):
  - Priority-1 drains queued live interactive requests (`solve_interactive_now`) before running bounded background learning ticks (`learning_slice_ms`), tracking non-overlapping subsystem CPU timers (`query_selection_ms`, `verification_ms`, `induction_ms`, `replay_ms`, `consolidation_ms`, `retrieval_ms`) and yielding `slice_budget_yield` before overrunning tight slices.
  - Enforces cryptographic anti-forgery / single-use `IndependentVerificationReceipt` consumption (`self_certified_promotions == 0`), mandatory `RegressionReplayManager` checks across pre-verified anchor domains, non-destructive `ContinualKnowledgeBase` retirement during `consolidate_mdl`, and `.freeze()` (`post_freeze_env_queries == 0`).
- `research/cpu_learning/evaluation/run_s9_experiments.py` — Reproducible 5-seed closed-loop runner verifying all **21 Stage S9 acceptance gates**, bit-exact S0–S9 manifests (`--verify-manifests`), AST/lexical & Python 3.11 compatibility checks, and interactive SLA monitoring (`mean = 4.155s/seed`, `max = 4.712s/seed <= 15.0s`).
- `research/cpu_learning/tests/test_s9_autonomous_controller.py` — Independent adversarial pytest suite (`8` test suites; **`76 / 76`** total tests passing across `test_s0`..`test_s9`) covering AST/Python 3.11 scans, anti-forgery/anti-replay receipt rejection, tight `1.5 ms` slice-budget yielding & exhausted-budget `0`-scan short-circuiting, KB duplicate retirement & alias resolution, poisoned anchor rollback, full-pool `RandomCurriculumController` fairness (`900` queries -> `>= 99.5%`), a **100%-alien 3-frontier autonomy test** (quantum relational + Cyrillic-Esperanto UTF-8 + modular synthesis), and committed + 2-seed `tmp_path` gate verification.

---

## 2. Independent Adversarial Audit Findings Resolved Before Sign-Off

Four concrete defects identified during independent adversarial review were fixed and regression-tested before Stage S9 sign-off:
1. **Anti-Forgery / Single-Use `IndependentVerificationReceipt` Enforcement & Grounding Repair Receipts (`meta_controller.py`):** `FreezeGatedVerifierGate` now guards `issue_receipt` with `_guard_mutable()`, rejects any `"self"` `verifier_kind`, tracks issued `receipt_id`s and marks them consumed on first use (`verify_and_consume_receipt`), and mints real receipts during grounding `induce_or_repair` — raising `PermissionError` on `None`, forged, digest-mismatched, or replayed receipts (`self_certified_promotions == 0`).
2. **Tight `learning_slice_ms` Deadline Yielding & Exhausted-Budget Short-Circuit (`curriculum.py`, `meta_controller.py`):** Removed the `8.0 ms` selection floor, added deadline checks across anomaly and frontier scoring loops, yielded `slice_budget_yield` before overrunning tight slices (verified `learning_slice_ms = 0.816 ms <= 1.5 ms` on `CurriculumBudget(42, 5.0, 1.5)`), and short-circuited immediately after Priority-1 interactive draining (`0` anomalies scanned, `0.0 ms` learning slice) once `max_total_cpu_ms` is exhausted.
3. **Non-Destructive `ContinualKnowledgeBase` Duplicate Retirement & Alias Resolution (`meta_controller.py`, `run_s9_experiments.py`):** When `consolidate_mdl` succeeds, `meta_controller` now transitions the duplicate `VersionedSkillEntry` (`move_dup_alias`) to `"retired"` inside `ContinualKnowledgeBase` under full `RegressionReplayManager` rollback protection, updates `_cached_library_bits` (`17,251.2` mean final library bits), and merges `kb` `alias_map` inside `solve_interactive_now`.
4. **Non-Overlapping Subsystem CPU Timers & `starvation_ms` Separation (`meta_controller.py`):** Closed `replay_ms` strictly prior to `retrieval_ms` so `sum(subsystems) <= total_learning_slice_ms` holds on `100%` of ticks (`644.40 ms <= 645.09 ms` mean), and separated `starvation_ms = 25.0` from `sla_ms = 5.0` (`0` starved and `0` dropped interactive requests across all runs).

---

## 3. Verified 5-Seed Stage S9 Results (`[101, 202, 303, 404, 505]`, `21 / 21` Gate Checks PASS)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s9_experiments --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s9_experiments
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s9_research_report.md`](../research/cpu_learning/reports/s9_research_report.md) · [`research/cpu_learning/reports/s9_summary.json`](../research/cpu_learning/reports/s9_summary.json) · [`research/cpu_learning/reports/raw_runs/s9_controller_traces.csv`](../research/cpu_learning/reports/raw_runs/s9_controller_traces.csv) · [`research/cpu_learning/reports/raw_runs/s9_seed_101.json`](../research/cpu_learning/reports/raw_runs/s9_seed_101.json) · [`research/cpu_learning/data/s9_preregistration.json`](../research/cpu_learning/data/s9_preregistration.json)

### A. Held-Out Multi-Frontier Accuracy Under Identical `42`-Query & CPU Budgets

| Metric / Frontier | Cold-Start Seed | `PassiveStreamController` (`0.0` queries) | `RandomCurriculumController` (`42.0` queries) | Mirror `AutonomousMetaController` (`31.0` queries, `Mean [95% CI]`) |
|---|---:|---:|---:|---:|
| **Relational World (All Unseen Splits)** | 66.46% (`3.22%` collat.) | 73.29% | 77.14% | **98.76% [98.76%, 98.76%]** (`0.080%` collat.) |
| — Compositional Depth 4 | 63.64% | 81.82% | 81.82% | **100.00%** |
| — Compositional Depth 6 | 45.45% | 45.45% | 52.73% | **100.00%** |
| **Raw UTF-8 Ambiguous Grounding** | 0.00% | 0.00% | 0.00% | **100.00% [100.00%, 100.00%]** |
| **Functional Math & List Synthesis** | 35.42% | 35.42% | 38.75% | **100.00% [100.00%, 100.00%]** |
| **Overall 3-Frontier Accuracy** | 33.96% | 36.24% | 38.63% | **99.59% [99.59%, 99.59%]** |
| **Anchor Old-Skill Retention (`Arcane D2 + Code AST`)** | 100.00% | 100.00% | 100.00% | **100.00%** (`0` hashes rewritten) |

- **Paired Lifts & Sample Efficiency:** `AutonomousMetaController` converges in **`31.0`** active queries and halts early (`idle_noop`) on verified frontiers vs `RandomCurriculumController` exhausting all `42.0` queries — achieving **`+65.63pp [+64.41pp, +66.85pp]`** over Cold-Start, **`+63.35pp [+62.13pp, +64.57pp]`** over `PassiveStreamController`, and **`+60.96pp [+55.83pp, +66.08pp]`** over `RandomCurriculumController` (`0.080% < 2.0%` relational collateral; mid-stream poisoned anchor update `100%` caught, rolled back, and quarantined across all `5 / 5` seeds).
- **Control Fairness Proof (`#90/#92`):** When `RandomCurriculumController` is granted the full `900`-query pool on Seed `101`, it reaches the exact same **`99.59%`** overall ceiling (`98.76%` relational, `100.00%` UTF-8 grounding, `100.00%` math/lists), confirming its `38.63%` score under the `42`-query cap reflects genuine uniform-sampling inefficiency across an `~890`-probe 3-frontier pool rather than any structural handicap.

### B. Interactive Non-Starvation SLA, Subsystem CPU Breakdown & MDL Accounting (`mean = 4.155s/seed`, `max = 4.712s/seed <= 15.0s`)

| Metric | Concurrent `AutonomousMetaController` | Idle Baseline Controller | Gate / Limit |
|---|---:|---:|---:|
| Live Interactive Queries Served | `120 / seed` (`600` total) | `120 / seed` (`600` total) | `>= 100 / seed` |
| Interactive `p95` Latency (`Mean [95% CI]`) | **`1.3903 ms [0.9288, 1.8517]`** (`max = 1.9008 ms`) | `1.1159 ms [0.7400, 1.4918]` | `<= 5.0 ms` |
| Starved (`> 25 ms`) / Dropped Interactive Queries | **`0` starved / `0` dropped** | `0` starved / `0` dropped | `0` |

- **Non-Overlapping Mean Per-Seed Subsystem CPU Breakdown (`644.40 ms <= 645.09 ms` total learning slice; `interactive_slice_ms = 63.56 ms`):** `query_selection = 345.83 ms`, `replay = 199.89 ms`, `induction = 72.75 ms`, `retrieval = 11.82 ms`, `consolidation = 7.94 ms`, `verification = 6.17 ms`.
- **MDL Duplicate Retirement & Stream Compression:** Compresses `481,189` raw stream bits into **`17,251.2` active library bits** (`ratio = 0.0359` = **`27.9×` compression**).

---

## 4. Remaining Limitations & Next Stage (S10 — Broad External Benchmark Evaluation & Final Research Synthesis)

### Remaining Limitations After S9
1. **Controlled Multi-Frontier Benchmark Environments So Far:** While Stages S0–S9 prove end-to-end CPU-native operator induction, active contrast/refutation, SHARD proposal speedup, cross-domain consolidation, raw-UTF-8 grounding, non-neural `2.5649 bpb` byte modeling, zero-forgetting continual replay, and non-starving autonomous self-curricula across relational, arcane, logistics, biolab, math, list, AST, and HumanEval-Micro suites, the unified architecture has not yet been re-evaluated head-to-head against the **external open benchmarks** where Mirror's pre-research organs struggled (`MBPP`, `LongMemEval`, `StoryCloze` from `scripts/run_five_organ_external_eval.py`).
2. **Open-Domain Commonsense & Unbounded Real-World Priors:** Pure CPU few-shot rule induction without web-scale pretrained corpora learns verifiable operational and algorithmic structure rapidly from examples and simulators, whereas ungrounded commonsense narrative priors not derivable from observed transitions remain bounded by observed evidence — a boundary Stage **S10** measures transparently.

### Next Stage: Stage S10 — Broad External Benchmark Re-Evaluation, Ablation Matrix & Final Synthesis
- **Goal:** Evaluate the complete unified `SHARD + ITL + RDC + Grounding + Continual + Autonomous` CPU-native learning stack against external open benchmarks (including `MBPP` execution-only synthesis, `LongMemEval` episodic/ledger retrieval + state tracking, and `StoryCloze` / causal narrative state tracking where applicable), compare directly against Stage S0 / `run_five_organ_external_eval.py` baselines and CPU neural references under matched hardware/time budgets, and report complete honest wins, failure boundaries, and reproducible end-to-end artifacts.
- **Acceptance Criteria:** Demonstrate statistically verified external benchmark improvements over pre-research Mirror baselines with zero test leakage, complete failure-mode transparency where external tasks require ungrounded world priors, and `100%` green verification across all S0–S10 manifests, pytest suites, and Linux Python 3.11 CI.
