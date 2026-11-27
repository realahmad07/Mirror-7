# 49 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S10 Broad Independent Evaluation & Honest Verdict

This document records the capstone **Stage S10** independent broad evaluation, complete **13-gate PDF p. 13 quantitative acceptance ledger** (governed strictly by 95% Student-$t$ confidence bounds across all 5 fixed seeds `[101, 202, 303, 404, 505]`), **11-system PDF p. 15 reference & ablation matrix**, live single-core CPU throughput microbenchmark (`>= 1,000 pred/sec/core`), `tracemalloc` heap and transparent TDP-proxy energy profiling, fresh post-freeze temporal stress test & 6-category OOD abstention calibration (`1,100` items total), whole-tree AST/Python 3.11 integrity audit, and the brutally honest **GPT-class readiness verdict** for Mirror 7's unified **SHARD + ITL + RDC** CPU-native learning architecture (`research/cpu_learning/`).

> **Research Integrity & Verdict Notice (`NARROW_VERIFIED_CPU_LEARNING_PROVEN__OPEN_WORLD_GPT_PARITY_NOT_CLAIMED`):** All **13 / 13** PDF p. 13 quantitative research gates and all **16 / 16** Stage S10 capstone checks pass across 5 fixed seeds on pure CPU (`0` GPU, `0` pretrained LLMs, `0` backpropagation in the candidate learner, `0` self-certified promotions, `0` post-freeze environment queries, `0` AST import/hardcode/comparison-leak/Python 3.12+ violations across `63+` non-evaluator modules and `37` learner modules). **Open-world GPT-class general language/coding parity is explicitly NOT claimed**: prior external evaluation findings (`scripts/run_five_organ_external_eval.py`: `0 / 20` sanitized MBPP, below-chance ROCStories Cloze, weak LongMemEval retrieval) are preserved verbatim alongside the exact scope of what SHARD+ITL+RDC has and has not proven.

---

## 1. Stage S10 Modules & Verification Artifacts (`research/cpu_learning/`)

- `research/cpu_learning/evaluation/run_s10_broad_eval.py` — Capstone 5-seed evaluator (`s10-broad-eval-v1`, pure Python 3.11 stdlib, `0` direct imports of `hidden_actions`/`oracle`):
  - Executes whole-tree AST isolation, zero hardcoded domain-literal checks across all 9 learner packages (`frames/`, `ledger/`, `itl/`, `shard/`, `rdc/`, `transfer/`, `grounding/`, `continual/`, `controller/`), `comparison_only=True` neural baseline isolation checks, and Python 3.11 compatibility checks.
  - Cross-verifies bit-exact S0–S10 pre-registered manifests (`--verify-manifests`) and all prior S0–S9 5-seed summaries/gates.
  - Runs live per-seed unified training, `.freeze()` isolation checks (`ControllerFrozenError`, `post_freeze_env_queries == 0`), a fresh 220-item post-freeze temporal stress & 6-category OOD abstention calibration suite (`1,100` total evaluations across 5 seeds), a live 4-workload single-core CPU throughput microbenchmark (`4,000` predictions/seed = `20,000` total), `tracemalloc` peak heap measurement, and transparent single-core TDP-proxy energy estimation (`estimated_tdp_proxy_not_hardware_rapl`).
- `research/cpu_learning/tests/test_s10_broad_evaluation.py` — Independent adversarial capstone pytest suite (`5` test suites; **`81 / 81`** total tests passing across `test_s0`..`test_s10`) covering whole-tree AST/Py3.11 isolation, ECE miscalibration detection, live seed-101 throughput/calibration/memory/energy bounds, saved 5-seed 16/16 gate + 13/13 PDF gate + 11-system matrix + honest verdict checks, bit-exact S0–S10 manifests, and isolated 2-seed `tmp_path` reproducibility.

---

## 2. Adversarial Audit & Stability Findings Resolved Before Capstone Sign-Off

Two real systems bugs uncovered during live 5-seed S10 stress testing were resolved and verified across both `test_s9` and `test_s10`:
1. **Lossless Frontier `scan_cursor` Rewind Under Wall-Clock Deadlines (`research/cpu_learning/controller/curriculum.py`):** In `SelfCurriculumScheduler.select_next_action`, the functional frontier scanner incremented `sstate.scan_cursor` eagerly while assembling a candidate batch and broke out on the `22.0 ms` selection deadline without rewinding `scan_cursor`, which could skip unscored discriminating probes (`list_keep_even`) on contended CPUs. All three frontier scanners (`relational`, `grounding`, `functional`) now collect new indices non-destructively and advance `scan_cursor` strictly past actually scored probes (`r_scored_to` / `g_scored_to` / `f_scored_to`), guaranteeing lossless 100% probe coverage under arbitrary OS/CI scheduling jitter.
2. **`tracemalloc` Profiler Hook Separation (`research/cpu_learning/evaluation/run_s10_broad_eval.py`):** Running `tracemalloc.start()` during Phase A autonomous training slowed pure-Python allocations by ~3x and distorted training CPU/energy measurements. `tracemalloc` is now isolated strictly to post-freeze unified engine construction and the 220-task evaluation phase (and stopped before Phase C throughput), yielding undistorted Phase A training CPU/energy (`20.48 J` at 15W), true inference/index peak heap (`166.6 KB`), and deterministic `100.00% [100.00%, 100.00%]` post-freeze temporal accuracy across all 5 seeds (`mean = 5.942s/seed <= 15.0s`).

---

## 3. Master 13-Gate Quantitative Acceptance Ledger (PDF p. 13 — 5 Fixed Seeds `[101..505]`)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s10_broad_eval --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s10_broad_eval
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s10_research_report.md`](../research/cpu_learning/reports/s10_research_report.md) · [`research/cpu_learning/reports/s10_summary.json`](../research/cpu_learning/reports/s10_summary.json) · [`research/cpu_learning/reports/raw_runs/s10_gate_ledger.csv`](../research/cpu_learning/reports/raw_runs/s10_gate_ledger.csv) · [`research/cpu_learning/reports/raw_runs/s10_baseline_matrix.csv`](../research/cpu_learning/reports/raw_runs/s10_baseline_matrix.csv) · [`research/cpu_learning/data/s10_preregistration.json`](../research/cpu_learning/data/s10_preregistration.json)

| Gate ID | PDF p. 13 Requirement | Mean | Governing 95% CI Bound | Status | Key Measured Evidence |
|---|---|---:|---:|:---:|---|
| `G01_initial_itl_composition` | Depth-4 `>= 95%` & Depth-6 `>= 90%` on unseen entities | `1.0000` | `1.0000` | **PASS** | Depth-4 `100.00%`, Depth-6 `100.00%` |
| `G02_operator_recovery` | Operator precision `>= 90%` & recall `>= 90%` | `1.0000` | `1.0000` | **PASS** | `P = 100.00%`, `R = 100.00%` (`41/41` clauses) |
| `G03_collateral_change` | `< 2%` collateral changes on unrelated facts | `0.0000` | `0.0000` | **PASS** | `0.0000%` (`ci95_high = 0.0000%`) |
| `G04_beat_episode_lookup` | Beat episode lookup by `>= +20pp` with `<= 8` demos/action | `+1.0000` | `+1.0000` | **PASS** | `+100.00pp` over lookup at `6` demos/action |
| `G05_hdc_proposal_value` | SHARD uses fewer proposals & lower CPU time at equal accuracy | `31.91x` | `15.60x` | **PASS** | Recovery `214.7x` fewer props / `31.91x` faster; Planning `27.3x` / `16.74x` |
| `G06_rdc_dream_value` | Verified dream-refutation beats plain MDL on OOD error | `+0.2391` | `+0.2391` | **PASS** | OOD error `23.91% -> 0.00%` (`0` self-certified) |
| `G07_rdc_blame_value` | Interventional blame beats random & uniform blame | `+0.6518` | `+0.6116` | **PASS** | Top-1 `100.0%` (`+65.18pp` vs random, `+100.00pp` vs uniform) |
| `G08_active_learning_value` | Max-disagreement querying uses fewer queries than random | `0.8237` | `0.5583` | **PASS** | S5 `82.37%` fewer queries (`14.0` vs `93.2`); S9 `+60.96pp` at `31` vs `42` |
| `G09_cross_domain_concept_transfer` | Saves bits in 2 source domains + lifts 3rd/4th + rejects 5th | `+0.3636` | `+0.3636` | **PASS** | Saved `1,080`/`1,904` bits; D3 & D4 `+36.36pp`; D5 `0.000%` neg transfer |
| `G10_continual_no_forgetting` | `>= 95%` retention on old domains after 5 continual phases | `0.9943` | `0.9943` | **PASS** | `99.43%` retention (`0.000%` drop) vs `13.27%` naive & `9.05%` neural |
| `G11_byte_language_model` | Lower held-out `bpb` than 3-gram KN, PPM-C order-5, CPU transformer | `2.5649` | `2.5841` | **PASS** | Mirror `2.5649 bpb` vs PPM-C `2.5910`, 3-gram `3.0243`, Transformer `4.3633` |
| `G12_execution_only_code_induction` | `>= 50%` execution-only HumanEval-Micro | `0.9500` | `0.9500` | **PASS** | `95.00%` (`19/20`; `count_vowels` honestly unsolved) |
| `G13_single_core_cpu_throughput` | `>= 1,000` predictions/sec/core on CPU | `1,676.2` | `1,440.95` | **PASS** | Composite `1,676.2 pred/s/core` (`[1,440.9, 1,911.4]`) |

---

## 4. Complete 11-System Reference & Ablation Matrix (PDF p. 15)

| System ID | Stage | Category | Unseen Rel Acc | Depth 4 | Depth 6 | Collateral | Continual Ret | Byte-LM (bpb) | Preds/s/core | Peak RAM (KB) | MDL Bits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `episode_lookup` | `S0` | `memory_baseline` | `0.00%` | `0.00%` | `0.00%` | `0.000%` | — | — | `4,267.6` | `19.6` | `11,581` |
| `structural_knn` | `S0` | `memory_baseline` | `2.98%` | `0.00%` | `0.00%` | `4.567%` | — | — | `719.6` | `168.2` | `30,357` |
| `shard_without_itl` | `S0` | `shard_baseline` | `3.57%` | `0.00%` | `0.00%` | `4.563%` | — | — | `426.5` | `75.3` | `249,262` |
| `itl_without_hdc` | `S4` | `ablation_no_shard_lsh` | `100.00%` | `100.00%` | `100.00%` | `0.000%` | — | — | `54.7` | `931.1` | `4,360` |
| `itl_without_contrasts` | `S2/S3` | `ablation_no_contrast_probes` | `91.30%` | `100.00%` | `100.00%` | `0.000%` | — | — | `4,399.9` | `907.1` | `4,440` |
| `itl_without_rdc` | `S5/S8` | `ablation_no_rdc_or_replay` | `76.09%` | `100.00%` | `100.00%` | `0.341%` | `13.27%` | — | `4,399.9` | `907.1` | `4,720` |
| `rdc_without_dreams` | `S5` | `ablation_mdl_without_dreams` | `76.09%` | `100.00%` | `100.00%` | `0.341%` | `99.43%` | — | `4,380.5` | `931.1` | `4,616` |
| `current_mirror_solver` | `S0` | `existing_mirror` | `0.00%` | `0.00%` | `0.00%` | `0.000%` | — | — | `185.0` | `33.1` | `0` |
| `hand_coded_hidden_oracle` | `S0` | `evaluator_ceiling` | `100.00%` | `100.00%` | `100.00%` | `0.000%` | — | — | `1,609.3` | `23.8` | `2,888` |
| `small_cpu_neural_reference` (`comparison_only=True`) | `S0/S7/S8` | `comparison_only_neural` | `2.74%` | `0.00%` | `0.00%` | `2.644%` | `9.05%` | `4.3633` | `658.8` | `56.5` | `23,019` |
| **`unified_mirror_shard_itl_rdc`** | **`S1–S10`** | **`unified_candidate_architecture`** | **`100.00%`** | **`100.00%`** | **`100.00%`** | **`0.000%`** | **`99.43%`** | **`2.5649`** | **`1,676.2`** | **`166.6`** | **`17,211`** |

---

## 5. Live Single-Core CPU Throughput, Post-Freeze Calibration & Estimated Energy

- **Single-Core CPU Throughput (`4,000` live iterations/seed = `20,000` total, after `tracemalloc.stop()`):**
  - Relational `OperatorCard.apply`: **`4,462.5 pred/sec/core`**
  - Compiled Functional Sandbox (`even_fn`): **`394,647.3 pred/sec/core`**
  - SHARD 10,240-bit LSH Proposal + ITL Verification: **`927.1 pred/sec/core`**
  - Raw UTF-8 `GroundedInstructionExecutor`: **`960.9 pred/sec/core`**
  - **Composite 4-Workload Throughput:** **`1,676.2 pred/sec/core [1,440.9, 1,911.4]`** (passes the PDF p. 13 `>= 1,000 pred/sec/core` future gate on lower 95% CI bound!).
- **Fresh Post-Freeze Temporal Stress & 6-Category OOD Abstention Calibration (`220` items/seed = `1,100` total):**
  - Post-freeze unseen temporal holdout accuracy (`120` in-scope items/seed: relational, guarded Depth-6, UTF-8, 9-digit addition, len 16–22 lists): **`100.00% [100.00%, 100.00%]`** (`0.000%` collateral).
  - Unanswerable OOD abstention recall (`100` items/seed: partial `ObservationMask`, unknown actions, malformed UTF-8, open-world utterances, uncontrasted ambiguous verb, role-type mismatch): **`100.00% [100.00%, 100.00%]`**.
  - In-scope false abstention: **`0.00%`** · Selective accuracy when answered: **`100.00%`** · Brier score / 10-bin ECE: **`0.000000 / 0.000000`** · Post-freeze environment queries: **`0`**.
- **Memory, MDL Economy & Transparent Energy Estimate (`estimated_tdp_proxy_not_hardware_rapl`, `hardware_rapl_available = False`):**
  - Peak traced heap (`tracemalloc`, post-freeze unified engine construction + 220-task evaluation): **`166.6 KB` (`0.163 MB`)**.
  - Unified MDL compression: **`17,211` active library bits** vs **`481,189` raw training stream bits** (`ratio = 0.0358`, **`28.0x` compression**).
  - Analytical TDP-proxy energy estimate (`time.process_time() * declared_single_core_TDP_watts`): Training **`20.48 J` (15W laptop core) / `38.24 J` (28W server core)**; Inference **`11.5504 mJ/pred` (15W) / `21.5608 mJ/pred` (28W)**.

---

## 6. Honest External Baseline Context & GPT-Class Readiness Verdict

### Preserved Pre-Research External Baseline (`scripts/run_five_organ_external_eval.py`)
- **Sanitized MBPP (20 tasks):** `0 / 20 (0.0%)`
- **ROCStories Cloze:** `Below-chance (< 50.0%)`
- **LongMemEval Oracle:** `Weak external long-memory retrieval`
- **Multi-File Architect Self-Checks:** `4 / 4` self-generated sandbox checks (explicitly weaker evidence than independent external tests)

### Verdict: `NARROW_VERIFIED_CPU_LEARNING_PROVEN__OPEN_WORLD_GPT_PARITY_NOT_CLAIMED`
- **What IS Experimentally Proven:** A single unified CPU-native **SHARD + ITL + RDC** architecture (`0` GPU, `0` backprop, `0` pretrained LLMs, `0` dense neural weight matrices) genuinely induces reusable, verifiable, compositional operators from `3–8` examples, accelerates symbolic search by `214.7x` fewer proposals (`31.91x` CPU speedup), corrects its own errors via independent dream-refutation/blame/active-querying without self-certification, transfers to multi-digit/novel-base math (`100%`), long list pipelines (`100%`), Python AST refactorings (`100%`), execution-only HumanEval-Micro (`95%`), and bounded raw UTF-8 grounding (`100%` semantic, `2.5649 bpb` Byte-LM), retains `99.43%` (`0%` forgetting) across 5 continual domains with `+36.36pp` cross-domain 2-shot transfer, runs an autonomous 11-step self-curriculum (`99.59%` in `31` queries, `1.39 ms` interactive `p95` SLA), and serves frozen unified predictions at `1,676.2 pred/sec/core` with `100.0%` calibrated OOD abstention.
- **What Is NOT Yet Achieved (Why GPT-Class Open-World Parity Is NOT Claimed):** Open-world unconstrained natural-language conversation and blind human-rated general fluency; unrestricted general Python synthesis beyond neutral typed DSL combinators (`he_micro_20_count_vowels` remains unsolved; prior open-ended MBPP `0 / 20`); open-domain commonsense narrative benchmarks (`ROCStories Cloze`) and uncurated long-horizon document QA (`LongMemEval`); web-scale open-vocabulary byte language modeling beyond structured grounded domains; and direct hardware RAPL/PMU energy telemetry.
