# 50 — Stage S11.0: Open-World Generalization Evaluation Foundation, Audits & Reproducible Baselines

- **Stage / Protocol:** `S11.0` / `s11.0-open-world-eval-v1` (Evaluation Foundation Only — **no new learning architecture implemented or promoted**)
- **Branch:** `research/open-world-generalization-s11`
- **Foundation Gate:** **`PASS` (`11 / 11` Stage S11.0 foundation checks)**
- **Honest Open-World Verdict:** **`S11_0_EVALUATION_FOUNDATION_ESTABLISHED__OPEN_WORLD_GENERALIZATION_NOT_YET_ACHIEVED`**
- **Hardware / Dependencies:** Pure CPU Python 3.11+ standard library (`0` GPU, `0` pretrained LLMs, `0` external AI APIs, `0` neural-network training, `0` modifications to production `.dcrystal` vaults)
- **Reproduce:**
  ```bash
  PYTHONHASHSEED=0 python -m research.open_world.evaluation.run_s11_0_foundation --verify-manifests
  PYTHONHASHSEED=0 python -m research.open_world.evaluation.run_s11_0_foundation
  PYTHONHASHSEED=0 python -m pytest -v research/open_world/tests
  ```

> **Research Integrity Notice:** Stage S11.0 explicitly does **not** claim open-world generalization. Its purpose is to establish a leak-free, license-verified, source-disjoint open-world evaluation protocol and measure five honest baselines **before** implementing Stage S11.1 (`docs/51-open-world-generalization-s11-1-raw-observation-learning-design.md`). Raw external dataset text is **never** committed or redistributed inside this Apache-2.0 repository; pinned public files are fetched at runtime into gitignored `.external_eval_cache/s11_open_world/` with exact SHA-256 verification.

---

## 1. Audit of Existing Stages S0–S10 (`research/cpu_learning/`) & Open-World Limitations

Programmatic audit artifact: [`research/open_world/reports/s11_0_s0_s10_and_pr43_audit.json`](../research/open_world/reports/s11_0_s0_s10_and_pr43_audit.json)

### 1.1 Verified Strengths Preserved Untouched
- All **11 stages (`S0–S10`)** and all **`13 / 13`** master quantitative gates pass (`81 / 81` pytest tests, `0` edits to `research/cpu_learning/` or its committed manifests).
- Reusable CPU-native building blocks proven in controlled environments:
  - Append-only hash-chained SQLite `EpisodeLedger` with SQL triggers & `.freeze()` write locks (S1)
  - 10,240-bit BSC + multi-table LSH `SHARDProposalEngine` (`214.7x` fewer recovery proposals, `31.91x` speedup) (S4)
  - Domain-agnostic ITL role-unification & invariant/guard induction (S2–S3)
  - RDC refutation, leave-one-out blame, active max-disagreement querying, and MDL compaction with `0` self-certification (S5)
  - Zero-forgetting `ContinualKnowledgeBase` (`99.43%` 5-phase retention, `100%` poison rollback) & anti-forgery `IndependentVerificationReceipt` controller (S8–S9)
  - Mirror Crystal `QuotientEGraph`, `TwinEGraphDoCalculus`, `KanervaSDMInt8`, and `MengerProvenanceVerifier`

### 1.2 Eight Structural Limitations Blocking Open-World Generalization (`L01–L08`)

| ID | Affected Scope | Severity | Structural Limitation Blocking Open-World Transfer |
|---|---|---|---|
| `L01` | S0–S5, S8, S9 | Critical | **Pre-segmented state transitions required:** ITL assumes clean `(before_state, action, after_state)` relational graphs from a simulator; real text/dialogues provide raw unstructured UTF-8 streams. |
| `L02` | S1–S5, S7, S8 | Critical | **Closed predicate & role schemas:** `TypedStateFrame` and `OperatorCard` assume finite discrete predicates/types; even S7 raw-UTF-8 grounding aligns strictly to already-verified discrete operator deltas. |
| `L03` | S3, S5, S9 | Major | **Interactive simulator oracle dependence:** Contrast refinement, dream-refutation, and active querying rely on noise-free `TransitionEnvironment.step()` calls unavailable in passive real-world corpora. |
| `L04` | S6, S10 | Major | **Bounded micro-DSL code gap:** HumanEval-Micro (`19/20`, `count_vowels` unsolved) uses a compact combinator DSL, whereas external open-ended sanitized MBPP remains `0 / 20`. |
| `L05` | S7, S10 | Critical | **Controlled grounding vs. organic language:** S7's `2.5649 bpb` Byte-LM and `100%` UTF-8 execution hold on structured synthetic worlds, not organic causal commonsense (Balanced COPA) or adversarial reading comprehension (SQuAD 2.0). |
| `L06` | S1, S8, Organ 3 | Major | **No bi-temporal belief revision over noisy transcripts:** Neither `EpisodeLedger` nor `KanervaSDMInt8` tracks evolving contradictory user facts (`knowledge-update`), date arithmetic (`temporal-reasoning`), or distractor-heavy multi-session coreference. |
| `L07` | Staged rule/blueprint packs | Governance & coverage | **Unverified legacy rule/blueprint packs:** The 997 staged roadmap blueprints, 110 authored domain seeds, and 26 source-supported rules (tests `not_run`) are unverified generic prompts with near-zero open-world coverage. |
| `L08` | S10 | Measurement | **Analytical TDP energy proxy:** S10 energy numbers are `process_time * TDP` estimates, not RAPL/PMU hardware readings; S11.0 reports raw CPU/wall time, throughput, and `tracemalloc` peak heap directly. |

---

## 2. Inspection & Gate Record of PR #43 (`research-track-a-native-semantic-gate`)

**Governance Verification:** PR #43 (head commit `2d70a161ba74c7dacdd884134ec7ce3267db29b3`) was inspected read-only without merging. Live git ancestry verification confirms **PR #43 is NOT merged into `main` or `research/open-world-generalization-s11`**, and it is **NOT treated as a passing capability**.

- **GitHub Check-Runs at Head (`2d70a16`):** `source-guard = success` · `fast-ci = success` · **`native-brain = failure`** (CI intentionally fails when no predeclared operator configuration passes all 3 seeds on the unchanged semantic gate).

### 2.1 Failed Semantic-Reasoning Gates Recorded Verbatim from PR #43
1. **`PR43_G1` (Exp 6, 60 epochs, 3 seeds):** `FAIL (0/3)` for both MLP (`2.61%` top-1, margin `-0.3131`) and Linear Operator (`2.06%` top-1, margin `-0.3037`).
2. **`PR43_G2` (Exp 6, 240 epochs, pre-correction historical evaluator):** `FAIL (0/3)` for both MLP (`34.96%` top-1, margin `-0.0852`) and Linear Operator (`40.72%` top-1, margin `-0.0649`). Both beat chance (`~0.75–0.81%`) and passed order/no-collapse checks, yet failed the positive held-out candidate-margin gate on all 6 seed runs.
3. **`PR43_G3` (Contrastive weight `0.25` ablation):** `FAIL (0/3)`, rejected (`34.10%` top-1, margin `-0.1024`, worse than `0.10`).
4. **`PR43_G4` (Self-fed rollout supervision weight `1.0`):** `FAIL (0/3)`, rejected (`31.98%` top-1, margin `-0.1112`).
5. **`PR43_G5` (Equivalent-target corrected evaluator reruns across seeds 17/29/43):** `FAIL (0/3)` across ALL 4 tested configurations:
   - MLP `c=0.10`: margins `-0.1182 / -0.0770 / -0.0602` (`31.34% / 35.34% / 38.21%`)
   - Linear `c=0.10`: margins `-0.0863 / -0.0301 / -0.0783` (`35.82% / 48.12% / 38.21%`)
   - Linear `c=0.25`: margins `-0.1215 / -0.0690 / -0.1167` (`28.36% / 40.60% / 33.33%`)
   - Linear `rollout=1.0`: margins `-0.1410 / -0.0733 / -0.1194` (`26.12% / 41.35% / 28.46%`)
6. **`PR43_G6` (Stratified composition-depth failure localization, commit `0ee5859`, Linear `c=0.10`):**
   - Seen concepts (`-0.0625`, `39.7%`) and unseen concepts (`-0.0694`, `40.3%`) both negative; sharpest collapse on **chain length 5 (`-0.1175`, `29.0%`; every seed between `-0.1326` and `-0.1015`)** and **held-out `4→5` suffix (`-0.1314`, `28.3%`)** vs chain length 3 (`+0.0097`, `62.3%`).
7. **`PR43_G7` (`native-brain` CI gate):** `FAIL`.

### 2.2 Unresolved Blockers in PR #43
- **`PR43_U1`:** Low-dose rollout supervision ablations (`0.10` and `0.25`) remained pending with no passing 3-seed gate.
- **`PR43_U2`:** Matched comparison against frozen Experiment 5/7 baselines under identical data, parameter, and compute budgets incomplete.
- **`PR43_U3`:** `CLUTRR` excluded due to `CC-BY-NC 4.0` vs unknown HF license; `RuleTaker` (`Apache-2.0`) pinned explicitly as templated/controlled language only (not organic natural text).
- **`PR43_U4`:** Legacy 10-case V5 smoke prompt export sets `promotion_eligible=false` (source-disjointness from Drive training data unaudited; lexical metrics do not measure semantic correctness).
- **`PR43_U5`:** Independent exact-answer verification and external reproduction incomplete; no checkpoint promoted.

---

## 3. External Dataset & License Audit (`14` Datasets Verified Directly Against Upstream APIs)

Full structured registry: [`research/open_world/reports/s11_0_dataset_license_audit.json`](../research/open_world/reports/s11_0_dataset_license_audit.json)

| Dataset | Role | Verified Upstream License | S11.0 Decision | Pinned Checksum / Handling Policy |
|---|---|---|---|---|
| **Balanced COPA** (`pkavumba/balanced-copa`) | Real-world causal commonsense (`cause`/`effect` + mirrored counterfactuals) | **`CC-BY-4.0`** (ext.) + `BSD-2-Clause` (orig.) | **APPROVED_ACTIVE** | Commit `813bd03c`; `train.csv` (`650ec76d...`) & `test.csv` (`b57415ab...`); runtime fetch only, `0` git redistribution |
| **SQuAD 2.0 Dev** (`rajpurkar.github.io/.../dev-v2.0.json`) | Real-world Wikipedia QA + exact gold spans + ~50% human unanswerables | **`CC-BY-SA-4.0`** | **APPROVED_ACTIVE** | `dev-v2.0.json` (`80a5225e...`, `4,370,528` B); eval-only runtime fetch, `0` git redistribution (ShareAlike respected) |
| **LongMemEval** (`xiaowu0162/longmemeval-cleaned`) | Real-world multi-session long-term memory (6 categories + `*_abs` abstention) | **`MIT`** (both GH SPDX & HF card) | **APPROVED_ACTIVE** | Commit `98d7416c`; `longmemeval_oracle.json` (`821a2034...`, `15.4 MB`); `longmemeval_s_cleaned.json` (`277 MB`) pinned for offline heavy runs |
| RuleTaker (`tasksource/ruletaker`) | Controlled templated deduction | `Apache-2.0` | APPROVED_DEFERRED | Templated synthetic language, not organic open-world text |
| bAbI QA (`facebook/babi_qa`) | Synthetic micro-world QA | `CC-BY-3.0` | APPROVED_DEFERRED | Synthetic templated stories; already probed in five-organ script |
| BoolQ (`google/boolq`) | Wikipedia yes/no QA | `CC-BY-SA-3.0` | APPROVED_DEFERRED | Parquet-only on HF; lacks unanswerables & span offsets vs SQuAD 2.0 |
| AI2 ARC (`allenai/ai2_arc`) | Science commonsense MCQA | `CC-BY-SA-4.0` | APPROVED_DEFERRED | Parquet-only on HF; lacks supporting evidence spans; reserved for S11.2+ |
| HotpotQA (`hotpotqa/hotpot_qa`) | Multi-hop supporting-fact QA | `CC-BY-SA-4.0` / `Apache-2.0` | APPROVED_DEFERRED | Primary CMU JSON DNS failed (`Errno 11001`); HF mirror Parquet-only |
| CommonsenseQA (`tau/commonsense_qa`) | 5-way commonsense MCQA | `MIT` (HF) / ConceptNet `CC-BY-SA-4.0` | APPROVED_DEFERRED | Upstream GH lacks LICENSE file; ConceptNet ShareAlike provenance review pending |
| WinoGrande (`allenai/winogrande`) | Commonsense coreference | `Apache-2.0` (GH; HF tag missing) | APPROVED_DEFERRED | Clean Apache-2.0; deferred until archive checksum pinned |
| **CLUTRR** (`facebookresearch/clutrr`) | Kinship multi-hop | **`CC-BY-NC-4.0`** / HF `unknown` | **REJECTED_LICENSE** | Non-commercial restriction; excluded (matches PR #43) |
| **ROCStories Cloze** (`qikp/rocstories-cloze`) | Narrative ending ranking | **`other`** / Rochester research-only | **REJECTED_LICENSE** | Unspecified/non-redistributable terms; excluded from S11 manifests |
| **LoCoMo** (`snap-research/locomo`) | Long-term dialogue memory | **`NOASSERTION`** (`LICENSE` HTTP 404) | **REJECTED_LICENSE** | Missing verifiable SPDX license; excluded in favor of MIT LongMemEval |
| **HellaSwag** (`rowanz/hellaswag`) | Grounded continuation | **HTTP 451** / HF missing license | **REJECTED_UNAVAILABLE** | Upstream GitHub returned HTTP 451 Unavailable For Legal Reasons |

---

## 4. Source-Disjoint Evaluation Design, Evaluator Isolation & Preregistration

Preregistered protocol manifest (committed prior to any S11.1 implementation): [`research/open_world/data/s11_0_preregistration.json`](../research/open_world/data/s11_0_preregistration.json)

### 4.1 Source-Disjoint Splitting & Real Leakage Hazards Caught Automatically
Every benchmark uses **connected-component cluster splitting** (`research/open_world/datasets/splitter.py`) verified mathematically for **zero item overlap, zero cluster overlap, and zero shared resource overlap** across `train` / `validation` / `hidden_eval`:

1. **Balanced COPA (`balanced_copa` — `698` train / `300` val / `502` hidden_eval):**
   - Union-Find clusters every original + mirrored counterfactual pair by shared `(choice1, choice2)` AND shared premise text.
   - **Real data-hygiene catch:** Our connected-component audit automatically discovered **2 rows in official `train.csv`** sharing an identical premise with rows in official `test.csv`, and quarantined them into the sealed `hidden_eval` component (`502` hidden items, `498` clusters; `0` train/val/hidden resource overlap!).
2. **SQuAD 2.0 (`squad_v2` — `420` train / `210` val / `420` hidden_eval):**
   - Partitioned strictly by **Wikipedia article title** (`14` train articles / `7` validation articles / `14` completely unseen hidden-evaluation articles; balanced `15` answerable + `15` human-authored unanswerable questions per article = `210` answerable + `210` unanswerable in `hidden_eval`).
   - Exact sentence-level gold evidence spans derived from `answer_start` character offsets (strictly sealed in `HiddenGoldRecord`).
3. **LongMemEval (`longmemeval` — `199` train / `99` val / `202` hidden_eval):**
   - **Real data-hygiene catch:** Empirical inspection found **8 session IDs shared across multiple questions** among the 940 unique sessions in `longmemeval_oracle.json`. Union-Find connected-component clustering over `haystack_session_ids` eliminated cross-split session leakage (`0` session-ID overlap across splits!).
   - **Distractor-Haystack Augmentation (Fixing Raw-Oracle Triviality):** Because `176 / 500` items in raw `longmemeval_oracle.json` contain only 1 session, each episode deterministically interleaves up to **`K = 10` distractor sessions drawn strictly from OTHER connected components inside the SAME split**, ordered chronologically by `session_date` (`~11–16` multi-turn sessions per hidden episode; `188` answerable + `14` genuine `*_abs` unanswerables; `0` cross-split session contamination).

### 4.2 Strict Evaluator / Learner Isolation & Freeze Locks
- `OpenWorldObservation` and `ConversationTurn` (`research/open_world/contracts.py`) contain **zero** gold answers, choice labels, `is_impossible`, `has_answer`, `answer_session_ids`, or `mirrored` fields.
- `HiddenGoldRecord` (`evaluator/hidden_eval.py`) and `EvaluatorOnlyOracle` (`evaluator/oracle.py`) live exclusively inside `research/open_world/evaluator/`.
- Whole-tree AST audit (`audit_s11_isolation_and_constraints`) verifies **`0` forbidden imports** in `baselines/`, **`0` ML/LLM/GPU imports**, **`0` Python 3.12+ stdlib calls**, **`0` post-freeze memory writes** (`SolverFrozenError` verified on all 5 baselines), and **bit-identical SHA-256** of `restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal` before and after evaluation.

---

## 5. Reproducible Raw Baseline Results (All 5 Baselines × 3 Real-World Benchmarks × 8 Metric Families)

Raw artifacts: [`s11_0_summary.json`](../research/open_world/reports/s11_0_summary.json) · [`s11_0_baseline_matrix.csv`](../research/open_world/reports/s11_0_baseline_matrix.csv) · [`s11_0_balanced_copa.json`](../research/open_world/reports/raw_runs/s11_0_balanced_copa.json) · [`s11_0_squad_v2.json`](../research/open_world/reports/raw_runs/s11_0_squad_v2.json) · [`s11_0_longmemeval.json`](../research/open_world/reports/raw_runs/s11_0_longmemeval.json)

### 5.1 Task 1 — Balanced COPA (Real-World Causal Commonsense, `502` Source-Disjoint Hidden Items)

| Baseline | Primary Acc (95% CI) | Selective Acc | Val Mirrored Pair Cons. | Cross-Rel Transfer | Evid F1 | False Abstain | Brier / ECE | MDL Bits | CPU ms/item | Peak RAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `existing_mirror_system` | `35.46%` `[31.5, 39.8]` | `50.57%` | `0.00%` | `35.45%` | `72.71%` | `29.88%` | `0.2588 / 0.1915` | `744` | `1.401` | `0.347` |
| `retrieval_only_bm25` | `48.21%` `[44.0, 53.2]` | `48.21%` | `19.46%` | `49.01%` | `74.10%` | `0.00%` | `0.2915 / 0.0890` | `74,856` | `3.299` | `0.458` |
| `existing_rule_blueprint` | `0.00%` `[0.0, 0.0]` | `—` | `0.00%` | `0.00%` | `0.00%` | `100.00%` | `0.4225 / 0.6500` | `21,784` | `0.156` | `0.015` |
| `simple_symbolic_extractor` | `47.21%` `[43.4, 52.2]` | `47.21%` | `22.15%` | `48.41%` | `73.61%` | `0.00%` | `0.2909 / 0.0913` | `545,216` | `0.311` | `1.671` |
| `evaluator_only_oracle` *(isolated)* | `100.00%` `[100, 100]` | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `0.00%` | `0.0000 / 0.0000` | `0` | `0.023` | `0.011` |

### 5.2 Task 2 — SQuAD 2.0 (Real-World Evidence-Grounded QA & Unanswerable Abstention, `420` Hidden Items Across `14` Unseen Wikipedia Articles)

| Baseline | Overall Token-F1 | **Answerable F1** (`n=210`) | **Unans Abstain Rec** (`n=210`) | False Abstain (Ans) | Evid F1 | Unsupp Ans Rate | Brier / ECE | MDL Bits | CPU ms/item | Peak RAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `existing_mirror_system` | `49.97%`* | **`0.89%`** | `99.05%`* | `95.71%` | `3.97%` | `18.18%` | `0.2511 / 0.0037` | `744` | `2.902` | `0.553` |
| `retrieval_only_bm25` | `22.31%` | `21.29%` | `23.33%` | `19.52%` | **`68.44%`** | `55.15%` | `0.3867 / 0.4565` | `276,912` | `0.632` | `0.969` |
| `existing_rule_blueprint` | `49.76%`* | **`0.00%`** | `99.52%`* | `100.00%` | `0.00%` | `100.00%` | `0.2726 / 0.1518` | `21,784` | `0.186` | `0.015` |
| `simple_symbolic_extractor` | **`30.95%`** | **`25.24%`** | **`36.67%`** | `23.33%` | `66.03%` | `51.70%` | `0.3864 / 0.3882` | `276,848` | `2.530` | `0.961` |
| `evaluator_only_oracle` *(isolated)* | `100.00%` | `100.00%` | `100.00%` | `0.00%` | `100.00%` | `0.00%` | `0.0000 / 0.0000` | `0` | `0.037` | `0.011` |

*\*Note: `existing_mirror_system` (`97.38%` overall abstention) and `existing_rule_blueprint` (`99.76%` overall abstention) reach `~50%` overall SQuAD 2.0 score solely by blanket abstention on the 50% unanswerable split while scoring `0.89%` and `0.00%` on answerable items! Accordingly, our preregistered S11.1 gate `S11_1_G2` requires beating the best baseline on **both** answerable Token-F1 AND overall Token-F1 with false abstention `<= 30%`.*

### 5.3 Task 3 — LongMemEval (Real-World Multi-Session Long-Term Memory, `202` Session-Disjoint Hidden Episodes + `K=10` Chronological Distractors)

| Baseline | Primary Score | Multi-Sess | Know-Update | Temporal | Cross-Cat Transfer | **Session Evid F1** (Any-Hit) | **Unans `_abs` Rec** (`n=14`) | False Abstain | Brier / ECE | MDL Bits | CPU ms/item | Peak RAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `existing_mirror_system` | `2.28%` | `0.41%` | `0.16%` | `3.28%` | `2.11%` | `11.77%` (`15.96%`) | `0.00%` | `18.62%` | `0.2691 / 0.5074` | `744` | **`849.78`** | `1.090` |
| `retrieval_only_bm25` | `4.27%` | `0.92%` | `0.43%` | `3.64%` | `2.29%` | **`73.33%`** (`89.36%`) | `0.00%` | `6.91%` | `0.6021 / 0.7568` | `1,120,784` | `66.45` | `4.608` |
| `existing_rule_blueprint` | `6.93%`* | `9.26%`* | `9.38%`* | `5.66%`* | `7.06%`* | `0.00%` (`0.00%`) | `100.00%`* | `100.00%` | `0.4017 / 0.5807` | `21,784` | `0.31` | `0.018` |
| `simple_symbolic_extractor` | `4.11%` | `6.17%` | `0.46%` | `4.90%` | `3.34%` | `66.98%` (`78.72%`) | **`21.43%`** | `17.55%` | `0.4595 / 0.6477` | `1,120,760` | `105.51` | `3.828` |
| `evaluator_only_oracle` *(isolated)* | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `100.00%` (`100.00%`) | `100.00%` | `0.00%` | `0.0000 / 0.0000` | `0` | `0.01` | `0.011` |

*\*Note: `existing_rule_blueprint` achieves `6.93%` (`14 / 202`) exclusively via `100.00%` blanket abstention matching the 14 `_abs` items (`0.00%` answerable score, `0.00%` evidence F1).*

---

## 6. Failure Analysis & Architectural Diagnostics

1. **Why All Non-Oracle Baselines Fail Causal Commonsense (`Balanced COPA <= 48.2%`):**
   - Balanced COPA's mirrored counterfactual pairs deliberately neutralize bag-of-words shortcuts (`retrieval_only_bm25` mirrored consistency = `19.46%`, below `25%` independent random chance!).
   - Mirror's shipped `LTAGKneserNeyRealizer.pdcst` was trained on a tiny 8-sentence prior, producing `29.88%` exact ties and coin-flip `50.57%` selective accuracy.
   - Surface unigram PMI (`simple_symbolic_extractor`) is too sparse across 698 train items (`14,844` raw pairs, `545 KB` MDL bits) to transfer to unseen causal event vocabulary.
2. **Why Pure Lexical Retrieval Fails Open-World Abstention (`SQuAD 2.0`: `76.67%` Unanswerable Hallucination):**
   - `retrieval_only_bm25` locates the right Wikipedia sentence on `70.48%` of answerable questions (`68.44%` evidence F1), yet hallucinates on **`76.67%`** of human-authored unanswerable questions because SQuAD 2.0 unanswerables preserve high keyword overlap while altering entities, dates, or polarity.
   - Adding simple WH-type + capitalized-entity + number + negation verification (`simple_symbolic_extractor`) immediately boosts answerable F1 (`21.29% → 25.24%`) and cuts unanswerable hallucination by `-13.33pp` (`76.67% → 63.33%`), proving that **structural entity/polarity verification** — not similarity alone — is the core bottleneck.
3. **Why Mirror's Shipped `KanervaSDMInt8` Struggles on Multi-Session Memory (`11.77%` Evidence F1, `849.8 ms/item`):**
   - **Latency bottleneck discovered empirically:** `FlyHash.text_to_input_vec` (`mirror_crystal/hippocampal_sdm_memory.py`) executes `3 * len(bytes)` pure-Python `hashlib.blake2b` calls per write. Even when bounded to a 2,048-character window (`1024` head + `1024` tail), per-episode SDM construction + 12 session writes took `849.8 ms/item` (`1.18 pred/s`), whereas unconstrained full-transcript hashing (`~25 KB/session`) was over 10× slower.
   - **Signal washout:** Compressing multi-turn multi-topic sessions into a single 64-dim character n-gram FlyHash SDR washes out specific query entities, yielding only `15.96%` session hit vs BM25's `89.36%`.
4. **The Session-Retrieval vs. Answer-Extraction Chasm on `LongMemEval` (`89.36%` Session Hit vs `4.27%` Answer Accuracy):**
   - `retrieval_only_bm25` retrieves the right session on **`89.36%`** of answerable episodes (`73.33%` session F1), yet reaches only **`4.27%`** answer score (`0.43%` on `knowledge-update`, `0.92%` on `multi-session`, `0.00%` on `_abs`).
   - Why? Because LongMemEval answers require (a) isolating the exact user/assistant fact turn inside a 15–30 turn dialogue, (b) resolving temporal ordering & date arithmetic (`temporal-reasoning`), (c) tracking superseded facts across weeks (`knowledge-update`), (d) aggregating scattered facts across multiple sessions (`multi-session`), and (e) abstaining when a queried entity is never mentioned (`_abs`).
5. **Existing Staged Rules & Blueprints (`1,133` Unverified Entries) Have `0.0%` Open-World Grounded Coverage:**
   - Across `1,124` unseen open-world hidden items (`502` COPA + `420` SQuAD 2.0 + `202` LongMemEval), `existing_rule_blueprint` abstained on `99.91%` of items (`0.00%` answerable score, `0.00%` evidence F1), confirming that static staged textbook prompts cannot substitute for open-world raw-observation learning.

---

## 7. Next Step: Stage S11.1 Raw-Observation Learning Proposal

With the S11.0 evaluation protocol preregistered and all five baselines measured reproducibly, the next stage is specified in **[`docs/51-open-world-generalization-s11-1-raw-observation-learning-design.md`](./51-open-world-generalization-s11-1-raw-observation-learning-design.md)** and governed by the four preregistered quantitative promotion gates in `research/open_world/data/s11_0_preregistration.json`.
