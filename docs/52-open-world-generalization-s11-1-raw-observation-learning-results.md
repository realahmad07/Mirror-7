# 52 — Stage S11.1 Experimental Results: Raw-Observation Open-World Learning

- **Stage / Protocol:** `S11.1` / `s11.1-raw-observation-learning-v1` (evaluated verbatim against frozen Stage S11.0 preregistration `s11.0-open-world-eval-v1`)
- **Branch:** `research/open-world-s11-1-raw-observation-learning`
- **Honest Verdict:** **`S11_1_PARTIAL_PROGRESS__PREREGISTERED_GATES_NOT_ALL_PASSED`**
- **S11.1 Complete:** **`False` (`1 / 4` preregistered gates passed; `0` gates weakened)**
  - `S11_1_G4` (CPU-Native Governance, Isolation, Freeze, RAM < 256 MB & Vault Integrity): **`PASS` (`7 / 7`)**
  - `S11_1_G3` (LongMemEval Long-Term Memory): **`FAIL` (`3 / 4` checks PASS; missed `unanswerable_abs_recall >= 60%` at `35.71%`)**
  - `S11_1_G2` (SQuAD 2.0 Unseen Articles): **`FAIL` (`4 / 7` checks PASS; missed `unanswerable_abstention_recall >= 70%` at `25.24%` and `overall_f1` margin)**
  - `S11_1_G1` (Balanced COPA Unseen Clusters): **`FAIL` (`1 / 4` checks PASS; `47.61%` vs `48.21%` BM25, `val` mirrored consistency `14.77% < 50%`)**
- **Hard CPU-Native Constraints Verified:** `0` GPU · `0` pretrained LLMs · `0` external AI APIs · `0` neural training · `0` manual domain action lists or event schemas · `0` AST isolation violations (`29` Python files scanned) · `0` Python 3.12+ stdlib incompatibilities · `0` post-freeze memory writes · peak `tracemalloc = 104.35 MB` (`< 256 MB`) · production `.dcrystal` vault SHA-256 bit-identical (`fc5adbee3e6adf640d65b67ebcc31dc65d77d235b71de90dc1abd33795de9fbc`).

---

## 1. Implemented 9-Stage Raw-Observation Learning Architecture (`research/open_world/learner/`)

All 9 stages specified in [`docs/51-open-world-generalization-s11-1-raw-observation-learning-design.md`](51-open-world-generalization-s11-1-raw-observation-learning-design.md) are implemented in pure Python 3.11 stdlib plus Mirror 7's pure-CPU 10,240-bit BSC primitives (`mirror_crystal.quotient_egraph_resonator`):

| Stage | Module | Implemented Mechanisms |
|---:|---|---|
| **1–2** | `learner/evidence_ledger.py` | Incremental $O(N)$ UTF-8 byte-span segmentation (`RawEvidenceUnit`), SHA-256 content hashes, timestamp parsing, and append-only hash-chained SQLite `RawEvidenceLedger` (`BEFORE UPDATE` / `BEFORE DELETE` abort triggers + `.freeze()` lock). |
| **3** | `learner/extraction.py` | Schema-free unsupervised candidate span/event discovery (`StructuralPivotTable`, proper/quoted/date/quantity/noun-chunk spans, footnote stripping, morphological stemming), 10,240-bit BSC context-HV coreference clustering preserving explicit uncertainty (`coref_confidence < 1.0`), and grammatical polarity/modality detection (`AFFIRMED`/`NEGATED`/`UNCERTAIN`, `FACTUAL`/`HYPOTHETICAL`/`PREFERENCE`/`QUERY`). |
| **4** | `learner/temporal_graph.py` | Non-destructive `BiTemporalGraph` resolving explicit & relative event dates (`yesterday`, `N days ago`, `last week`), emitting `SUPERSEDES`, `CONTESTED`, and `CORROBORATES` edges via `TropicalJTMS` without ever overwriting or deleting earlier observations. |
| **5** | `learner/causal_miner.py` | `RawCausalMiner`: multi-scale distributional `doc_cos` + 2-hop sparse PMI + grammatical discourse causal connectives + contrastive asymmetric `CausalLift(A -> B)` with exact Wilson 95% CIs, multi-cluster verification promotion (`HYPOTHESIS_UNVERIFIED` vs `VERIFIED_MULTI_CLUSTER`), and `TwinEGraphDoCalculus` counterfactual checks. |
| **6 & 9** | `learner/capability_inducer.py` | Unified **SHARD + ITL + RDC** loop: `OpenWorldSHARDIndex` (10,240-bit BSC Multi-Table LSH), ITL structural anti-unification into `OpenWorldCapabilityCard`s, RDC MDL compression scoring, cross-cluster promotion gate (`k_min=3`), and `CapabilityVersionStore` with regression replay & automatic poison-update quarantine/rollback. |
| **7** | `learner/raw_observation_learner.py` | `RawObservationOpenWorldLearner(OpenWorldSolver)` with all 6 ablation toggles (`AblationConfig`), validation-calibrated 2-parameter Platt confidence (`ECE`), hard Stage 8 verification veto, and strict `.freeze()` propagation across every internal store. |
| **8** | `learner/verifier.py` | Independent zero-self-certification verifier: raw-span grounding, grammatical answer-type verification (`temporal`/`percentage`/`numeric`), proper-entity & numeric grounding, polarity consistency, domain-agnostic scalar/antonym contradiction checks, local-IDF focal predicate grounding, bi-temporal audit, and executable calendar date-difference & multi-session numeric arithmetic. |

---

## 2. Main Results: Stage S11.1 vs. Frozen Stage S11.0 Baselines (`hidden_eval`)

All metrics were computed by the sealed S11.0 evaluator (`research/open_world/evaluator/hidden_eval.py`) on the exact bit-for-bit verified source-disjoint splits (`PYTHONHASHSEED=0`, total full + 6-ablation wall time `806.9s` on CPU):

### 2.1 Balanced COPA (`502` Source-Disjoint Hidden Items)

| System | Primary Acc (95% CI) | Val Mirr Cons | Cross-Dir Transfer | Evid F1 | Brier | ECE | CPU ms/item | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `existing_mirror_system` (S11.0) | `35.46%` (`50.57%` sel.) | `0.00%` | `35.24%` | `52.59%` | `0.2995` | `0.2143` | `1.39` | `1.2 MB` |
| `retrieval_only_bm25` (S11.0 best) | **`48.21% [44.0, 53.2]`** | `19.46%` | `47.41%` | `74.10%` | `0.2498` | `0.0365` | `0.62` | `0.5 MB` |
| `simple_symbolic_extractor` (S11.0) | `47.21%` | `22.15%` | `48.21%` | `73.61%` | `0.2649` | `0.0888` | `0.38` | `0.5 MB` |
| **`s11_1_full`** | `47.61% [43.2, 51.8]` | `14.77%` | `46.62%` | `73.80%` | `0.2529` | `0.0612` | `6.26` | `104.35 MB` |
| `A3_no_asymmetric_causal_lift` (ablation) | `48.61% [43.8, 52.8]` | `15.44%` | `48.81%` | `74.30%` | `0.2504` | `0.0272` | `4.67` | `0.42 MB` |
| `evaluator_only_oracle` (ceiling) | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `0.0000` | `0.0000` | `0.01` | `< 0.1 MB` |

### 2.2 SQuAD 2.0 (`420` Hidden Items Across `14` Unseen Wikipedia Articles: `210` Ans + `210` Unans)

| System | Answerable F1 | Overall Token-F1 (95% CI) | Evidence F1 | Unans Recall | False Abs | Brier | ECE | CPU ms/item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `existing_mirror_system` (S11.0) | `0.89%` | `49.97%` *(97.4% blanket abstain)* | `2.38%` | `99.05%` | `95.71%` | `0.3600` | `0.3882` | `6.22` |
| `existing_rule_blueprint` (S11.0) | `0.00%` | `49.76%` *(99.8% blanket abstain)* | `0.00%` | `99.52%` | `100.00%` | `0.4519` | `0.4476` | `1.14` |
| `retrieval_only_bm25` (S11.0) | `21.29%` | `22.31%` | `68.44%` | `23.33%` | `19.52%` | `0.3519` | `0.3712` | `1.97` |
| `simple_symbolic_extractor` (S11.0) | `25.24%` | `30.95%` | `66.03%` | `36.67%` | `22.38%` | `0.3435` | `0.3587` | `2.35` |
| **`s11_1_full`** | **`36.10% (+10.85pp)`** | `30.67% [26.2, 34.8]` | **`75.40% (+6.96pp)`** | `25.24%` | **`13.33%`** | **`0.1821`** | **`0.0381`** | `3.76` |
| `evaluator_only_oracle` (ceiling) | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `0.00%` | `0.0000` | `0.0000` | `0.01` |

*Compressed MDL model size on SQuAD 2.0: `10,512 bits` (`6,167 bytes`), `266.14 pred/s/core`, peak `tracemalloc = 1.18 MB` (`161.1 KB` predict).*

### 2.3 LongMemEval (`202` Session-Disjoint Hidden Episodes: `188` Ans + `14` `_abs` Unanswerables, `K=10` Distractors)

| System | Primary Score (95% CI) | Cross-Cat Transfer | Session Evid F1 | `_abs` Recall | False Abs | Brier | ECE | CPU ms/item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `existing_mirror_system` (S11.0) | `2.28%` | `0.00%` | `11.77%` | `14.29%` | `84.57%` | `0.5836` | `0.6845` | `849.80` |
| `existing_rule_blueprint` (S11.0) | `6.93%` *(100% blanket abstain)* | `7.06%` | `0.00%` | `100.00%` | `100.00%` | `0.8195` | `0.7568` | `2.70` |
| `retrieval_only_bm25` (S11.0) | `4.27%` | `3.82%` | `73.33%` | `0.00%` | `0.00%` | `0.5256` | `0.6477` | `46.50` |
| `simple_symbolic_extractor` (S11.0) | `4.11%` | `4.57%` | `66.98%` | `21.43%` | `14.89%` | `0.5798` | `0.6650` | `49.94` |
| **`s11_1_full`** | **`22.16% [16.8, 27.2] (+15.23pp)`** | **`24.99% (3.54x)`** | **`74.95% (+1.62pp)`** | `35.71% (5/14)` | **`10.64%`** | **`0.1508`** | **`0.0616`** | `127.94` |
| `evaluator_only_oracle` (ceiling) | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `0.00%` | `0.0000` | `0.0000` | `0.02` |

**LongMemEval Hidden Category Breakdown (`s11_1_full` vs. S11.0 `retrieval_only_bm25`):**
- `single-session-user` (`n=28`): **`37.14%`** (vs `12.22%`, `3.0×`)
- `temporal-reasoning` (`n=53`): **`31.08%`** (vs `4.90%`, **`6.3×`**)
- `knowledge-update` (`n=32`): **`22.92%`** (vs `0.46%`, **`50×`**)
- `single-session-assistant` (`n=23`): **`19.57%`** (vs `5.80%`, `3.4×`)
- `multi-session` (`n=54`): **`10.19%`** (vs `6.17%`, `1.7×`)
- `single-session-preference` (`n=12`): **`4.58%`** (open-ended generative advice — structurally bounded for extractive spans)
- *Compressed MDL model size: `9,512 bits` (`9,686 bytes`), `7.82 pred/s/core` (**`6.6×` faster** than S11.0 `existing_mirror_system`), peak `tracemalloc = 7.33 MB` (`2.35 MB` predict).*

---

## 3. All Six Controlled Ablations (`A1`–`A6`)

| Ablation | Disabled Mechanism | Measured Hidden-Split Impact vs. `s11_1_full` | Scientific Conclusion |
|---|---|---|---|
| **`A1`** | Evidence-grounding verifier (Stage 8) | SQuAD F1 **`-2.56pp`** (`30.67% → 28.11%`), Unans Rec **`-4.29pp`**, ECE `0.0381 → 0.0552`; LongMemEval `_abs` Recall crashes **`-28.57pp`** (`35.71% → 7.14%`), Primary **`-2.54pp`** (`22.16% → 19.62%`). | **PROVEN ESSENTIAL:** Independent Stage 8 verification drives `80%` of LongMemEval `_abs` detection and SQuAD grounding calibration. |
| **`A2`** | Bi-temporal supersession (Stage 4) | LongMemEval Primary crashes **`-5.89pp`** (`22.16% → 16.27%`), Session Evid F1 crashes **`-10.30pp`** (`74.95% → 64.65%`), and **Cross-Category Transfer is cut in half (`-12.36pp`: `24.99% → 12.63%`)**. | **PROVEN ESSENTIAL:** Non-destructive bi-temporal supersession & event-date resolution are core drivers of long-term memory reasoning. |
| **`A3`** | Asymmetric causal lift (Stage 5) | Balanced COPA Primary **`+1.00pp`** (`47.61% → 48.61%`), Cross-Dir Transfer **`+2.19pp`** (`46.62% → 48.81%`), Val Mirr `14.77% → 15.44%`, ECE `0.0612 → 0.0272`. | **HONEST NEGATIVE RESULT:** On `5.58 MB` of non-narrative training text, directional causal lift adds estimation variance rather than signal. |
| **`A4`** | ITL capability anti-unification (Stage 6) | LongMemEval Primary **`-1.48pp`** (`22.16% → 20.68%`), ECE worsens `0.0616 → 0.0800`; SQuAD ECE doubles `0.0381 → 0.0762`; SHARD candidates scanned explode **`19.54×`** on SQuAD (`9.88 → 193.04`), **`8.65×`** on LongMemEval (`16.36 → 141.51`), **`2.41×`** on COPA (`46.70 → 112.71`). | **PROVEN ESSENTIAL:** Anti-unification prevents memorization bloat and stabilizes cross-cluster calibration. |
| **`A5`** | SHARD 10,240-bit LSH indexing (Stage 6) | Balanced COPA (`361` indexed causal frames/cards): scans **`7.73×` more candidates** (`46.70 → 361.00`) at identical accuracy (`47.61%`). | **PROVEN SCALABLE:** Multi-table BSC LSH cuts candidate scans by `87%` with zero accuracy loss as card counts grow. |
| **`A6`** | Calibrated abstention gate (Stage 7) | SQuAD Overall F1 crashes **`-11.78pp`** (`30.67% → 18.89%`) with **`100%` unanswerable hallucination**; LongMemEval Primary drops **`-1.81pp`** (`22.16% → 20.35%`) with **`100%` `_abs` hallucination**. | **PROVEN ESSENTIAL:** Removing abstention destroys epistemic safety across both benchmarks. |

---

## 4. Unseen-Combination & Stage 9 Rollback Probes (`5 / 5 PASS`)

Executed on completely unseen synthetic combinations outside all 3 benchmark vocabularies (`s11_1_summary.json`):
1. **`unseen_bitemporal_supersession_latest_and_historical` — `PASS`:** Stored `"blue umbrella"` in `"garage"` (`2024-03-01`), then updated to `"attic"` (`2024-04-12`). `BiTemporalGraph` emitted `1` non-destructive `SUPERSEDES` edge, returning `"attic"` for `"currently"` and `"garage"` for `"initially"`.
2. **`unseen_contradictory_observation_marked_contested` — `PASS`:** Two conflicting same-day observations (`2024-06-01`, `"Bern"` vs `"Zurich"`) emitted `1` `CONTESTED` edge with both claims preserved non-destructively.
3. **`unseen_executable_date_arithmetic` — `PASS`:** Resolved relative `"yesterday"` offset (`2024-05-01` → `2024-04-30`) against `question_date = 2024-05-20`, computing exact `"20 days"`.
4. **`unseen_adversarial_antonym_and_entity_swap_abstention` — `PASS`:** Abstained on both scalar antonym flip (`ungrounded_scalar_superlative_modifier`) and unseen entity substitution (`ungrounded_entities:Borealis Submarine`).
5. **`stage9_poison_quarantine_and_non_destructive_rollback` — `PASS`:** Injected poisoned card `poison_card_bad` failed regression replay (`committed_poison=False`), was moved to `quarantined_cards`, and triggered non-destructive rollback (`rollback_count=1`).

---

## 5. Preregistered S11.1 Gate Ledger & Scientific Root-Cause Analysis

| Gate | Verdict | Sub-Checks Passed | Exact Root Cause of Failure (No Gate Weakening) |
|---|---|---:|---|
| **`S11_1_G1`** (Balanced COPA) | **`FAIL`** | `1 / 4` | **Corpus Scale & Mirrored-Pair Structure:** Balanced COPA `train` contains only `698` short sentences (`~7 KB`, `59.2%` `val` vocabulary coverage). Even augmented with all `5.58 MB` of S11.0 training Wikipedia + assistant text (`88.9%` vocab coverage), non-narrative prose is orders of magnitude too small and topically skewed to cover everyday physical/social commonsense causality. Moreover, mirrored COPA pairs share subject nouns (causing correlated same-choice predictions on `~70%` of pairs), and reaching `>= 50%` pair consistency mathematically requires `>= ~71%` single-item accuracy even under independent errors. |
| **`S11_1_G2`** (SQuAD 2.0) | **`FAIL`** | `4 / 7` | **Adversarial Entailment Ceiling Without Semantic Taxonomies:** `s11_1_full` beats every non-oracle S11.0 baseline on Answerable F1 (`+10.85pp`), Evidence F1 (`+6.96pp`), False Abstention (`13.33% <= 30%`), and ECE (`0.0381 <= 0.15`). However, validation ROC audits reveal that human-authored SQuAD 2.0 unanswerables preserve **higher** exact lexical overlap (`73.3%` vs `71.2%`), **higher** entity passage coverage (`95.7%` vs `94.4%`), and **25% higher** exact bigram overlap (`30.2%` vs `24.1%`) than answerables while executing subtle taxonomic/relational swaps (*"wrote about in 1893"* vs *"born in 1893"*, *"Sicily"* vs *"city in Sicily"*). Purely surface/distributional features yield validation ROC AUC `~0.60`, mathematically precluding `unans_rec >= 70%` at `false_abs <= 30%`. |
| **`S11_1_G3`** (LongMemEval) | **`FAIL`** | `3 / 4` | **Subtle False-Premise `_abs` Detection Gap:** `s11_1_full` passes Primary Score (`22.16%`, `+15.23pp`), Cross-Category Transfer (`24.99%`, `3.54×`), and Session Evidence F1 (`74.95%`, `+1.62pp`) with `ECE = 0.0616` (`10×` better than S11.0). Only `_abs` abstention recall (`35.71% = 5/14` vs `>= 60%`) fails: remaining `_abs` episodes query an unrecorded attribute or conjunctive detail inside a session that genuinely discusses the same topic/entity (mirroring the SQuAD 2.0 entailment gap), while `val` contains only `5` `_abs` calibration examples (`60%` val recall). |
| **`S11_1_G4`** (Governance & Isolation) | **`PASS`** | **`7 / 7`** | All CPU-native, zero-LLM, zero-manual-schema, AST isolation (`29` files / `0` violations), freeze lock, `0` post-freeze write, `< 256 MB` peak RAM (`104.35 MB`), and `.dcrystal` vault SHA-256 checks passed. |

---

## 6. Reproducibility & Next Recommendation

### Reproduce Everything
```bash
PYTHONHASHSEED=0 python -m pytest -v research/open_world/tests
PYTHONHASHSEED=0 python -m research.open_world.evaluation.run_s11_1_experiment --offline-smoke
PYTHONHASHSEED=0 python -m research.open_world.evaluation.run_s11_1_experiment --no-download
```
- **Raw Machine-Readable Artifacts:**
  - [`research/open_world/reports/s11_1_summary.json`](../research/open_world/reports/s11_1_summary.json)
  - [`research/open_world/reports/s11_1_ablation_matrix.csv`](../research/open_world/reports/s11_1_ablation_matrix.csv)
  - [`research/open_world/reports/s11_1_research_report.md`](../research/open_world/reports/s11_1_research_report.md)
  - `research/open_world/reports/raw_runs/s11_1_{balanced_copa,squad_v2,longmemeval}.json`

### Concrete Next Recommendation (Stage S11.1b / S11.2 — Do NOT Weakly Declare S11.1 Complete)
Stage S11.1 proved that the 9-stage CPU-native **SHARD + ITL + RDC + Bi-Temporal + Stage 8 Verifier** loop yields large real-world gains on multi-session episodic memory (`+15.23pp`, `3.54×` cross-category transfer, `50×` on knowledge-update) and extractive reading comprehension (`+10.85pp` answerable F1, `+6.96pp` evidence F1, `10×` better ECE), while isolating the **exact two missing capabilities** blocking `G1`, `G2`, and `G3`:
1. **Predicate-Argument & Relational Entailment Verification (targeting `G2` and `G3`'s `_abs` recall):** Replace bag-of-focal-tokens grounding in Stage 8 with unsupervised dependency/predicate-argument role alignment (verifying whether the exact `(agent, verb_frame, object/attribute)` relation asked by the question is actually asserted in the cited span, catching relational swaps like *"wrote about in 1893"* vs *"born in 1893"* and conjunctive missing-attribute `_abs` questions without any pretrained LLM).
2. **License-Verified Open-Domain Raw Narrative & Relational-Corpus Scaling on CPU (targeting `G1` Balanced COPA):** Scale Stage 5 `RawCausalMiner` ingestion from `5.58 MB` of Wikipedia/assistant text to a pinned, license-verified (`CC0`/`Public Domain`/`MIT`) raw narrative + commonsense event corpus (e.g., Project Gutenberg narratives + Wiktionary/WordNet-grade open lexical-hyponymy graphs, strictly verified for zero COPA/SQuAD/LongMemEval test leakage) with streaming disk-backed sparse PMI and contrastive event-frame bundles in 10,240-bit BSC space.