# 46 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S7 Raw UTF-8 Language Grounding, Compositional Realization & Byte-LM Comparison

This document records the Stage **S7** implementation, 5-seed evaluation across **unlabelled raw-UTF-8 construction grammar induction**, **minimal-contrast ambiguity resolution & epistemic abstention**, **truth-locked plan realization with strictly separated semantic vs fluency metrics**, and **zero-leakage multi-domain UTF-8 byte-level language modeling (`bits_per_byte`)**, adversarial audit findings, and unified MDL accounting for Mirror 7's CPU-native **Language Grounding & Byte-LM** subsystem (`research/cpu_learning/grounding/`).

> **Research Integrity Notice:** Every construction schema, negation marker, 3-valued conditional rule, and sequential clause connector in Stage S7 is induced strictly from paired `(raw_utf8_bytes, before_state, after_state)` episodes by aligning observed transitions through Stage S3/S4 `SHARDProposalEngine` + verified `OperatorCard`s — **ZERO GPU, ZERO pretrained LLMs, ZERO backpropagation in Mirror's learners, ZERO pre-written verb synonym tables, and ZERO hardcoded S0/D2/D3 literals**. Automated AST & lexical auditing across `52` non-evaluator modules (`31` learner, `5` grounding) certifies **0 import violations**, **0 hardcoded domain literals**, and strict isolation of `SmallCPUByteTransformerRef` (`comparison_only == True`, never imported by any Mirror learner). Semantic correctness (`hallucination_rate`) and surface readability (`surface_fluency_score`, `reference_perplexity_bpb`) are reported strictly separately. Stages **S8–S10** (continual learning, autonomy, and broad external comparisons) are NOT claimed here.

---

## 1. Stage S7 Modules Implemented (`research/cpu_learning/`)

- `research/cpu_learning/grounding/__init__.py` — Public Stage S7 grounding, realization, and Byte-LM exports.
- `research/cpu_learning/grounding/byte_aligner.py` — Multi-byte-safe UTF-8 byte-span tokenizer (`s7-byte-aligner-v1`) and unlabelled `(before_state, after_state)` transition aligner using `SHARDProposalEngine` + verified `OperatorCard`s to recover candidate `(action_id, role_bindings)` and exact global UTF-8 byte-span offsets.
- `research/cpu_learning/grounding/construction_learner.py` — `RawByteConstructionLearner` and `GroundedInstructionExecutor` (`s7-construction-learner-v1`):
  - Induces reusable `ConstructionCard`s, negation prefixes, 3-valued epistemic `LearnedConditionRule`s (`OBSERVED_PRESENT` / `OBSERVED_ABSENT` / `UNOBSERVED`), and sequential clause connectors directly from paired raw UTF-8 bytes and state transitions.
  - Preserves all competing parses (`is_ambiguous=True`) via `observe_utterance_in_scene` before contrast and resolves role alignment from minimal before/after contrast demos.
  - Enforces `100%` honest abstention on OOD utterances, malformed UTF-8 byte sequences, and partial `ObservationMask` (`UNOBSERVED` conditional predicates or occluded entities), plus `.freeze()` mutation locking.
- `research/cpu_learning/grounding/plan_realizer.py` — `GroundedPlanRealizer` (`s7-plan-realizer-v1`) generating natural-language explanations conditioned on verified `(before_state, plan_steps, after_state, block_reasons, abstained)` structures, scored by strictly separated semantic accuracy (`semantic_correctness`, `hallucination_rate` under `ObservationMask`) vs surface readability (`surface_fluency_score`, `reference_perplexity_bpb`), alongside `telegraphic_tuple_dump` and `ungated_fluent_assumer` ablations and `.freeze()` / `PlanRealizerFrozenError` protection.
- `research/cpu_learning/grounding/byte_lm.py` — Zero-leakage multi-domain UTF-8 corpus generator (`~29.7 KB` train / `~9.5 KB` held-out test per seed across disjoint entities and phrasings; `s7-byte-lm-v1`) and exact normalized 256-way (`0x00..0xFF`) frozen cross-entropy `evaluate_byte_lm_bpb` comparing `mirror_ctw_construction_byte_lm` (`0` backprop / `0` GPU / `0` dense neural matrices: Context-Tree-Weighting / variable-order Kneser-Ney + construction-skeleton context mixer), `ppm_c_order5_byte_lm`, `byte_3gram_kneser_ney`, comparison-only `small_cpu_byte_transformer_ref` (`comparison_only=True`, analytical backprop verified `< 1e-5` vs finite differences), and `uniform_byte_floor` (`8.0000 bpb`).
- `research/cpu_learning/evaluation/run_s7_experiments.py` — Reproducible 5-seed benchmark runner verifying all **24 Stage S7 acceptance gates**, bit-exact S0–S7 manifests (`--verify-manifests`), AST/lexical zero-leakage & transformer isolation audits, and unified single-stream MDL accounting (`mean = 20.05s/seed <= 30s`).
- `research/cpu_learning/tests/test_s7_grounding_and_byte_lm.py` — Independent adversarial pytest suite (`8` test suites; **`60 / 60`** total tests passing across `test_s0`..`test_s7`) including a **100% non-English synthetic alien world** (Cyrillic/accented multi-byte UTF-8 entities, alien OSV verbs `lumos`/`xylar`, alien negation `zot!`, alien conditional `krel...vax`, alien connector `koda` -> `100%` exact state & global UTF-8 byte spans), conditional+negation symmetry, multi-byte UTF-8 & occluded-entity realizer hallucination checks, 256-way Byte-LM normalization & construction-skeleton ablation (`-0.232 bpb` gain over raw-10 KT alone), and committed + subset-seed gate verification.

---

## 2. Independent Adversarial Audit Findings Resolved Before Sign-Off

Four concrete defects uncovered during independent adversarial review were fixed and regression-tested before Stage S7 sign-off:
1. **Compositional Conditional + Negation Symmetry (`construction_learner.py`):** Previously, `execute_raw_utf8` stripped learned negation prefixes before `parse_atomic_clause` when a conditional guard evaluated `True` (`negated_noop`), but skipped negation-prefix stripping when the conditional guard evaluated `False` (falsely abstaining with `ungrounded_construction_pattern`). `execute_raw_utf8` now strips learned negation prefixes (updating `target_body_offset`) symmetrically across both `True` and `False` conditional branches.
2. **Multi-Byte UTF-8 Entity Regex, Partial-Mask Occlusion & Realizer Freeze Guard (`plan_realizer.py`):** Upgraded `_ENTITY_ID_REGEX` from ASCII `[a-zA-Z]` to Unicode `r"\b[^\W\d_]+_[0-9]+\b"` (catching hallucinated multi-byte UTF-8 IDs such as `élan_99` / `ключ_42`), restricted `world_eids` in `evaluate_realization` by `ObservationMask.is_entity_observed(eid)` (lifting detected hallucinations on `ungated_fluent_assumer` from `46.58%` -> **`48.83%`** while `mirror_truth_locked` remains `0.00%`), added automatic template refresh in `realize()`, and added `.freeze()` / `PlanRealizerFrozenError`.
3. **Pass-1 Forwarding for Blocked Step-2 Multi-Clause Episodes (`construction_learner.py`):** Multi-clause training episodes where clause 1 mutates state and clause 2 is a blocked no-op without `;` (failing the single-clause mention/role count check in Pass 1) are now forwarded to `remaining_for_pass2` so connectors such as `"then"` are induced cleanly.
4. **Subset-Seed Prior Manifest Verification (`run_s7_experiments.py`):** Added `verify_prior_manifests: bool = True` validating committed S0–S6 manifests against `DEFAULT_SEEDS` so subset-seed test runs (`seeds=[101, 202]`) pass cleanly alongside committed 5-seed runs.

---

## 3. Verified 5-Seed Stage S7 Results (`[101, 202, 303, 404, 505]`, `24 / 24` Gate Checks PASS)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s7_experiments --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s7_experiments
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s7_research_report.md`](../research/cpu_learning/reports/s7_research_report.md) · [`research/cpu_learning/reports/s7_summary.json`](../research/cpu_learning/reports/s7_summary.json) · [`research/cpu_learning/reports/raw_runs/s7_splits.csv`](../research/cpu_learning/reports/raw_runs/s7_splits.csv) · [`research/cpu_learning/reports/raw_runs/s7_seed_101.json`](../research/cpu_learning/reports/raw_runs/s7_seed_101.json) · [`research/cpu_learning/data/s7_preregistration.json`](../research/cpu_learning/data/s7_preregistration.json)

### A. Raw-UTF-8 Construction Induction & Compositional Execution (`156` Unseen Held-Out Tasks/Seed = `780` Total)

| Unseen Split (`HELDOUT_POOL` + `TEMPORAL_POOL`) | Tasks/Seed | Mirror Exact (`Mean [95% CI]`) | Mirror Inter (`Mean`) | Global Byte-Span Match | Collateral | Lexical Baseline (`Mean`) |
|---|---:|---:|---:|---:|---:|---:|
| `s7_unseen_entity_substitution` | 20 | **100.00% [100.00%, 100.00%]** | **100.00%** | **100.00%** | **0.000%** | 100.00% |
| `s7_unseen_paraphrase_novel_verbs` (`3` demos/verb) | 20 | **100.00% [100.00%, 100.00%]** | **100.00%** | **100.00%** | **0.000%** | 100.00% |
| `s7_unseen_word_orders_passive_sov` | 24 | **100.00% [100.00%, 100.00%]** | **100.00%** | **100.00%** | **0.000%** | 100.00% |
| `s7_negated_instructions` | 20 | **100.00% [100.00%, 100.00%]** | **100.00%** | **100.00%** | **0.000%** | 0.00% |
| `s7_conditional_clauses` (`if` / `unless` + negated) | 24 | **100.00% [100.00%, 100.00%]** | **100.00%** | **100.00%** | **0.000%** | 0.00% |
| `s7_chained_multistep_depth2` | 16 | **100.00% [100.00%, 100.00%]** | **100.00%** | **100.00%** | **0.000%** | 0.00% |
| `s7_chained_multistep_depth4` | 16 | **100.00% [100.00%, 100.00%]** | **100.00%** | **100.00%** | **0.000%** | 0.00% |
| `s7_chained_multistep_depth6` | 16 | **100.00% [100.00%, 100.00%]** | **100.00%** | **100.00%** | **0.000%** | 0.00% |

- **Overall Compositional Grounding:** **`100.00% ± 0.00%`** (`0.000%` collateral) vs **`41.03%`** task-weighted (`37.50%` unweighted split mean; `0.00%` on all negated, conditional, and chained multi-step splits) for the lexical nearest-episode baseline.
- **Ambiguity Disambiguation & Epistemic Abstention (`5 / 5` Seeds):** Pre-contrast ambiguity preserved **`100.00%`** (`100.00%` honest abstention) -> **`100.00%`** exact after `2` minimal contrast demos vs **`50.00%`** first-guess ablation; OOD utterances, malformed UTF-8 bytes, and partial `ObservationMask` (`UNOBSERVED` facts / occluded entities) achieve **`100.00%`** honest abstention (`0.000%` hallucinated mutations).

### B. Plan-Conditioned Truth-Locked Realizer (Strictly Separated Dual Metrics)

| Realizer Method | Semantic Correctness | Hallucination Rate | Surface Fluency Score | Reference Perplexity (`bpb`) |
|---|---:|---:|---:|---:|
| `mirror_truth_locked` | **100.00%** | **0.00%** | **0.9576** | **2.7659** |
| `telegraphic_tuple_dump` | 100.00% | 0.00% | 0.3073 | 9.5297 |
| `ungated_fluent_assumer` | 37.50% | 48.83% | 0.9158 | 4.2368 |

- Demonstrates empirically why semantic fidelity and surface fluency must never be conflated: `telegraphic_tuple_dump` is `100.00%` semantically faithful yet disfluent (`0.3073` fluency, `9.5297 bpb`), `ungated_fluent_assumer` sounds fluent (`0.9158`) yet hallucinates on **`48.83%`** of states/occlusions (`37.50%` semantic correctness), while `mirror_truth_locked` achieves **`100.00%`** semantic correctness (`0.00%` hallucination) AND **`0.9576`** fluency (`2.7659 bpb`).

### C. Zero-Leakage Multi-Domain UTF-8 Byte-LM `bits_per_byte` Benchmark (`~29.7 KB` Train / `~9.5 KB` Held-Out)

| Model ID | Comparison Only | Uses Backprop | Held-Out `bits_per_byte` (`Mean [95% CI]`) | Model Bits | Train Sec | Eval Sec |
|---|:---:|:---:|---:|---:|---:|---:|
| `mirror_ctw_construction_byte_lm` | `False` | `False` | **2.5649 [2.5457, 2.5841]** | `1,919,059` | `3.087s` | `0.602s` |
| `ppm_c_order5_byte_lm` | `False` | `False` | 2.5910 [2.5747, 2.6074] | `307,614` | `0.261s` | `0.183s` |
| `byte_3gram_kneser_ney` | `False` | `False` | 3.0243 [3.0021, 3.0466] | `40,790` | `0.086s` | `0.063s` |
| `small_cpu_byte_transformer_ref` | `True` | `True` | 4.3633 [4.2305, 4.4961] | `111,680` | `6.242s` | `6.948s` |
| `uniform_byte_floor` | `False` | `False` | 8.0000 [8.0000, 8.0000] | `8` | `0.000s` | `0.015s` |

- **Non-Neural CPU Byte-LM Advantage:** Under exact 256-way probability normalization (`|sum(P) - 1| == 0.0` across `0x00..0xFF`), `mirror_ctw_construction_byte_lm` (`0` backprop) beats `ppm_c_order5_byte_lm` (`2.5649` vs `2.5910 bpb`, winning on all `5 / 5` seeds; `-0.232 bpb` construction-skeleton gain over raw-10 KT alone) and beats both `byte_3gram_kneser_ney` (`3.0243 bpb`) and `small_cpu_byte_transformer_ref` (`4.3633 bpb` despite `2×` longer training time) on strictly non-overlapping 95% CIs.
- **Unified Grounding + Operator Archive MDL (`20.05s` mean/seed <= `30s`):** Compresses `198,693` raw grounding training bits (`11,442` single-stream compressed training corpus bits) into **`9,616` unified induced archive bits** (`0.8408` vs compressed training corpus, `0.0484` = **`20.7×`** vs raw).

---

## 4. Remaining Limitations & Next Stage (S8 — Continual Non-Stationary Learning & Zero-Forgetting Consolidation)

### Remaining Limitations After S7
1. **Batch-Per-Stage Training Regime So Far:** Stages S2–S7 train and freeze learners within stationary stage episodes. Continual multi-phase streams introducing new domains, shifting operator exceptions, contradictory/corrupted episodes, and long-horizon episodic recall without catastrophic forgetting belong to Stage **S8**.
2. **Autonomous Exploration & Broad External Re-Evaluation Deferred:** Closed-loop bounded curiosity/goal selection (**S9**) and end-to-end re-evaluation against external open benchmarks (`MBPP`, `LongMemEval`, `StoryCloze` in **S10**) remain unimplemented until each gate passes.

### Next Stage: Stage S8 — Continual Non-Stationary Learning & Zero-Forgetting Consolidation
- **Goal:** Run the unified SHARD + ITL + RDC + Grounding system across sequential non-stationary phases (relational worlds, novel domains, rule revisions/exceptions, math/list/code operators, and corrupted distractors) under bounded memory/MDL budgets — proving zero catastrophic forgetting on earlier phases, targeted versioned exception splitting on rule drift, and long-horizon episodic/rule retrieval scaling.
- **Acceptance Criteria:** Achieve `>= 98%` backward retention across all prior phases (`0%` catastrophic forgetting vs naive overwrite baselines), clean isolation/quarantine of corrupted episodes, bounded MDL growth, and zero regression on all S0–S7 manifests and test suites across all 5 fixed seeds.
