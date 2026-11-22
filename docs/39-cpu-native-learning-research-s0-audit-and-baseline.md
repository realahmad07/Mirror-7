# 39 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S0 Audit & Baseline

This document records the Stage **S0** architecture audit, reusable-component map, conflict analysis, pre-registered hidden relational simulator benchmark, 5-seed baseline measurements, and independent SHARD arithmetic canary reproduction for Mirror 7's unified **SHARD + ITL + RDC** CPU-native learning research program.

> **Research Integrity Notice:** Stage S0 establishes the reproducible hidden-simulator benchmark and pre-learner baselines **only**. No ITL operator induction (S2), SHARD search guidance (S4), RDC consolidation (S5), or GPT-level language capability is claimed or implemented here.

---

## 1. Unified Architecture Contract (SHARD + ITL + RDC)

Rather than building competing subsystems, the research plan assigns each mechanism a single role inside one CPU-native learning loop (with existing QAAC / E-Graph components kept strictly as specialist reasoning tools):

| Mechanism | Job in the Combined System | Prohibited Use |
|---|---|---|
| **SHARD** | Represent situations with CPU-native HDC/bit structures; retrieve related episodes; propose likely role bindings and candidate primitives. | Similarity is not proof; a nearest memory must never be accepted as a learned rule. |
| **ITL** | Infer reusable typed state-transition `OperatorCard`s (roles, preconditions, effects, preserved invariants, exception guards) from before/action/after observations. | Never use compression alone to certify an operator; correctness requires passing held-out interventions. |
| **RDC** | Refute wrong hypotheses via independent simulators/checkers, assign interventional rule blame, select maximum-disagreement queries, and compress already-verified knowledge via description length (`J = L(library) + L(examples|library) + beta*verified_failures + gamma*collateral_changes`). | Self-generated dreams/tests must never certify themselves; compression must never override a failed transition test. |
| **QAAC / E-Graph** | Specialist algebraic equivalence, rewriting, and discrete clause tools (`mirror_crystal/quotient_egraph_resonator.py`). | Do not introduce a fourth competing general learning loop without ablation proof. |

---

## 2. Live Repository Audit (`main` at `f41599b`)

Before creating isolated research code, Mirror 7's live codebase was audited across all eight target layers (`L0`–`L7`):

| Target Layer | Existing Mirror-7 Modules Inspected | Reuse Plan | Identified Conflicts & Missing Mechanisms |
|---|---|---|---|
| **L0 Raw Input** | `mirror_crystal/byte_intent_router.py`, `mirror_crystal/hdc_associative_engine.py` | Reuse UTF-8 hashing patterns; retain exact raw UTF-8 bytes + byte-span offsets in `research/cpu_learning/`. | Existing routers normalize/fold text early and discard exact byte-span offsets required for S7 language grounding. |
| **L1 SHARD (HDC & LSH)** | `mirror_crystal/quotient_egraph_resonator.py` (`D_UNIFIED=10240`, `hv10240_of`, `bind10240`, `BitSlicedBundle10240`, `hamming_sim10240`), `mirror_crystal/hdc_associative_engine.py` (`D=4096`, `FastBitSlicedBundle`, `hv_of`, `hamming_sim`), `mirror_crystal/lsh_rule_store.py` (`LSHRuleIndex`), `mirror_crystal/bit_signatures.py` | **Reused directly in S0:** `quotient_egraph_resonator` powers `SHARDHDCRetrievalBaseline`; `hdc_associative_engine` powers the 5-seed SHARD arithmetic canary. | **Conflict 1:** Two incompatible HDC widths exist (`4096` vs `10240`). **Conflict 2:** `LSHRuleIndex` returns stored string responses as answers; in SHARD+ITL+RDC, similarity may only propose candidates. |
| **L2 State Frames** | `mirror_crystal/universal_representation.py` (`RepresentationGraph`), `mirror_crystal/cortical_active_inference.py` (`BilingualCCGHDCParser`, `Hyperedge5Role`), `mirror_crystal/semantic_frame_vm.py` | Reuse deterministic canonical JSON/SHA-256 discipline; keep `BilingualCCGHDCParser` for parser-only baselines. | **Measured Gap:** `BilingualCCGHDCParser` only exposes `(who, act, targ)` and achieved **22.20%** exact role-parse accuracy on S0 raw instructions (collapsing 3-role `give` and 4-role `move` into a single `targ` string and misparsing `"Have <actor> ..."`). **Missing:** lightweight diffable relational `StateGraph` + `ObservationMask` (distinguishing unobserved facts from false facts). |
| **L3 Episode Ledger** | `mirror_crystal/hippocampal_sdm_memory.py` (`KanervaSDMInt8`, `CompactRoaringBitmap`), `mirror_crystal/long_term_memory.py`, `mirror_crystal/disk_vault.py` | Candidate S4/S8 episodic indexes; production `.dcrystal` vaults (`63,510` primary + `74,741` multi-shard concepts) remain untouched. | **Missing (S1 target):** Append-only before/action/after `EpisodeLedger` with sparse deltas, observation masks, provenance, split IDs, and test-time retrieval auditing. |
| **L4 ITL Operator Learner** | `mirror_crystal/multifile_cegis_architect.py` (`FirstOrderASTAntiUnifier`), `mirror_crystal/conversational_blueprint_engine.py`, `mirror_crystal/capability_synthesizer.py`, `mirror_crystal/native_intelligence/context_reasoner.py` | Reuse anti-unification concepts (Plotkin LGG) as design reference; keep existing UOL/AST synthesizers isolated. | **Missing (S2–S3 target):** No existing module induces relational `OperatorCard`s (role variables, preconditions, add/delete/attribute effects, preserved invariants, guards) from 3–8 demonstrations or runs single-factor contrast probes. `ContextReasoner` covers 6 fixed in-prompt families and abstains (`100%`) on relational state-graph tasks. |
| **L5 RDC Controller** | `mirror_crystal/multifile_cegis_architect.py` (`StitchMDLLibraryLearner`), `mirror_crystal/hippocampal_sdm_memory.py` (`SubmodularMDLRuleDistiller`), `mirror_crystal/quotient_egraph_resonator.py` (`TropicalJTMS`, `TwinEGraphDoCalculus`) | Retain `QuotientEGraph` & `TropicalJTMS` as specialist tools only. | **Missing (S5 target):** No unified RDC controller implementing simulator-backed dream-refutation, leave-one-operator-out interventional blame, maximum-disagreement queries, or reversible MDL consolidation. |
| **L6 Verifiers & Simulators** | `mirror_crystal/verification.py`, `mirror_crystal/grammar_and_sandbox.py` (`ZeroTrustSandboxJudge`), `mirror_eval/generalization/` (`NovelGen` sealed runner), `scripts/run_five_organ_external_eval.py` | Reuse `ZeroTrustSandboxJudge` in S6 code induction and sealed-evaluation patterns. | **Built in S0:** Isolated deterministic hidden relational simulator (`move`, `give`, `open`, `lock`), generator-level splits, retrieval audit, bit-budget audit, and 5-seed statistical scoring. |
| **L7 Planner / Realizer** | `mirror_crystal/uol_graph_planner.py`, `mirror_crystal/native_realizer.py`, `mirror_crystal/ternary_lut_realizer.py`, `mirror7_backend/cognitive_orchestrator.py` | Do not modify production routing/realization during S0–S5. | Production `UnifiedCognitiveOrchestrator` routes to fixed handlers/vaults and pre-baked LUT priors; integration waits until S7–S10 gates pass. |

### Preserved External Evaluation Context

Per research governance rules, prior external evaluation findings from `scripts/run_five_organ_external_eval.py` are preserved explicitly as baseline context and are **not** replaced by internal test counts:
- Useful constrained parsing and symbolic-probe behavior on structured patterns.
- Weak long-memory retrieval on external LongMemEval oracle tasks.
- `0 / 20` sanitized MBPP task passes.
- Below-chance ROCStories Cloze ranking.
- Multi-file architect passed four self-generated sandbox checks (weaker evidence than independent external task tests).

---

## 3. Stage S0 Implementation (`research/cpu_learning/`)

All S0 code lives in the isolated package `research/cpu_learning/` with zero modifications to production Mirror organs or `.dcrystal` vaults:

- `research/cpu_learning/simulator/state_graph.py` — Strictly public representation types (`Entity`, `StateGraph`, `ObservationMask`, `SparseDelta`, `ActionInvocation` with verified UTF-8 byte spans, `TrainingEpisode`, `EvaluationTaskPublic`, `EvaluationTaskHiddenLabel`). Zero hidden action rules.
- `research/cpu_learning/simulator/hidden_actions.py` — **Evaluator-only** deterministic hidden transition semantics for `move`, `give`, `open`, and `lock` with typed preconditions, container open/locked/accessible guards, ownership checks, portability checks, non-commuting pairs, and `HIDDEN_OPERATOR_SCHEMAS`. Enforced by AST static import-isolation tests so no learner-side module can import it.
- `research/cpu_learning/simulator/generators.py` — Deterministic generator-level split suite (`s0-relworld-v1.0`, `s0-temporal-v2.0`) over 5 pre-registered seeds `[101, 202, 303, 404, 505]` with **100% disjoint entity and color pools** across `TRAIN_POOL`, `HELDOUT_POOL`, and `TEMPORAL_POOL`. Generates 6 demos/action (`24` train episodes/seed) and 9 unseen generator splits (`168` unseen tasks/seed = `840` total unseen evaluations) plus 1 exact-memorization retrieval canary (`8` tasks/seed, excluded from learning score).
- `research/cpu_learning/evaluation/metrics.py` — Exact final-state accuracy (abstention never scores as exact match), collateral change rate on unrelated facts/attributes, independent test-time retrieval audit, operator-recovery precision/recall/F1, raw/compressed MDL bit accounting, and exact Student-$t$ 95% confidence intervals ($df = 4, t_{0.975,4} = 2.776445$).
- `research/cpu_learning/evaluation/baselines.py` — 6 isolated baselines (`NoOpIdentityBaseline`, `EpisodeLookupBaseline`, `StructuralKNNBaseline`, `SHARDHDCRetrievalBaseline`, `CurrentMirrorSolverBaseline`, and comparison-only `SmallCPUNeuralReferenceBaseline`) with `.freeze()` enforcement before temporal holdout.
- `research/cpu_learning/evaluation/oracle.py` — Evaluator-only `HiddenOracleBaseline` verifying the 100% accuracy / 0% collateral / 1.00 operator-recovery ceiling.
- `research/cpu_learning/evaluation/shard_canary.py` — Independent 5-seed CPU reproduction of the SHARD HDC arithmetic canary (`comparison`, `carry`, and discontinuous `modular_addition`) with a bit-exact 2's-complement vertical bit-sliced accumulator (`SignedBitSlicedAccumulator`, 0 floats, 0 backprop).
- `research/cpu_learning/evaluation/run_experiments.py` — One-command reproducible S0 runner writing pre-registration manifests, raw per-seed JSON/CSV logs, `summary.json`, and `research_report.md`.
- `research/cpu_learning/tests/test_s0_simulator_and_harness.py` — Automated pytest verification of import isolation, guard semantics, non-commutation, byte spans, pool disjointness, retrieval exclusion, freeze enforcement, bit-exactness, and end-to-end gate verification.

---

## 4. Measured Stage S0 Baseline Results (5 Fixed Seeds: `[101, 202, 303, 404, 505]`)

**Reproduce command:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_experiments
```

**Raw artifacts:** [`research/cpu_learning/reports/summary.json`](../research/cpu_learning/reports/summary.json) · [`research/cpu_learning/reports/research_report.md`](../research/cpu_learning/reports/research_report.md) · [`research/cpu_learning/reports/raw_runs/s0_per_split.csv`](../research/cpu_learning/reports/raw_runs/s0_per_split.csv) · [`research/cpu_learning/data/preregistration.json`](../research/cpu_learning/data/preregistration.json)

### A. Unseen Generator-Split Baseline Matrix (9 Unseen Splits, 168 Tasks/Seed = 840 Total)

| Solver | Category | Unseen Exact Acc (Mean ± 95% CI) | Depth-4 Acc | Depth-6 Acc | Temporal Frozen | Abstain Rate | Collateral Rate | Preds/s (1-core) | Peak Heap (KB) | Model MDL (bits) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hand_coded_hidden_oracle` | `evaluator_ceiling` | **100.00% ± 0.00%** | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 742.6 | 98.2 | 2,256 |
| `noop_identity_floor` | `sanity_floor` | 5.95% ± 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 1,479.0 | 89.9 | 0 |
| `episode_lookup` | `memory_baseline` | **0.00% ± 0.00%** | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 985.3 | 61.1 | 11,483 |
| `structural_knn` | `memory_baseline` | 2.98% ± 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.23% | 408.0 | 233.1 | 29,768 |
| `shard_hdc_retrieval_no_itl` | `shard_baseline` | **3.21% ± 0.40%** | 0.00% | 0.00% | 0.00% | 0.00% | 10.22% | 299.6 | 355.6 | 249,176 |
| `current_mirror_solver` | `existing_mirror` | **0.00% ± 0.00%** | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 141.8 | 641.8 | 0 |
| `small_cpu_neural_reference` | `comparison_only_neural` | 3.57% ± 1.05% | 0.00% | 0.00% | 0.00% | 0.00% | 6.39% | 383.1 | 122.1 | 22,749 |

Key S0 findings:
1. **Retrieval vs. Learning Verified:** On `canary_exact_memorization` (8 exact training copies/seed), `episode_lookup`, `structural_knn`, and `shard_hdc_retrieval_no_itl` achieve `100.00%` raw exact match, and the evaluator flags `100.00%` as retrieval hits (`learning_scored_tasks = 0`), preventing memorization from inflating learning scores.
2. **Memory, Pure HDC Similarity, Current Mirror, and Small Neural Reference All Collapse on Unseen Entities & Compositions:** Every non-oracle baseline achieves `0.00%` exact final-state accuracy on `heldout_comp_depth4`, `heldout_comp_depth6`, and `heldout_temporal_frozen`. Non-abstaining retrieval baselines (`structural_knn`, `shard_hdc_retrieval_no_itl`) also inflict `> 10%` collateral corruption by applying concrete training entity deltas into unseen worlds.
3. **Current Mirror Safety Preserved:** `current_mirror_solver` abstains cleanly (`100.00%` abstention, `0.00%` collateral corruption) rather than hallucinating state changes, while its `BilingualCCGHDCParser` achieves `22.20%` exact role-parse accuracy on the raw simulator instructions.

### B. Independent 5-Seed SHARD HDC Arithmetic Canary (PDF p. 3 Reproduction)

Archived raw log: [`research/cpu_learning/reports/raw_runs/shard_canary_5seed.json`](../research/cpu_learning/reports/raw_runs/shard_canary_5seed.json)

| Task (Unseen Digit Pairs `{0..9}^2`, 60/40 Split) | Classes | Chance | Pure HDC Similarity (Mean ± 95% CI) | SHARD Bit-Perceptron (Mean ± 95% CI) | Learned Readout Bits |
|---|---:|---:|---:|---:|---:|
| `comparison` (`a > b`) | 2 | 50.0% | 71.00% ± 18.18% | **89.00% ± 9.96%** | 8,192 |
| `carry` (`a + b >= 10`) | 2 | 50.0% | 95.00% ± 5.81% | **98.00% ± 3.40%** | 8,192 |
| `modular_addition` (`(a + b) % 10`) — **Key Canary** | 10 | 10.0% | **4.00% ± 1.70%** | **7.00% ± 2.60%** | 40,960 |

This independent CPU reproduction confirms both conclusions from the research handoff:
- Local zero-float CPU bit-updates over graded 4,096-bit HDC codes learn smooth linear/threshold decision boundaries (`comparison`: 89.00%, `carry`: 98.00%), while explicitly storing `num_classes * 4,096` learned readout bits (not literally parameter-free).
- Both pure HDC similarity (`4.00%`) and the local bit-perceptron (`7.00%`) fail completely on discontinuous modular arithmetic `(a + b) % 10` (below 10% chance), establishing the exact canary that S6 ITL rule induction must later solve from examples without hand-coding.

---

## 5. Missing Mechanisms & Stage S1 Acceptance Criteria

S0 passes all 7 pre-registered acceptance checks (`deterministic_seed_replay`, `pool_disjointness_verified`, `hidden_oracle_exact_accuracy_100pct`, `hidden_oracle_collateral_zero`, `hidden_oracle_operator_recovery_f1_100pct`, `episode_lookup_canary_retrieval_audited`, `five_fixed_seeds_present`).

The next stage is **S1 — Typed State-Frame + Append-Only Episode Ledger** (not yet implemented). Its acceptance gate requires:
1. Learner-facing typed state-frame and observation-mask ingestion module receiving ONLY public episode observations (zero access to `hidden_actions.py` or gold schemas).
2. Append-only, provenance-tracked before/action/after `EpisodeLedger` with canonical sparse-delta extraction, observation-mask preservation invariants, disk persistence, and test-time access auditing hooks.
3. Separate parser-only scoring harness measuring raw UTF-8 instruction-to-frame role alignment accuracy independently from operator induction.
4. Full regression pass of all S0 gates and existing Mirror-7 test suites before advancing to S2 (ITL operator induction).
