# 43 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S4 SHARD-Guided Proposal Generation

This document records the Stage **S4** implementation, 5-seed head-to-head proposal & planning benchmarks on a scaled **48-operator** verified library (`4` S3-induced + `44` deterministic distractors), bind-depth geometry stress probes (Depths `1–6`), bit-budget accounting, adversarial audit findings, and the formal **`KEEP_SHARD_AS_PROPOSER`** decision for Mirror 7's CPU-native **SHARD-Guided Proposal Generation** subsystem (`research/cpu_learning/shard/`).

> **Research Integrity Notice (Similarity Is NOT Proof):** Stage S4 reuses Mirror 7's CPU-native 10,240-bit Binary Spatter Code (BSC) hypervector primitives (`mirror_crystal.quotient_egraph_resonator`: `D_UNIFIED`, `BitSlicedBundle10240`, `bind10240`, `permute10240`, `hamming_sim10240`, `hv10240_of`) strictly to rank candidate operators, role bindings, and episodic warm-starts — **ZERO GPU, ZERO backprop, ZERO dense neural nets**. ITL exact verification (`OperatorCard.apply()` + exact state/delta equality) ALWAYS decides acceptance. Automated AST auditing certifies **0 import violations** across `31` non-evaluator files (`16` learner, `7` ITL, `4` SHARD) and **0 hardcoded action or predicate literals** anywhere inside `research/cpu_learning/itl/` or `research/cpu_learning/shard/`. Stages **S5–S10** (RDC consolidation, math/code transfer, language grounding, continual learning, autonomy, and broad external comparisons) are NOT claimed here.

---

## 1. Stage S4 Modules Implemented (`research/cpu_learning/`)

- `research/cpu_learning/shard/__init__.py` — Public Stage S4 exports.
- `research/cpu_learning/shard/encoders.py` — Domain-agnostic 10,240-bit BSC hypervector encoders (`s4-shard-bsc10240-v1`) with dynamically sized exact-majority bit-plane bundling (`_majority_bundle10240`):
  - `encode_typed_state_hv(state)` — structural entity-type, predicate, and relation bindings.
  - `encode_sparse_delta_hv(delta, before_state)` and `encode_operator_card_hv(card)` — entity-ID-invariant structural delta and operator effect/role-type signatures mapped into the **same 10,240-bit subspace** so observed transitions match candidate operators directly via `POPCNT` Hamming similarity.
  - `encode_role_bindings_hv`, `encode_delta_entity_slots_hv`, and `unbind_role_slot_hv` — XOR binding and role-slot unbinding/ranking.
  - `encode_sequence_hv` and `unbind_step_from_sequence` — role-permuted sequence bundling and clean-up decoding for Depth `1–6` bind-depth geometry stress probing.
- `research/cpu_learning/shard/episodic_index.py` — Multi-table LSH bit-projection + exact `POPCNT` Hamming rerank indexes (`s4-shard-lsh-index-v1`): `SHARDOperatorIndex` (over verified `OperatorCard` libraries) and `SHARDEpisodicIndex` (ingesting S1's `EpisodeLedger` via `index_from_ledger` with `0` subsequent ledger reads), tracking `memory_accesses`, `bucket_candidates_scanned`, `index_bits`, and enforcing `.freeze()` (`IndexFrozenError` on post-freeze insertions).
- `research/cpu_learning/shard/proposer.py` — `SHARDProposalEngine` (`s4-shard-proposer-v1`) implementing warm-started candidate proposal across three benchmarks:
  - **Task A (Unlabelled Transition Operator + Role-Slot Recovery):** Given `(before_state, observed_delta)` without `action_id` or `roles`, ranks `(OperatorCard, role_binding)` candidates via SHARD LSH/Hamming + slot unbinding and stops on the first ITL-verified exact match (`card.apply(before, roles).delta == observed_delta`).
  - **Task B (Multi-Step Goal-Directed Plan Search, Depths 2 & 4 + Temporal Frozen):** Given `(initial_state, target_final_state)` without the action sequence, ranks operators via HDC goal-residual similarity (`hamming_sim10240(hv(current_to_goal_delta), hv(card))`) + LSH episodic warm-start under strict `OperatorCard.apply()` verification.
  - **Task C (Bind-Depth Geometry Stress Probe):** `run_bind_depth_geometry_probe` contrasting role-permuted clean-up decoding against naive unpermuted superposition across Depths `1–6` (`codebook=64`).
- `research/cpu_learning/evaluation/run_s4_experiments.py` — Reproducible 5-seed benchmark runner (`0` direct imports of `hidden_actions.py` / `oracle.py`) verifying all **20 Stage S4 acceptance gates** on exact Student-$t$ 95% CI bounds, bit-exact S0/S1/S2/S3/S4 manifest checks (`--verify-manifests`), AST isolation + zero-hardcoding audits, bit-budget accounting, and the formal `KEEP_SHARD_AS_PROPOSER` decision.
- `research/cpu_learning/tests/test_s4_shard_proposals.py` — Independent adversarial pytest suite (`7` test suites; **`36 / 36`** total tests passing across `test_s0`..`test_s4`) covering AST isolation, expanded 23-word domain-literal scanning across `shard/`, bit-exact majority and overflow regression up to `N=512`, entity-ID invariance, XOR slot unbinding, LSH full-scan fallback re-sorting regression, on-disk S1 `EpisodeLedger` indexing with `0` subsequent reads & freeze enforcement, a **100% novel 36-operator synthetic arcane domain** (recovery, planning, canceling-enabler fallback, unsolvable-delta rejection), no-episodic / distractor-target / unguided-best-first / similarity-not-proof ablations, Task C Depths `1–6`, and committed + isolated gate runs.

---

## 2. Independent Adversarial Audit Findings Resolved Before Sign-Off

Two reproducible bugs inside `research/cpu_learning/shard/` were uncovered during independent adversarial testing and fixed + regression-tested before Stage S4 sign-off:
1. **LSH Full-Scan Fallback & Combined Re-Sort (`episodic_index.py`, `_MultiTableLSHCore.query_ranked_indices`):** Previously, when `include_fallback=True`, the fallback scan skipped unbucketed vectors whenever `>= top_k` low-similarity decoys hit LSH buckets, and appended fallback scores without re-sorting. Now whenever `include_fallback=True` and `len(candidate_indices) < n_total`, all missing vectors are unconditionally scored, merged into `scored`, and re-sorted by `(-sim, tie_key)` before slicing `[:top_k]` (verified against 6 bucket-colliding decoys at `sim=-10,208` + 1 unbucketed near-exact vector at `sim=+10,208`, which now ranks #1).
2. **Bit-Plane Wraparound Overflow Prevention (`encoders.py`, `_majority_bundle10240`):** Previously, fixed `BitSlicedBundle10240(bits=8)` silently wrapped modulo `2**8` once `total_w * 2 + 3 >= 256` (inverting `5,086 / 10,240` bits at `N=128` copies). `_majority_bundle10240` now dynamically sizes `needed_bits = max(int(bits), (total_w * 2 + 3).bit_length())` before instantiation, verified with `0` inverted bits across `N = 128, 256, 512, 2048`.

---

## 3. Verified 5-Seed Stage S4 Results (`[101, 202, 303, 404, 505]`, `20 / 20` Gate Checks PASS)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s4_experiments --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s4_experiments
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s4_research_report.md`](../research/cpu_learning/reports/s4_research_report.md) · [`research/cpu_learning/reports/s4_summary.json`](../research/cpu_learning/reports/s4_summary.json) · [`research/cpu_learning/reports/raw_runs/s4_per_split.csv`](../research/cpu_learning/reports/raw_runs/s4_per_split.csv) · [`research/cpu_learning/reports/raw_runs/s4_seed_101.json`](../research/cpu_learning/reports/raw_runs/s4_seed_101.json) · [`research/cpu_learning/data/s4_preregistration.json`](../research/cpu_learning/data/s4_preregistration.json)

### A. Task A — Unlabelled Transition Operator & Role-Slot Recovery (`48` Operators, `280` Unseen Tasks)

| Method | Exact Acc (Mean ± 95% CI) | Mean Proposals (Mean ± 95% CI) | CPU Time/Seed (ms) | Collateral |
|---|---:|---:|---:|---:|
| `shard_guided_proposer` | **100.00% ± 0.00%** | **1.00 ± 0.00** | **35.14** | **0.000%** |
| `unguided_exhaustive_proposer` | 100.00% ± 0.00% | 214.66 ± 0.12 | 1,085.62 | 0.000% |
| `random_proposer` | 100.00% ± 0.00% | 248.24 ± 24.86 | 1,246.31 | 0.000% |
| `shard_retrieval_only_no_itl` | 0.00% ± 0.00% | 1.00 ± 0.00 | 12.07 | 2.565% |

- **vs Unguided Exhaustive:** **`99.53%`** fewer proposals (**`214.7×`**, 95% CI low `214.5×`) and **`31.91×`** real single-core CPU speedup (95% CI low `26.64×`).
- **vs Random Proposer:** **`99.59%`** fewer proposals (**`248.2×`**, 95% CI low `223.4×`) and **`37.81×`** real single-core CPU speedup (95% CI low `28.54×`).
- **vs Similarity-Only Retrieval (`shard_retrieval_only_no_itl`):** **`+100.00pp`** exact accuracy (`100.00%` vs `0.00%`, eliminating `2.565%` collateral corruption), confirming experimentally that similarity is NOT proof without ITL verification.

### B. Task B — Multi-Step Goal-Directed Plan Search (Depths `2` & `4` + Temporal Frozen, `48` Operators, `120` Unseen Tasks, Budget `600` Proposals)

| Planner | Overall Goal Acc (Mean ± 95% CI) | Depth-2 Acc | Depth-4 Acc | Temporal Acc | Mean Proposals | CPU Time/Seed (ms) | Collateral |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shard_guided_planner` | **100.00% ± 0.00%** | **100.00%** | **100.00%** | **100.00%** | **22.00** | **154.59** | **0.000%** |
| `unguided_bfs_planner` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 600.00 | 2,567.70 | 0.000% |
| `random_walk_planner` | 0.00% ± 0.00% | 0.00% | 0.00% | 0.00% | 600.00 | 2,961.76 | 0.000% |

- **vs Unguided BFS:** **`96.33%`** fewer proposals (**`27.3×`**, 95% CI low `25.6×`) and **`16.74×`** real single-core CPU speedup (95% CI low `15.60×`), while solving `100.00%` of Depth-2, Depth-4, and Frozen Temporal goals where unguided BFS times out at `0.00%`.
- **vs Random Walk:** **`96.33%`** fewer proposals (**`27.3×`**, 95% CI low `25.6×`) and **`19.56×`** real single-core CPU speedup (95% CI low `15.81×`).

### C. Task C — Bind-Depth Geometry Stress Probe (Depths `1–6`, `codebook=64`, `25` Trials/Depth/Seed)

| Depth | Role-Permuted Clean-Up Acc (Mean ± 95% CI) | Permuted Mean Sim (`/10,240`) | Naive Superposition Acc | Naive Mean Sim (`/10,240`) |
|---:|---:|---:|---:|---:|
| 1 | **100.00% ± 0.00%** | 10,240.0 | 100.00% | 10,240.0 |
| 2 | **100.00% ± 0.00%** | 5,127.4 | 50.80% | 5,252.1 |
| 3 | **100.00% ± 0.00%** | 5,116.4 | 34.67% | 5,415.5 |
| 4 | **100.00% ± 0.00%** | 3,840.9 | 26.40% | 4,150.7 |
| 5 | **100.00% ± 0.00%** | 3,842.8 | 22.40% | 4,396.9 |
| 6 | **100.00% ± 0.00%** | 3,199.7 | 20.27% | 3,955.5 |

### D. Bit-Budget Accounting & Formal Keep/Remove Decision (`per_seed_mean = 8.57s`, `max = 11.72s < 30s`)
- **`raw_training_data_bits`:** `704,301` (`compressed_training_data_bits`: `52,427`)
- **`operator_library_bits` (48 cards raw):** `199,528` (compressed operator MDL: `12,104` bits)
- **`active_bundle_bits` (48 × 10,240-bit BSC + 12-table LSH index):** `509,952` bits
- **`episode_store_bits` (SHARD episodic LSH index):** `333,824` bits
- **`compressed_model_mdl_bits` (operators + BSC index):** `503,624` bits (`MDL / Raw Train = 0.7151`)
- **Decision (`KEEP_SHARD_AS_PROPOSER`):** Because SHARD cuts both search proposals (`214.7×` on Task A, `27.3×` on Task B) and real single-core CPU runtime (`31.91×` on Task A, `16.74×` on Task B) by more than an order of magnitude at `100.00%` exact ITL-verified accuracy and `0.000%` collateral, SHARD is formally retained strictly in the **proposal** role inside the unified SHARD + ITL + RDC loop.

---

## 4. Remaining Limitations & Next Stage (S5 — Refutation-Driven Consolidation (RDC))

### Remaining Limitations After S4
1. **No Unified RDC Consolidation Controller Yet:** Leave-one-operator-out multi-step interventional blame, dream-refutation probe synthesis backed by independent execution (never self-certifying), maximum-disagreement active query selection, and reversible description-length library compaction (`J = L(library) + L(examples|library) + beta*verified_failures + gamma*collateral_changes`) belong to Stage **S5**.
2. **Bit Overhead of Dense 10,240-bit Index:** Adding 48 operators' 10,240-bit BSC vectors plus 12-table LSH buckets costs `509,952` active bundle bits (`503,624` compressed MDL bits vs `4,360` MDL bits for the 4 core S3 operators alone); compacting redundant operators and pruning unused proposals is a direct target of Stage S5 consolidation.
3. **Domain Scope:** Validated on the deterministic relational world (scaled to 48 operators) and an independent 36-operator synthetic arcane world; math/code transfer (including the discontinuous `(a + b) % 10` canary), language grounding, and continual learning remain deferred to **S6–S10**.

### Next Stage: Stage S5 — Refutation-Driven Consolidation (RDC)
- **Goal:** Implement the unified RDC controller combining independent simulator-backed dream-refutation proposals, leave-one-operator-out interventional blame attribution, maximum-disagreement active queries, exception splitting, and reversible MDL library consolidation where compression can never override a failed transition test.
- **Acceptance Criteria:** Prove measurable blame localization, active query efficiency, exception-refutation repair, and verified MDL reduction with zero regression across all S0–S4 gates and manifests across all 5 fixed seeds.
