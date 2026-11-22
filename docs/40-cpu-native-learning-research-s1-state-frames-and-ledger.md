# 40 — CPU-Native Learning Research (SHARD + ITL + RDC): Stage S1 Typed State Frames & Episode Ledger

This document records the Stage **S1** implementation, independent 5-seed parser-only benchmark, partial `ObservationMask` epistemic preservation proofs, and append-only SQLite `EpisodeLedger` audit for Mirror 7's CPU-native learning program (`research/cpu_learning/`).

> **Research Integrity Notice:** Stage S1 verifies learner-facing state representation (`TypedStateFrame` + `ObservationMask`), raw-UTF-8-to-frame role parsing (`TypedRoleFrameParser`), and append-only provenance storage (`EpisodeLedger` with test-time memory-access auditing). **No ITL operator induction (S2), SHARD proposal search (S4), or RDC consolidation (S5) is implemented or claimed in Stage S1.**

---

## 1. Stage S1 Modules Implemented (`research/cpu_learning/`)

All S1 learner-side code lives strictly inside `research/cpu_learning/frames/` and `research/cpu_learning/ledger/` with **0 violations** under AST import-isolation auditing (`hidden_actions.py` and `oracle.py` are unreachable by all learner modules):

- `research/cpu_learning/frames/state_frame.py` — Learner-facing `TypedEntityFrame`, `TypedStateFrame`, `FactEpistemicStatus` (`OBSERVED_PRESENT`, `OBSERVED_ABSENT`, `UNOBSERVED`), `CandidateActionFrame`, `FrameParseResult`, `assert_no_hidden_fields`, `extract_masked_sparse_delta`, `verify_unobserved_preservation`, and canonical state-action key hashing.
- `research/cpu_learning/frames/frame_parser.py` — Raw UTF-8 byte tokenizer (`tokenize_utf8_bytes` preserving exact byte offsets), `BilingualCCGFrameAdapter` (wrapping Mirror's existing `BilingualCCGHDCParser` for head-to-head comparison), and `TypedRoleFrameParser` (state-typed disambiguation, multi-parse ambiguity preservation, and clean OOD abstention).
- `research/cpu_learning/ledger/episode_ledger.py` — Append-only SQLite `EpisodeLedger` (`s1-ledger-v1`) with SQL `BEFORE UPDATE` / `BEFORE DELETE` immutability triggers, SHA-256 chain hashing, `.freeze()` write lock, provenance/split/seed/generator/byte-span persistence, and synchronous test-time `AuditScope` + `MemoryAccessEvent` hooks.
- `research/cpu_learning/evaluation/parser_harness.py` & `research/cpu_learning/evaluation/run_s1_experiments.py` — Separate parser-only scoring harness (530 cases/seed × 5 seeds = 2,650 evaluations across unseen S0 instructions, frozen temporal splits, unseen paraphrases, state-typed disambiguation, partial-mask occlusion, genuine ambiguity, and OOD unparseables) plus 5-seed ledger/mask/audit verification.
- `research/cpu_learning/tests/test_s1_state_frames_and_ledger.py` — Automated pytest suite verifying AST isolation, hidden-field guards, partial-mask unobserved-is-not-false invariants, SQLite immutability/hash-chain/freeze/audit hooks, parser ablations, and bit-exact manifest verification.

---

## 2. Verified 5-Seed Stage S1 Results (`[101, 202, 303, 404, 505]`, `16 / 16` Gate Checks PASS)

**Reproduce commands:**
```bash
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s1_experiments --verify-manifests
PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s1_experiments
PYTHONHASHSEED=0 python -m pytest -v research/cpu_learning/tests
```

**Raw artifacts:** [`research/cpu_learning/reports/s1_research_report.md`](../research/cpu_learning/reports/s1_research_report.md) · [`research/cpu_learning/reports/s1_summary.json`](../research/cpu_learning/reports/s1_summary.json) · [`research/cpu_learning/reports/raw_runs/s1_parser_per_split.csv`](../research/cpu_learning/reports/raw_runs/s1_parser_per_split.csv) · [`research/cpu_learning/reports/raw_runs/s1_ledger_audit_5seed.json`](../research/cpu_learning/reports/raw_runs/s1_ledger_audit_5seed.json)

### Separate Parser-Only Scoring Matrix (5 Seeds, 2,650 Evaluations)

| Parser | S0 Unseen Exact Role | Temporal Frozen | Unseen Paraphrases | State-Typed Disambig. | Partial-Mask Occluded | Overall Byte-Span Exact | Ambiguous Flag / Top-K | OOD Abstain | Parses/s (1-core) | Peak Heap (KB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `existing_mirror_bilingual_ccg` | 20.83% | 26.67% | 0.00% | 0.00% | 0.00% | 18.96% | 0.00% / 0.00% | 60.00% | 2,390.3 | 16.7 |
| `s1_typed_frame_parser_no_types` | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 98.33% | 75.00% / 75.00% | 100.00% | 7,321.8 | 5.5 |
| `s1_typed_frame_parser_full` | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00% / 100.00%** | **100.00%** | **7,068.9** | **5.5** |

### Append-Only `EpisodeLedger` & Partial-Mask Audit (5 Seeds)
- **Immutability & Hash Chain:** `5/5` seeds PASS (`UPDATE`, `DELETE`, duplicate IDs, and post-freeze appends blocked; disk reload SHA-256 hash chain verified).
- **Partial/Asymmetric `ObservationMask` Preservation:** `8/8` checks PASS per seed (occluded facts marked `UNOBSERVED` and never misclassified as deleted; revealed facts never fabricated as added).
- **Memory-Access Audit Hooks:** `100.00%` canary exact-key hits logged vs `0.00%` false hits on unseen tasks.

---

## 3. Next Stage: Stage S2 — ITL Operator Induction

With all 16 S1 checks and all 11 S0 checks passing cleanly, the program advances to **Stage S2 (ITL Operator Induction)**: inducing reusable typed `OperatorCard`s for `move`, `give`, `open`, and `lock` strictly from the 6 before/action/after training episodes per action stored in `EpisodeLedger` (zero access to `hidden_actions.py`), evaluated against the PDF p. 13 Initial ITL Go Gates (`>=95%` Depth-4 exact final-state accuracy, `>=90%` Depth-6, `>=90%` hidden precondition/effect precision & recall, `<2%` collateral corruption, and `>=20` percentage points above episode lookup across all 5 fixed seeds).
