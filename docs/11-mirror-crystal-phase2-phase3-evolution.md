# 11 — Mirror Crystal: Phase-2 Cognitive Core, Phase-3 & Phase-4 Bottleneck Defenses

This page documents the **Phase-2 Cognitive Core**, **Phase-3 Third-Horizon Guards**, and **Phase-4 Split-Brain & Geometric Defenses** added to [`mirror_crystal/`](../mirror_crystal/) on top of the foundational zero-weight engine ([`10-mirror-crystal-weightless-engine.md`](10-mirror-crystal-weightless-engine.md)).

Every subsystem runs in **100% pure Python 3.11+ standard library** (`0` floating-point neural weights, `0` backpropagation, `0` PyTorch/CUDA dependencies) on commodity CPUs (`Intel Core i5-10210U` @ `1.60 GHz`, `16 GB RAM`) and ultra-light client devices (`< 20 MB` RAM).

---

## 1. Iterative Bottleneck-Driven Architecture Evolution

Instead of stopping after initial retrieval and storage, Mirror Crystal was hardened through four consecutive engineering cycles—where each cycle explicitly hunted and solved the new bottlenecks introduced by the previous cycle:

| Horizon | Primary Question Asked | Modules Introduced |
|---|---|---|
| **Phase 1** *(Storage, Edge & Trust)* | How do weak devices (`4 GB` RAM, HDD, no GPU) tap a `130 GB+` brain safely without loading full JSON in RAM? | `DiskCrystalVault` (`.dcrystal` SQLite WAL + `mmap`), `PocketCrystalClient`, `CloudCrystalNode`, `SignedRuleCapsule` (`HMAC-SHA256`), `OnDevicePrivacyScrubber`, `trust_tier` supersession, negation polarity bits, long-log focus windows. |
| **Phase 2** *(Cognitive Breadth)* | How does a weightless CPU engine see images/signals, synthesize brand-new code, track pronouns/senses, discover new shortcuts while idle, and understand `200` languages without `200x` storage? | `PerceptualVQLens`, `SandboxMCTSPlanner`, `CoreferenceAndSenseTracker`, `NightlyDreamSynthesizer`, `UniversalConceptHub`, `FluentByteRealizer`. |
| **Phase 3** *(Third-Order Guards)* | What breaks *after* adding Phase 2? (`O(N^2)` dream bloat/cycles, sticky old pronouns on topic switch, `while True:` sandbox hangs, mobile keyboard typos, false-cousin partial overlaps). | `DreamUtilityPruner`, `TopicShiftDecayGuard`, `BoundedLoopStepGuard` (`_BoundedStepInjector`), `MorphologicalFuzzySpellBridge`, `EpistemicConfidenceCalibrator`. |
| **Phase 4** *(Fourth-Order Edge Interactions)* | What breaks when Phase 1 + Phase 2 + Phase 3 interact on real mobile devices? (Rotated/mirrored cameras, Roman-Urdu/Hinglish code-switching, and offline Turn-1 -> online Turn-2 split-brain coreference). | `DihedralCanonicalPatchHash` (`D4` symmetry), transliterated + multilingual `slot_formula_lock`, `SplitBrainCorefBridge` + PII-immune entity extraction. |

---

## 2. Phase-2 Cognitive Core (`mirror_crystal/phase2_cognitive_core.py`)

### P2-1: `PerceptualVQLens` — Weightless Multimodal Vision & Waveform Tokenizer
Raw pixel numbers shift drastically under illumination changes (`pixel * 2.5 + 85`), and 1D sensor/audio waveforms drift under DC offsets and gain changes. `PerceptualVQLens` converts non-text modalities into deterministic discrete tokens indexed directly inside `2,048`-bit `POPCNT` signatures:

- **2D Image / Matrix Grids (`encode_image_grid`):** Normalizes each `patch_size x patch_size` spatial block by its local mean $\mu$ and mean absolute deviation $\text{MAD}$, quantizing into `5` discrete `INT8` structural levels `{-2, -1, 0, +1, +2}` hashed via `BLAKE2b` (`vq2d_r{r}c{c}_{hex}`, `vqorient_*`, `vqshape_*`, `vqmacro_*`). Mathematically invariant to positive contrast scaling and additive brightness shifts.
- **1D Waveforms & Telemetry (`encode_waveform`):** Normalizes arbitrary-length numeric series into zero-crossing density (`vqwave_zc*`), peak crest (`vqwave_crest*`), and segment level/slope tokens (`vq1d_s{i}_l{lvl}_{slope}`).

### P2-2: `SandboxMCTSPlanner` — CPU Best-First / UCB1 Code Synthesis & Self-Repair
When a prompt + unit-test specification (`test_code`) requires combining multiple separate rules or fixing an off-by-one bug:
1. Retrieves top candidate functions and atomic helper rules (`def is_prime`, `def digit_sum`, etc.) from `MirrorCrystalEngine` or `DiskCrystalVault`.
2. Expands compositional program templates (nested predicate filtering, functional pipelines, range aggregations) and AST self-repair mutations (`range(a, n)` -> `range(a, n + 1)`, `<` <-> `<=`, defensive empty/zero guards).
3. Scores partial assertion progress under Upper-Confidence-Bound (`UCB1`) selection inside `ZeroTrustSandboxJudge`.
4. **Self-Crystallization:** Immediately crystallizes any `100%`-verified discovery as `trust_tier=1` (`domain="mcts_synthesized"`), turning future queries into `< 1 ms` direct locks.

### P2-3: `CoreferenceAndSenseTracker` — Multi-Turn Pronouns & Polysemy Disambiguation
- **Anaphoric Pronoun Tracking:** Tracks per-session active topic anchors so follow-up turns like `"What is its time complexity?"` automatically resolve `"its"` to the active subject (`"grover quantum search algorithm"`).
- **Context-Cue Homonym Disambiguation:** Injects unambiguous sense tokens (`sense_river_bank` vs. `sense_finance_bank`, `sense_reptile_python` vs. `sense_code_python`, `sense_botany_tree` vs. `sense_cs_tree`, `sense_bio_cell` vs. `sense_battery_cell`) into signatures and deduplication hashes so homonyms coexist in the same vault without overwriting one another.

### P2-4: `NightlyDreamSynthesizer` — Idle-CPU Autonomous Shortcut Discovery
Scans the directed causal graph (`_causal_edges` in RAM or `causal_edges` in SQLite) during idle CPU windows to discover multi-hop transitive consequences (`A -> B -> C -> D`) and crystallize high-utility `1`-hop `CrystalRule` shortcuts with explicit `<think>` provenance traces.

### P2-5: `UniversalConceptHub` & `FluentByteRealizer` — Zero-Duplication Multilingual Locking
Instead of duplicating a `130 GB` knowledge vault across `200` languages (`26 TB`), `UniversalConceptHub` maps non-ASCII scripts and foreign scientific terms (**Urdu**, **Arabic**, **Chinese**, **Hindi**, **Spanish**, **French**, plus extensible `register_translations`) onto canonical concept tokens *before* `2,048`-bit barcode encoding. A **single** canonical rule answers prompts across all supported scripts while `FluentByteRealizer` strips internal `<think>` blocks and adds localized verified framing without altering numbers or code.

---

## 3. Phase-3 Third-Horizon Bottlenecks & Engineering Solutions

Adding Phase 2 exposed five third-order failure modes, each solved and regression-tested:

| Phase-3 Bottleneck | Failure Mode Without Guard | Implemented Solution |
|---|---|---|
| **P3-1: Dream Graph Explosion & Cycles** | Naive transitive closure over `N` causal edges creates `O(N^2)` trivial rules and hangs on `A -> B -> A` loops. | **`DreamUtilityPruner`:** Per-path cycle rejection (`cycles_pruned`), novelty-weighted hop utility scoring (`hops * log2(2 + novel_tokens)`), and hard per-cycle cap (`max_dreams_per_cycle`). |
| **P3-2: Sticky Stale Pronouns on Topic Switch** | Asking `"How does binary search split an array when it runs?"` after discussing Grover search would wrongly inject `"grover"` because `"it"` appeared. | **`TopicShiftDecayGuard`:** Expands pronouns *only* when the new turn introduces `0` overriding topic entities, overrides anchors immediately on topic switches, and expires anchors after `ttl_turns = 3`. |
| **P3-3: Infinite Loops in Untrusted / MCTS Code** | `while True: pass` or `for i in range(10**12):` inside `exec()` hangs a CPU worker thread indefinitely. | **`BoundedLoopStepGuard` (`_BoundedStepInjector`):** Rewrites AST `While`, `For`, and `FunctionDef` bodies to call reserved `_mc_step_tick()` + `_safe_range`, aborting runaway code in **`< 10 ms`**. |
| **P3-4: Mobile Keyboard Typos Missing Inverted Buckets** | Typing `"Grvoer quantm serach"` yields `df = 0` on all three words, missing exact token postings (and hashing all candidates on cold start costs `> 500 ms`). | **`MorphologicalFuzzySpellBridge` (`ContextBitTable.resolve_typo_token`):** Bucketed `(first_char, len)` + exact triangle-inequality character-set symmetric-difference filter (`|set(a) ^ set(b)| <= 2 * max_ed`) + bounded OSA Damerau-Levenshtein + `POPCNT` 3-byte n-gram tie-break (`< 10 ms` cold on `32K+` vaults, `< 0.1 ms` cached). |
| **P3-5: False-Cousin Single-Word Overlaps** | On `32,000+` rules, a specific 5-concept query sharing just `1` word with a tangential rule could return a weak guess instead of abstaining. | **`EpistemicConfidenceCalibrator`:** Rejects single-token overlaps on `>= 4`-concept non-slot queries when `score < 0.34` and labels every response with `epistemic_tier` (`verified_exact`, `high_confidence`, `moderate_confidence`, `honest_abstain`). |

---

## 4. Phase-4 Fourth-Horizon Emergent Interactions & Solutions

Testing Phase 1 (`PocketCrystalClient` + `DiskCrystalVault`) together with Phases 2 & 3 revealed three subtle cross-subsystem bottlenecks:

1. **P4-1 — Camera Rotation & Selfie-Flip Blindness (`DihedralCanonicalPatchHash`):**
   - *Problem:* Brightness invariance alone fails when a mobile camera is rotated `90°/180°/270°` or mirrored horizontally.
   - *Solution:* `PerceptualVQLens._d4_canonical_hex` computes the minimal `BLAKE2b` digest across all `8` symmetries of the dihedral group $D_4$ (`vqd4_*`, `vqmacrod4_*`), recognizing rotated and mirrored fiducials while preserving spatial tokens (`vq2d_*`) when orientation matters.
2. **P4-2 — Roman-Urdu / Hinglish Code-Switching & Multilingual Slot Math:**
   - *Problem:* Millions of mobile users mix Latin transliteration (`"41^53 mod 1000000007 ka hisab"`) or non-ASCII scripts (`"计算 41^53 mod 1000000007"`, `"41^53 mod 1000000007 کا حساب"`) with mathematical operators.
   - *Solution:* Extended `UniversalConceptHub` with Roman-Urdu/Hinglish transliterations and preserved raw numeric spans prior to `SlotAbstractor.mask_query_numbers`, enabling zero-duplication `slot_formula_lock` execution across scripts.
3. **P4-3 — Hybrid Edge-Cloud Split-Brain Coreference & PII Token Interference (`SplitBrainCorefBridge`):**
   - *Problem:* If Turn 1 (`"Explain Grover quantum search"`) is answered `100%` locally in `PocketCrystalClient` RAM (`0` Cloud traffic) and Turn 2 (`"What is its time complexity for student@uni.edu?"`) misses Pocket RAM and queries `CloudCrystalNode`, Cloud never saw Turn 1—and the email address `student@uni.edu` looked like new topic nouns (`student`, `uni`, `edu`)!
   - *Solution:* `CoreferenceAndSenseTracker.extract_topic_entities` strips email/URL/placeholder tokens before topic-shift detection, and `PocketCrystalClient.ask` bridges the locally resolved query through `OnDevicePrivacyScrubber.scrub` before calling `CloudCrystalNode` (`0` PII leaked, `100%` cross-tier coreference continuity).
4. **P4-4 — Corpus-Scale Frame Collision & Possessive `'s` Fragmentation (`49,372`-Row Real Corpus Audit):**
   - *Problem:* On `98,000+` observed documents, a static `min_shared_specific_frames = 2` threshold folded `17,237` coincidental frame pairs across common words, while English possessives (`"Grover's"` vs. `"Grover"`) split inverted posting buckets.
   - *Solution:* `ContextBitTable.tokenize` normalizes possessive `'s` suffixes (`grover's -> grover`), and `fold_equivalences` scales its frame-support threshold logarithmically with `doc_count` while excluding generic high-DF words (`df > doc_count // 35`).
5. **P4-5 — Verified Code Axiom Priority in `SandboxMCTSPlanner` Amidst `32K+` Prose Rules:**
   - *Problem:* When `32,000+` prose math rules contain `"prime"`, a small `top_k=5` retrieval window can crowd out `snake_case` helper functions like `def is_prime_fast(n)`.
   - *Solution:* `SandboxMCTSPlanner` splits `snake_case` identifier stems and directly scans `trust_tier == 1` (`pass_count >= 6`) sandbox-verified code axioms alongside widened `top_k=12` inverted candidates, guaranteeing helper discovery at full corpus scale.
6. **P4-6 — In-Place Foreign Particle Stripping in `UniversalConceptHub`:**
   - *Problem:* Appending canonical tokens while leaving Roman-Urdu/Romance particles (`"ka"`, `"hisab"`, `"calcula"`) in the surface string diluted lexical salience against `0`-slot prose rules.
   - *Solution:* `UniversalConceptHub.normalize_to_canonical` replaces translated Latin/Romanized tokens in-place and strips foreign functional particles whenever a concept translation triggers.

---

## 5. Fifth-Horizon Outlook (Planetary Trillion-Rule Scale)

When scaling Mirror Crystal beyond a single `130 GB` NVMe node to hundreds of millions of concurrent devices and `10^9+` rules, the next architectural horizon consists of:
1. **Geo-Sharded Domain Routers:** Partitioning `DiskCrystalVault` shards by domain (`math`, `code`, `quantum`, `biomed`, `law`) across stateless CPU nodes routed by a `2 KB` top-level domain bitmask.
2. **WebAssembly (`WASM` SIMD `i8x16.popcnt`) Pocket Engine:** Compiling `ContextBitTable` and `PocketCrystalClient` into a `< 150 KB` `.wasm` module so browsers and ultra-budget phones execute `POPCNT` locks natively at C speed with zero server dependency.
3. **Proof-Carrying Rule Capsules:** Attaching machine-checkable AST unit tests (`test_code`) directly inside `SignedRuleCapsule`s so edge clients independently verify both cryptographic origin (`HMAC-SHA256`) and mathematical truth on arrival.

---

## 6. Quickstart & Verification Commands

```python
from mirror_crystal import (
    DiskCrystalVault,
    NightlyDreamSynthesizer,
    PerceptualVQLens,
    SandboxMCTSPlanner,
)

vault = DiskCrystalVault("artifacts/mirror_crystal_all_datasets.dcrystal")

# 1. Typo-tolerant + multilingual + slot-generalized query on 32,450+ disk rules:
print(vault.respond("41^53 mod 1000000007 کا حساب").answer)
print(vault.respond("How does Grvoer quantm serach achieve a quadratic speedup?").answer)

# 2. CPU UCB1 multi-rule code synthesis & AST verification:
planner = SandboxMCTSPlanner(max_nodes=24, max_wall_ms=350.0)
res = planner.solve_and_verify(
    vault,
    prompt="Write solve_prime_digits(limit) returning primes <= limit whose sum_dec_digits is prime.",
    test_code="assert solve_prime_digits(30) == [2, 3, 5, 7, 11, 23, 29]",
)
print(res.mode, res.epistemic_tier, res.answer)
vault.close()
```

```bash
# Run all 35 unit, bottleneck, Phase-2/3/4, and out-of-core regression tests (~1.5s on CPU):
python -m pytest mirror_crystal/tests/ -v
```

---

## 7. Full 5-Dataset Out-of-Core Ingestion & 18-Scenario Live Regression Summary

All `5` local JSONL corpora in `training/data/` (`94.331 MB` raw, `49,372` train rows) were streamed through `DiskCrystalVault` on CPU (`Intel Core i5-10210U` @ `1.60 GHz`, `0` GPU) into `artifacts/mirror_crystal_all_datasets.dcrystal` (full telemetry saved in [`artifacts/mirror_crystal_all_datasets_regression_report.json`](../artifacts/mirror_crystal_all_datasets_regression_report.json)):

- **Streaming Throughput & Deduplication:** `49,372` rows processed in `113.50 s` -> **`32,458` unique rules** crystallized on disk (`44.00 MB` `.dcrystal` vault — `< 50%` of raw JSONL size with inverted indices included) and **`16,925` exact duplicates** skipped automatically (`100%` of the `6,264` overlap rows in `mirror_quantum_math_reasoning_v1.jsonl` caught in `0` new writes).
- **Zero-RAM Cold Boot:** `0` rules loaded into RAM at boot (`18.03 MB` Python heap; `51.78 MB` peak query heap across all `18` test scenarios; `256` hot rules in bounded LRU).
- **Live Regression Pass Rate:** **`18 / 18` (`100%`) scenarios passed**, covering pre-ingestion canary survival (`0%` catastrophic forgetting), unseen big-integer modular exponentiation (`20.9 ms`), multi-hop causal chaining (`15.5 ms`), `PerceptualVQLens` lighting + `90°` rotation invariance (`19.2 ms`), `SandboxMCTSPlanner` multi-rule program synthesis (`166.5 ms`), `BoundedLoopStepGuard` `while True` anti-hang abort (`5.8 ms`), `TopicShiftDecayGuard` (`14.8 ms`), homonym sense separation, `NightlyDreamSynthesizer` + `DreamUtilityPruner` (`2.8 ms`, `2` cycles pruned), `UniversalConceptHub` Urdu/Chinese/Roman-Urdu zero-duplication locking (`17.5–25.3 ms`), and `MorphologicalFuzzySpellBridge` + `SplitBrainCorefBridge` (`57.4 ms` cold / `8.1 ms` warm, `0` PII leaked).
- **Next Evolution (Phase 5 & Phase 6):** See **[`12-mirror-crystal-phase5-phase6-bicameral-scale.md`](12-mirror-crystal-phase5-phase6-bicameral-scale.md)** for Bicameral `1.58`-bit `TernaryFluidComposer`, `SymbolicCASOrgan`, `ActiveCuriosityPlanner`, `DocumentAutoCrystallizer`, `.png`/`.wav` `SensoryCodec`, `ShardedCrystalRouter`, `184`-byte `.wasm` `POPCNT` kernel, and Proof-Carrying Rule Capsules.
