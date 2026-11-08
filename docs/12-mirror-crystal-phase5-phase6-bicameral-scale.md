# 12 — Mirror Crystal: Phase-5 Bicameral Intelligence, Symbolic CAS, Sensory Codecs, Sharded Scale & Phase-6 Guards

This page documents the **Phase-5 Cognitive, Sensory & Planetary-Scale Expansion** and **Phase-6 Emergent Bottleneck Guards** added to [`mirror_crystal/`](../mirror_crystal/), building directly upon [`10-mirror-crystal-weightless-engine.md`](10-mirror-crystal-weightless-engine.md) and [`11-mirror-crystal-phase2-phase3-evolution.md`](11-mirror-crystal-phase2-phase3-evolution.md).

All eight Phase-5 subsystems and nine Phase-6 defenses run in **100% pure Python 3.11+ standard library** (`0` GPU, `0` PyTorch/NumPy runtime dependencies), plus a **`184`-byte spec-compliant WebAssembly 1.0 (`mirror_crystal_pocket_popcnt.wasm`)** kernel verified inside Google V8.

---

## 1. Phase-5 Subsystems (`phase5_bicameral_and_cas.py` & `phase5_sensory_shards_wasm.py`)

| Organ / Subsystem | Architectural Role | Measured CPU Latency (`i5-10210U`) |
|---|---|---|
| **P5-1: `SymbolicCASOrgan`** | Exact `fractions.Fraction` symbolic differentiation (polynomial + product/quotient/chain rules on `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`), polynomial indefinite & definite integration, quadratic & `N x N` linear equation systems (Gauss-Jordan), and dimensional unit conversions—independently checked via finite differences / Simpson quadrature / root substitution. | **`0.65 – 0.81 ms`** |
| **P5-2: `PackedTernaryProjection` & `TernaryFluidComposer`** | **Bicameral Left+Right Brain Fusion:** Left-Brain (`DiskCrystalVault`) provides `100%` grounded facts/numbers; Right-Brain uses a packed **`1.58`-bit `{-1, 0, +1}` integer projection** (`4` weights per byte, `0` float multiplies) + online collocation lattice to style verified facts (`email`, `apology`, `executive_brief`, `eli5`) or synthesize open-ended stories/poems. | **`3.20 – 4.01 ms`** |
| **P5-3: `ActiveCuriosityPlanner`** | Diagnostic Information-Gain Clarifier: detects under-specified queries straddling `>= 2` genuinely disjoint knowledge branches (`grpc` vs. `http`, `postgresql` vs. `sqlite`), asks a targeted 1-turn clarifying question, and fuses the user's reply on Turn 2. | **`3.03 ms` -> `1.03 ms`** |
| **P5-4: `DocumentAutoCrystallizer`** | Ingests raw unstructured `.md`, `.txt`, and `.py` files without `.jsonl` formatting—extracting headed sections, inline definitions, causal edges, and AST-verified `trust_tier=1` Python functions. | **`> 400` rules/s** |
| **P5-5: `SensoryCodec` (`.png` & `.wav`)** | Zero-dependency RFC-2083 `.png` encoder/decoder (Grayscale/RGB/RGBA, all `5` PNG scanline filters: `None`, `Sub`, `Up`, `Average`, `Paeth`) and RIFF `.wav` PCM codec (`8/16/24/32`-bit mono/stereo) wired directly into `PerceptualVQLens`. | **`< 5 ms`** |
| **P5-6: `ShardedCrystalRouter`** | Partitions rules across domain `.dcrystal` vaults (`code`, `math_quantum`, `systems`, `general`) routed via `256`-byte (`2,048`-bit) shard summary bitmasks (`< 8 KB` RAM) + unmounted cold-shard hints (`gdrive://...`). | **`< 0.15 ms`** route |
| **P5-7: `PackedSimdBitKernel` & `WasmPocketCompiler`** | Batch C-int `POPCNT` scanner + zero-dependency WebAssembly 1.0 binary emitter ([`artifacts/mirror_crystal_pocket_popcnt.wasm`](../artifacts/mirror_crystal_pocket_popcnt.wasm), **`184` bytes**) executing hardware `i64.popcnt` (`0x7B`) over `32 x 64`-bit lanes. | **`11.18 ms` / `100,000` `2,048`-bit `POPCNT`s** (`8.94M/s` in V8) |
| **P5-8: Proof-Carrying `SignedRuleCapsule`** | Extends `HMAC-SHA256` rule capsules with signed `proof_test_code` verified locally inside `ZeroTrustSandboxJudge` on arrival at `PocketCrystalClient`, blocking buggy or unsafe code even under key compromise. | **`< 3 ms`** |

---

## 2. Phase-6 Emergent Bottlenecks Discovered & Solved

Testing Phase 5 both in unit suites and against the **`32,464`-rule full-corpus `.dcrystal` vault** uncovered nine higher-order failure modes, each permanently guarded:

| Bottleneck ID | Failure Mode Discovered | Implemented Guard |
|---|---|---|
| **B6-1: Fact Drop / Number Hallucination During Styling** | Right-Brain re-styling a verified fact into an email could drop `"not"`, alter `"O(sqrt(N))"`, or invent an ungrounded number; or creative stories could leak into factual memory. | **`FactFidelityGuard` & `CreativeFactualBleedGuard`:** Verifies `100%` preservation of numbers, `O(...)` complexities, and negations (falling back to verbatim rule text on violation), and tags creative prose `creative_synthesis` with `0` vault writes. |
| **B6-2: `.png` Zip Bombs & Giant Audio Streams** | A `20 KB` `.png` claiming `10000 x 10000` in `IHDR` or expanding to `20 MB+` in `zlib` crashes low-RAM laptops. | **`PNGDecompressionBombGuard` & `WAVDurationCap`:** Enforces `<= 2048 x 2048` (`4.19M` px) before buffer allocation, uses bounded `zlib.decompressobj().decompress(..., max_length=16MB)` rejecting `unconsumed_tail`, and caps WAV reads at `480,000` frames. |
| **B6-3: Symbolic Polynomial / Rational Explosion** | Asking `Differentiate (x + 1)^500` expands `501` huge binomial coefficients; `Convert 5 kg to km` mixes incompatible physical dimensions. | **`CASComplexityExplosionGuard` & `DimensionalMismatchGuard`:** Caps AST nodes (`<= 128`), polynomial degree (`<= 64`), and rational bit-length (`<= 2,048` bits), and rejects cross-dimension unit conversions cleanly. |
| **B6-4: Endless Clarification Loops & Paraphrase False Alarms** | Naive ambiguity detection would ask clarifications when two Grover paraphrases tie in score, or trap users in multi-turn questioning loops. | **`GenuineAmbiguityFilter` & `CuriosityLoopGuard`:** Requires `0.0` non-query remainder token Jaccard (`rem0 & rem1 == empty`) so paraphrases never trigger clarification, and caps clarifications at `1` turn per topic. |
| **B6-5: Cross-Domain Shard Boundary Misses** | `"Python function for Grover quantum speedup"` spans both `code.dcrystal` and `math_quantum.dcrystal`. | **Adaptive Multi-Shard Fan-Out:** Automatically fans out to `top_2` shards whenever secondary shard affinity is within `68%` of the primary shard, plus surfaces `cold_shards` hints for unmounted archives. |
| **B6-6: Boilerplate & Embedded Unsafe Code in Raw Docs** | Ingesting a raw README with `# License` sections or an unsafe `import os` example inside a prose section could pollute or quarantine the section rule. | **`DocBoilerplateAndSafetyGuard`:** Skips TOC/License headings, strips fenced code from prose bodies after AST extraction, and verifies extracted functions via `ZeroTrustSandboxJudge`. |
| **B6-7: Stale Creative Template Hijacking (`32K` Corpus Discovery)** | On `32,464` rules (including OpenAssistant chats), `"Write a short story about a quantum lighthouse on Europa"` matched `"write + short + story"` (`0.616`) on an old story about a *ceramic cup*! | **`CreativeSubjectCoverageGuard`:** Raises the Left-Brain lock threshold to `0.85` on explicit creative prompts so novel story topics route cleanly to `TernaryFluidComposer.synthesize_creative`. |
| **B6-8: Unknown Proper-Entity Partial Match (`32K` Corpus Discovery)** | Asking `"What is the atomic mass of Xylofluoride-992?"` matched `"atomic + mass"` (`0.429`) on Carbon/Oxygen rules instead of abstaining on unknown `Xylofluoride` (`df = 0`). | **`extract_oov_proper_entities`:** Detects mid-sentence capitalized/alphanumeric OOV entities (`df == 0` after typo bridging) and blocks both partial factual locks and false clarifications -> clean `honest_abstain`. |
| **B6-9: Non-ASCII Word Order in Multilingual Symbolic CAS** | `"求导 x^3 + 5*x"` and `"x^3 + 5*x کا مشتق"` appended `"differentiate"` at the end via `UniversalConceptHub`, missing prefix calculus regexes. | **Cross-Lingual CAS Bridge:** Maps calculus/algebra verbs across **Urdu, Arabic, Chinese, Hindi, Spanish, and French** and canonicalizes prefix operator order inside `SymbolicCASOrgan.try_solve`. |

---

## 3. Verification & Benchmark Commands

Full live benchmark telemetry against the `32,464`-rule `.dcrystal` vault is recorded in [`artifacts/mirror_crystal_phase5_phase6_benchmark.json`](../artifacts/mirror_crystal_phase5_phase6_benchmark.json), and the compiled `184`-byte WASM kernel is saved at [`artifacts/mirror_crystal_pocket_popcnt.wasm`](../artifacts/mirror_crystal_pocket_popcnt.wasm).

```bash
# Run all 50 unit, bottleneck, Phase-2..6, sensory, sharded, and V8 WASM tests (~2.4s on CPU):
python -m pytest mirror_crystal/tests/ -v
```
