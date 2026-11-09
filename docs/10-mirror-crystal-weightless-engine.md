# 10 — Mirror Crystal: Zero-Weight, Zero-Gradient CPU Intelligence Engine

**Mirror Crystal** (`mirror_crystal/`) is an experimental, standalone, from-scratch intelligence track inside Mirror 7 exploring **complete GPU independence**: replacing floating-point parameter matrices and iterative backpropagation (`loss.backward()`) with **hardware-native 2,048-bit Context Barcodes (`POPCNT`), One-Shot Variable/Formula Abstraction, Locality-Sensitive Rule Crystallization (`LSH`), Multi-Hop Lock Chaining, and CPU Reality Execution**.

> **Additive Architecture Track:** Mirror Crystal complements (and does not replace) `mirror7_backend` and the learned V5/V6 neural prototypes (`training/`). It runs on pure **Python 3.11+ standard library (`0` PyTorch, `0` CUDA, `0` GPU)** so any commodity laptop CPU can ingest `.jsonl` and `.txt`/`.py` corpora in a single pass.

---

## 1. Core Architectural Mechanism (How It Learns Without Weights)

Instead of adjusting decimal weights across multiple GPU epochs, `MirrorCrystalEngine` maintains four integer/bit data structures in CPU RAM and updates them in **one streaming pass**:

| Structure | Module | Role & Hardware Primitive |
|---|---|---|
| **1. `ContextBitTable`** | [`bit_signatures.py`](../mirror_crystal/bit_signatures.py) | Encodes raw UTF-8 byte morphology (`bits 0..1023`) + surrounding neighbor byte frames (`bits 1024..2047`) into native `2,048`-bit integers. Words appearing inside identical specific neighbor frames merge context bits automatically via `fold_equivalences()`, and similarity runs in C via CPU `POPCNT` (`int.bit_count()`). |
| **2. `SlotAbstractor`** | [`slot_abstractor.py`](../mirror_crystal/slot_abstractor.py) | Distinguishes entity-key facts from parameterized laws. Discovers exact arithmetic/modular formulas (`pow(a, b, m)`, `+`, `-`, `*`, `%`, `gcd`, `**`) and replaces concrete numbers with `{v0}, {v1}` slots so **one** training example generalizes to arbitrary unseen integers. |
| **3. `LSHRuleIndex`** | [`lsh_rule_store.py`](../mirror_crystal/lsh_rule_store.py) | Indexes crystallized `CrystalRule` records in `array('I')` inverted salient-token postings and 16-band LSH bit-slices for sub-millisecond retrieval. Enforces **Surprise-Only Learning** (skips duplicate rules when existing locks already predict the target) and **Multi-Hop Causal Chaining** (`A -> B` + `B -> C` => `A -> C`). |
| **4. `ByteMarkovGrammar` & `ZeroTrustSandboxJudge`** | [`grammar_and_sandbox.py`](../mirror_crystal/grammar_and_sandbox.py) | Tracks variable-order UTF-8 byte transitions (`0` float weights) and runs a zero-trust static `ast` + isolated CPU execution check on code/math rules—rejecting broken or unsafe code automatically. |

---

## 2. Supported Data Formats (How to Feed Data)

`MirrorCrystalEngine` streams files line-by-line in constant (`O(1)`) transient RAM.

### A. `.jsonl` (Primary Structured Teaching Format)
Compatible out-of-the-box with all existing `training/data/*.jsonl` datasets (`user_text`/`target_text`, `user`/`response`, `prompt`/`completion`, `instruction`/`output`):

```json
{"user_text": "What is a qubit?", "target_text": "A qubit is a two-level quantum state |psi> = alpha|0> + beta|1> with |alpha|^2 + |beta|^2 = 1."}
{"user_text": "Compute 17^19 mod 1000000007", "target_text": "The exact modular result of 17^19 mod 1000000007 is 286765645."}
{"user_text": "Implement binary_search(arr, x)", "target_text": "def binary_search(arr, x):\n  ...", "test": "assert binary_search([1, 3, 5], 3) == 1"}
```

### B. Multi-Turn Session `.jsonl`
```json
{"turns": [{"user": "My server limit is 120 requests/sec.", "response": "Saved constraint limit = 120."}, {"user": "Are 3 bursts of 45 safe?", "response": "Unsafe: 3 * 45 = 135 > 120."}]}
```

### C. Raw Unlabeled `.txt`, `.md`, and `.py` Files
Raw text and source-code files can be fed directly without JSON conversion to build `ContextBitTable` neighbor-frame synonym barcodes, causal fact edges, and `ByteMarkovGrammar` fluency statistics.

---

## 3. Command-Line Usage (`python -m mirror_crystal`)

### 1. Train (Single-Pass CPU Ingestion)
```bash
python -m mirror_crystal train \
  --input training/data/mirror_complete_master_v2.jsonl \
  --save artifacts/mirror_crystal_v1.mcrystal
```

### 2. Continue Teaching an Existing `.mcrystal` Brain (`0%` Forgetting)
```bash
python -m mirror_crystal train \
  --load datasets/crystal_archive/raw/crystal_4.zip \
  --input training/data/mirror_quantum_math_reasoning_v1.jsonl \
  --save artifacts/mirror_crystal_v1.mcrystal
```

### 3. Single-Prompt Query
```bash
python -m mirror_crystal query \
  --brain datasets/crystal_archive/raw/crystal_4.zip \
  --prompt "How does Grover's quantum search algorithm achieve a quadratic speedup?"
```

### 4. Interactive Multi-Turn CLI Chat (With Live `/teach` Instant Learning)
```bash
python -m mirror_crystal chat --brain artifacts/mirror_crystal_v1.mcrystal
```
Inside interactive chat, type `/teach Question ||| Answer` to crystallize a new permanent rule in `< 1 ms` and save to disk immediately.

### 5. Run Hardware Efficiency & Reasoning Benchmark + Unit Tests
```bash
python -m pytest -q mirror_crystal
python -m mirror_crystal benchmark --max-rows 7125
```

---

## 4. Python API Example

```python
from mirror_crystal import MirrorCrystalEngine

engine = MirrorCrystalEngine()

# 1. Learn word equivalence from unlabeled text (0 weights)
engine.observe_unlabeled_text("""
configure gateway server_limit before launch
configure gateway traffic_cap before launch
monitor gateway server_limit under load
monitor gateway traffic_cap under load
""")
engine.bit_table.fold_equivalences()

# 2. Teach 1 concrete modular exponentiation example -> generalizes to unseen numbers
engine.ingest_pair(
    "Compute 17^19 mod 1000000007",
    "The exact modular result of 17^19 mod 1000000007 is 286765645.",
)
print(engine.respond("Compute 31^43 mod 1000000007").answer)

# 3. Multi-turn session constraint memory + verified CPU reality check
engine.respond("My server limit is 120 requests/sec.", session_id="demo")
reply = engine.respond("Are 3 bursts of 45 safe? Give me a limiter.", session_id="demo")
print(reply.answer)
```

---

## 5. Verified Hardware Benchmark (`Intel Core i5-10210U` Laptop CPU • `0` GPU)

Measured end-to-end via `python -m mirror_crystal benchmark --max-rows 7125` on commodity laptop hardware (`Intel Core i5-10210U` 4C/8T @ `1.60 GHz`, `16 GB RAM`, Windows, Python `3.11`, pure standard library — telemetry saved in [`artifacts/mirror_crystal_hardware_benchmark.json`](../artifacts/mirror_crystal_hardware_benchmark.json)):

| Metric | Verified Result (`0` GPU) | Engineering Significance |
|---|---:|---|
| **Floating-Point Neural Weights** | **`0`** | Zero matrices, zero `loss.backward()`, zero optimizer state |
| **Streaming Ingestion Throughput** | **`663.3 rows/sec`** | Ingested all **`6,697` train rows** (`mirror_complete_master_v2.jsonl`, `6.84 MB`) in **`10.097 s`** |
| **Total Rules Crystallized** | **`6,798` Locks** | Indexed in `array('I')` inverted salient postings + 16-band `2,048`-bit LSH |
| **Context-Frame Synonym Bridges** | **`10,444` Merged** | Discovered automatically via `POPCNT` neighbor-frame intersection (`0` embeddings) |
| **Compressed `.mcrystal` Brain Size** | **`4.005 MB`** | [`artifacts/mirror_crystal_v1.mcrystal`](../artifacts/mirror_crystal_v1.mcrystal) (`1.7x` smaller than raw `.jsonl`) |
| **Mean CPU Query Latency** | **`3.452 ms`** | `2.1 ms – 4.9 ms` end-to-end retrieval + slot math + AST verification |
| **1-Shot Unseen Integer Generalization** | **`PASS` (`True`)** | Taught `17^19 mod 1000000007` once -> solved unseen `41^53 mod 1000000007 = 354849446` |
| **100-Step Continual Zero-Forgetting** | **`PASS` (`True`)** | `100%` exact recall of Rule #0 and Rule #100 after sequential online updates |
| **Multi-Hop Causal Lock Chaining** | **`PASS` (`True`)** | Chained `A -> B` + `B -> C` -> `Therefore, A -> C` in `2.13 ms` |
| **Multi-Turn Session Reality Check** | **`PASS` (`True`)** | Recalled `120 req/s` cap across turns, detected `3 * 45 = 135` overflow, verified `TokenBucketLimiter` |

---

## 6. Honest Scope & Engineering Boundaries

- **Strengths:** Runs `100%` on commodity CPUs (`0` GPU / `0` PyTorch), ingests thousands of `.jsonl` rules in `~10` seconds, compresses knowledge into a single `4.0 MB` `.mcrystal` file, guarantees `0%` catastrophic forgetting on distinct keyed facts, generalizes exact slot arithmetic in 1 shot, chains explicit `A -> B -> C` causal edges, and never executes unsafe AST constructs.
- **Current Research Boundary:** Open-ended creative prose generation and noisy raw pixel/audio perception still benefit from coupling Mirror Crystal with a compact perceptual encoder ("Semantic Lens")—the active next step in the Mirror CPU-first research roadmap.

---

## 7. Scaling to Billion / Trillion-Byte Corpora Without RAM Overflow (`DiskCrystalVault` / `.dcrystal`)

Loading an entire `.mcrystal` JSON archive into RAM works well for compact brains (`4 MB – 50 MB`), but would exhaust laptop RAM on multi-gigabyte or trillion-byte corpora. [`mirror_crystal/disk_vault.py`](../mirror_crystal/disk_vault.py) solves both the **RAM ceiling** and **CPU scan cost** via an out-of-core **SQLite WAL + OS `mmap` B-Tree Vault (`.dcrystal`)**:

1. **Zero Full-Brain RAM Load (`0` Rules in RAM at Boot):** Rules stay on SSD/NVMe inside indexed B-Trees (`rules`, `token_postings`, `band_postings`) with bounded `OrderedDict` LRU caching (`lru_rule_cache_size`, e.g. `256` hot rules) and capped vocabulary/frame tables (`max_vocab_entries=100,000`, `max_tracked_frames=65,536`).
2. **Sub-Linear CPU Candidate Lookup:** Even across millions of disk rules, a query reads **only** the `8–16` matching salient-token / LSH-band posting blobs (`array('I')`, `4` bytes per ID) and fetches **at most `64` candidate rows** in a single batched `WHERE rule_id IN (...)` B-Tree read before running C-level `POPCNT`.
3. **Surprise Gate Deduplication at the Door:** A 16-byte `Blake2b` normalized-prompt index (`prompt_hash UNIQUE`) skips redundant/duplicate training rows in `O(log N)` B-Tree time without creating duplicate rules.
4. **Lock-Free Horizontal Sharding:** Because Crystal Rules do not share coupled dense weight matrices, multi-terabyte corpora partition cleanly into independent `.dcrystal` domain/hash shards (`math.dcrystal`, `code.dcrystal`, `shard_00..shard_63.dcrystal`) queried in parallel across CPU cores.

### CLI Usage for Out-of-Core `.dcrystal` Mode (Whole-Directory Streaming)
```bash
# Stream an entire directory of .jsonl files directly into a disk-paged .dcrystal vault:
python -m mirror_crystal train \
  --input training/data \
  --save artifacts/mirror_crystal_all_datasets.dcrystal

# Query or chat directly from disk (loads 0 rules into RAM at startup):
python -m mirror_crystal query \
  --brain artifacts/mirror_crystal_all_datasets.dcrystal \
  --prompt "Compute 73^89 mod 1000000007"

python -m mirror_crystal chat --brain artifacts/mirror_crystal_all_datasets.dcrystal
```
Full 5-dataset (`94.33 MB` / `49,372` train rows) ingestion and 10-scenario regression telemetry is archived in [`artifacts/mirror_crystal_all_datasets_regression_report.json`](../artifacts/mirror_crystal_all_datasets_regression_report.json).

---

## 8. Micro-Crystal Teleportation & Production Bottleneck Defenses (`pocket_teleport.py`)

For consumer apps running on weak CPUs, low-RAM mobile devices, or non-SSD PCs backed by a zero-GPU cloud vault, [`mirror_crystal/pocket_teleport.py`](../mirror_crystal/pocket_teleport.py) and upgraded core modules provide 8 built-in defenses (verified in [`mirror_crystal/tests/test_bottleneck_solutions.py`](../mirror_crystal/tests/test_bottleneck_solutions.py)):

| Bottleneck Addressed | Implemented Defense |
|---|---|
| **1. Stale / Conflicting Rules** | `trust_tier` (`1=verified`, `2=curated`, `3=user_local`) + monotonic `updated_seq` automatically supersedes outdated facts while blocking low-trust overrides. |
| **2. Weak User Hardware (`<20 MB` RAM, No SSD)** | `PocketCrystalClient` keeps `<= max_ram_rules` hot rules in bounded LRU RAM (`~0.27 ms` offline hits) and teleports `1.3 KB` rules on demand from `CloudCrystalNode`. |
| **3. Network Tampering & `/teach` Poisoning** | `SignedRuleCapsule` enforces constant-time `hmac.compare_digest` (`HMAC-SHA256`), and `teach_private_local()` isolates user notes to local `trust_tier=3` RAM. |
| **4. Cloud Privacy Leaks** | `OnDevicePrivacyScrubber` masks emails, API keys/tokens, and IPs locally before network fallback and rehydrates placeholders on-device. |
| **5. Multi-Part Compound Prompts** | `split_compound_intents()` decomposes multi-question prompts and stitches verified sub-answers (`mode="compound_multi_intent"`). |
| **6. `2,048`-Bit Log Saturation** | `ContextBitTable.split_focus_windows()` isolates diagnostic error lines across multi-hundred-line tracebacks. |
| **7. Affirmative vs. Negated Collisions** | Dedicated negation polarity bits (`bits 2040..2047`) prevent `"enable X"` and `"NEVER enable X"` from swapping answers. |
| **8. Viral Traffic & Bot Scraping** | `CloudCrystalNode` combines an `O(1)` deterministic Edge CDN cache with per-device `TokenBucket` rate limiting, plus rarest-first `memoryview('I')` SQLite posting scans (`~3.4–4.6 ms` warm disk queries on `32,445` rules). |

---

## 9. Phase-2 Cognitive Core, Phase-3 & Phase-4 Bottleneck Defenses

See **[`11-mirror-crystal-phase2-phase3-evolution.md`](11-mirror-crystal-phase2-phase3-evolution.md)** for the full architecture and regression benchmarks of:
- **Phase 2:** `PerceptualVQLens` (`INT8` image/waveform VQ patch tokenizer), `SandboxMCTSPlanner` (UCB1 CPU code synthesis & AST self-repair), `CoreferenceAndSenseTracker` (multi-turn pronouns + homonym sense disambiguation), `NightlyDreamSynthesizer` (autonomous multi-hop shortcut discovery), and `UniversalConceptHub` + `FluentByteRealizer` (zero-duplication cross-lingual locking).
- **Phase 3 & Phase 4:** `DreamUtilityPruner`, `TopicShiftDecayGuard`, `BoundedLoopStepGuard` (`_BoundedStepInjector`), `MorphologicalFuzzySpellBridge` (`ContextBitTable.resolve_typo_token`), `EpistemicConfidenceCalibrator`, `D4` camera-rotation invariance, and `SplitBrainCorefBridge` hybrid Edge-Cloud coreference continuity.

### Portable Crystal archive loading

The portable distribution is the four ZIP archives under datasets/crystal_archive/raw/. The CPU MCRYSTAL payload is embedded in crystal_4.zip. MirrorCrystalEngine.load() reads that payload directly from the ZIP, so a separate checked-in artifacts/mirror_crystal_v1.mcrystal copy is not required. Query/chat/continue-teaching accept crystal_4.zip as their brain/load path. See datasets/crystal_archive/DATA_CARD.md and LICENSES.md for the data inventory and source-specific terms.
