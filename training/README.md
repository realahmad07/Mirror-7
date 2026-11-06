# Mirror 7 Training Foundation & Independent Intelligence Architecture

This directory contains the training and research infrastructure for learned language/response realization, native semantic-state research, knowledge transfer from teacher models, and independent evaluation.

## Architectural Boundaries

The native neural models and reasoning discovery engines are modular layers designed to operate in harmony with Mirror 7's core symbolic runtime (`phase140_unified_cognitive_runtime`). They do **not** replace the backend runtime, nor do they create a permanent dependency on external pretrained models during production inference.

---

## Component Directory Map

| Module / File | Role |
|---|---|
| `bytes.py` | Control-token/byte encoding primitives (256 raw bytes + 4 specials: PAD, BOS, EOS, UNK). |
| `schema.py` | Canonical `TrainingExample` schema with SHA-256 fingerprinting. |
| `dataset.py` | JSONL loading, split isolation validation, and dataset summarization. |
| `native_conversation.py` | Native Conversation V3 encoder-decoder cross-attention model (990,915 params). |
| `native_conversation_v5.py` | Native Conversation V5 Input-Conditioned Recurrent Decoder (957,251 params). |
| `gpu_train.py` | Portable GPU/CPU trainer with mixed precision (AMP), dynamic bucket batching, throughput reporting, and resume support. |
| `colab/mirror7_gpu_training.ipynb` | Google Colab GPU training notebook (T4/V100/A100). |
| `knowledge_transfer/` | Pretrained teacher distillation pipeline across 13 core knowledge categories. |
| `evaluation_suite/` | 10-dimension independent evaluation suite and teacher disconnection audit. |
| `mirror7_backend/reasoning_discovery/` | Autonomous problem exploration, algorithmic strategy synthesis, and safety sandbox. |

---

## 1. Native Conversation V5 Architecture

`native_conversation_v5.py` resolves the context drift bottleneck observed in V3:
- **Recurrent Context Injection:** The decoder utilizes an `nn.GRUCell(embedding_dim + attention_dim, hidden_dim)`. At every autoregressive generation step, cross-attention context computed over encoder states is concatenated directly with the current byte embedding.
- **Unrolled Forward Pass:** The training `forward()` method unrolls step-by-step with the exact same numerical logic as free-running `generate()`, verified by `test_teacher_forced_vs_step_by_step_generation_consistency`.
- **Precomputed Keys/Values:** Encoder keys and values are projected once before decoding, ensuring efficient $O(T_{target} \cdot T_{source})$ attention execution.
- **Strict Parameter Efficiency:** Total parameter count is **957,251**, comparable to V3's 990,915 parameters while providing direct recurrent conditioning.

---

## 2. Knowledge Transfer System (`training/knowledge_transfer/`)

Transfers structured knowledge and reasoning demonstrations from capable teacher models into Mirror 7's native student representation without permanent dependencies.

### 13 Knowledge Categories:
1. `general_conversation`
2. `language_and_writing`
3. `mathematics`
4. `logical_reasoning`
5. `programming_and_debugging`
6. `instruction_following`
7. `planning`
8. `problem_decomposition`
9. `tool_use`
10. `self_correction`
11. `factual_explanation`
12. `ambiguous_requests`
13. `multi_step_problem_solving`

### Pipeline Flow:
```
TaskGenerator (Seed / Variations / Corpus)
       ↓
TeacherProvider (Qwen2.5-7B-Instruct / DeepSeek-R1-Distill-7B / MockTeacher)
       ↓
ResponseValidator (Syntax parse, step verification, length bounds, repetition & deduplication)
       ↓
DatasetNormalizer (Transforms to canonical TrainingExample with provenance metadata)
       ↓
DistillationManager (SFT demonstrations, verified self-corrections, preference pairs)
```

---

## 3. Google Colab GPU Training Pipeline

### Features:
- **CUDA / MPS / CPU Detection**: Automatically configures precision and device allocation.
- **Mixed Precision**: Automatic FP16/BF16 `torch.amp.autocast` with `GradScaler`.
- **Dynamic Bucket Batching**: Sequences grouped by target byte length, eliminating padding waste.
- **Longer Sequence Ceiling**: Supports `max_bytes = 1024` or `2048` on GPU, preserving >91.7% of full conversational responses (eliminating the 80.7% truncation loss caused by CPU 256-byte limits).
- **Execution**: Run locally or in Google Colab:
  ```powershell
  python -m training.gpu_train --dataset training/data/mirror_oasst2_curated_v1.jsonl --output artifacts/mirror7_gpu_v5.pt --epochs 5 --batch-size 64 --max-bytes 1024
  ```

---

## 4. Reasoning Discovery Engine (`mirror7_backend/reasoning_discovery/`)

Enables Mirror 7 to develop, test, and improve its own algorithmic methods rather than purely imitating teacher outputs:
- **Problem Explorer**: Formalizes objectives, separates constraints from assumptions, builds dual representations.
- **Strategy Generator**: Synthesizes inspectable code procedures across algorithmic paradigms (brute force, greedy, bisection, dynamic programming, wheel factorization).
- **Safe Sandbox**: Enforces strict AST security scanning (no OS, filesystem, sockets, or dangerous builtins) with execution timeouts.
- **Strategy Comparator**: Measures empirical speedup ratios and correctness deltas against baselines.
- **Novelty Evaluator**: AST structural similarity metrics and unseen generalization verification.
- **Strategy Memory**: Persistent storage with version lineage and domain indexing.

---

## 5. Independent Evaluation & Teacher Disconnection Suite

Comprehensive evaluation across 10 distinct capability dimensions:
1. Conversational Quality
2. Prompt Understanding
3. Factual Accuracy
4. Mathematical Reasoning
5. Multi-Step Reasoning
6. Coding Ability
7. Planning
8. Tool Use Correctness
9. Generalization
10. Strategy Discovery

### Teacher Disconnection Verification:
To verify complete production independence from teachers:
```powershell
python -m pytest training/evaluation_suite/test_independent_evaluation.py -k test_teacher_disconnection_verifier -v
```
Verifies:
- Checkpoint loads standalone without network or external tokens.
- Zero teacher API calls occur during inference.
- Multi-turn cognitive steps, state snapshots, and backend actions remain 100% operational.

---

## 6. Full Regression Suite Execution

Run all 75 automated test suites:
```powershell
pytest -q
```
*Current test pass rate: 75/75 (100% pass rate in 25.82s).*
