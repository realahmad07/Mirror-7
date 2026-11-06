# PHASE 2 — DATA AND NATIVE MODEL IMPROVEMENT REPORT

## 1. Empirical Dataset Audit

An exhaustive audit of `training/data/mirror_oasst2_curated_v1.jsonl` was conducted on all 14,627 records.

### Corpus Summary
| Metric | Value |
|---|---|
| Total Examples | 14,627 |
| Splits | Train: 11,703 (80.0%), Validation: 1,455 (9.95%), Test: 1,469 (10.05%) |
| Unique IDs | 14,627 (100% unique) |
| Unique Targets | 14,627 (100% unique) |
| Unique Inputs | 14,341 (98.0% unique) |
| Cross-Split Leakage | **0 examples** (strict split isolation verified) |
| Quality Scores | Uniform 1.0 |

### Length Distribution Analysis
| Text Type | Min | 25th Pct | Median | 75th Pct | 95th Pct | Max | Mean |
|---|---|---|---|---|---|---|---|
| **User Prompt (Bytes)** | 1 | 47.0 | 79.0 | 141.0 | 342.7 | 9,890 | 130.5 |
| **Target Response (Bytes)** | 1 | 345.0 | 867.0 | 1,498.5 | 2,621.7 | 9,950 | **1,037.8** |

### Critical Truncation Bottleneck Uncovered
The historical CPU training runs configured `max_bytes = 256` or `512` due to CPU compute limits. Our empirical audit reveals the severe cost of this constraint:

| Cutoff Setting | Targets Truncated | Total Target Bytes Discarded | Impact on Learning |
|---|---|---|---|
| **256 bytes** (CPU default) | **80.7%** (11,809 / 14,627) | **77.8%** (11.80 MB lost) | **Severe truncation collapse**: The model is forced to place `<EOS>` mid-sentence on 80.7% of examples, training the network that sentences terminate abruptly without proper closure or punctuation. |
| **512 bytes** (V5 run) | **65.6%** (9,591 / 14,627) | **59.8%** (9.08 MB lost) | Over half of conversational context is lost. |
| **1024 bytes** | **43.4%** (6,352 / 14,627) | **33.1%** (5.03 MB lost) | Acceptable for short-to-medium dialogues. |
| **2048 bytes** (GPU target) | **11.8%** (1,722 / 14,627) | **8.3%** (1.27 MB lost) | **91.7% of all information preserved**. Natural paragraphs, step-by-step reasoning, and complete code blocks survive intact. |

---

## 2. Algorithmic Root-Cause Analysis

Beyond dataset truncation, our audit analyzed why from-scratch byte-level generation struggles on conversational quality:

### A. The Byte-Level Prediction Horizon
A word of 6 characters requires 6 sequential autoregressive predictions. A 100-word response requires ~600 steps. In recurrent networks (GRU/LSTM):
1. Hidden state drift accumulates over 600 steps.
2. Gradient flow across 600 steps suffers from vanishing gradients.
3. In V3, cross-attention context was only fed to the final linear projection, not into the recurrent cell. After step 30, the recurrent state drifted into an unconditional language prior.
4. V5 solved this structurally by feeding `[embedding; context]` into `GRUCell` at every step, but 600-step sequences still require sufficient parameter capacity and length preservation.

### B. Teacher-Forcing vs. Free-Running Discrepancy (Exposure Bias)
During training (teacher-forcing), the decoder always receives ground-truth previous bytes. During inference, any single mispredicted byte (e.g. an incomplete UTF-8 byte sequence or typo) pushes the hidden state into out-of-distribution space, triggering cascade failure.
- **Solution**: Implement **Scheduled Sampling** (decaying ground-truth probability $p_{tf}$ from 1.0 to 0.7 during training) to expose the model to its own previous predictions.

### C. Byte-Level Repetition Penalty Distortions
Standard token-level repetition penalties divide logit probabilities of previously emitted tokens. However, in **byte-level models**:
- English text frequently repeats common ASCII bytes (`e`, `t`, `a`, `o`, space, `\n`) within normal non-repetitive prose.
- Multi-byte UTF-8 characters repeat continuation bytes (`0x80`-`0xBF`).
- Penalizing individual bytes distorts the fundamental letter frequency of the language, producing misspelled words and gibberish.
- **Solution**: Repetition penalties must be applied at the **character n-gram level** (e.g., 4-gram blocking or word-level tracking) rather than single-byte down-weighting.

### D. Structured Semantic-State Fusion
Mirror 7 already has a rich semantic state inducer (`phase343_semantic_state`). Currently, native conversation models receive only the raw `user_text`.
- **Solution**: Concatenate canonical context (`TrainingExample.canonical_context()`, containing goal, state, and entities) into the encoder input so the neural model is conditioned on Mirror 7's structured symbolic understanding.

---

## 3. Scalable Dynamic Batching Pipeline

To eliminate truncation loss and prepare for GPU execution:
1. **Bucketed Length Batching**: Group sequences of similar length together to minimize padding waste.
2. **Dynamic Cutoff**: Set sequence ceiling to 2048 bytes on GPU (retaining 91.7% of target content).
3. **Complete Word & EOS Boundary Invariance**: Use `_clean_truncate` to ensure truncation never cuts inside UTF-8 multi-byte sequences or inside alphanumeric words.
