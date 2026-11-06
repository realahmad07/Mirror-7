# Training and Model Development

## Training philosophy

The first training cycle followed one rule: quality over quantity.

Every route was treated as an experiment with controlled data preparation, held-out evaluation, diagnostics, explicit failure criteria, preserved checkpoints, and no deliberate benchmark leakage.

The source-control notebook under training/colab/ contains the detailed Colab record. A sanitized copy is maintained in the repository.

## Dataset

### Curated corpus

| Property | Value |
|---|---|
| Source family | OASST2-derived curated corpus |
| File | mirror_oasst2_curated_v1.jsonl |
| Total examples | 14,627 |
| Train | 11,703 |
| Validation | 1,455 |
| Test | 1,469 |
| Fingerprint | e7d66d84725bfc9347ea7b8656bfb815625f80adc81cd05d1ea868e6585c4715a |

The data contract contains a stable example id, user text, target text, optional goal/state/evidence/actions, an explicit split, and optional quality metadata.

### Data-quality rule

Repetition heuristics were intentionally conservative. A suspicious repeated n-gram was not automatically deleted without measuring whether the filter improved the validation regime.

## Scratch response model

### Model family

The first native-language attempt was a byte-level GRU using UTF-8 bytes plus a small control-symbol vocabulary rather than a subword tokenizer.

### 10.57M-class target

| Parameter | Value |
|---|---:|
| Target parameter count | 10,574,855 |
| Embedding | 256 |
| Hidden | 1,024 |
| GRU layers | 2 |
| Dropout | 0.10 |
| Maximum sequence | 4,096 bytes |
| Hardware | NVIDIA A100-SXM4-40GB |
| Environment | Google Colab |

Checkpoint recorded in the training session:

/content/drive/MyDrive/Mirror7/mirror7_response_10m_v1.pt

## Scratch-model evaluation

The 10M-class model produced non-empty held-out output consistently, but the important acceptance signal was poor free-running quality.

- 100/100 held-out generations were non-empty in the sampled evaluation.
- Exact match remained 0.
- Generic or repetitive collapse appeared.
- Teacher-forced response prediction was much stronger than free-running continuation.
- Prompt conditioning existed in diagnostics.
- Tiny-data overfitting succeeded.

The decisive issue was therefore not simply “the code cannot learn”.

## Diagnostic program

The training notebook records a long sequence of targeted tests.

### Confirmed healthy paths

1. Encoding and loss alignment were correct.
2. Teacher-forced and first-generation-step logits agreed.
3. Hidden states changed with the prompt.
4. A tiny 20-example dataset could be overfit.
5. Semantic embedding diagnostics showed useful signal between correct prompt/response pairs and random pairs.

### Confirmed weak paths

1. First-byte prediction on the full distribution was weak.
2. Prompt conditioning weakened during free-running generation.
3. Teacher-forced accuracy was far above free-running exact-prefix performance.
4. Scheduled sampling and prefix noise gave only modest gains.
5. Two-stage response planning did not remove collapse.
6. Hand-coded response plans were weak.
7. Retrieval and reranking did not become sufficient generation methods.
8. BPE compression did not solve free-running collapse.
9. A small non-autoregressive Transformer did not solve the task.
10. Explicit response-state and curriculum variants did not solve the central problem.

## Native semantic-state branch

The research then shifted from long-form text generation toward compact learned state.

### Default configuration

| Parameter | Value |
|---|---:|
| Byte vocabulary | 257 |
| Embedding | 96 |
| Hidden | 256 |
| GRU layers | 2 |
| State dimension | 128 |
| Dropout | 0.10 |
| Max input bytes | 512 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Batch size | 64 |
| Epochs | 5 |
| Loss | Symmetric in-batch contrastive + transition cosine |
| Parameter budget | Below 1.5M |

Checkpoint:

/content/drive/MyDrive/Mirror7/mirror7_native_brain_v1.pt

## Native-brain outcome

Experiment 1 established reproducible semantic-state structure.

Experiment 2 strengthened transition prediction at a retrieval trade-off.

Experiments 3–5 established explicit and variable-depth transformations on controlled synthetic data.

Experiments 6–7 tested opaque operations and residual updates, exposing a remaining depth/generalization bottleneck.

## Experiment 6 parameters

| Parameter | Value |
|---|---:|
| Concepts | 24 |
| Opaque operations | 6 |
| Maximum chain length | 5 |
| Sequences per concept/length | 16 |
| Concept state dimension | 64 |
| Operation embedding dimension | 32 |
| Training epochs | 60 |
| Batch size | 128 |
| Learning rate | 5e-4 |
| Negative margin | 0.2 |

Checkpoint:

/content/drive/MyDrive/Mirror7/mirror7_exp6_pilot.pt

## Experiment 7

The residual transition used:

z(next) = Normalize(z(current) + Residual(z(current) + z(operation)))

The frozen Exp6 comparison baseline was 63.33% on length-5 and 65.33% on unseen concepts. Exp7 matched length-5 and reached 65.56% on unseen concepts.

Checkpoint:

/content/drive/MyDrive/Mirror7/mirror7_exp7_pilot.pt

## Training conclusion

The first cycle did not produce a ChatGPT-class standalone language model.

It did produce a useful intermediate result: compact semantic-state learning and bounded state transformation are trainable and reproducible, while strong free-running native language generation remains the central unresolved model problem.
