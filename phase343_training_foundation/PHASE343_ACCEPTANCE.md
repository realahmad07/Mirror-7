# Phase 343 — Training Foundation Acceptance

## Objective

Prepare the learned response realization layer for GPU/Colab training without replacing the existing Mirror 7 runtime.

## Acceptance

- Deterministic JSONL dataset schema exists.
- Train/validation/test split validation exists.
- User-input leakage guard exists.
- Byte-level response encoding exists without a subword tokenizer dependency.
- Trainable GPU-compatible baseline model exists.
- Checkpoint contains model configuration, dataset fingerprint, and validation history.
- Held-out evaluation command exists.
- Colab notebook reproduces the training flow.
- GitHub CI runs the training-foundation tests without requiring a GPU.

## Boundary

This phase establishes training infrastructure. It does not claim that the learned model is capable of broad natural-language competence, and it does not establish AGI.

The first real training run must use a separately prepared, reviewed dataset and must be evaluated on held-out data before integration.
