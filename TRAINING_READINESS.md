# Mirror 7 Training Readiness

## What is ready now

The repository now contains a complete first training pipeline for an optional learned response realization layer.

Flow:

validated JSONL
      |
      v
byte-level context/goal/user encoding
      |
      v
ByteGRU language model
      |
      v
GPU training
      |
      v
checkpoint + dataset fingerprint
      |
      v
held-out evaluation
      |
      v
post-training regression
      |
      v
workspace integration

## What still requires a human action

The physical steps are intentionally minimal:

1. Open the Colab notebook.
2. Select a GPU runtime.
3. Put the reviewed real training JSONL where the notebook can read it.
4. Run the training/evaluation cells.
5. Preserve the checkpoint and evaluation output.

The repository already contains the code and notebook needed for those steps.

## Important Colab constraint

Colab provides GPU access but resource availability and limits vary over time. The project therefore treats Colab as the training environment, not as the permanent production backend.

## Dataset requirement

The smoke dataset is only for pipeline verification. It is not the production training corpus and should never be presented as evidence of model competence.


## Current training session — 20 September 2026

The exact continuation point is recorded in [TRAINING_SESSION_HANDOFF_2026-09-20.md](./TRAINING_SESSION_HANDOFF_2026-09-20.md).

Current status:
- Colab Pro + NVIDIA A100-SXM4-40GB verified.
- OASST2 filtered corpus validated and preserved.
- Curated training set: 14,627 examples.
- 1.7M baseline trained but not accepted because held-out generation remained incoherent.
- Tiny 5-example diagnostic reached 100% training exact match, proving the core training/checkpoint path works.
- Training/EOS/shuffling/evaluation/validator fixes have been pushed.
- Next target: 10.57M-parameter ByteGRU using the curated dataset, followed by held-out evaluation.

Quality rule: quality over quantity. Dataset heuristics are reviewed conservatively; simple repetition flags are not automatic rejection rules.
