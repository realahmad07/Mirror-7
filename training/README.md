# Mirror 7 Training Foundation

This directory contains the pre-training and training infrastructure for learned language/response realization and native semantic-state research.

## Boundaries

The training layer is optional research infrastructure. It does not replace the tested Mirror 7 runtime.

## Main code

| File | Role |
|---|---|
| bytes.py | Control-token/byte encoding primitives. |
| schema.py | Training-record schema. |
| dataset.py | JSONL loading, fingerprints, and split handling. |
| model.py | ByteGRU language model and checkpoint boundary. |
| train.py | Scratch response-model trainer. |
| evaluate.py | Held-out response evaluation. |
| validate_dataset.py | Dataset contract checks. |
| native_brain.py | Compact semantic-state learner. |
| experiment6_learned_operations.py | Opaque-operation research model. |

## Dataset contract

A training record can contain:

- stable example id;
- user text;
- target text;
- optional goal;
- optional state;
- optional evidence;
- optional actions;
- train/validation/test split;
- optional quality score.

The pipeline keeps split identity explicit and computes a deterministic dataset fingerprint.

## Model boundary

Scratch response models emit bytes or text.

Native-brain experiments learn compact state and transition representations.

Neither silently replaces the backend runtime.

## Colab

The canonical interactive training notebook is training/colab/mirror7_training.ipynb.

GPU access is treated as an experimental resource. Persistent checkpoints should remain in Drive or a dedicated artifact store, not in source control.

## Research rule

Do not accept a training run because the loss improved alone. Require held-out behavior, diagnostics, controls, and comparison with a frozen baseline.

## Native Conversation V5

Native Conversation V5 is integrated in `native_conversation_v5.py`, with the portable GPU trainer in `gpu_train.py`, the Colab notebook in `colab/mirror7_gpu_training.ipynb`, knowledge-transfer helpers under `knowledge_transfer/`, and independent evaluation under `evaluation_suite/`. The enhanced configuration has 974,019 parameters. Fast CI, Security Guard, and Native Brain Training checks passed on the integration PR head. A local CPU smoke covered checkpoint save/resume and changed-dataset rejection. A full GPU run and improved free-running response quality have not yet been established. Read [the V5 native model review](../docs/V5_NATIVE_MODEL_REVIEW_2026-11-01.md) and [Phase 367 acceptance](../PHASE367_ACCEPTANCE.md).
