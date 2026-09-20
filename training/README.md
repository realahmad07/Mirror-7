# Mirror 7 Training Foundation

This directory contains the pre-training and training infrastructure for the learned language/response layer.

## Architecture boundary

The training layer is an optional learned realization component. It does not replace the tested Mirror 7 runtime.

    user text + Mirror state/goal/evidence
                     |
                     v
           learned response realization
                     |
                     v
             response text / bytes

Mirror 7's explicit mechanisms remain responsible for state, prediction, reasoning, memory, planning, actions, and other tested capabilities.

## Byte-level design

The baseline model operates directly on UTF-8 bytes plus a small set of control symbols. It does not depend on a subword tokenizer.

This keeps the first training experiment aligned with the project's exploration of alternative representations while still giving the system a trainable language interface.

## Dataset contract

Training data is JSONL. Each record contains:

- example_id: globally unique stable identifier.
- user_text: user input.
- target_text: desired response.
- goal: optional goal string.
- state: JSON-serializable Mirror state summary.
- evidence: optional JSON-serializable reasoning/evidence summary.
- actions: optional legal/selected actions.
- split: train, validation, or test.
- quality: optional 0..1 data-quality weight.

See training/schema.py.

## First training rule

Do not train directly on generated benchmark answers or leak held-out evaluator information into the training set.

The training pipeline keeps train/validation/test identifiers separate and reports split counts before training.

## Colab

Open training/colab/mirror7_training.ipynb in Google Colab, connect a GPU runtime, and run the cells in order.

The notebook is intentionally a normal interactive training notebook rather than a background service.

## Training artifact

The trainer writes a checkpoint containing:

- model configuration
- model state
- training arguments
- vocabulary/control-symbol definition
- dataset fingerprint
- validation metrics

The artifact is a learned component, not a replacement for the Mirror 7 backend or cognitive runtime.
