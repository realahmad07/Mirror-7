# Native V5 Kaggle Training Run — 2026-11-02

## Purpose

This run tests the 974,019-parameter Native V5 conversation model as a standalone student initialized from its own native architecture. Qwen2.5-1.5B-Instruct is used temporarily as a teacher to produce text targets; its weights are unloaded before native student initialization and are not part of Mirror inference.

## Environment

- Platform: Kaggle
- Accelerators: 2 × NVIDIA Tesla T4 (16 GB class each)
- Repository checkout: `/kaggle/working/Mirror-7`
- Curated data directory: `/kaggle/working/Mirror7_Curated_v5_Final/curated_v5_final`
- Dataset release: Curated V5.4
- Native model configuration: 974,019 parameters

## Data preparation snapshot

The V5.4 manifest reports 12,213 curated records across conversation, logic, science, quantum, math, and coding domains. The pipeline converted the data into native prompt/target pairs and removed 1,061 prompt collisions.

Teacher-target generation completed for 582 of 600 selected prompts. The pipeline reused these saved targets on its retry rather than generating them again. The resulting native distillation dataset passed validation with 20,110 examples.

These counts describe the prepared experiment data; they do not establish model quality or benchmark performance. The V5.4 manifest also marks the corpus as not yet training-ready pending human review and benchmark gates. Treat this run as experimental.

## Training status

At the latest recorded checkpoint (2026-11-02), **native model training had not started**. Dataset preparation and teacher-target generation completed, Qwen was unloaded, and the native dataset validation passed. Training import then failed because the T4 patch had inserted duplicated `if return_attentions:` lines.

The launcher patch was corrected to normalize the attention guard idempotently. Both `training/native_conversation_v5.py` and `Mirror_Native_Training_FINAL.py` passed Python syntax compilation after the repair. This is a syntax check only; it is not evidence that GPU training has run successfully.

## Checkpoint policy

Training checkpoints and logs for this run belong under this directory, separate from the source model and curated datasets. The repository is public: any checkpoint committed here is publicly downloadable.

- Do not commit raw datasets, Hugging Face caches, Kaggle credentials, access tokens, or private user data.
- Commit model checkpoints only after confirming the file size is supported by ordinary Git. For files above GitHub's regular per-file limit, use Git LFS and verify its quota before training.
- Preserve the best-validation checkpoint and the latest resumable checkpoint separately.
- Include the training step/epoch, dataset fingerprint, model configuration, optimizer/scheduler/scaler state availability, and validation metrics in each checkpoint manifest.
- A successful GitHub push is a backup, not proof that the checkpoint can resume. Test restore from a downloaded copy.

## Recovery and synchronization

The Kaggle working directory is ephemeral when the session ends. The training process must save checkpoints locally first, then synchronize completed checkpoint files to this run directory and push them to the repository. Sync only after a checkpoint file is fully written; use a temporary file plus atomic rename where the trainer supports it.

Recommended repository layout:

```text
training_runs/
  2026-11-02-native-v5-kaggle/
    README.md
    checkpoints/
      best/
      latest/
    logs/
    manifests/
```

Do not push on every optimizer step. Use periodic resumable checkpoints (for example, every 100 optimizer steps) plus end-of-epoch/best checkpoints, and push only newly completed files. The actual save interval must be configured in the trainer and verified from its source before launch.

## Current limitations

- No successful native training epoch is recorded yet.
- No validation loss, generation quality, or benchmark result is claimed for this run.
- Teacher-generated targets are not independently verified for factual or mathematical correctness.
- The dataset remains experimental until the stated human-review and benchmark gates are completed.
