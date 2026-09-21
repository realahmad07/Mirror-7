# Mirror 7 — Training Session Handoff

## Handoff state

The September 2026 first training cycle is complete. This file is retained as the historical operational handoff; the authoritative current research summary is under docs/04-training.md and docs/research/.

## Compute

- Google Colab with NVIDIA A100-SXM4-40GB was used.
- Google Drive stored datasets and checkpoints.
- GitHub is the source of truth for code and documentation.

## Dataset

Current curated corpus:

- mirror_oasst2_curated_v1.jsonl
- 14,627 total examples
- 11,703 train
- 1,455 validation
- 1,469 test
- fingerprint e7d66d84725bfc9347ea7b8656bfb815625f80adc81cd05d1ea868e6585c4715a

The earlier filtered corpus was preserved rather than overwritten. Conservative curation removed 174 records from the 14,801-record filtered version.

## Scratch response-model route

A byte-level GRU route was tested first.

Verified target configuration:

- 10,574,855 parameters
- embedding 256
- hidden 1,024
- 2 layers
- dropout 0.10
- maximum sequence 4,096 bytes

Checkpoint:

/content/drive/MyDrive/Mirror7/mirror7_response_10m_v1.pt

The route improved training/validation loss but did not pass held-out free-running generation. The evaluation showed non-empty output but exact match 0 and repetitive/generic collapse in the sampled held-out evaluation.

Diagnostics showed:

- encoding/loss alignment correct;
- prompt-sensitive hidden states;
- tiny-set overfitting works;
- teacher-forced prediction substantially stronger than free-running continuation;
- scheduled sampling and other recovery attempts did not solve the central problem.

## Native semantic-state route

The native-brain model used:

- byte vocabulary 257;
- embedding 96;
- hidden 256;
- 2 GRU layers;
- state dimension 128;
- dropout 0.10;
- max input 512 bytes;
- AdamW;
- learning rate 3e-4;
- batch size 64;
- five default epochs.

Checkpoint:

/content/drive/MyDrive/Mirror7/mirror7_native_brain_v1.pt

The seven experiments are documented individually under docs/research/.

## Final training-cycle conclusion

The cycle established reproducible semantic-state and bounded transformation learning but did not produce a strong standalone native language model.

Do not restart the same failed scratch-language route merely by increasing parameter count.

Start the next cycle only after a new mechanism, frozen benchmark, and explicit failure hypothesis are defined.

## Security

The original Colab notebook contained an exposed GitHub credential in a cell. The repository copy is sanitized. The credential should be revoked/rotated independently of source cleanup.
