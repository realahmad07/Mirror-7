# Mirror 7 Training Readiness — Updated

## Current state

The first full training cycle is complete.

### Dataset

- Curated file: mirror_oasst2_curated_v1.jsonl
- Examples: 14,627
- Train / validation / test: 11,703 / 1,455 / 1,469
- Fingerprint: e7d66d84725bfc9347ea7b8656bfb815625f80adc81cd05d1ea868e6585c4715a

### Hardware

- Google Colab
- NVIDIA A100-SXM4-40GB
- Google Drive for persistent datasets/checkpoints

### Scratch response model

- 10,574,855 parameters
- 256 embedding
- 1,024 hidden
- 2 GRU layers
- 0.10 dropout
- 4,096-byte maximum sequence
- byte-level vocabulary with control symbols

The model trained successfully but did not pass the important free-running held-out generation requirement.

### Native semantic-state model

- 257 byte vocabulary
- 96 embedding
- 256 hidden
- 2 GRU layers
- 128-dimensional state
- 0.10 dropout
- 512-byte maximum input
- AdamW
- 3e-4 learning rate
- batch 64
- five default epochs
- parameter budget below 1.5M

Experiments 1–7 are complete and documented under docs/research/.

## Current checkpoint policy

Recorded Colab/Drive checkpoints include:

- mirror7_response_10m_v1.pt
- mirror7_native_brain_v1.pt
- mirror7_exp6_pilot.pt
- mirror7_exp7_pilot.pt

These remain research artifacts rather than production assets.

## Next training gate

Do not start another expensive A100 run until:

1. the new hypothesis is explicit;
2. the benchmark is frozen;
3. the baseline is frozen;
4. the expected failure signal is measurable;
5. the evaluation code is fixed before training.

## Notebook

The sanitized source-control notebook is:

training/colab/mirror7_training.ipynb

The publication copy has been scrubbed for the credential found in the original Colab session.
