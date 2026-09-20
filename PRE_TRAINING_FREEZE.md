# Mirror 7 — Pre-Training Freeze

## Purpose

This document marks the GitHub-side engineering boundary before GPU training begins.

## Frozen before training

- Mirror 7 cognitive/runtime interfaces.
- Backend session lifecycle and HTTP API contract.
- Browser-facing CORS/security boundary.
- Checkpoint and action-gateway boundaries.
- Package metadata and container entrypoint.
- Existing phase acceptance suites and regression policy.
- The separation between explicit Mirror mechanisms and learned components.

## Training is the next major stage

GPU/Colab work begins only after this engineering boundary is accepted. Training may add learned language, representations, priors, response behavior, or other bounded learned components, but it must integrate through the existing interfaces rather than silently replacing the tested Mirror 7 architecture.

## Not claimed

This freeze does not mean Mirror 7 is already a finished trained assistant, nor does it establish AGI. It means the software-side foundation is prepared for the training stage.

## Launch sequence after the freeze

```text
engineering complete
       ↓
pre-training freeze
       ↓
GPU/Colab training
       ↓
post-training evaluation
       ↓
regression + security + load testing
       ↓
release candidate
       ↓
public launch
```
