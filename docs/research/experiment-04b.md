# Experiment 4B — Transformation-Specific Contrastive Objective

## Objective

Fix the collapse observed in Experiment 4 by rewarding the transformed state for matching its target while penalizing similarity to the original input state.

## Training design

Recorded objective:

L = (1 - sim(z', z_target)) + max(0, sim(z', z_input) - sim(z', z_target) + gamma)

with gamma = 0.2.

Recorded training settings:

| Parameter | Value |
|---|---:|
| Learning rate | 5e-4 |
| Batch size | 64 |
| Model parameters | 757,984 |

## Three-seed result

| Metric | Result |
|---|---:|
| Compositional margin | 0.2967 mean |
| Held-out Top-1 | 87.18% |
| Order margin | 0.0949 |

## Interpretation

This was the clearest native-brain compositional result in the first training cycle.

The result is specific to the defined state-transformation benchmark, not open-domain reasoning.
