# Experiment 6 — Learned Opaque Operations

## Objective

Remove semantic operation names and instead use opaque operation identifiers.

The question is whether the model can learn operation semantics from examples rather than relying on meaningful labels.

## Benchmark

| Parameter | Value |
|---|---:|
| Concepts | 24 |
| Opaque operations | 6 |
| Chain lengths | 1–5 |
| Sequences per concept/length | 16 |
| State dimension | 64 |
| Operation embedding | 32 |
| Default epochs | 60 |
| Batch size | 128 |
| Learning rate | 5e-4 |
| Negative margin | 0.2 |

The benchmark held out selected concepts and longer/higher-order operation patterns.

## Immutable Exp7 baseline

For the frozen Experiment 7 comparison benchmark, the Experiment 6 baseline was:

| Metric | Baseline |
|---|---:|
| Length-5 accuracy | 63.33% |
| Unseen-concept accuracy | 65.33% |

## Interpretation

The model can learn opaque operation semantics without meaningful operation names, but deeper composition remains the limiting factor.

That result motivated the residual transition tested in Experiment 7.
