# Experiment 1 — Native Semantic State

## Objective

Test whether a compact recurrent model can map user and target text into a useful shared semantic state and learn a predictive transition from user state to target state.

## Configuration

| Parameter | Value |
|---|---:|
| Embedding | 96 |
| Hidden | 256 |
| GRU layers | 2 |
| State dimension | 128 |
| Max bytes | 512 |
| Batch | 64 |
| Learning rate | 3e-4 |
| Epochs | 5 |
| Loss | Symmetric contrastive + transition cosine |

## Result

| Metric | Mean | Std |
|---|---:|---:|
| Top-1 retrieval | 14.27% | 0.65% |
| Top-5 retrieval | 25.47% | 0.69% |
| Semantic margin | 0.2209 | 0.0003 |
| Transition margin | 0.1150 | 0.0048 |

## Interpretation

The important result is reproducible learned structure above the random baseline, especially the semantic and transition margins.

It is not evidence of general reasoning or long-form language generation.
