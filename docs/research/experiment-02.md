# Experiment 2 — Transition-Weighted Objective

## Objective

Increase the pressure on transition prediction while keeping the architecture at comparable scale to Experiment 1.

## Result

| Metric | Experiment 1 | Experiment 2 |
|---|---:|---:|
| Transition margin | 0.1150 | 0.1605 |
| Top-1 retrieval | 14.27% | 12.93% |

The transition margin improved by roughly 39.5% relative to Experiment 1.

## Interpretation

The objective can trade retrieval alignment for stronger transition prediction.

This validated that changing the objective changes what the compact state model learns, but it did not establish an overall win because retrieval weakened.
