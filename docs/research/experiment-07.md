# Experiment 7 — Residual State Transition

## Objective

Test whether a residual update can make repeated state transformations more stable.

Conceptually:

z(next) = Normalize(z(current) + Residual(z(current) + z(operation)))

## Immutable comparison

| Metric | Exp6 baseline | Exp7 residual |
|---|---:|---:|
| Length-5 accuracy | 63.33% | 63.33% |
| Unseen-concept accuracy | 65.33% | 65.56% |

Exp7 three-seed length-5 standard deviation: 0.0056.

The pilot also reached 63.89% length-5 and 66.67% unseen concepts before the final three-seed validation.

## Interpretation

The residual design did not produce a decisive improvement on the primary depth benchmark.

It matched the length-5 baseline and slightly improved unseen-concept performance.

The correct conclusion is that residual transitions are a viable stability mechanism while the depth/generalization bottleneck remains unresolved.

## Checkpoint

/content/drive/MyDrive/Mirror7/mirror7_exp7_pilot.pt

## Research handoff

Experiment 7 is the end of the first native-brain training cycle. Further work should introduce a new hypothesis or benchmark rather than scaling the same architecture without new evidence.
