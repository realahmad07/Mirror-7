# Experiment 5 — Variable-Length Transformation Chains

## Objective

Move beyond fixed two-step compositions and test chains of length 1–4 with inversion, intensification, decrease, and reset operations.

## Held-out design

The benchmark withheld selected concepts and longer operation sequences, including defined collision cases.

## Pilot

The post-training held-out pilot reported:

| Length | Margin | Accuracy |
|---:|---:|---:|
| 3 | 0.2343 | 56.25% |
| 4 | 0.2539 | 68.10% |

## Three-seed validation

The three seeds produced mean accuracy 60.71% with standard deviation 0.0226. Mean margin remained positive.

## Interpretation

The model retained useful multi-step state-transition behavior as depth increased, but accuracy remained imperfect. The result supports bounded multi-step generalization, not general reasoning depth.
