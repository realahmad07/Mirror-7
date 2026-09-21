# Experiment 3 — Explicit Learned State Transformations

## Objective

Test whether the state model can learn explicit operations such as inversion and intensification instead of only matching user/target pairs.

## Controlled task

The benchmark used operation identifiers representing transformations such as INVERT and INTENSIFY, with held-out pairings and negative comparisons.

## Pilot evidence

The early pilot exposed a useful failure signal: predicted state variance was only 0.002532, mean pairwise similarity was 0.6759, transformation distinctness was 0.3241, transformation accuracy was 0%, and the negative margin was -0.3019.

That failure was used to change the objective rather than being hidden.

## Gated result

Three-seed validation produced a mean transformation margin of approximately 0.2720 with low seed variance and passed the defined margin gate.

## Interpretation

This supports learnable explicit state transformation on the defined synthetic benchmark.

It does not support a general reasoning claim outside that benchmark.
