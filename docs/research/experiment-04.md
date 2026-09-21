# Experiment 4 — Sequential State Transformation

## Objective

Compose multiple transformations through repeated updates:

z0 → Op1 → z1 → Op2 → z2

## Finding

The pilot showed representation collapse.

Target similarity increased, but similarity to the original input also remained high. The model was not cleanly separating the transformed state from the source state.

## Why it mattered

Success at one transformation step does not imply compositional depth.

The collapse finding led directly to Experiment 4B, which changed the loss to explicitly separate transformed state from the input state.
