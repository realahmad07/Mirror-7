# MIRROR7 Bootstrap Progress

This branch is the active engineering line for closing the bootstrap chain.

## Required order

1. Close Phase 27 structured relocation.
2. Integrate the Phase 28 surface parser into the actual MIRR compiler.
3. Establish a compiler implementation that is entirely represented in MIRR source.
4. Separate source-dictionary execution from fresh target-dictionary generation.
5. Replace hard-coded absolute branch operands with symbolic relocation.
6. Prove fresh-stage bootstrap.
7. Prove self-recompile.
8. Prove byte-identical fixed point.
9. Prove independent rebuild.
10. Prove independent verification.

## Verification rule

A checkbox is promoted only after an executable test demonstrates the corresponding invariant. Source presence, a successful host build, or a historical report is not sufficient evidence.

## Current boundary

The repository currently contains the Phase 27 unfinished compiler and a separately verified Phase 28 parser component. The active work must integrate these into the real compiler path rather than treating the parser as a standalone substitute.
