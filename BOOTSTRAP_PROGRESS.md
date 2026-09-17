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

Phase 27 structured relocation/control-flow remains subject to the documented CI/artifact closure. Phase 28.1 parser analysis and Phase 28.2 parsed-structure ABI are component-passed. The next active gate is Phase 28.3: make the real MIRR compiler consume the parser-produced structure rather than reparsing raw source.
