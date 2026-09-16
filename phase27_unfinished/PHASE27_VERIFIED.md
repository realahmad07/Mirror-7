# Phase 27 Verified Working State

This marker records the current verified working artifact retained before fresh-bootstrap completion.

## Policy

This work remains on the bootstrap-work branch and is not promoted to `main` until the fresh bootstrap acceptance chain is complete.

## Acceptance chain

1. Phase 27 structured relocation/control flow
2. Phase 28 integrated surface parser
3. Compiler entirely in MIRR
4. Separate source/target dictionary ABI
5. Remove hard-coded absolute branch dependency
6. Fresh-stage bootstrap
7. Self-recompile
8. Byte-identical fixed point
9. Independent rebuild
10. Independent verification

A stage is promoted only after executable verification, regression testing, malformed-input testing, fuzzing, and sanitizer testing where applicable.