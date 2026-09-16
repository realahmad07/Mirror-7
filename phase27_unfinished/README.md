# Phase 27 — Structured Relocation

**Status: active / unfinished until executable acceptance tests pass.**

This stage adds structured `IF / ELSE / THEN` compilation and relocation. The compiler must propagate every malformed descendant failure to the top-level result and must not depend on fixed absolute branch addresses.

The source in this directory is retained as the active implementation baseline. The acceptance test specification is in `PHASE27_BOOTSTRAP_TESTS.md`.

## Promotion rule

Phase 27 becomes PASS only when the actual compiler path demonstrates:

1. valid structured programs compile;
2. malformed control flow fails at the top level;
3. nested failures propagate outward;
4. generated branch operands are relocated from emitted positions rather than copied from stale absolute addresses;
5. strict compilation, regression, randomized/fuzz, and sanitizer checks pass.

Until then, this directory deliberately remains labelled unfinished.
