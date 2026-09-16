# Mirror-7 Phase 13 Heritage

This directory preserves the pre-MIRR/Nucleus development line from the Phase 13 final archive.

## Why it is kept

Phase 13 contains architectural evidence that remains useful to the current bootstrap program:

- learned native-encoding artifacts and deterministic raw-ELF emission
- data-driven compiler semantics, where source-language rules live in an artifact rather than hard-coded compiler branches
- malformed-artifact rejection and randomized data-only extension tests
- explicit documentation of the boundary that was still external: the generic compiler engine itself
- V122-V129 bootstrap experiments leading toward artifact-driven self-construction

## Integration boundary

The historical Python implementation is **reference material, not part of the active MIRR compiler build**. Keeping it under `phase13_heritage/` avoids contaminating the current compiler/runtime while making the earlier invariants available for comparison and regression work.

The current acceptance path remains the Phase 24+ MIRR/Nucleus chain. The Phase 13 material strengthens that chain by preserving prior evidence and by providing a concrete reference model for future artifact/bootstrap work.
