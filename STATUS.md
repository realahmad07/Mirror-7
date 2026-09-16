# MIRROR7 — Detailed Source and Phase Status

This is the canonical human-readable status page for the repository. It deliberately distinguishes implementation history, component verification, and bootstrap completion.

## Legend

- ☑ **PASS recorded** — preserved artifacts contain evidence of a successful historical test.
- ☑ **Component verified** — the named component has passed its own tests, but the larger integration gate remains open.
- ☑ **Implemented; verification required** — implementation exists but is not currently promoted to PASS.
- ☐ **UNFINISHED / NOT VERIFIED** — the acceptance gate is still open.

## Acceptance board

| Gate | State | Requirement |
|---|---:|---|
| Phase 24 — runtime dictionary | ☑ | Runtime dictionary implementation and preserved tests. |
| Phase 25 — tokenization + lookup + compilation | ☑ | Implementation preserved; clean-room verification remains a separate requirement. |
| Phase 26 — integrated native compiler path | ☑ | Implementation preserved; clean-room verification remains a separate requirement. |
| Phase 27 — structured relocation/control flow | ☐ | Must pass actual compiler-level valid, malformed, nested, stress and sanitizer tests. |
| Phase 28 — surface parser | ☑ | Standalone parser component verified by strict build, regression/fuzz, and sanitizers; end-to-end compiler integration remains part of the bootstrap gate. |
| Compiler entirely in MIRR | ☐ | No hidden host compiler implementation may remain in the accepted self-hosting path. |
| Separate source/target dictionary ABI | ☐ | Compiler dictionary and fresh generated target dictionary must be independently selectable. |
| Remove hard-coded absolute branch dependency | ☐ | Control-flow targets must be generated/relocated from symbolic structure or equivalent position-independent metadata. |
| Fresh-stage bootstrap | ☐ | A fresh stage must rebuild the compiler from the accepted bootstrap substrate. |
| Self-recompile | ☐ | The MIRR compiler must compile its own source through the same bootstrap path. |
| Byte-identical fixed point | ☐ | Repeated self-recompilation must produce identical bytes under fixed build inputs. |
| Independent rebuild | ☐ | A separate rebuild path must reproduce the same accepted artifact. |
| Independent verification | ☐ | Verification must be independently executable and not rely solely on builder assertions. |
| Bootstrap complete | ☐ | All preceding gates must be green. |

## Current engineering boundary

The next implementation work is compiler-in-MIRR self-hosting. Two architectural defects must be eliminated before that can be promoted:

1. **Absolute branch dependency:** old generated branch offsets can become stale when primitive/dictionary layout changes. This must be replaced or contained by real relocation based on the current generated layout.
2. **Single-dictionary ABI:** compiler execution and compiler output cannot safely share an implicit dictionary when the goal is a fresh target rebuild. The runtime needs explicit source/executable-dictionary and target/generated-dictionary context.

These are bootstrap correctness issues, not documentation issues.

## Phase 28 component evidence

The surface parser component in `phase28_surface_parser/parser.c` is tested for:

- valid definitions and nested control flow;
- invalid names and malformed numeric literals;
- `ELSE` outside `IF`;
- duplicate `ELSE`;
- unmatched `THEN`;
- unclosed `IF`;
- empty branches;
- malformed nested constructs;
- randomized parser input;
- strict C17 warnings-as-errors;
- AddressSanitizer and UndefinedBehaviorSanitizer execution.

The CI workflow records the component result. It is not used as evidence that the entire self-hosting compiler has already closed the Phase-28/29 boundary.

## Repository/UI policy

Source is stored as individual repository files, never as a ZIP dependency. Documentation is Markdown so it remains inspectable and versioned. The browser UI is a presentation layer over repository status; it does not replace executable tests or acceptance evidence.

## Debugging protocol

For every failure:

```text
reproduce
  ↓
minimize
  ↓
trace exact boundary
  ↓
inspect ABI / layout / ownership assumptions
  ↓
consult authoritative technical references when useful
  ↓
minimal justified fix
  ↓
rerun failing case
  ↓
regression
  ↓
combination / fuzz / sanitizer
  ↓
promote only with evidence
```

## Final rule

No file, README, generated artifact, or successful isolated demonstration is sufficient to claim **BOOTSTRAP COMPLETE**. The accepted result requires a genuinely fresh self-hosted rebuild followed by self-recompile, byte-identical fixed-point verification, independent rebuild, and independent verification.
