# MIRROR7 — Detailed Source and Phase Status

This is the canonical human-readable status page for the repository. It deliberately distinguishes implementation history, component verification, integration verification, and bootstrap completion.

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
| Phase 28 — surface parser component | ☑ | Strict C17, regression/fuzz, and ASan/UBSan component tests are implemented. |
| Phase 28 — surface parser/compiler integration | ☑ | End-to-end gate is implemented: parser acceptance/rejection is compared with the real Phase-27 compiler path. CI result is still required for promotion. |
| Whole-project deep audit | ☑ | Static consistency audit and strict host builds are implemented; CI result is still required. |
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

The next implementation work is compiler-in-MIRR self-hosting. Two architectural defects remain acceptance blockers:

1. **Absolute branch dependency:** generated branch offsets are still encoded as absolute addresses. Current Phase 27 generation verifies them against the current layout, but a fresh dictionary layout can make them stale. The long-term fix is real relocation based on symbolic structure or equivalent position-independent metadata.
2. **Single-dictionary ABI:** compiler execution and compiler output still share an implicit dictionary context. A fresh target rebuild requires explicit source/executable-dictionary and target/generated-dictionary separation.

These are bootstrap correctness issues, not documentation issues.

## Phase 28 integration evidence

`phase28_surface_parser/parser.c` now has a normal CLI entrypoint in addition to its strict test mode. `phase28_surface_parser/verify_phase28.py` exercises two layers:

1. the surface parser accepts valid definitions and rejects malformed control flow, names, and numeric literals;
2. the actual Phase-27 compiler is rebuilt from its current builder and exercised with the parser-valid programs, while parser-invalid programs must fail compilation.

This is an integration contract, not a claim that the parser C implementation has become part of the eventual self-hosted MIRR compiler. The latter remains an explicit future gate.

## Deep-audit evidence

`audit_mirror7.py` checks required repository inputs, exact builder/artifact reproducibility, primitive-count/layout assumptions, duplicate definitions, u16 code-size limits, branch-target containment and instruction-boundary validity, and strict C17 builds of the nucleus and parser. It deliberately reports absolute-address relocation and dictionary-ABI separation as remaining architectural debt rather than hiding them behind a green local check.

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
