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
| Phase 27 — structured relocation/control flow | ☐ | Current implementation has been corrected and synchronized with its builder; actual current end-to-end CI evidence is still required. |
| Phase 28 — surface parser component | ☑ | Strict C17, regression/fuzz, and ASan/UBSan component tests are implemented. |
| Phase 28 — surface parser/compiler integration | ☐ | Integration harness is implemented; current CI must prove parser-valid programs compile and malformed programs are rejected through the real compiler path. |
| Whole-project deep audit | ☐ | Static audit implementation is committed; current CI execution is still required before promoting the audit gate. |
| Compiler entirely in MIRR | ☐ | No hidden host compiler implementation may remain in the accepted self-hosting path. |
| Separate source/target dictionary ABI | ☐ | Compiler dictionary and fresh generated target dictionary must be independently selectable. |
| Remove hard-coded absolute branch dependency | ☐ | Control-flow targets must be generated/relocated from symbolic structure or equivalent position-independent metadata. |
| Fresh-stage bootstrap | ☐ | A fresh stage must rebuild the compiler from the accepted bootstrap substrate. |
| Self-recompile | ☐ | The MIRR compiler must compile its own source through the same bootstrap path. |
| Byte-identical fixed point | ☐ | Repeated self-recompilation must produce identical bytes under fixed build inputs. |
| Independent rebuild | ☐ | A separate rebuild path must reproduce the same accepted artifact. |
| Independent verification | ☐ | Verification must be independently executable and not rely solely on builder assertions. |
| Bootstrap complete | ☐ | All preceding gates must be green. |

## Phase 27 corrections

The current Phase 27 builder and generated artifact include corrections discovered by tracing the structured-control path:

1. **Target-word ownership:** `create-pass` retains the word ID returned by `word-new` so `if-open`, `else-open`, and `then-close` operate on the word actually being compiled rather than stale dictionary cells.
2. **Relocation boundary:** the generated image accounts for the exact insertion point and shifts only absolute branch targets at or after that point. Targets before the insertion remain unchanged.
3. **Generated-artifact reproducibility:** the committed `compiler_phase27_words.mirr` is maintained as generated output from the committed builder.
4. **Verification:** `verify_phase27.py` rebuilds in a temporary directory, compares bytes, performs a strict C17 nucleus build, and runs structured valid/nested cases. CI execution is the authoritative end-to-end gate.

The repository history contains the debugging commits that led to these corrections, including the latest alignment commit `9447b98c62fcfaa765d831a10ed1fdab4cb340e8`.

## Phase 28 integration evidence

`phase28_surface_parser/parser.c` has a normal CLI entrypoint in addition to strict test mode. Its local component tests cover valid surface syntax and malformed control flow, names, and numeric boundaries, with ASan/UBSan execution.

`phase28_surface_parser/verify_phase28.py` is the integration gate. It is intended to exercise two layers:

1. the surface parser accepts valid definitions and rejects malformed input;
2. parser-valid programs and parser-invalid programs are then checked against the actual Phase-27 compiler path.

This is an integration contract, not a claim that the parser C implementation has become part of the eventual self-hosted MIRR compiler. That self-hosting transition remains a separate acceptance gate.

## Deep-audit evidence

`audit_mirror7.py` checks required repository inputs, exact builder/artifact reproducibility, primitive-count/layout assumptions, duplicate definitions, u16 code-size limits, branch-target containment and instruction-boundary validity, and strict C17 builds of the nucleus and parser.

The audit deliberately does **not** convert architectural debt into a green result. In particular, it reports these as remaining blockers:

- generated branch offsets are still absolute addresses tied to the current dictionary layout;
- compiler execution and compiler output still require an explicit source/target dictionary ABI separation;
- fresh-stage bootstrap, self-recompile, byte-identical fixed point, independent rebuild, and independent verification are not yet established.

## Current acceptance boundary

```text
Phase 27 implementation correction     ☑
Phase 27 artifact synchronization      ☑
Phase 27 actual CI promotion           ☐
Phase 28 parser component              ☑
Phase 28 compiler integration          ☐
Whole-project audit CI promotion       ☐
        ↓
compiler entirely in MIRR              ☐
        ↓
source/target dictionary ABI           ☐
        ↓
symbolic relocation                 ☐
        ↓
fresh-stage bootstrap                ☐
        ↓
self-recompile                         ☐
        ↓
byte-identical fixed point             ☐
        ↓
independent rebuild                   ☐
        ↓
independent verification              ☐
        ↓
BOOTSTRAP COMPLETE                    ☐
```

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
