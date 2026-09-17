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
| Phase 25 — tokenization + lookup + compilation | ☑ | Implementation preserved; clean-room verification remains separate. |
| Phase 26 — integrated native compiler path | ☑ | Implementation preserved; clean-room verification remains separate. |
| Phase 27 — structured relocation/control flow | ☐ | Implementation is corrected and locally exercised, but the latest GitHub Actions debug run failed; CI/artifact closure is required. |
| Phase 28.1 — surface parser component | ☑ | Strict C17, regression/fuzz, and ASan/UBSan component tests are implemented. |
| Phase 28.2 — parsed-structure ABI | ☑ | Versioned surface IR is emitted and independently validated by a strict C17 ABI regression. |
| Phase 28 — surface parser/compiler integration | ☐ | **NOT VERIFIED:** the parser now produces a parsed structure, but the real compiler does not yet consume it. |
| Whole-project deep audit | ☐ | Static audit is implemented; current CI execution is still required. |
| Phase 13 heritage preservation | ☑ | Pre-MIRR artifact/bootstrap evidence is preserved under `phase13_heritage/` and included in the deep audit. |
| Phase 19 heritage preservation | ☑ | RAW-free self-language, semantic self-source, fixed-point evidence, native-seed source, and artifact hashes are preserved under `phase19_heritage/` and included in the deep audit. |
| Phase 69 portable-carrier heritage | ☑ | V69 execution-carrier methodology is preserved under `phase69_heritage/` with a fresh verification gate; it is reference infrastructure, not current self-hosting. |
| Compiler entirely in MIRR | ☐ | Must compile the compiler without a host-side compiler implementation dependency. |
| Separate source/target dictionary ABI | ☐ | Compiler dictionary and fresh generated target dictionary must be independently selectable. |
| Remove hard-coded absolute branch dependency | ☐ | Control-flow targets must be generated/relocated from symbolic structure or equivalent metadata. |
| Fresh-stage bootstrap | ☐ | A fresh stage must rebuild the compiler from the accepted bootstrap substrate. |
| Self-recompile | ☐ | The MIRR compiler must compile its own source through the same bootstrap path. |
| Byte-identical fixed point | ☐ | Repeated self-recompilation must produce identical bytes under fixed build inputs. |
| Independent rebuild | ☐ | A separate rebuild path must reproduce the same accepted artifact. |
| Independent verification | ☐ | Verification must be independently executable and not rely solely on builder assertions. |
| Bootstrap complete | ☐ | All preceding gates must be green. |

## Phase 28 audit

The supplied `phase28.zip` was compared against the repository. Its claimed final Phase 28 report overstated the integration status. The parser and compiler were previously separate stages.

Phase 28.2 now closes the missing data-contract definition: `phase28_surface_parser/surface_ir.h` defines a versioned, bounded, source-ordered token IR with typed tokens, source byte offsets, and explicit 16-bit numeric values. `mirror7_parse_ir()` produces it and `mirror7_surface_ir_validate()` validates it. `test_surface_ir.c` exercises the contract.

The actual compiler integration remains open. The current architecture is now:

```text
MIRR SOURCE
     ↓
SURFACE PARSER
     ↓
PARSED SURFACE IR       ← Phase 28.2 PASS
     ↓
MIRR COMPILER           ← Phase 28.3 OPEN
     ↓
DICTIONARY + RELOCATION
     ↓
MIRR EXECUTABLE
     ↓
VM
     ↓
OUTPUT
```

## Phase 28.2 verification evidence

- `surface_ir.h` defines ABI version 1 and fixed bounds.
- `parser.c` emits typed tokens for `:`, names, numbers, `;`, `IF`, `ELSE`, and `THEN` while retaining the existing grammar validation.
- Numeric tokens carry a `uint16_t` value and source byte position; name/control tokens retain bounded text and source byte position.
- `mirror7_surface_ir_validate()` rejects unsupported versions and malformed numeric/token fields.
- `test_surface_ir.c` passes under strict C17 (`-Wall -Wextra -Wpedantic -Werror`) and reports `PHASE28_2_ABI_PASS`.
- The existing parser regression/fuzz test also passes and reports `PHASE28_PARSER_TEST_PASS`.

## Phase 27/28 engineering notes

- Phase 27 structured-control fixes include create-pass target retention and insertion-boundary relocation correction.
- The current Phase 27 verifier includes generated-artifact reproducibility, strict nucleus build, and structured execution cases.
- The latest GitHub Actions run used to close the Phase 27 boundary (`35255177479`, commit `f260011ad95b4ac2f0c55e7de6e42eaa3c5ce01c`) completed with failure. It is therefore not represented as a current green CI gate.
- Phase 28 parser tests include valid/invalid grammar cases and a fuzz loop with strict and sanitizer builds.
- Phase 28.2 adds the first explicit parser → compiler data contract, but does not yet claim compiler consumption.
- `verify_phase28.py` must be extended or redesigned in Phase 28.3 to consume parser output through the real compiler.

## Debugging protocol

```text
reproduce → minimize → trace exact boundary → inspect ABI/layout/ownership → consult references when useful → minimal fix → rerun → regression → fuzz/sanitizer → promote only with evidence
```

## Final rule

No file, README, generated artifact, or successful isolated demonstration is sufficient to claim **BOOTSTRAP COMPLETE**. The accepted result requires a genuinely fresh self-hosted rebuild followed by self-recompile, byte-identical fixed-point verification, independent rebuild, and independent verification.
