# MIRROR 7 — Phase 28 Audit

## Scope

The supplied `phase28.zip` was compared with the current `main` branch. The package contains the Phase 28 surface parser, its verifier, Phase 28 workflow, bootstrap/status documentation, and a claimed Phase 28 final report.

## Findings

### Existing

- **Surface parser:** `phase28_surface_parser/parser.c` — validates surface syntax and returns success/failure.
- **Parsed structure:** **None.** The parser does not currently expose an AST, token stream, or other structured representation to the compiler.
- **Compiler:** Phase 27 MIRR compiler in `phase27_unfinished/compiler_phase27_words.mirr`.
- **Dictionary:** maintained by `phase25_26/nucleus.c`.
- **Relocation:** implemented in the MIRR compiler using `word-patch-u16` and related primitives.
- **Executable generation:** performed in the nucleus dictionary/code area.
- **VM:** `phase25_26/nucleus.c`.
- **Verification:** `phase28_surface_parser/verify_phase28.py` tests parser acceptance/rejection and separately exercises the compiler/runtime path.

## Critical integration gap

The supplied Phase 28 report claims:

```text
28.3 Compiler integration PASS
28.12 Acceptance gate PASS
```

That claim is **not supported by the implementation currently present in the package**.

The verifier first runs the standalone parser against its test cases, then separately builds/runs the compiler. The parser's result is not passed as a parsed structure into the compiler. The compiler continues to consume the source text through its own token-reading path.

Therefore the current architecture is:

```text
MIRR SOURCE ──► Surface Parser ──► PASS/FAIL
      │
      └────────► MIRR Compiler ──► VM
```

not the required Phase 28 architecture:

```text
MIRR SOURCE
     ↓
SURFACE PARSER
     ↓
PARSED STRUCTURE
     ↓
MIRR COMPILER
     ↓
DICTIONARY + RELOCATION
     ↓
EXECUTABLE
     ↓
VM
     ↓
OUTPUT
```

## Additional verification issue

The supplied Phase 28 integration verifier also depends on regenerating the Phase 27 artifact. A clean local reproduction showed that the builder output differed from the packaged artifact only in line endings (LF versus CRLF), causing the byte-for-byte reproducibility check to fail. This is an artifact synchronization/build reproducibility issue inherited from Phase 27, not evidence that Phase 28 integration is complete.

## Comparison with current GitHub `main`

- `phase28_surface_parser/parser.c` is already present on `main` and matches the supplied package implementation.
- `phase28_surface_parser/verify_phase28.py` is already present on `main` and has the same architectural limitation described above.
- `.github/workflows/phase28.yml` is already present on `main` and runs strict parser tests, sanitizer tests, and the current integration verifier.
- The supplied package adds Phase 28 audit/report documentation that was not present on `main`.
- The repository's previous Phase 27 push triggered GitHub Actions run `35255177479`, which completed with **failure**; therefore Phase 27 CI closure must not be represented as green.

## Decision

**Phase 28 is NOT VERIFIED yet.**

The correct next engineering task is to close the real parser → parsed-structure → compiler boundary, then rerun the complete Phase 28 gate and Phase 27 regression. The supplied final report must not be used as proof of completion until that happens.

## Required next path

```text
28.1 parser analysis                    PASS
        ↓
28.2 define parsed-structure ABI       OPEN
        ↓
28.3 connect structure to compiler     OPEN
        ↓
28.4 simple end-to-end programs
        ↓
28.5 definitions
        ↓
28.6 control flow
        ↓
28.7 nested control flow
        ↓
28.8 relocation preservation
        ↓
28.9 real VM execution
        ↓
28.10 held-out programs
        ↓
28.11 Phase 27 regression
        ↓
28.12 acceptance gate
```

No later step is promoted until the preceding step passes.