# MIRROR7

## Open-source bootstrap and self-hosting project

MIRROR7 is an experimental open-source computational architecture whose bootstrap path is being developed from a small C runtime toward a self-hosted MIRR compiler and reproducible self-rebuild.

> **Verification rule:** source presence is not a PASS. A stage is promoted only after the actual implementation path has been built and exercised with regression, malformed-input, stress/fuzz, and sanitizer testing where applicable.

## Current bootstrap chain

```text
Stage-0 C nucleus
    ↓
VM / memory / dictionary
    ↓
runtime dictionary creation
    ↓
tokenization + lookup
    ↓
MIRR compilation machinery
    ↓
integrated compiler path
    ↓
structured relocation / control flow
    ↓
Phase 28 surface-parser integration
    ↓
compiler entirely in MIRR
    ↓
source/target dictionary ABI
    ↓
symbolic relocation (no generated absolute-branch dependency)
    ↓
fresh-stage bootstrap
    ↓
self-recompile
    ↓
byte-identical fixed point
    ↓
independent rebuild
    ↓
independent verification
```

## Status

| Gate | Status | Evidence / meaning |
|---|---|---|
| Phase 24 — runtime dictionary | ☑ PASS recorded | Preserved Phase 24 implementation/test artifacts. |
| Phase 25 — tokenization + lookup + compilation | ☑ Implemented; fresh verification required | Source preserved; historical result is not treated as a new clean-room PASS. |
| Phase 26 — integrated native compiler path | ☑ Implemented; fresh verification required | Source preserved; historical result is not treated as a new clean-room PASS. |
| Phase 27 — structured relocation/control flow | ☐ CI closure required | Implementation is corrected and locally exercised, but the latest GitHub Actions debug run failed; no current green CI promotion is claimed. |
| Phase 28 — surface parser component | ☑ Component verified | Strict C17 plus parser regression/fuzz and ASan/UBSan execution are implemented. |
| Phase 28 — parser/compiler integration | ☐ NOT VERIFIED | Audit found no parsed-structure ABI: the parser validates source separately while the compiler still consumes raw source through its own token path. |
| Whole-project deep audit | ☑ Static audit implemented; CI promotion pending | `audit_mirror7.py` checks repository inputs, generated-artifact reproducibility, layout/branch invariants, u16 limits, and strict C17 host builds. |
| Compiler entirely in MIRR | ☐ Not verified | Must compile the compiler without a host-side compiler implementation dependency. |
| Separate source/target dictionary ABI | ☐ Not verified | Compiler execution dictionary and generated target dictionary must be independently selectable. |
| Remove hard-coded absolute branch dependency | ☐ Not verified | Branch targets must derive from relocation/symbolic structure rather than fixed offsets. |
| Fresh-stage bootstrap | ☐ Not verified | A clean stage must rebuild the compiler from the bootstrap substrate. |
| Self-recompile | ☐ Not verified | The MIRR compiler must successfully compile its own source. |
| Byte-identical fixed point | ☐ Not verified | Recompilation must produce identical output under the defined reproducible-build conditions. |
| Independent rebuild | ☐ Not verified | A separate build path must reproduce the same artifact. |
| Independent verification | ☐ Not verified | Verification must not rely solely on the builder's own assertions. |
| Bootstrap complete | ☐ Not verified | All preceding gates must be green. |

## Phase 28 audit result

The supplied Phase 28 package was compared against the repository. The package's parser implementation is present on `main`, but its claimed final report overstated the integration state.

The current implementation proves this separate behavior:

```text
MIRR SOURCE ──► Surface Parser ──► PASS/FAIL
      │
      └────────► MIRR Compiler ──► VM
```

It does **not yet** prove the required integrated path:

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

The detailed findings are recorded in `PHASE_28_AUDIT.md` and `PHASE_28_REPORT.md`. Phase 28 therefore remains open until a real parser-output-to-compiler boundary is implemented and verified.

A separate reproducibility issue was also observed in the supplied Phase 28 verifier: the Phase 27 builder output and packaged artifact differed in line endings (LF versus CRLF), causing the byte-for-byte artifact check to fail. This is a build/artifact synchronization issue and is not being hidden behind the Phase 28 claim.

## Recent Phase 27/28 engineering work

The current branch contains the following verification infrastructure and corrections:

- Phase 27 generated output was synchronized with `build_phase27.py` after a stale absolute branch target was found.
- `create-pass` retains the newly created target word ID so structured-control patching does not depend on stale dictionary cells.
- Relocation adjustment uses the actual insertion boundary; targets before it remain unchanged and targets at/after it move by the inserted size.
- `verify_phase27.py` rebuilds the generated artifact, checks byte-for-byte reproducibility, performs a strict nucleus build, and exercises simple, IF/ELSE/THEN, and nested structured programs.
- Phase 28's surface parser has a CLI entry point and strict/sanitizer verification.
- `verify_phase28.py` validates parser acceptance/rejection and separately exercises the Phase 27 compiler/runtime path; it is not yet a proof of parser-output-to-compiler integration.
- `audit_mirror7.py` provides a repository-wide static consistency gate and reports remaining bootstrap blockers.

## Repository layout

```text
Mirror-7/
├── README.md
├── STATUS.md
├── PHASE_28_AUDIT.md
├── PHASE_28_REPORT.md
├── audit_mirror7.py
├── .github/workflows/phase27.yml
├── .github/workflows/phase28.yml
├── phase24/
├── phase25_26/
├── phase27_unfinished/
└── phase28_surface_parser/
```

The repository stores source files directly. **No ZIP archive is used as the code representation.**

## Engineering method

For every failure:

1. reproduce the smallest failing case;
2. identify the exact failing boundary;
3. inspect the implementation and ABI assumptions;
4. consult authoritative/public technical references when an external solution is useful;
5. implement the smallest justified correction;
6. rerun the failing case;
7. rerun regression and combination cases;
8. run stress/fuzz/sanitizer tests where applicable;
9. only then promote the gate.

## Languages

- **C17:** bootstrap nucleus and low-level runtime.
- **MIRR:** compiler/runtime source being moved toward self-hosting.
- **Python:** build and test tooling.
- **HTML/CSS/JavaScript:** repository status UI.

## Important scope boundary

MIRROR7 is a research/engineering project. The bootstrap work is about establishing a reproducible self-hosted computational substrate. Future learning, world-model, planning, tool-use, capability-acquisition, and AGI experiments are architectural goals and are not claimed merely because the bootstrap compiler exists.

## Next acceptance sequence

```text
Resolve Phase 27 CI/artifact reproducibility
  → define Phase 28 parsed-structure ABI
  → connect parser output to the real MIRR compiler
  → Phase 28 end-to-end acceptance
  → compiler entirely in MIRR
  → separate source/target dictionary ABI
  → symbolic relocation
  → fresh-stage bootstrap
  → self-recompile
  → byte-identical fixed point
  → independent rebuild
  → independent verification
```
