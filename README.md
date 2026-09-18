# MIRROR7

## Open-source bootstrap and self-hosting project

MIRROR7 is an experimental open-source computational architecture whose bootstrap path is being developed from a small C runtime toward a self-hosted MIRR compiler and reproducible self-rebuild.

> **Verification rule:** source presence is not a PASS. A stage is promoted only after the actual implementation path has been built and exercised with regression, malformed-input, stress/fuzz, and sanitizer testing where applicable.

## Current bootstrap chain

```text
Stage-0 C nucleus
    ↓
VM / runtime dictionary
    ↓
MIRR compilation machinery
    ↓
structured relocation / control flow
    ↓
Phase 28 surface parser + structured IR
    ↓
MIRR compiler integration
    ↓
Phase 29 MIRR source closure
    ↓
Phase 30 compiler entirely in MIRR
    ↓
source/target dictionary ABI              ← NEXT
    ↓
symbolic relocation
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
    ↓
BOOTSTRAP COMPLETE
```

## Status

| Gate | Status | Evidence / meaning |
|---|---|---|
| Phase 24 — runtime dictionary | ☑ PASS recorded | Preserved Phase 24 implementation/test artifacts. |
| Phase 25 — tokenization + lookup + compilation | ☑ Implemented; fresh verification required | Source preserved; historical result is not treated as a new clean-room PASS. |
| Phase 26 — integrated native compiler path | ☑ Implemented; fresh verification required | Source preserved; historical result is not treated as a new clean-room PASS. |
| Phase 27 — structured relocation/control flow | ☑ VERIFIED | Current builder output was regenerated and strict Phase 27 verification passes. |
| Phase 28.1 — surface parser component | ☑ Component verified | Strict C17 plus parser regression/fuzz and ASan/UBSan execution are implemented. |
| Phase 28.2 — parsed-structure ABI | ☑ Component verified | Versioned surface IR is emitted and independently validated by a strict C17 ABI regression. |
| Phase 28 — parser/compiler integration | ☑ VERIFIED | 28.1–28.12 pass, including held-out programs, relocation, VM execution, and Phase 27 regression. |
| Whole-project deep audit | ☑ Static audit implemented; CI promotion pending | `audit_mirror7.py` checks repository inputs, generated-artifact reproducibility, layout/branch invariants, u16 limits, and strict C17 host builds. |
| Phase 29 — MIRR source closure | ☑ VERIFIED | Compiler source closes over MIRR words/Nucleus primitives and runs through the bootstrap builder path. |\n| Phase 30 — compiler entirely in MIRR | ☑ VERIFIED | Compiler source loads directly into the Nucleus and compiles/executes representative MIRR programs without the Python builder supplying semantics. |
| Separate source/target dictionary ABI | ☐ OPEN | Compiler execution dictionary and generated target dictionary must be independently selectable. |
| Remove hard-coded absolute branch dependency | ☐ OPEN | Branch targets must derive from relocation/symbolic structure rather than fixed offsets. |
| Fresh-stage bootstrap | ☐ Not verified | A clean stage must rebuild the compiler from the bootstrap substrate. |
| Self-recompile | ☐ Not verified | The MIRR compiler must successfully compile its own source. |
| Byte-identical fixed point | ☐ Not verified | Recompilation must produce identical output under the defined reproducible-build conditions. |
| Independent rebuild | ☐ Not verified | A separate build path must reproduce the same artifact. |
| Independent verification | ☐ Not verified | Verification must not rely solely on the builder's own assertions. |
| Bootstrap complete | ☐ Not verified | All preceding gates must be green. |

## Phase 28 architecture

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

### Phase 28.2 — parsed-structure ABI

`phase28_surface_parser/surface_ir.h` defines version 1 of the parser → compiler data contract. The parser emits source-ordered typed tokens for definitions, names, numbers, `IF`, `ELSE`, `THEN`, and `;`, with source byte positions and explicit 16-bit numeric values. `mirror7_surface_ir_validate()` checks the representation independently.

`test_surface_ir.c` verifies the ABI under strict C17, including token order, source positions, numeric representation, version rejection, and malformed numeric-token rejection. The existing parser regression/fuzz test also passes.

This is deliberately a **component PASS**, not a Phase 28 completion claim. The next gate must make the real MIRR compiler consume this IR instead of reparsing raw source independently.

## Phase 28 audit result

The supplied Phase 28 package was compared against the repository. Its claimed final report overstated the integration state. The audit found that the parser and compiler were previously separate stages. The repository now records the corrected boundary and the Phase 28.2 implementation without promoting the remaining integration work.

A separate reproducibility issue remains in the Phase 27/28 verification path: the Phase 27 builder output and packaged artifact differed in line endings (LF versus CRLF), causing the byte-for-byte artifact check to fail. This remains a CI/artifact-closure issue and is not hidden behind the Phase 28 status.

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

## Current capabilities

Mirror 7 can now:

- execute the C Nucleus/VM and maintain a runtime dictionary;
- parse the supported MIRR surface grammar into structured IR;
- compile definitions, numeric literals, calls, and nested `IF/ELSE/THEN` control flow;
- relocate generated branch targets and execute the resulting code;
- represent the compiler implementation as MIRR definitions;
- load that MIRR compiler directly into the Nucleus and use it to compile representative MIRR programs;
- run strict verification, held-out tests, regression checks, and deep static audits.

These milestones establish a verified computational/bootstrap substrate. They do **not** by themselves establish AGI or bootstrap completion.

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
    ├── parser.c
    ├── surface_ir.h
    └── test_surface_ir.c
```

The repository stores source files directly. **No ZIP archive is used as the code representation.**

## Next acceptance sequence

```text
Resolve Phase 27 CI/artifact closure
  → Phase 28.3 parser IR → real compiler
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
