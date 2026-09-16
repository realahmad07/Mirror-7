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
| Phase 27 — structured relocation/control flow | ☐ Verification required | Remains a bootstrap gate until the actual current source is re-run end-to-end. |
| Phase 28 — integrated surface parser | ☐ Verification required | Standalone parser CI exists; integration must be verified through the real compiler path. |
| Compiler entirely in MIRR | ☐ Not verified | Must compile the compiler without a host-side compiler implementation dependency. |
| Separate source/target dictionary ABI | ☐ Not verified | Compiler execution dictionary and generated target dictionary must be independently selectable. |
| Remove hard-coded absolute branch dependency | ☐ Not verified | Branch targets must derive from relocation/symbolic structure rather than fixed offsets. |
| Fresh-stage bootstrap | ☐ Not verified | A clean stage must rebuild the compiler from the bootstrap substrate. |
| Self-recompile | ☐ Not verified | The MIRR compiler must successfully compile its own source. |
| Byte-identical fixed point | ☐ Not verified | Recompilation must produce identical output under the defined reproducible-build conditions. |
| Independent rebuild | ☐ Not verified | A separate build path must reproduce the same artifact. |
| Independent verification | ☐ Not verified | Verification must not rely solely on the builder's own assertions. |

## Repository layout

```text
Mirror-7/
├── README.md
├── STATUS.md
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
Phase 27 closure
  → Phase 28 compiler integration
  → compiler entirely in MIRR
  → separate source/target dictionary ABI
  → symbolic relocation
  → fresh-stage bootstrap
  → self-recompile
  → byte-identical fixed point
  → independent rebuild
  → independent verification
```
