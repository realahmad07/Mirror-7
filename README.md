# MIRROR7

## Open-source bootstrap and self-hosting project

MIRROR7 is an experimental open-source computational architecture whose bootstrap path is developed from a small C nucleus toward a self-hosted MIRR compiler and reproducible self-rebuild.

> **Verification rule:** source presence is not a PASS. A stage is promoted only after the implementation path has been executed and its acceptance tests pass. Verification distinguishes builder output, independent rebuild, and independent verification.

## Bootstrap status — COMPLETE

The final bootstrap tree has been exercised from a clean extraction. The complete acceptance chain is green:

```text
Phase 27 structured relocation/control flow
        ↓
Phase 28 surface parser/compiler integration
        ↓
Phase 29 MIRR source closure
        ↓
Phase 30 compiler entirely in MIRR
        ↓
source/target dictionary separation
        ↓
symbolic / relocatable control-flow targets
        ↓
fresh-stage bootstrap
        ↓
self-recompile
        ↓
byte-identical fixed point
        ↓
genuinely independent rebuild
        ↓
independent verification
        ↓
BOOTSTRAP COMPLETE
```

## Acceptance board

| Gate | Status | Evidence |
|---|---|---|
| Phase 27 deep audit | ☑ PASS | `MIRROR7_PHASE19_HERITAGE_AUDIT_PASS` and `MIRROR7_DEEP_AUDIT_STATIC_PASS`. |
| Phase 27 verification | ☑ PASS | `PHASE27_VERIFICATION_PASS`; regenerated artifact is reproducible and executable. |
| Phase 28 | ☑ PASS | `PHASE28_INTEGRATION_PASS`; strict C17 parser/compiler integration and regression. |
| Phase 29 | ☑ PASS | `PHASE29_MIRR_SOURCE_CLOSURE_PASS`; 29 MIRR compiler words verified. |
| Phase 30 | ☑ PASS | `PHASE30_COMPILER_ENTIRELY_IN_MIRR_PASS`; direct Nucleus execution verified. |
| Source/target dictionary separation | ☑ PASS | `STEP_2_DICTIONARY_SEPARATION_PASS`. |
| Symbolic / position-independent relocation | ☑ PASS | `STEP_3_RELOCATION_PASS`. |
| Fresh-stage bootstrap | ☑ PASS | `STEP_4_FRESH_STAGE_BOOTSTRAP_PASS`. |
| Self-recompile | ☑ PASS | `STEP_5_SELF_RECOMPILE_PASS`; Compiler A compiles the MIRR compiler source and the resulting Compiler B executes held-out programs. |
| Byte-identical fixed point | ☑ PASS | `STEP_6_BYTE_IDENTICAL_FIXED_POINT_PASS`; two independent compilation runs produced identical compiled-B bytes and identical held-out behavior. |
| Independent rebuild | ☑ PASS | `STEP_7_INDEPENDENT_REBUILD_PASS`; standalone reconstruction matches the primary artifact byte-for-byte. |
| Independent verification | ☑ PASS | `STEP_8_INDEPENDENT_VERIFICATION_PASS`; independent structural checks plus adversarial corruption rejection. |
| Bootstrap complete | ☑ PASS | All bootstrap acceptance gates are green. |

## Reproducibility evidence

- Primary artifact SHA256: `ec48f82db766b3fa4bbcad83bc5f9193aedb2212329122e8b380a1a68d3a5c50`
- Independent rebuild SHA256: `ec48f82db766b3fa4bbcad83bc5f9193aedb2212329122e8b380a1a68d3a5c50`
- Fixed-point compiled-B SHA256: `4d395c63b6e4364fcad2f198e74e305b639996cc649bbf58d66bb2e46e597d25`

## Adversarial verification

- **ADV1:** corrupted out-of-bounds `0branch:99999` → rejected, exit code `71`.
- **ADV2:** truncated/malformed word definition → rejected, exit code `69`.

## Phase 28 architecture

```text
MIRR SOURCE
     ↓
SURFACE PARSER
     ↓
PARSED SURFACE IR
     ↓
MIRR COMPILER
     ↓
SOURCE/TARGET DICTIONARY SEPARATION
     ↓
SYMBOLIC RELOCATION
     ↓
MIRR EXECUTABLE
     ↓
VM
     ↓
OUTPUT
```

## Verification method

For every bootstrap gate:

1. execute the implementation rather than trusting source presence or a previous report;
2. reproduce failures at the smallest boundary;
3. make the smallest justified correction;
4. rerun the failed gate;
5. rerun the relevant regression suite;
6. require byte-level equality where reproducibility is claimed;
7. use a separate build/verification path for the corresponding gates;
8. retain adversarial rejection tests.

## What Mirror 7 can do now

Mirror 7 can now:

- execute the C Nucleus/VM and maintain a runtime dictionary;
- parse the supported MIRR surface grammar into structured IR;
- compile definitions, numeric literals, calls, and nested `IF/ELSE/THEN` control flow;
- generate and execute relocated dictionary code;
- represent the compiler implementation as MIRR definitions;
- load and execute the MIRR compiler directly through the Nucleus;
- separate compiler execution state from generated target state;
- perform fresh-stage bootstrap and self-recompile;
- reproduce a byte-identical fixed point;
- reproduce the accepted artifact through an independent rebuild path;
- independently verify artifact structure and reject adversarial corruption.

These milestones establish a verified bootstrap/self-hosting computational substrate. **They do not by themselves establish AGI.**

## Repository layout

```text
Mirror-7/
├── README.md
├── STATUS.md
├── BOOTSTRAP_PROGRESS.md
├── audit_mirror7.py
├── phase25_26/
├── phase27_unfinished/
├── phase28_surface_parser/
├── phase29_mirr_compiler/
├── phase30_compiler_mirr/
├── phase31_bootstrap/
└── test_step2_dict_sep.py … test_step8_independent_verifier.py
```

The repository stores source and verification files directly. ZIP archives are release/audit bundles, not the source representation.

## Bootstrap result

**BOOTSTRAP COMPLETE — verified by executable acceptance gates.**


## Phase 31 — Discovered Representation, Stable State, and Temporal Identity

Phase 31 is verified locally with the supplied Phase 31 bundle. The pipeline is:

```text
raw observation → discovered representation → stable structural state
→ temporal state identity → cross-encoding invariance → integrated raw→state pipeline
```

| Gate | Status | Evidence |
|---|---|---|
| Phase 31.1 — discovered representation | ☑ PASS | `phase31_bootstrap/test_phase31_step1.py` |
| Phase 31.2 — stable state extraction | ☑ PASS | `phase31_bootstrap/test_phase31_step2.py` |
| Phase 31.3 — temporal state identity | ☑ PASS | `phase31_bootstrap/test_phase31_step3.py` |
| Phase 31.4 — cross-encoding invariance | ☑ PASS | `phase31_bootstrap/test_phase31_step4.py` |
| Phase 31.5 — raw→state integration | ☑ PASS | `phase31_bootstrap/test_phase31_step5.py` |
| Phase 31.6 — full acceptance gate | ☑ PASS | `phase31_bootstrap/test_phase31_gate.py` |

**Phase 31 result: 37 / 37 tests passed locally.** See `phase31_bootstrap/README.md` for scope, metrics, and limitations.
