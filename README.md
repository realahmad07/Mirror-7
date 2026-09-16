# MIRROR7 — Open-Source Self-Hosted Intelligence Substrate

MIRROR7 is an open-source experimental computing architecture designed to progress from a small bootstrap runtime toward a self-hosted, learning-capable computational system. The project is being developed primarily through programming, compiler/runtime research, and reproducible experiments rather than by assuming access to large training budgets.

The central engineering principle is: **do not declare a stage complete merely because files exist.** A stage is complete only when implementation and verification evidence support that claim. Unfinished work is kept visible and explicitly marked unfinished.

## Project direction

```text
Stage-0 C bootstrap nucleus
        ↓
VM / runtime / memory / dictionary
        ↓
runtime dictionary creation
        ↓
tokenization + dictionary lookup
        ↓
MIRR compilation machinery
        ↓
integrated native compiler path
        ↓
structured relocation / control flow
        ↓
complete surface parser
        ↓
compiler entirely in MIRR
        ↓
fresh-stage bootstrap
        ↓
self-recompile
        ↓
byte-identical fixed point
        ↓
independent rebuild
        ↓
bootstrap verification
        ↓
self-hosted substrate
        ↓
learning and representation systems
        ↓
world modelling / planning / tools
        ↓
controlled capability acquisition and self-improvement experiments
        ↓
AGI evaluation
```

The later stages are goals, not claims that they have already been achieved.

## Repository status

| Stage | Status | Interpretation |
|---|---|---|
| Phase 24 — runtime dictionary creation | ☑ PASS recorded | Phase 24 implementation and test artifacts are preserved. |
| Phase 25 — tokenization, lookup, compilation | ☑ Implemented; fresh verification required | Source is preserved; a fresh clean-room PASS is not asserted here. |
| Phase 26 — integrated native compiler path | ☑ Implemented; fresh verification required | Source is preserved; a fresh clean-room PASS is not asserted here. |
| Phase 27 — structured relocation/control flow | ☐ UNFINISHED | Active work and tests are preserved under `phase27_unfinished/`. |
| Complete surface parser | ☐ Not verified | Not claimed as complete. |
| Compiler entirely in MIRR | ☐ Not verified | Not claimed as complete. |
| Fresh-stage bootstrap | ☐ Not verified | Not claimed as complete. |
| Self-recompile / fixed point | ☐ Not verified | Not claimed as complete. |
| Independent rebuild | ☐ Not verified | Not claimed as complete. |
| Bootstrap complete | ☐ Not verified | Not claimed as complete. |

## Repository layout

```text
Mirror-7/
├── README.md
├── STATUS.md
├── phase24/
│   ├── nucleus.c
│   ├── nucleus_debug.c
│   ├── compiler_words.mirr
│   ├── test_phase24.py
│   ├── stress_phase24.py
│   ├── README.md
│   └── PHASE24_REPORT.md
├── phase25_26/
│   ├── nucleus.c
│   ├── compiler.mirr
│   ├── phase25.mirr
│   ├── phase26.mirr
│   ├── compiler_phase26_words.mirr
│   ├── compiler_nucleus_words.mirr
│   ├── make_phase25.py
│   ├── test_phase25_26.py
│   ├── compiler_gen.txt
│   └── compiler_gen26.txt
├── phase27_unfinished/
│   ├── nucleus.c
│   ├── compiler_phase27.mirr
│   ├── compiler_phase27_words.mirr
│   ├── build_phase27*.py
│   ├── stress_phase27*.py
│   └── Phase 27 MIRR test programs
└── ui/
    └── index.html
```

## Languages used

- **C** — bootstrap nucleus and low-level runtime/compiler substrate.
- **MIRR** — project language used for compiler/runtime source and generated dictionary structures.
- **Python** — build, stress-test, regression-test, and experiment tooling.
- **HTML/CSS/JavaScript** — lightweight project UI in `ui/index.html`.

Python is tooling around the bootstrap system; it is not being presented as the self-hosting implementation language.

## Verification philosophy

MIRROR7 follows an explicit verification discipline:

1. Reproduce the smallest failing case.
2. Trace the failure to its exact propagation or implementation boundary.
3. Make the smallest justified change.
4. Re-run the previously failing test.
5. Re-run regression tests for valid programs.
6. Exercise malformed inputs and combinations of constructs.
7. Run stress/fuzz/sanitizer-style testing where applicable.
8. Record the result without upgrading an unfinished stage to PASS.

For Phase 27, malformed-control-flow tests are intentionally preserved because failure propagation is part of compiler correctness. Cases such as `ELSE` outside an `IF` and duplicate `ELSE` must fail all the way to the top-level compilation result rather than being silently converted into success.

## What this repository does and does not claim

This repository is a source snapshot of the MIRROR7 work that has been packaged. It does **not** claim that MIRROR7 is already a general artificial intelligence system or AGI. The long-term architecture includes learning, representation, program synthesis, world modelling, planning, tool use, capability acquisition, and self-improvement research, but each of those stages still requires its own implementation and empirical verification.

Historical phase labels are preserved for provenance. Where fresh verification is still required, this README says so explicitly rather than converting historical implementation records into new PASS claims.

## Working with the source

The phase directories are intentionally separated so that the evolution of the bootstrap can be inspected. Start with `phase24/` for the recorded runtime-dictionary baseline, then inspect `phase25_26/` for the tokenizer/compiler path, and finally `phase27_unfinished/` for the current structured-relocation work.

The repository contains the **individual source files**, not a ZIP archive of the code. The implementation, tests, reports, and unfinished work are directly inspectable through GitHub.

## Contribution and research direction

Changes should preserve the distinction between:

- implementation present in source,
- tests that have actually been run,
- historical results,
- current verification results, and
- future architectural goals.

That distinction matters for a bootstrap project because a compiler can appear to work while still depending on hidden generated artifacts, stale absolute addresses, an incomplete relocation mechanism, or an ABI assumption that prevents a genuine fresh-stage rebuild.

## Current next engineering target

The immediate target is **Phase 27: close structured relocation/control-flow correctness**. The phase should remain marked unfinished until malformed descendants reliably propagate compilation failure through every enclosing compiler layer, valid structured control flow remains regression-safe, and the compiler path survives combination/stress testing.

Only after that gate is genuinely closed should the project advance to the complete surface parser and later self-hosting bootstrap stages.
