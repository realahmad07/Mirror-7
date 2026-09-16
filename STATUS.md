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
| Phase 27 — structured relocation/control flow | ☐ | Current implementation is corrected and synchronized with its builder; current end-to-end CI evidence is still required. The latest traced defect was an end-of-dictionary branch target in `find-word`; the Phase 26 source has been corrected from `0branch:367` to `0branch:365` so the Phase 27 builder emits the real `drop/exit` boundary. |
| Phase 28 — surface parser component | ☑ | Strict C17, regression/fuzz, and ASan/UBSan component tests are implemented. |
| Phase 28 — surface parser/compiler integration | ☐ | Integration harness exists; current CI must prove real compiler-path acceptance/rejection. |
| Whole-project deep audit | ☐ | Static audit is implemented; current CI execution is still required. |
| Phase 13 heritage preservation | ☑ | Pre-MIRR artifact/bootstrap evidence is preserved under `phase13_heritage/` and included in the deep audit. |
| Phase 19 heritage preservation | ☑ | RAW-free self-language, semantic self-source, fixed-point evidence, native-seed source, and artifact hashes are preserved under `phase19_heritage/` and included in the deep audit. |
| Phase 69 portable-carrier heritage | ☑ | V69 execution-carrier methodology is preserved under `phase69_heritage/` with a fresh verification gate; it is reference infrastructure, not current self-hosting. |
| Compiler entirely in MIRR | ☐ | **Phase 30 in progress:** the MIRR compiler source now has a dedicated direct-execution acceptance gate under `phase30_compiler_mirr/`. No promotion to PASS until that gate succeeds on CI. |
| Separate source/target dictionary ABI | ☐ | Compiler dictionary and fresh generated target dictionary must be independently selectable. |
| Remove hard-coded absolute branch dependency | ☐ | Control-flow targets must be generated/relocated from symbolic structure or equivalent metadata. |
| Fresh-stage bootstrap | ☐ | A fresh stage must rebuild the compiler from the accepted bootstrap substrate. |
| Self-recompile | ☐ | The MIRR compiler must compile its own source through the same bootstrap path. |
| Byte-identical fixed point | ☐ | Repeated self-recompilation must produce identical bytes under fixed build inputs. |
| Independent rebuild | ☐ | A separate rebuild path must reproduce the same accepted artifact. |
| Independent verification | ☐ | Verification must be independently executable and not rely solely on builder assertions. |
| Bootstrap complete | ☐ | All preceding gates must be green. |

## Phase 69 portable-carrier integration
The supplied V69 archive was reviewed against the current MIRR/Nucleus roadmap. The retained material captures an important intermediate portability pattern: serialize an explicit execution-kernel contract with a bounded machine artifact, replay it through independent implementations, and check cross-runner equivalence without importing the symbolic Mirror evaluator. The native C runner provides a second implementation family.

This heritage is intentionally not promoted as current MIRR self-hosting. Its value is methodological: it makes hidden execution assumptions explicit and provides a template for future independent rebuild and bootstrap verification. The supplied archive's historical V69 report recorded a targeted 17/17 pass; the present repository uses a fresh, narrower verifier rather than treating that historical result as current CI evidence.

## Phase 13 heritage integration
The repository preserves the pre-MIRR/Nucleus Phase 13 development line under `phase13_heritage/`. It provides earlier artifact/bootstrap experiments and deterministic testing as reference material only.

## Phase 19 heritage integration
The repository preserves the supplied Phase 19 RAW-free self-language experiment under `phase19_heritage/`. The retained material documents a 20-generation byte-identical fixed point for an older semantic VM, a semantic self-source with no legacy RAW token, randomized/adversarial test discipline, tamper/hash checks, and the native-seed source.

This is intentionally **not** promoted as current MIRR self-hosting. Its value is the acceptance methodology it contributes to the current roadmap: deterministic artifact reconstruction, semantic self-source validation, repeated fixed-point checking, malformed-input rejection, and explicit tamper detection.

## Phase 27 corrections
The current Phase 27 builder and generated artifact include corrections discovered by tracing the structured-control path: create-pass retains the `word-new` result, relocation shifts only targets at/after the insertion point, and the committed generated artifact is maintained from the committed builder. The latest dictionary-loop correction fixes the `find-word` end test so a true end-of-dictionary condition lands on `drop/exit` instead of two bytes past it.

## Phase 30 — compiler entirely in MIRR
Phase 30 adds `phase30_compiler_mirr/verify_phase30.py`, which checks the compiler-source closure and then executes the MIRR compiler source directly through the Nucleus on simple and structured MIRR programs. The Phase 27 Python builder remains bootstrap/staging infrastructure and is not treated as the compiler semantics.

## Current acceptance boundary
```text
Phase 13 heritage preservation         ☑
Phase 19 RAW-free heritage             ☑
Phase 69 portable-carrier heritage     ☑
Phase 27 implementation correction     ☑
Phase 27 artifact synchronization      ☑
Phase 27 actual CI promotion           ☐
Phase 28 parser component              ☑
Phase 28 compiler integration          ☐
Whole-project audit CI promotion       ☐
Phase 30 direct MIRR compiler gate     ☐
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
```text
reproduce → minimize → trace exact boundary → inspect ABI/layout/ownership → consult references when useful → minimal fix → rerun → regression → fuzz/sanitizer → promote only with evidence
```

## Final rule
No file, README, generated artifact, or successful isolated demonstration is sufficient to claim **BOOTSTRAP COMPLETE**. The accepted result requires a genuinely fresh self-hosted rebuild followed by self-recompile, byte-identical fixed-point verification, independent rebuild, and independent verification.
