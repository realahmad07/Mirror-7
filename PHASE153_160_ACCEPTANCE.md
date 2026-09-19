# Mirror 7 — Phases 153–160 Open-World Evaluation Acceptance

## Scope

Phases 153–160 move the project from isolated capability integration toward a sealed evaluation boundary around the combined Phase 141–152 runtime.

| Phase | Focus |
|---|---|
| 153 | Sealed observation/state/action/outcome interface |
| 154 | Unseen environment and conservative uncertainty |
| 155 | Structural transfer across renamed entities |
| 156 | Noisy observations with irrelevant perturbations |
| 157 | 32-step long-horizon prediction replay |
| 158 | Independent-style evaluator contract |
| 159 | Bounded stress run with larger state/noise load |
| 160 | Cross-phase final gate |

## Acceptance protocol

- Seeds: **0, 1, 2**
- Every phase must pass for all three seeds.
- Unknown actions must remain low-confidence rather than being invented.
- Memory and workspace remain within explicit bounds.
- The external evaluator accepts only fixed numeric metrics and does not import Mirror internals.
- Phase 160 requires every preceding phase to pass.

## Verification

The GitHub regression workflow executes:

1. Phases 141–152 integration gate.
2. Phases 153–160 tests across seeds 0/1/2.
3. External evaluator positive and negative controls.
4. Final bounded integration gate.

## Boundary

This block demonstrates a stricter, sealed, bounded evaluation of the existing architecture. Phase 154 measures conservative handling of genuinely unseen dynamics and subsequent adaptation; it is **not** a claim of zero-shot generalization to arbitrary environments.

Phase 155 tests structural transfer over a controlled entity-renaming transformation; it is **not** evidence of unrestricted cross-domain knowledge transfer.

Phase 157 tests a repeated 32-step learned transition chain; it is **not** unrestricted real-world long-horizon planning.

Phase 159 checks explicit resource bounds; it is **not** a scaling law.

No result in this acceptance file is evidence of AGI or human-level general intelligence.
