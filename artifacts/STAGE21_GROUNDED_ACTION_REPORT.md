# Stage 21 — Grounded Autonomous Action & Feedback Acceptance Report

**Status:** Implementation committed; final acceptance pending branch CI and Security Guard  
**Implementation:** `mirror_crystal/stage21_grounded_action.py`  
**Tests:** `mirror_crystal/tests/test_stage21_grounded_action.py`  
**Focused test cases:** 28

## Delivered

- Immutable host-registered action definitions, strict JSON input schemas and SHA-256 definition fingerprints.
- Proposals bound to exact inputs, selected Stage 20 observation IDs, a recomputed Stage 20 report fingerprint and explicit predicted postconditions.
- Two distinct internal checker identities: evidence provenance and action-contract policy.
- Evidence checks require authenticated observation origin, source/observation digest binding, and selection from one verified cross-modal agreement spanning distinct modalities and sources.
- Stage 19 isolated container supervisor and child runner reused for single action executions, with bounded output and PID-1 supervisor evidence.
- Predicted-versus-observed postcondition matching and failure/mismatch feedback.
- Single-use proposal executions and engine-bound verification and approval records.
- Separate human approval gate for consequential action risk; proposer self-approval is rejected.
- Package exports, README/STATUS roadmap entries, and design documentation.

## Verification coverage

28 focused tests cover schema and input limits, immutable exact proposal binding, signed and unsigned provenance, tampered report fingerprints, insufficient modality evidence, unknown/replaced action definitions, distinct checker identities, engine-bound verification, successful and mismatched postconditions, failed sandbox execution, replay prevention, consequential human approval, author/reviewer separation, approval reuse, and execution history immutability.

Two of the 26 tests use a real Docker/Podman container where CI requires the runtime. They check the Stage 19 supervisor integration, PID 1, candidate stdout manifest spoofing and candidate module-mutation rejection.

## Safety boundaries

Action code runs inside the Stage 19 isolated container path. No network, host mounts, production credentials, direct hardware or external service interface are exposed. The engine returns JSON observations only.

Consequential approval is a binding in-engine record, not identity authentication and not a host-side actuator. Reviewer authentication, durable tamper-evident audit storage, host enforcement of approval, actual external actuation and rollback remain integration responsibilities. Execution feedback never mutates trusted evidence or knowledge.

## Acceptance rule

Stage 21 is accepted only when:
1. The focused test suite passes.
2. Full `mirror_crystal` and `mirror_eval` regression passes.
3. Fast CI, backend tests, smoke/release checks, training checks and package build pass.
4. Security Guard passes on the exact final PR head.
5. The PR is merged and `main` is confirmed at the resulting merge commit.

No production actuator or automatic real-world execution is claimed by this stage.
