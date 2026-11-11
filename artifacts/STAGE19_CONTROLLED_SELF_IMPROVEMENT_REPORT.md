# Stage 19 — Controlled Self-Improvement Acceptance Report

**Status:** PASS within the bounded container-evaluation and human-approval scope  
**Implementation:** `mirror_crystal/stage19_controlled_improvement.py`  
**Tests:** `mirror_crystal/tests/test_stage19_controlled_improvement.py`

## Delivered

- Immutable code/rule proposal objects bound to baseline revision, exact source digests and suite fingerprint.
- Frozen JSON regression suite with at least three preservation cases and one explicit improvement case.
- Environment fingerprint binds Docker/Podman runtime, pinned image, timeout, memory/PID settings and sandbox/harness policy.
- Baseline and candidate run in separate fresh containers through the Stage 17 container boundary.
- A trusted supervisor remains in container PID 1 and starts a separate short-lived candidate process for every frozen test case. Candidate stdout is bounded to 8 KiB, JSON results to 16 KiB, and candidate output cannot forge the supervisor's result manifest.
- Fail-closed rejection for baseline failure, candidate regression, missing/duplicate result IDs, misbound artifact/suite/environment evidence, malformed sandbox results, PID namespace mismatch, JSON-harness monkey-patching attempts, stdout/result size violations, result-manifest spoofing, and unavailable container runtime.
- Candidate policy violations are preserved as per-case failed execution evidence from the trusted supervisor; the supervisor emits the complete manifest without trusting candidate stdout or aborting before evidence capture.
- Explicit independent human approval gate. Passing only becomes READY_FOR_HUMAN_REVIEW; approval returns a scoped authorization bound to the exact candidate artifact, evaluation, suite and environment.

## Verification commands

```bash
python -m pytest -q mirror_crystal/tests/test_stage19_controlled_improvement.py
python -m pytest -q mirror_crystal/tests/test_container_sandbox.py
python -m pytest -q mirror_crystal mirror_eval
python -m compileall -q mirror_crystal
```

The real-container integration test is required by CI via `MIRROR_REQUIRE_CONTAINER_SANDBOX=1`; local environments without Docker/Podman skip it unless that requirement is enabled.

## Honest boundaries

The current artifact interface is a bounded Python `run(inputs) -> JSON` contract. This engine does not patch arbitrary repository files, mutate production registries, create/merge pull requests or deploy. Host applications must authenticate reviewer identity and persist approval/evaluation evidence. A human approval authorization is a scoped policy artifact, not a cryptographic signature or deployment action.
