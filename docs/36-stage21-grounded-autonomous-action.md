# Stage 21 — Grounded Autonomous Action & Feedback

## Goal and safety boundary

Stage 21 adds a bounded path from verified Stage 20 evidence to an action proposal, independent pre-execution checks, isolated execution, and comparison of predicted versus observed postconditions.

The implementation is deliberately **not** a general-purpose actuator. Candidate action code runs in an isolated container using the Stage 19 PID-1 supervisor and per-case child guard. It has no network access, host filesystem mount, hardware access or production credential. The result is a JSON observation from the sandbox. A trusted host integration must separately enforce a human approval token before any consequential external side effect.

Implementation: `mirror_crystal/stage21_grounded_action.py`.

## Action contract

A host allow-list stores immutable `GroundedActionDefinition` records. Each definition binds:

- A safe action ID and descriptive purpose.
- A Python `run(inputs)` source entrypoint and source fingerprint.
- A strict JSON input schema with required fields, types, bounds and no undeclared additional properties by default.
- A risk level: `read_only`, `reversible` or `consequential`.
- A SHA-256-pinned container image, runtime selection, timeout, memory and PID limits.

The registry refuses to replace an existing action ID with different content. A `GroundedActionProposal` is immutable and fingerprints the exact action definition, JSON inputs, Stage 20 report fingerprint, selected observation IDs, declared expected postconditions, proposer and timestamp.

## Independent pre-execution checks

Every proposal requires both separately identified checks to pass:

1. **Evidence provenance:** recomputes the Stage 20 report fingerprint; verifies each selected observation is present and bound to the source asset digest; requires integrity and authenticated observation origin; and requires all selected observations to belong to one `verified_agreement` conclusion spanning at least two modalities and distinct sources.
2. **Action contract and policy:** checks the exact registered definition digest, validates inputs against the registered schema, and requires explicit expected postconditions and bounded risk/resource settings.

Unsigned corroboration, insufficient evidence, contradictory evidence, a missing observation, an altered report, a different action definition or a malformed input blocks execution. The verification record is bound to the proposal and evidence report. Only a passing verification recorded by that engine instance can proceed.

These are independent checker paths within Mirror's trusted codebase, not external certification bodies. The host remains responsible for reviewing the trustworthiness and deployment of the Mirror package itself.

## Controlled execution

`ContainerActionExecutor` reuses `run_python_regression_in_container` from Stage 19. The existing supervisor executes the action in a separate child process inside a fresh container and returns a supervisor-authored result manifest. Candidate stdout is bounded and cannot forge that manifest.

The executor requires:
- Docker or Podman; no process-mode fallback.
- Pinned image digest and bounded timeout, memory, PID and loop-step limits.
- Network disabled, read-only root, non-root UID, dropped capabilities and no-new-privileges.
- Container PID 1, zero supervisor exit code, exactly one result record and a finite JSON object no larger than 16 KB.

Failure to start the container, invalid PID evidence, malformed output, candidate policy violation or missing result fails closed.

## Predicted-versus-observed feedback

Every action input field must also declare which selected Stage 20 observation IDs supplied its value. Each input binding is checked for at least two distinct modalities and sources, matching normalized value, canonical unit and dimension. Unbound, altered or merely adjacent evidence does not ground an action input.

The action invocation also declares a trusted supervisor output contract requiring one bounded JSON object. The Stage 19 child harness must mark that per-case result as passed before Stage 21 consumes it; the action engine does not ignore a failed supervisor case merely because an output string exists.

The proposal declares one or more expected postconditions. After sandbox execution, Mirror compares each declared key/value against the returned JSON object and records:

- Conditions that matched.
- Conditions that were absent or differed.
- The observed sandbox output and execution status.
- A bound feedback record and deterministic execution fingerprint.

Statuses are `succeeded`, `postcondition_mismatch` and `failed`. Mismatches and failures are retained for diagnosis. They do not update the Stage 20 evidence report, knowledge registry, capability registry or trusted facts. A revised plan must be a new proposal, with new inputs/evidence and a new fingerprint.

The engine exposes an immutable tuple view of its in-memory execution history. Durable append-only storage, tamper-evident retention and authenticated reviewer identities remain host integration responsibilities.

## Human approval

A `consequential` action cannot execute without a human approval record bound to the exact proposal and passing verification fingerprint. The proposal author cannot approve their own proposal. Approval is single-use within the engine instance; proposal execution is also single-use. Read-only and reversible classifications do not require a human approval token in this bounded execution layer.

A reviewer ID is an asserted identity label, not an authentication credential. The host application must authenticate the reviewer and retain the approval/audit record. The host must independently enforce the approval token before carrying out any consequential real-world operation.

## Acceptance tests

Run:

- `python -m pytest -q mirror_crystal/tests/test_stage21_grounded_action.py`
- `python -m pytest -q mirror_crystal mirror_eval`
- `python -m compileall -q mirror_crystal`

The focused suite contains 30 deterministic cases covering schema rejection, proposal immutability, report fingerprint tampering, signed and unsigned evidence, modality floors, distinct pre-execution checks, registry replacement, engine-bound verification, postcondition success/mismatch, sandbox failure, replay prevention, consequential approval, proposer self-approval, approval replay, result-manifest spoofing, hostile module mutation and real container execution. CI sets the container-required mode, so the two real-container adversarial tests must not silently skip there.

## Explicit limitations

- Stage 21 executes bounded pure computations inside a sandbox and returns observations; it does not operate real hardware, external services, networks, production databases or the host filesystem.
- No host-side actuation adapter, durable event store, rollback system, authenticated reviewer identity provider or production authorization enforcement is implemented here.
- Two distinct checker IDs are enforced internally; this is not a substitute for external independent security review.
- Passing a frozen set of checks is not proof of universal correctness, real-world success or safety outside the checked scope.
- Execution feedback is not trusted knowledge. It remains a separate record until a future, explicitly verified knowledge-admission process accepts it.

Stage 21 is therefore **PASS only within bounded grounded proposals, independent internal checks, container-isolated computation, feedback capture and a consequential-action approval gate** once full CI and Security Guard pass. Real-world actuation remains a separate host integration.
