# Stage 19 — Controlled Self-Improvement

## Goal and security boundary

Stage 19 adds an evidence-gated path for proposing and evaluating code or rule improvements. It deliberately does not provide unrestricted self-modification. A proposal is an immutable source artifact paired with an exact baseline, a frozen regression/improvement suite, and an environment fingerprint.

Implementation: `mirror_crystal/stage19_controlled_improvement.py`.

## Candidate contract

The current bounded artifact interface is a Python source module exposing:

```python
def run(inputs: dict) -> object:
    # Return finite JSON-compatible data.
    ...
```

The same interface can represent code changes or a rule interpreter/compiled rule adapter. Source is limited to 512 KB per artifact. Artifact paths must be normalized relative paths and cannot contain drive prefixes or traversal segments. Baseline revision and both source digests are recorded in the proposal fingerprint.

## Frozen regression suite

A `FrozenRegressionSuite`:
- Is bound to a 40- or 64-character lowercase Git revision and exact SHA-256 environment fingerprint.
- Contains at least three regression-preservation cases and at least one explicit improvement case.
- Uses unique case IDs and immutable JSON input/expected-output values.
- Requires baseline and candidate expected outputs to match for regression cases; improvement cases must declare a changed expected result.
- Is capped at 512 KB and bound to container runtime, image, CPU/time, memory and PID configuration.

The environment fingerprint is computed from the selected Docker/Podman runtime, pinned container image, resource settings, sandbox policy version and Stage 19 harness version. The executor verifies that its runtime settings exactly match the frozen suite.

## Evaluation flow

1. Validate proposal ID, baseline/candidate source digests, artifact path, Git revision and suite fingerprint.
2. Run the baseline artifact in a new container against the baseline expected outputs.
3. Stop if any baseline case fails; comparison against a broken baseline is invalid.
4. Run the candidate artifact in a second new container against candidate expected outputs. The harness and candidate use separate namespaces and separately cloned import-module maps.
5. Require every frozen case to pass. A failure, missing case, duplicate case, non-zero exit, runtime mismatch, malformed harness result or missing container runtime rejects the candidate.
6. Produce a deterministic evaluation record. A passing candidate is marked `READY_FOR_HUMAN_REVIEW`, never deployed automatically.
7. Require an explicit independent reviewer decision. The proposal author cannot approve their own change. The engine accepts only evaluation objects it recorded in that same engine instance and permits one decision per evaluation.
8. An approval returns a scoped `PromotionAuthorization` bound to proposal ID, evaluation ID, candidate SHA-256, suite fingerprint and environment fingerprint.

## Sandbox policy

Production evaluation calls `run_python_regression_in_container` and fails closed when Docker/Podman is unavailable. The existing Stage 17 container runner supplies network isolation, read-only root, non-root UID, dropped capabilities, no-new-privileges, CPU/memory/PID/file limits, and a bounded timeout. It does not fall back to a process-only runner.

## Acceptance tests

```bash
python -m pytest -q mirror_crystal/tests/test_stage19_controlled_improvement.py
python -m pytest -q mirror_crystal/tests/test_container_sandbox.py
python -m pytest -q mirror_crystal mirror_eval
```

The Stage 19 tests cover immutable proposals/suites, minimum test composition, dual baseline/candidate comparisons, baseline failure, candidate regression, wrong artifact/suite/environment evidence, self-approval, cross-engine forged evaluation, approval replay, explicit rejection and real container execution.

## Explicit limitations

- The artifact runner is currently a bounded Python `run(inputs)` contract, not an arbitrary repository patch applicator.
- Passing tests are evidence about the frozen test suite only; they do not prove a candidate is universally correct or safe.
- Reviewer identity must be authenticated by the host application. The engine checks distinct reviewer IDs but does not verify identity credentials.
- Approval/evaluation ledgers are in-memory. The host must persist approvals, logs, artifacts and rollback metadata before using authorization for deployment.
- The authorization object does not apply code, mutate a capability registry, create/merge a PR, or deploy. A separate trusted host integration must enforce it.
- Fake executors are test injection points only; production evaluation must use `ContainerRegressionExecutor`.

Stage 19 is **PASS for bounded, isolated candidate evaluation and explicit approval gating**. Durable audit storage, authenticated identity, promotion and rollback remain host/deployment integration work.
