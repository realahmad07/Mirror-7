# Testing, GitHub Checks, and Release Discipline

## Audit finding

The repository had accumulated a very large set of phase-specific GitHub Actions workflows. That was useful during construction, but it made ordinary pushes enqueue many overlapping checks.

The repository is now divided into:

- active CI for current engineering;
- manual deep regression for the complete historical suite;
- archived legacy workflows preserved outside .github/workflows/.

## Active checks

| Workflow | Trigger | Purpose |
|---|---|---|
| CI | push / PR | Fast compile, backend smoke, training foundation, build. |
| Native Brain Training Checks | training changes / PR | Native-brain and Exp6 structural tests. |
| Deep Regression | manual only | Complete historical pytest suite. |
| Release | version tag / manual | Distribution build. |
| Security Guard | push / PR | Credential-pattern and compile guard. |

## Why deep regression is manual

A prior full backend-focused regression reached 50 passing tests after roughly 1 hour 47 minutes before it was interrupted. No failure was reported during that run, but it was too expensive to make every documentation or code push wait on the complete historical suite.

## Fast CI gate

The fast workflow verifies:

- compilation of active backend/training Python;
- backend tests;
- backend smoke and release-boundary tests;
- training foundation tests;
- smoke dataset generation and validation;
- package build.

The native-brain workflow installs training dependencies and runs the focused native-brain and Exp6 structural suites.

## Legacy workflow policy

Files under docs/archive/legacy-workflows/ are historical copies. GitHub does not treat them as active workflows because only files under .github/workflows/ are workflow definitions.

This preserves provenance while preventing old phase jobs from flooding the current Checks UI.

## What a green check means

A green CI run means the current fast engineering gate passed.

It does not mean every historical research phase was re-executed on that commit.

A green Deep Regression run is the stronger signal when a complete repository re-audit is intentionally requested.

## Local verification

~~~bash
python -m compileall -q mirror7_backend training
python -m pytest -q mirror7_backend
python -m pytest -q phase324_backend_smoke phase342_release_boundary
python -m pytest -q training/test_training_foundation.py
python -m pytest -q training/test_native_brain.py training/test_experiment6_learned_operations.py
python -m build
~~~

## Release discipline

Normal development pushes are not release artifacts. The release workflow is tag-oriented so a versioned package can be tied to an explicit commit.

## Universal intelligence core acceptance

The nine concrete engineering layers (knowledge, representation, task understanding, capability contracts, matching, composition, controlled execution, independent verification and evidence-gated learning) passed the combined end-to-end test. The accepted code commit ran 291 tests across the Crystal/evaluation, backend, release-boundary and training suites. The final documentation commit also passed full CI and the security guard. See [PHASE1_9_ACCEPTANCE.md](../PHASE1_9_ACCEPTANCE.md) for exact counts and both code/documentation workflow runs.

This engineering-layer numbering is distinct from the original strategic nine-phase roadmap. Original Phases 1–5 passed within the bounded test scope; Phases 6–7 remain partial, and cross-domain transfer (Phase 8) and open-world continuous expansion (Phase 9) are not yet accepted. A green fast-CI run does not prove those broader research outcomes or re-execute every historical phase.
