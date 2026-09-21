# Contributing to Mirror 7

## Start here

Read:

1. docs/01-project-overview.md
2. docs/02-system-architecture.md
3. docs/07-repository-organization.md
4. docs/06-testing-ci.md

## Principles

- Keep UI code separate from the backend.
- Prefer small, testable changes.
- Preserve research provenance.
- Add or update tests with behavior changes.
- Do not publish credentials or tokens.
- Avoid benchmark leakage.
- Record dataset fingerprints and training configuration.
- Treat failed experiments as useful evidence.

## Fast local gate

~~~bash
python -m compileall -q mirror7_backend training
python -m pytest -q mirror7_backend
python -m pytest -q phase324_backend_smoke phase342_release_boundary
python -m pytest -q training/test_training_foundation.py
python -m pytest -q training/test_native_brain.py training/test_experiment6_learned_operations.py
python -m build
~~~

## Research changes

Include:

- hypothesis;
- falsifiable prediction;
- frozen baseline;
- dataset fingerprint;
- model configuration;
- random seeds;
- held-out evaluation;
- negative controls where applicable;
- checkpoint/artifact location;
- explicit interpretation of improvement and failure.

## Pull requests

Explain what changed, why it changed, which tests were run, what evidence supports the change, and known limitations.
