# Phase 333 — Production Reliability Audit

## Scope
Final backend reliability pass over session concurrency, checkpoint integrity, and lifecycle behavior.

## Acceptance
- Checkpoint path traversal is rejected.
- Corrupted checkpoint payloads are rejected by integrity validation.
- Concurrent steps retain an exact session sequence with no lost increments.
- Closing a session removes it only after active work completes.
- No UI dependency is introduced.

## Verification
`python -m pytest -q phase333_backend_reliability/test_phase333.py phase332_backend_finalization/test_phase332.py phase331_backend_e2e/test_phase331.py phase330_backend_api/test_phase330.py phase329_backend_integration/test_phase329.py mirror7_backend/test_http_server.py`

This phase is a backend reliability audit. It does not establish AGI or completion of the historical Mirror 7 research gate.
