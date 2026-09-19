# Phase 330 — Complete Backend API/Product Layer

## Acceptance
- Session listing and snapshots are available through the service layer.
- Checkpoint deletion is available through the service layer.
- HTTP API exposes health, status, session listing, create, step, save, restore, and close.
- CORS origin is configurable.
- Internal server errors are sanitized.
- The API remains UI-independent and delegates cognition to the existing runtime.

## Verification
`python -m pytest -q phase330_backend_api/test_phase330.py phase329_backend_integration/test_phase329.py mirror7_backend/test_http_server.py`
