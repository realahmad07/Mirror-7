# Phase 329 — Production Backend Integration & Hardening

## Acceptance gate

- Session lifecycle is safe against close/step races.
- HTTP request bodies are bounded to 1 MiB.
- JSON content type is enforced for request bodies.
- Session IDs are bounded and control characters are rejected.
- HTTP exposes health/status, create, step, checkpoint save/restore, and close.
- Missing sessions are reported as HTTP 404.
- Conflicts/runtime lifecycle errors are reported as HTTP 409.
- Dedicated tests cover lifecycle, persistence, HTTP validation, and body limits.

## Verification command

`python -m pytest -q phase329_backend_integration/test_phase329.py mirror7_backend/test_http_server.py`

## Scope

This phase hardens the backend transport/orchestration boundary. It does not claim or modify the underlying cognitive algorithm.
