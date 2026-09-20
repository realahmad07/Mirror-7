# Phase 342 — Browser/Release Boundary

## Objective

Close the remaining GitHub-side product boundary before GPU training.

## Acceptance

1. Browser preflight returns HTTP 204 and the configured origin.
2. API responses expose defensive browser/security headers.
3. A real session can be created, stepped, inspected, and closed through the HTTP API.
4. The production package still builds and the backend test suite remains green.
5. Docker image construction remains part of the release gate.

## Scope

This phase does not change Mirror 7 cognition rules and does not claim language competence, AGI, or training success. It makes the already-tested backend safe and usable as a browser-facing product boundary.

## Promotion rule

Promote only when the focused phase suite, backend regression suite, package build, compile check, and Docker build all pass.
