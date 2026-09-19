# Phase 319 — Production Multi-Capability Autonomous Upgrade Integration

Phase 319 closes the integration boundary between the multi-capability frontier (307–314) and the production self-modification sandbox (315–318).

## Pipeline

```text
shared capability frontier
        ↓
select largest weighted gap
        ↓
capability adapter
        ↓
bounded improvement
        ↓
production sandbox for source-changing path
        ↓
sealed train / held-out / regression evaluation
        ↓
cross-capability regression guard
        ↓
promote OR rollback
        ↓
update frontier
        ↺
```

## What is integrated

- Four existing capability adapters remain registered under one shared frontier.
- Planning, compositional-language, and symbolic-reasoning adapters keep their existing bounded mechanisms.
- Sequence source redesign is replaced by a production adapter backed by `ProductionSelfRedesign`.
- Every source evaluation therefore passes through the Phase 315 Docker boundary.
- The Phase 303 cross-capability regression guard remains the outer promotion decision.
- If an adapter reports a change but the outer gate rejects it, the production source adapter rolls back the promoted source.
- Step counts, redesign rounds, and candidate counts are explicitly bounded.

## Acceptance boundary

The dedicated workflow runs Phases 307–319 together, verifies Docker image availability, and executes the complete Phase 319 suite.

A successful Phase 319 result demonstrates an integrated, bounded production upgrade pipeline. It does not establish unrestricted self-improvement, autonomous architecture invention, frontier-model equivalence, or AGI.
