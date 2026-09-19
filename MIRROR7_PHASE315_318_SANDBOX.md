# Mirror 7 Phases 315–318 — Production Self-Modification Sandbox

## Purpose

Phases 315–318 move self-modification evaluation from a bounded host subprocess to a fail-closed production sandbox.

### Phase 315 — Hardened sandbox

The candidate runs inside Docker with:

- network disabled
- read-only container root
- read-only candidate workspace bind mount
- all Linux capabilities dropped
- no-new-privileges
- non-root UID/GID 65534:65534
- bounded PIDs
- bounded memory and CPU
- bounded file size
- bounded writable /tmp
- bounded candidate output
- outer execution timeout
- no Docker socket or repository mount

### Phase 316 — Sandboxed evaluator

ProductionSealedSourceEvaluator injects the Phase 315 sandbox into the existing sealed evaluator.

There is no silent fallback to the legacy host-process evaluator.

### Phase 317 — Sandboxed self-redesign

ProductionSelfRedesign injects the production evaluator into AutonomousRedesignEngine.

Promotion still requires the existing train / held-out / regression gate. Failed candidates never replace active source, and rollback restores the exact previous source.

### Phase 318 — Launch gate

Dedicated CI workflow: .github/workflows/phases-315-318.yml

The workflow pulls the declared sandbox image, verifies Docker, and runs the complete sandbox acceptance suite.

## Security boundary

This is a hardened application-level sandbox, not a claim of perfect isolation against arbitrary kernel or container-runtime vulnerabilities. Production deployments should use a trusted, pinned sandbox image and a hardened or rootless Docker host where practical.
