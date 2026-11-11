# Strategic Phase 9 — Continuous, Evidence-Gated Expansion

**Implementation:** `mirror_crystal/phase9_continuous_expansion.py`  
**Tests:** `mirror_crystal/tests/test_phase9_continuous_expansion.py`  
**Status:** bounded expansion epochs and persistent audit ledger implemented; unrestricted open-world discovery remains unaccepted.

## Goal

Create a repeatable admission loop that can add newly learned target-domain capabilities only after a Phase 8 transfer benchmark passes and the currently accepted system survives a frozen regression suite. Every attempt—accepted or rejected—is recorded in an append-only, hash-chained SQLite ledger.

## Epoch gates

1. **Integrity preflight:** verify the existing ledger chain and reject reused epoch IDs.
2. **Novel evidence:** require a passing Phase 8 report spanning at least two distinct target domains, at least one learned target capability, distinct task IDs within the epoch, and no reuse of task IDs or previously accepted capability IDs from prior epochs.
3. **Regression freeze:** snapshot all regression case IDs, capability IDs, inputs and expected outputs as canonical JSON before executing any baseline callback. Store the SHA-256 digest of that exact regression manifest in the ledger.
4. **Preservation check:** invoke the host-supplied baseline runner for every frozen regression. A mismatch or exception rejects the epoch; no existing capability is silently modified.
5. **Durable decision:** append the decision, Phase 8 manifest digest, new task/capability identities, capability fingerprints, regression results and blocker reasons to the SQLite chain.
6. **Explicit promotion:** accepted proposals are returned to the caller but are not registered automatically. Promotion requires an intact current ledger, an exact matching accepted ledger entry, matching proposal fingerprints, correct learned-program provenance and exact handler/program binding.

The SQLite ledger uses a transaction, `synchronous=FULL`, WAL for disk-backed databases, and triggers rejecting application-level UPDATE/DELETE statements. Each entry includes the previous entry digest. `verify_chain()` checks sequence continuity, previous-digest links and each entry digest. It detects out-of-band content modification but is not a secure audit boundary against an attacker who can rewrite the database and its triggers or recompute the entire chain; protect the database file with normal OS access controls and external backups/signatures for stronger tamper evidence.

## API

```python
from mirror_crystal import (
    ContinuousExpansionEngine,
    ExpansionLedger,
    FrozenRegressionCase,
)

ledger = ExpansionLedger("mirror-expansion-ledger.sqlite3")
engine = ContinuousExpansionEngine(ledger)

result = engine.run_epoch(
    epoch_id="epoch-0001",
    benchmark=passing_phase8_report,
    regression_cases=(
        FrozenRegressionCase(
            case_id="existing-math-smoke-1",
            capability_id="existing:math:capability",
            inputs={"x": 10},
            expected=11,
        ),
    ),
    baseline_runner=host_baseline_runner,
)

if result.accepted:
    # Registration is a separate, explicit operator decision.
    engine.promote(
        result, "mathematics",
        capability_registry, operation_registry,
    )
```

The baseline runner is host-supplied and must invoke the currently accepted baseline implementation for the declared capability. Its correctness and isolation are operator responsibilities. The regression suite must cover the existing behaviors that the deployment is unwilling to lose.

## Tests and negative controls

The Phase 9 test module covers:
- acceptance after a passing Phase 8 benchmark and full regression;
- explicit promotion, with no automatic registry mutation;
- frozen regression failure and callback exceptions;
- reused task evidence across epochs;
- duplicate epoch IDs and empty regression suites;
- update/delete rejection and hash-chain tamper detection;
- rejection of a forged result not present in the engine's ledger;
- regression-manifest digest changes when expected outputs change.

## Scope boundary

Phase 9 implements a **bounded, repeatable, evidence-gated expansion mechanism**. It does not autonomously discover arbitrary new knowledge, invent authenticated ground truth, create new domain verifiers, decide that a task is novel in the open world, or continuously run unattended. Candidate generation, target examples, verifiers, baseline runners and the decision to promote remain explicit host/operator inputs. The phase does not establish unrestricted self-modification, continuous open-world intelligence, AGI or guaranteed preservation beyond the frozen regression set.
