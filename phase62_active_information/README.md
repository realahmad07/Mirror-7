# Mirror 7 — Phase 62: Partial Observability + Active Information Seeking

Phase 62 removes the full-state visibility assumption from Phase 61.

The agent receives:

```text
partial observation
goal
legal actions
transition / observation feedback
```

Hidden values appear as `None`. The learner does not receive sensor labels
or an information ontology. It infers an action as information-producing when
the action increases visibility without changing already-visible state.

## Mechanism

```text
partial observation
        ↓
detect hidden goal-relevant variables
        ↓
infer information-producing actions from consequences
        ↓
rank information actions by learned relevance / gain
        ↓
reveal enough state to make the goal decidable
        ↓
learn / use full-state transition model
        ↓
plan + act
        ↓
observe discrepancy
        ↓
revise / fail closed
```

Unknown legal actions are bounded-probed once before repetition. When no safe
information path or model-supported action remains, the agent terminates
without inventing an action.

## Acceptance

- 3 progressively harder task families × 3 seeds;
- opaque held-out hidden configuration;
- learned goal-relevant sensor selection;
- information-path dropout;
- contradictory transition evidence / replanning;
- no repeated unknown-action probing at the same partial state;
- full-state transition learning after reveal;
- Phase 61 handoff compatibility;
- malformed-input rejection;
- 9/9 executable gate tests.

Run:

```bash
python -m phase62_active_information.run_phase62_gate
python -m pytest phase62_active_information/test_phase62.py -q
```

Expected:

```text
PHASE 62 ACCEPTANCE GATE: PASS
progressive: 3 task families × 3 seeds
held-out: 2/2
adversarial: 3/3
integration: Phase 61 handoff contract
regression: 9/9
```

## Boundary

Phase 62 establishes bounded partial-observation control and active
information seeking in deterministic structured environments. It does not
establish unrestricted belief-state inference, real-world sensing, or AGI.
