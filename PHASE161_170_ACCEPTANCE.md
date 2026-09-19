# Mirror 7 — Phases 161–170 Black-Box Evaluation Acceptance

## Purpose

This block removes more evaluator scaffolding by restricting the agent-facing protocol to raw bytes and opaque action bytes.

~~~text
raw observation bytes
        ↓
Mirror-discovered representation
        ↓
opaque action bytes
        ↓
predicted raw outcome
        ↓
external environment
        ↓
raw outcome bytes
        ↓
update / replan
~~~

| Phase | Focus |
|---|---|
| 161 | Black-box protocol |
| 162 | Raw representation discovery |
| 163 | Opaque action tokens + abstention |
| 164 | Multiple hidden domains |
| 165 | Hidden-goal action selection |
| 166 | Representation-independent policy transfer |
| 167 | Bounded irreversible long-horizon task |
| 168 | Contamination controls |
| 169 | Scaling measurements |
| 170 | Final acceptance gate |

Acceptance:
- seeds 0, 1, 2;
- every phase must pass on all three seeds;
- unknown opaque actions must be rejected conservatively;
- raw representation is discovered from transition evidence rather than semantic labels;
- evaluator controls test contamination behavior;
- scaling phase records measurements rather than imposing an arbitrary latency target;
- the external evaluator accepts only fixed summary metrics.

## Verification

The GitHub workflow re-runs Phases 141–152 and 153–160, then executes 161–170 across seeds 0/1/2 and runs independent evaluator positive/negative controls.

## Boundary

This is a bounded black-box/raw-byte evaluation. It does not establish arbitrary real-world perception, unrestricted action-space discovery, unrestricted cross-domain transfer, or AGI.
