# Phases 191–200 — Open Action / Affordance Discovery

This block removes action names and accepts opaque action bytes. Mirror records
before/after outcomes and derives bounded affordances from evidence.

Delivered:
191 opaque action ingestion
192 effect discovery
193 precondition evidence
194 failure evidence
195 safe action selection
196 conservative unknown-action probing
197 context-gated action choice
198 goal-effect matching
199 bounded action composition
200 fail-closed composition

No action meaning is supplied by the environment. The learner cannot execute
unknown effects in its internal model; it must observe an outcome first.

Run:
```bash
python -m pytest phase191_200_affordance_discovery/test_phase191_200.py -q
```

Boundary: bounded affordance discovery, not unrestricted real-world agency.
