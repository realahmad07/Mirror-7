# Mirror 7 — Phase 58: Hierarchical Concept Abstraction

Phase 58 turns recurring low-level concepts into recursively discovered
higher-level concepts.

```text
stable concepts
      ↓
recurrent compositions
      ↓
compact abstraction
      ↓
higher-level concept
      ↓
repeat recursively
      ↓
hierarchy
```

No hierarchy labels are supplied. Candidates must recur across independent
episodes and produce a positive bounded compression gain.

## Acceptance

- 3 progressively harder families × 3 seeds
- held-out recombination
- hierarchy-depth test
- negative/noise control
- distractor robustness
- deterministic regression
- malformed-input rejection

Run:

```bash
python -m phase58_hierarchical_abstraction.run_phase58_gate
python -m pytest phase58_hierarchical_abstraction/test_phase58.py -q
```

Expected: **8/8**.

Boundary: learned hierarchical structural composition, not semantic abstraction or AGI.
