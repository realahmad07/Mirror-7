# Independent Evaluation Protocol

This directory documents the protocol used for the 2026-09-19 external-style evaluation.

The evaluator was written outside the repository before the evaluation and hashed:
`c821c963ab8b80f5a28611c79f971deec6133eeee9836d210f293f325fcaed7c`

The source was published only after the evaluation result was locked.

The protocol intentionally used task families not present in the Phase 53 benchmark:
- reset + additive
- copy relation
- capped increment
- three-variable cycle
- sign-dependent conditional transformation
- mixed swap + increment

Acceptance was strict: all episodes must solve with zero invalid actions.

The recorded result was 14/18 solved, 9/9 held-out, and 184 invalid actions.
Therefore the gate is FAIL.
