# Phase 343 — Semantic State Induction

Adds a raw-text to structured-state layer addressing the representation gap
identified by the earlier raw lexical diagnostic.

The state separates variable entities, operations, constraints, conversational
context, and desired output type. It is independent of response generation and
can be consumed by reasoning/planning layers.

The implementation is deterministic and inspectable. It uses bounded evidence
extraction plus support-based schema induction; it does not claim that lexical
rules alone provide general semantic understanding.

Run:

    python -m pytest -q phase343_semantic_state/test_phase343.py

Acceptance requires all tests to pass, deterministic behavior across seeds,
adversarial operation substitution to alter task state, and support thresholding
to prevent singleton schema promotion.
