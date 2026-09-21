# Phase 352 — Semantic Desired Output → Execution

Connects the semantic state's desired-output field to a separate execution/output intent.

## Boundary
- Uses desired output only as metadata for downstream response realization.
- Planning still uses the explicit world state and learned transition model.
- No actions are invented, removed, or fabricated from output intent.
- Missing/unknown desired output remains unknown.

## Verification
Run:
`python -m pytest phase352_semantic_output_execution/test_phase352.py -q`
