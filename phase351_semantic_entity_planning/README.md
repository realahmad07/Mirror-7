# Phase 351 — Semantic Entity and Context to Planning

Resolves semantic entities only against facts explicitly present in world state.

Safety boundary:
- no world-state facts are fabricated;
- missing entities remain unresolved rather than guessed;
- explicit entity/goal contradictions reject planning;
- context carryover remains available;
- planner actions remain exclusively transition-model actions.
