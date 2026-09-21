# Phase 349 — Semantic Constraints to Planning

Connects explicit semantic step constraints to planner search depth.

Safety boundary:
- semantic state remains task metadata;
- world state is never fabricated or modified;
- only an explicitly parsed step/action upper bound can tighten search depth;
- actions still come exclusively from the learned transition model.
