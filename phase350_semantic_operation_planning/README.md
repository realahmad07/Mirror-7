# Phase 350 — Semantic Operation to Planning

Connects the induced semantic operation to planner action ordering.

Safety boundary:
- semantic operation is metadata, not world state;
- operation hints never create or remove transition-model actions;
- explicit step constraints from Phase 349 remain active;
- planner state is never fabricated or modified.
