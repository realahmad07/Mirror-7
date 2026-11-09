# Phase 6 — Typed Capability Composition Planning

## Purpose

Phase 6 turns candidate sets into a proposed dependency-ordered plan. It is the first layer that combines reusable capabilities to address a task, while remaining strictly separate from execution and verification.

## Composition rules

The planner performs a bounded deterministic search over the highest-ranked eligible candidates for each task unit. It connects outputs to inputs only when port names, types, and declared units/dimensions are compatible. A named task input or task-context value may bind only when its runtime value satisfies the input contract. Missing required inputs, ambiguous producers, missing capability dependencies, unsatisfied preconditions, and cycles block the plan.

Task dependencies are preserved. Capability dependencies are added to the step graph when the prerequisite capability is explicitly selected elsewhere in the same plan. Search limits are explicit so large candidate spaces cannot create an unbounded planning operation.

The output contains plan steps, bindings, dependencies, blockers, and expected postconditions. Expected postconditions are verification requirements only: a planned operation does not make them true. A plan is marked ready only when every required binding and declared prerequisite is structurally satisfied and every selected capability passes the Phase 4 evidence gate.

## Boundary

This phase never executes code, calls tools, performs physical operations, or upgrades evidence. A ready plan is a proposal for a later controlled executor and independent verifier, not proof that the requested result has been achieved.
