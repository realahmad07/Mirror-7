# Phase 5 — Universal Capability Matching

## Purpose

Phase 5 connects a Phase 3 TaskSpec to the Phase 4 capability catalog. The same matching mechanism works across domains; domain labels and task intents are constraints on compatibility, not separate intelligence engines.

## Matching behavior

`CapabilityMatcher` evaluates each task subtask independently (or the root task when no subtasks exist). It filters by supported domain and intent, ranks exact domain/intent matches ahead of generic matches, and reports compatible named input values.

A missing required input is reported separately because a prior task step may produce it. A supplied value with the wrong declared type rejects that candidate. Matching never invents values, aliases ports, executes a capability, or infers that a candidate is safe because it resembles the request.

The result explicitly separates compatibility from execution eligibility. A capability with insufficient evidence may appear in the report for inspection, but its `eligible` flag remains false and it cannot enter Phase 6 composition.

## Boundary

This phase returns ranked candidate sets and missing-input information. It does not select and run an algorithm, connect outputs across steps, prove preconditions, compose a plan, or change capability evidence.
