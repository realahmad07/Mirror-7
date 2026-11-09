# Phase 3 — Universal Task Understanding

## Purpose

Phase 3 turns a request into an explicit description of what needs to be achieved. It uses the shared representation model from Phase 2, but does not select or execute capabilities. The same task contract applies across mathematics, programming, physics, biology, chemistry, engineering, finance, writing, and future domains.

## Task contract

The TaskSpec stores:

- the original request and goal, preserving the user's wording by default;
- the recognized intent and its lexical evidence;
- a domain hint or a conservative domain estimate;
- known inputs and caller-supplied context;
- requested output formats;
- explicit constraints and their source;
- explicit assumptions, unknowns, and acceptance criteria;
- any subtasks and their dependency graph;
- schema and interpreter versions.

A request that has no recognized action is not silently assigned a made-up action. A short unresolved reference such as "fix it" is flagged for clarification unless context or known inputs provide a referent.

## One interpreter, multiple domains

TaskUnderstandingEngine uses one shared set of contracts and parsing pipeline. Domain inference is a conservative lexical cue, and callers can provide an authoritative domain_hint. A general domain remains valid when no domain cue is strong enough.

This is bounded native request parsing, not a claim of unrestricted natural-language understanding. The original request remains attached to the task so later layers can revisit or improve the interpretation without losing information.

## Decomposition boundary

Explicit numbered/bulleted action lists and clear sequential action clauses may become subtasks. Only explicit sequential language creates dependency edges. Parallel list items remain independent.

TaskSpec validates unique identifiers, dependency references, and cycles, and exposes a stable dependency-first order. It does not assign capability IDs, choose algorithms, or decide how a subtask will be executed.

## Persistence

Tasks can be explicitly stored in DiskCrystalVault using the Phase 2 representation graph. The root task and subtasks are represented as goal nodes; decomposes_into and depends_on relations preserve the structure. Storing is an explicit call: requests are not automatically copied into long-term memory.

Identical tasks are idempotent in storage. The task graph is stored transactionally using the representation layer's endpoint and identifier protections.

## Verification boundary

The test suite covers domain cues, explicit domain hints, intent recognition and abstention, unresolved references, typed user context, explicit constraints, requested outputs, assumptions, acceptance criteria, parallel/sequential decomposition, cycle rejection, deterministic round trips, persistence, and privacy-conscious explicit storage.

Phase 3 does not yet identify required capabilities, plan execution, compose solutions, run code, prove mathematics, verify scientific hypotheses, or learn reusable methods. These are subsequent phases.
