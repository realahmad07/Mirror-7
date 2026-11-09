# Phase 7 — Controlled Capability Execution

## Purpose

Phase 7 takes a ready Phase 6 plan and executes it through explicitly registered native operation handlers. A capability definition is not executable source code. The host application must separately register a handler for the selected capability identifier.

## Execution boundary

`NativeOperationRegistry` is an allow-list. Handlers must be importable and process-serializable. Every plan step runs in a disposable child process with a wall-clock timeout, CPU and address-space limits where supported, and bounded JSON input/output. Inputs and outputs are checked against the capability's typed port contracts. Unknown or missing required outputs, wrong types, missing handlers, missing bindings, stale dependencies, or an ineligible capability stop the plan. No later step runs after the first failure.

A blocked composition plan is never executed. Execution reports preserve each step's inputs, outputs, duration, and failure. The result fingerprint binds later verification to the exact execution record.

## Trust limitation

The default runner provides process isolation and resource limits for trusted, host-supplied handlers. It is not a security sandbox for arbitrary untrusted code: it does not independently restrict filesystem or network access. Production execution of untrusted operations still requires a hardened container/VM runner supplied by the host. The engine deliberately does not evaluate user-provided source strings or dynamically import names from task input.

## Boundary

Phase 7 executes a previously approved plan. Successful execution is only an observation; it does not prove declared postconditions or upgrade capability evidence.
