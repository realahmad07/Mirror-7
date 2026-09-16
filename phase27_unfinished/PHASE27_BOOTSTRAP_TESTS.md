# Phase 27 Bootstrap Tests

This file defines the executable acceptance boundary for structured relocation.

## Invalid programs that must fail at the top level

- `ELSE`
- `THEN`
- `IF 1 ELSE ELSE THEN`
- `IF 1 THEN ELSE`
- `IF 1`
- `IF 1 ELSE`
- nested malformed constructs where an inner compiler invocation fails

## Valid programs that must compile

- `IF 1 THEN`
- `IF 1 ELSE THEN`
- nested `IF 1 IF 2 THEN THEN`
- nested `IF 1 IF 2 ELSE THEN ELSE THEN`

## Required invariant

A malformed descendant must never be converted into successful completion by an enclosing compiler layer. The final compile result must be failure.

## Relocation invariant

Branch operands must be produced from the current emitted code position / relocation records. Tests must not depend on fixed byte offsets from a particular primitive-prefix size.
