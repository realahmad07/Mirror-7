# Phase 28 — Complete Surface Parser

Phase 28 extends the Phase 27 structured-control-flow compiler into a complete surface-syntax validation layer.

## Required grammar

A source program is a sequence of definitions:

```text
program   := { definition }
definition := ':' name { term } ';'
term      := integer | word | 'IF' | 'ELSE' | 'THEN'
```

The parser must reject:

- empty or missing definition names
- stray `;`
- unterminated definitions
- unknown words when compilation requires immediate resolution
- malformed numbers
- `ELSE` without an open `IF`
- duplicate `ELSE` in one `IF`
- `THEN` without an open `IF`
- unclosed `IF` blocks
- malformed nested definitions

## Verification rule

Presence of the parser source is not a PASS. Phase 28 is PASS only after the implementation is executed against valid, invalid, nested, boundary, and fuzz-generated inputs and the regression suite succeeds.

## Next gate

After Phase 28 passes, analyze the compiler boundary for Phase 29: moving the complete compiler implementation into MIRR without relying on a host-language compiler for compiler logic.
