# Phase 28 — Surface Parser → MIRR Compiler Integration

Phase 28 connects the surface syntax layer to the real MIRR compiler. The parser now emits a versioned, validated parsed-structure IR that is designed to be consumed by the compiler; compiler integration remains the next gated step.

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
- malformed numbers
- `ELSE` without an open `IF`
- duplicate `ELSE` in one `IF`
- `THEN` without an open `IF`
- unclosed `IF` blocks
- malformed nested definitions

## Phase 28.2 — Parsed-structure ABI

The parser/compiler boundary is explicitly defined by `surface_ir.h`.

The ABI is:

- versioned (`MIRROR7_SURFACE_IR_VERSION`)
- bounded (`MIRROR7_SURFACE_IR_MAX_TOKENS`)
- source ordered
- typed by `mirror7_ir_kind_t`
- source-position aware
- explicit about 16-bit numeric values
- independently validated by `mirror7_surface_ir_validate()`

`mirror7_parse_ir()` is the parser producer API. `mirror7_parse()` remains as a compatibility validation wrapper.

The boundary is deliberately not yet connected to the compiler:

```text
MIRR SOURCE
     ↓
SURFACE PARSER
     ↓
versioned surface IR
     ↓
MIRR COMPILER  ← Phase 28.3
```

`test_surface_ir.c` verifies token ordering, source positions, typed tokens, numeric representation, version validation, and malformed-IR rejection.

## Verification rule

Phase 28 is NOT complete merely because the parser or IR component passes. The final gate requires the actual parser output to be consumed by the real MIRR compiler and then exercised through dictionary generation, relocation, executable generation, VM execution, held-out programs, negative cases, and Phase 27 regression.

## Next gate

**Phase 28.3 — connect the parser-produced surface IR to the real MIRR compiler without a parallel raw-source bypass.**
