# Mirror 7 Algorithm 02: Compositional Program Induction

## Purpose
Infers reusable procedures from input-output examples and composes primitives.

## Mechanism
Uses beam search over an expression tree of typed primitives to synthesize programs. Scores programs based on example accuracy.

## API
- `ProgramSynthesis(max_depth, beam_width)`: Initialize synthesis engine.
- `synthesize(examples)`: Return best Program matching input-output tuples.
- `execute(program, inputs)`: Run program on inputs.
- `explain(program)`: String representation of program.

## Limitations
- Bounded search space may fail on very complex logic.
- Basic typing system currently supports simple compositions.
