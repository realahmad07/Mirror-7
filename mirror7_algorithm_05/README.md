# Semantic Memory Graph

## Purpose
Acts as the long-term knowledge store, representing relationships, confidence, and contradictions.

## Mechanism
- Typed knowledge graph (nodes/edges)
- Bounded memory with utility-based eviction
- Contradiction detection for conflicting facts

## API
- `add_node`
- `add_edge`
- `query`
- `find_path`
- `detect_contradictions`

## Limitations
- Graph traversals are depth-bounded to prevent unbounded recursion
