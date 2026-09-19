# Mirror 7 Algorithm 01: Open-World Concept Discovery

## Purpose
Discovers concepts dynamically from unstructured observations without predefined cluster sizes or categories.

## Mechanism
Uses an incremental concept tree based on Cobweb. Evaluates structural operations (incorporate, create new) by maximizing Category Utility.

## API
- `ConceptTree(max_depth, max_nodes)`: Initialize tree.
- `observe(observation)`: Incorporate a new observation.
- `classify(observation)`: Classify an observation into an existing concept node.
- `get_concepts()`: List all learned concept node IDs.

## Limitations
- Numeric features must be pre-discretized.
- Memory scales with the number of unique features and categories unless bounded by max_nodes.
