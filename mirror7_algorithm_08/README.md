# Mirror 7 Knowledge Consolidation Engine (Algorithm 08)

## Purpose
Convert useful repeated experiences into durable knowledge while resisting contradictory noise, memory explosion, and corruption.

## Mechanism
- **Two-Tier Memory**: Short-Term Buffer (STB) for recent experiences and Long-Term Store (LTS) for consolidated knowledge.
- **Consolidation**: Clusters experiences, extracts common content/schemas, and promotes them if they meet a threshold.
- **Noise Resistance**: Uses contradictions to adjust confidence, overriding knowledge only if evidence is stronger.
- **Memory Management**: Enforces bounded memory limits via FIFO for STB and utility-based eviction for LTS.

## API
- `add_experience(exp)`: Adds an experience, checking for immediate reinforcement or contradiction.
- `consolidate()`: Processes STB to extract knowledge.
- `query_knowledge(key_values)`: Fetches LTS knowledge matching given fields.
- `get_stats()`: Returns sizes of STB and LTS.

## Limitations
- Contradiction logic is based on simple key overlap.
- Schema extraction uses strict string matches to determine wildcards.
