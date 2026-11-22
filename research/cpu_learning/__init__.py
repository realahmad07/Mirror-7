"""Mirror 7 CPU-Native Learning Research Package (SHARD + ITL + RDC).

Isolated experimental package implementing staged, evidence-gated CPU-native learning
research without modifying production Mirror-7 organs or vaults.

Current implemented stage: S0 (Benchmark harness, hidden relational simulator,
generator-level splits, retrieval/MDL audits, baselines, and SHARD arithmetic canary).
"""
from __future__ import annotations

RESEARCH_STAGE = "S0"
__all__ = ["RESEARCH_STAGE"]
