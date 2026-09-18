"""
Mirror 7 — Phase 31.2: Representation -> Stable State

Derives a stable, canonical structural state from discovered representations.
State identity is independent of raw symbol vocabulary, deterministic, and
sufficiently expressive to distinguish distinct structural dynamics while
grouping equivalent structural encodings into identical states.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Tuple

try:
    from .mirror7_representation import DiscoveredRepresentation, discover_representation
except (ImportError, ValueError):
    from mirror7_representation import DiscoveredRepresentation, discover_representation


@dataclass(frozen=True)
class StructuralState:
    state_id: str
    length: int
    vocab_size: int
    num_runs: int
    max_run_length: int
    run_profile: Tuple[int, ...]
    transition_count: int
    unique_transitions: int
    motif_signature: Tuple[Tuple[Tuple[int, ...], int], ...]
    state_hash: str


def extract_state(representation: Any) -> StructuralState:
    if not isinstance(representation, DiscoveredRepresentation):
        raise TypeError(
            f"extract_state requires a DiscoveredRepresentation instance, got {type(representation).__name__}"
        )

    run_profile = tuple(count for _, count in representation.runs)
    max_run_length = max(run_profile) if run_profile else 0
    unique_transitions = len(set(representation.transitions))

    hasher = hashlib.sha256()
    hasher.update(f"REP_CHECKSUM:{representation.checksum}|".encode("ascii"))
    hasher.update(f"RUN_PROF:{','.join(map(str, run_profile))}|".encode("ascii"))
    hasher.update(f"TRANS_COUNT:{len(representation.transitions)}|U_TRANS:{unique_transitions}|".encode("ascii"))
    state_hash = hasher.hexdigest()

    state_id = f"STATE_{state_hash[:16]}"

    return StructuralState(
        state_id=state_id,
        length=representation.raw_length,
        vocab_size=representation.vocab_size,
        num_runs=len(representation.runs),
        max_run_length=max_run_length,
        run_profile=run_profile,
        transition_count=len(representation.transitions),
        unique_transitions=unique_transitions,
        motif_signature=representation.motifs,
        state_hash=state_hash,
    )


def are_states_identical(s1: StructuralState, s2: StructuralState) -> bool:
    if not isinstance(s1, StructuralState) or not isinstance(s2, StructuralState):
        raise TypeError("Both arguments must be StructuralState instances")
    return s1.state_hash == s2.state_hash
