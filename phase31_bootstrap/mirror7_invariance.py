"""
Mirror 7 — Phase 31.4: Cross-Encoding Invariance

Provides verification and testing machinery to prove that identical underlying structures
encoded using completely different raw symbol vocabularies produce identical
discovered representations, identical structural states, and identical temporal trajectories.
Simultaneously proves that structurally non-isomorphic observations (even with identical
length or symbol histograms) are strictly distinguished.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

try:
    from .mirror7_representation import DiscoveredRepresentation, discover_representation, are_structurally_equivalent
    from .mirror7_state import StructuralState, extract_state, are_states_identical
    from .mirror7_temporal import TemporalStateTracker, StateTransition
except (ImportError, ValueError):
    from mirror7_representation import DiscoveredRepresentation, discover_representation, are_structurally_equivalent
    from mirror7_state import StructuralState, extract_state, are_states_identical
    from mirror7_temporal import TemporalStateTracker, StateTransition


def permute_vocabulary(raw_observation: bytes, target_symbols: Sequence[int]) -> bytes:
    """
    Generate an isomorphic observation by projecting the unique symbols of raw_observation
    onto a completely distinct target vocabulary.
    """
    unique_symbols = []
    for b in raw_observation:
        if b not in unique_symbols:
            unique_symbols.append(b)

    if len(target_symbols) < len(unique_symbols):
        raise ValueError(
            f"Target vocabulary has {len(target_symbols)} symbols, but observation requires {len(unique_symbols)}"
        )

    mapping = {orig: target_symbols[i] for i, orig in enumerate(unique_symbols)}
    return bytes(mapping[b] for b in raw_observation)


def verify_cross_encoding_identity(observations: Sequence[bytes]) -> Tuple[bool, str]:
    """
    Verify that all provided raw observations produce byte-identical representations,
    states, and temporal signatures despite using different raw byte vocabularies.
    Returns (True, state_hash) on success, or (False, reason) on mismatch.
    """
    if not observations:
        return False, "No observations provided"

    representations = [discover_representation(obs) for obs in observations]
    ref_rep = representations[0]

    for idx, rep in enumerate(representations[1:], start=1):
        if rep.checksum != ref_rep.checksum:
            return False, f"Representation mismatch at observation {idx}"
        if rep.canonical_tokens != ref_rep.canonical_tokens:
            return False, f"Canonical tokens mismatch at observation {idx}"
        if rep.runs != ref_rep.runs:
            return False, f"Runs mismatch at observation {idx}"
        if rep.transitions != ref_rep.transitions:
            return False, f"Transitions mismatch at observation {idx}"

    states = [extract_state(rep) for rep in representations]
    ref_state = states[0]

    for idx, state in enumerate(states[1:], start=1):
        if state.state_hash != ref_state.state_hash:
            return False, f"State hash mismatch at observation {idx}"
        if state.state_id != ref_state.state_id:
            return False, f"State ID mismatch at observation {idx}"

    # Verify temporal trajectory equivalence
    tracker1 = TemporalStateTracker.replay(states)
    tracker_ref = TemporalStateTracker.replay([ref_state] * len(states))
    if tracker1.temporal_signature() != tracker_ref.temporal_signature():
        return False, "Temporal trajectory mismatch under isomorphic sequence"

    return True, ref_state.state_hash


def verify_structural_divergence(obs1: bytes, obs2: bytes) -> bool:
    """
    Verify that two structurally different observations are correctly separated at
    both representation and state levels.
    """
    rep1 = discover_representation(obs1)
    rep2 = discover_representation(obs2)
    state1 = extract_state(rep1)
    state2 = extract_state(rep2)

    return (
        rep1.checksum != rep2.checksum
        and state1.state_hash != state2.state_hash
        and state1.state_id != state2.state_id
    )
