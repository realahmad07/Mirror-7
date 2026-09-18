#"Mirror 7 — Phase 31.1: Raw Observation -> Structural Representation

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence, Tuple, Union


@dataclass(frozen=True)
class DiscoveredRepresentation:
    """
    Immutable, deterministic structural representation discovered from raw observation.
    """
    raw_length: int
    vocab_size: int
    canonical_tokens: Tuple[int, ...]
    runs: Tuple[Tuple[int, int], ...]
    transitions: Tuple[Tuple[int, int], ...]
    motifs: Tuple[Tuple[Tuple[int, ...], int], ...]
    checksum: str

    def __repr__(self) -> str:
        return (
            f"DiscoveredRepresentation(len={self.raw_length}, vocab={self.vocab_size}, "
            f"motifs={len(self.motifs)}, checksum={self.checksum[:8]}...)"
        )


def _validate_raw_input(raw: Any) -> bytes:
    """
    Validates that input is raw byte-like data (bytes, bytearray, or Sequence[int] in 0..255).
    Rejects strings, floats, None, dicts, or other types explicitly.
    """
    if raw is None:
        raise TypeError("raw observation cannot be None")
    if isinstance(raw, (str, dict, set)):
        raise TypeError(f"raw observation must be byte-like, got {type(raw).__name__}")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, (list, tuple)):
        out = bytearray()
        for idx, val in enumerate(raw):
            if not isinstance(val, int) or isinstance(val, bool):
                raise TypeError(f"Element at index {idx} is not an integer: {type(val).__name__}")
            if val < 0 or val > 255:
                raise ValueError(f"Element at index {idx} out of byte range 0..255: {val}")
            out.append(val)
        return bytes(out)
    raise TypeError(f"unsupported observation type: {type(raw).__name__}")


def canonicalize_symbols(raw_bytes: bytes) -> Tuple[Tuple[int, ...], int]:
    """
    Map raw symbol identities to canonical IDs ordered by first appearance (0, 1, 2, ...).
    This guarantees that identical structures with different raw vocabularies
    (e.g., [65, 65, 66] vs [9, 9, 7]) produce the exact same canonical token sequence.
    """
    symbol_map = {}
    canonical = []
    next_id = 0

    for b in raw_bytes:
        if b not in symbol_map:
            symbol_map[b] = next_id
            next_id += 1
        canonical.append(symbol_map[b])

    return tuple(canonical), next_id


def extract_runs(canonical_tokens: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    """
    Extract run-length structure as a tuple of (canonical_symbol, count).
    """
    if not canonical_tokens:
        return ()

    runs = []
    current_sym = canonical_tokens[0]
    current_count = 1

    for sym in canonical_tokens[1:]:
        if sym == current_sym:
            current_count += 1
        else:
            runs.append((current_sym, current_count))
            current_sym = sym
            current_count = 1
    runs.append((current_sym, current_count))

    return tuple(runs)


def extract_transitions(canonical_tokens: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    """
    Extract directed transitions (bigrams) between adjacent canonical symbols.
    """
    if len(canonical_tokens) < 2:
        return ()

    return tuple((canonical_tokens[i], canonical_tokens[i + 1]) for i in range(len(canonical_tokens) - 1))


def extract_motifs(canonical_tokens: Tuple[int, ...], min_len: int = 2, max_len: int = 4) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    """
    Discover repeated subsequences (motifs) occurring at least 2 times.
    Sorted deterministically by frequency descending, length descending, then motif content.
    """
    n = len(canonical_tokens)
    if n < min_len:
        return ()

    counts = {}
    for mlen in range(min_len, min(max_len + 1, n + 1)):
        for i in range(n - mlen + 1):
            sub = canonical_tokens[i : i + mlen]
            counts[sub] = counts.get(sub, 0) + 1

    repeated = [(motif, count) for motif, count in counts.items() if count >= 2]
    repeated.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))

    return tuple(repeated)


def compute_representation_checksum(
    raw_length: int,
    vocab_size: int,
    canonical_tokens: Tuple[int, ...],
    runs: Tuple[Tuple[int, int], ...],
    transitions: Tuple[Tuple[int, int], ...],
    motifs: Tuple[Tuple[Tuple[int, ...], int], ...],
) -> str:
    """
    Compute a deterministic SHA256 checksum over all discovered structural dimensions.
    """
    hasher = hashlib.sha256()
    hasher.update(f"L:{raw_length}|V:{vocab_size}|".encode("ascii"))
    hasher.update(f"TOK:{','.join(map(str, canonical_tokens))}|".encode("ascii"))
    hasher.update(f"RUNS:{';'.join(f'{s}:{c}' for s, c in runs)}|".encode("ascii"))
    hasher.update(f"TRANS:{';'.join(f'{a}->{b}' for a, b in transitions)}|".encode("ascii"))
    hasher.update(f"MOTIFS:{';'.join(f'{list(m)}:{c}' for m, c in motifs)}|".encode("ascii"))
    return hasher.hexdigest()


def discover_representation(raw_observation: Any) -> DiscoveredRepresentation:
    """
    Discover the structural representation from an undifferentiated raw observation.
    """
    raw_bytes = _validate_raw_input(raw_observation)
    raw_length = len(raw_bytes)

    canonical_tokens, vocab_size = canonicalize_symbols(raw_bytes)
    runs = extract_runs(canonical_tokens)
    transitions = extract_transitions(canonical_tokens)
    motifs = extract_motifs(canonical_tokens)
    checksum = compute_representation_checksum(
        raw_length=raw_length,
        vocab_size=vocab_size,
        canonical_tokens=canonical_tokens,
        runs=runs,
        transitions=transitions,
        motifs=motifs,
    )

    return DiscoveredRepresentation(
        raw_length=raw_length,
        vocab_size=vocab_size,
        canonical_tokens=canonical_tokens,
        runs=runs,
        transitions=transitions,
        motifs=motifs,
        checksum=checksum,
    )


def are_structurally_equivalent(rep1: DiscoveredRepresentation, rep2: DiscoveredRepresentation) -> bool:
    """
    Test whether two discovered representations are structurally identical.
    """
    if not isinstance(rep1, DiscoveredRepresentation) or not isinstance(rep2, DiscoveredRepresentation):
        raise TypeError("Both arguments must be DiscoveredRepresentation instances")
    return rep1.checksum == rep2.checksum
