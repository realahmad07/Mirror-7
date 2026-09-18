"""Phase 31: deterministic raw-observation representation discovery.

The module deliberately accepts only bytes. It does not assume a schema, field names,
entity labels, or numeric object structure. It extracts repeated local structure and
returns a canonical representation suitable for later state learning.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    value: bytes


@dataclass(frozen=True)
class Representation:
    length: int
    runs: tuple[tuple[int, int, int], ...]
    transitions: tuple[tuple[int, int], ...]
    motifs: tuple[tuple[bytes, tuple[int, ...]], ...]
    checksum: str

    def canonical_bytes(self) -> bytes:
        out = bytearray()
        out.extend(self.length.to_bytes(4, "big"))
        for v, s, e in self.runs:
            out.extend(bytes((v,)))
            out.extend(s.to_bytes(4, "big"))
            out.extend(e.to_bytes(4, "big"))
        out.append(0xff)
        for a, b in self.transitions:
            out.extend(bytes((a, b)))
        out.append(0xfe)
        for motif, positions in self.motifs:
            out.extend(len(motif).to_bytes(2, "big"))
            out.extend(motif)
            out.extend(len(positions).to_bytes(2, "big"))
            for p in positions:
                out.extend(p.to_bytes(4, "big"))
        return bytes(out)


def _canonicalize(obs: bytes) -> bytes:
    mapping = {}
    next_id = 0
    out = bytearray()
    for x in obs:
        if x not in mapping:
            mapping[x] = next_id
            next_id += 1
        out.append(mapping[x])
    return bytes(out)


def _runs(obs: bytes) -> tuple[tuple[int, int, int], ...]:
    if not obs:
        return ()
    result = []
    start = 0
    value = obs[0]
    for i, x in enumerate(obs[1:], 1):
        if x != value:
            result.append((value, start, i))
            start, value = i, x
    result.append((value, start, len(obs)))
    return tuple(result)


def _transitions(obs: bytes) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(set(zip(obs, obs[1:]))))


def _motifs(obs: bytes, widths: Iterable[int] = (2, 3, 4)) -> tuple[tuple[bytes, tuple[int, ...]], ...]:
    found = []
    for width in widths:
        if len(obs) < width:
            continue
        buckets: dict[bytes, list[int]] = {}
        for i in range(len(obs) - width + 1):
            motif = obs[i:i + width]
            buckets.setdefault(motif, []).append(i)
        for motif, positions in buckets.items():
            if len(positions) >= 2:
                found.append((motif, tuple(positions)))
    return tuple(sorted(found, key=lambda x: (len(x[0]), x[0], x[1])))


def discover_representation(observation: bytes | bytearray | memoryview) -> Representation:
    if not isinstance(observation, (bytes, bytearray, memoryview)):
        raise TypeError("observation must be bytes-like")
    obs = _canonicalize(bytes(observation))
    runs = _runs(obs)
    transitions = _transitions(obs)
    motifs = _motifs(obs)
    provisional = Representation(len(obs), runs, transitions, motifs, "")
    checksum = sha256(provisional.canonical_bytes()).hexdigest()
    return Representation(len(obs), runs, transitions, motifs, checksum)


def similarity(a: Representation, b: Representation) -> float:
    """Structural similarity independent of absolute byte vocabulary."""
    if a.length == 0 or b.length == 0:
        return 1.0 if a.length == b.length else 0.0
    ar = [(e - s) for _, s, e in a.runs]
    br = [(e - s) for _, s, e in b.runs]
    run_score = _sequence_score(ar, br)
    at = set((x, y) for x, y in a.transitions)
    bt = set((x, y) for x, y in b.transitions)
    trans_score = _set_score(at, bt)
    am = {len(m): len(pos) for m, pos in a.motifs}
    bm = {len(m): len(pos) for m, pos in b.motifs}
    motif_score = _set_score(set(am.items()), set(bm.items()))
    return (run_score + trans_score + motif_score) / 3.0


def _set_score(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _sequence_score(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    equal = sum(x == y for x, y in zip(a[:n], b[:n]))
    length_score = 1.0 - abs(len(a) - len(b)) / max(len(a), len(b))
    return 0.5 * (equal / n) + 0.5 * length_score
