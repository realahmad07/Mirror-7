
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    value: bytes

@dataclass(frozen=True)
class Relation:
    left: int
    right: int
    gap: int

@dataclass(frozen=True)
class Representation:
    segments: Tuple[Segment, ...]
    relations: Tuple[Relation, ...]
    motifs: Tuple[bytes, ...]

class GeneralRepresentation:
    """Bounded raw-stream representation with noise-tolerant repeated motifs."""

    def __init__(self, max_motif: int = 8, min_support: int = 2):
        if max_motif < 1 or min_support < 1:
            raise ValueError("invalid representation bounds")
        self.max_motif = max_motif
        self.min_support = min_support
        self.history: List[bytes] = []

    def _motifs(self, samples: Iterable[bytes]) -> Tuple[bytes, ...]:
        counts = Counter()
        for raw in samples:
            limit = min(len(raw), self.max_motif)
            seen = set()
            for n in range(1, limit + 1):
                for i in range(len(raw) - n + 1):
                    m = raw[i:i+n]
                    if m not in seen:
                        counts[m] += 1
                        seen.add(m)
        return tuple(sorted((m for m,c in counts.items() if c >= self.min_support),
                            key=lambda x: (-len(x), x)))

    def _segments(self, raw: bytes, motifs: Sequence[bytes]) -> Tuple[Segment, ...]:
        separators = {0, 10, 32, 124}
        out: List[Segment] = []
        i = 0
        while i < len(raw):
            if raw[i] in separators:
                i += 1
                continue
            best = None
            for motif in motifs:
                if raw.startswith(motif, i):
                    best = motif
                    break
            if best is None:
                start = i
                i += 1
                while i < len(raw) and raw[i] not in separators:
                    i += 1
                out.append(Segment(start, i, raw[start:i]))
            else:
                out.append(Segment(i, i + len(best), best))
                i += len(best)
        return tuple(out)

    def discover(self, raw: bytes) -> Representation:
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("raw must be bytes")
        raw = bytes(raw)
        self.history.append(raw)
        motifs = self._motifs(self.history)
        segments = self._segments(raw, motifs)
        relations = tuple(
            Relation(i, i + 1, segments[i + 1].start - segments[i].end)
            for i in range(len(segments) - 1)
        )
        return Representation(segments, relations, motifs)

    def discover_batch(self, samples: Sequence[bytes]) -> List[Representation]:
        for raw in samples:
            if not isinstance(raw, (bytes, bytearray)):
                raise TypeError("all samples must be bytes")
            self.history.append(bytes(raw))
        motifs = self._motifs(self.history)
        return [
            Representation(
                self._segments(bytes(raw), motifs),
                tuple(
                    Relation(i, i + 1, segs[i + 1].start - segs[i].end)
                    for i in range(len(segs) - 1)
                ),
                motifs,
            )
            for raw in samples
            for segs in [self._segments(bytes(raw), motifs)]
        ]
