"""Phase 31.2: extract a stable state from a raw structural representation."""
from __future__ import annotations
from dataclasses import dataclass
from .mirror7_representation import Representation


@dataclass(frozen=True)
class State:
    length: int
    run_lengths: tuple[int, ...]
    transition_count: int
    motif_count: int
    repeated_motif_count: int
    density: float
    checksum: str

    def canonical(self) -> tuple:
        return (
            self.length, self.run_lengths, self.transition_count,
            self.motif_count, self.repeated_motif_count,
            round(self.density, 12),
        )


def extract_state(rep: Representation) -> State:
    if not isinstance(rep, Representation):
        raise TypeError("rep must be a Representation")
    runs = tuple(end - start for _, start, end in rep.runs)
    motif_count = len(rep.motifs)
    repeated_motif_count = sum(1 for _, positions in rep.motifs if len(positions) >= 2)
    density = len(rep.transitions) / max(1, rep.length - 1)
    return State(
        length=rep.length,
        run_lengths=runs,
        transition_count=len(rep.transitions),
        motif_count=motif_count,
        repeated_motif_count=repeated_motif_count,
        density=density,
        checksum=rep.checksum,
    )
