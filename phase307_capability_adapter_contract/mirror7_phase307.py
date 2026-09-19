from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class AdapterResult:
    score: float
    changed: bool
    variant: str

class CapabilityAdapter(Protocol):
    name: str
    def improve(self, seed: int, rounds: int, candidates: int) -> AdapterResult: ...
