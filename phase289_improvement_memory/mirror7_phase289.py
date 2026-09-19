from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, List

@dataclass(frozen=True)
class MemoryRecord:
    fingerprint:str
    accepted:bool
    score:float
    reason:str

class ImprovementMemory:
    """Remembers evaluated algorithm fingerprints and avoids identical repeats."""
    def __init__(self): self._records:List[MemoryRecord]=[]

    @staticmethod
    def fingerprint(operations:Iterable[str])->str:
        return sha256("|".join(operations).encode()).hexdigest()

    def seen(self,operations:Iterable[str])->bool:
        fp=self.fingerprint(operations)
        return any(r.fingerprint==fp for r in self._records)

    def add(self,operations:Iterable[str],accepted:bool,score:float,reason:str):
        fp=self.fingerprint(operations)
        if not any(r.fingerprint==fp for r in self._records):
            self._records.append(MemoryRecord(fp,bool(accepted),float(score),reason))

    @property
    def records(self)->tuple[MemoryRecord,...]: return tuple(self._records)

    def successful(self)->tuple[MemoryRecord,...]:
        return tuple(r for r in self._records if r.accepted)
