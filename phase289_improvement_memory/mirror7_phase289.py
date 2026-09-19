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
    """Remembers evaluated algorithm fingerprints and allows verified status upgrades."""
    def __init__(self): self._records:List[MemoryRecord]=[]

    @staticmethod
    def fingerprint(operations:Iterable[str])->str:
        return sha256("|".join(operations).encode()).hexdigest()

    def seen(self,operations:Iterable[str])->bool:
        fp=self.fingerprint(operations)
        return any(r.fingerprint==fp for r in self._records)

    def add(self,operations:Iterable[str],accepted:bool,score:float,reason:str):
        fp=self.fingerprint(operations)
        existing=next((i for i,r in enumerate(self._records) if r.fingerprint==fp),None)
        record=MemoryRecord(fp,bool(accepted),float(score),reason)
        if existing is None:
            self._records.append(record)
        elif accepted and not self._records[existing].accepted:
            self._records[existing]=record

    @property
    def records(self)->tuple[MemoryRecord,...]: return tuple(self._records)

    def successful(self)->tuple[MemoryRecord,...]:
        return tuple(r for r in self._records if r.accepted)
