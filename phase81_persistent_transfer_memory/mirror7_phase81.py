"""Phase 81: bounded persistent memory with provenance and rejection."""
from dataclasses import dataclass, asdict
import json, math
from pathlib import Path

@dataclass(frozen=True)
class MemoryRecord:
    environment: str
    signature: tuple[float,...]
    state: tuple[float,...]
    action: str
    result: tuple[float,...]
    confidence: float

class PersistentTransferMemory:
    def __init__(self, path: str | Path, max_records: int = 512, tolerance: float = 1e-6):
        if max_records < 1 or tolerance < 0: raise ValueError("invalid bounds")
        self.path=Path(path); self.max_records=max_records; self.tolerance=tolerance; self.records=[]
        if self.path.exists(): self.load()

    @staticmethod
    def _vec(x):
        y=tuple(float(v) for v in x)
        if not y or not all(math.isfinite(v) for v in y): raise ValueError("invalid vector")
        return y

    def add(self, environment, signature, state, action, result, confidence):
        c=float(confidence)
        if not 0<=c<=1: raise ValueError("confidence out of range")
        rec=MemoryRecord(str(environment),self._vec(signature),self._vec(state),str(action),self._vec(result),c)
        self.records.append(rec); self.records=self.records[-self.max_records:]

    def retrieve(self, environment, signature, min_confidence=0.0):
        sig=self._vec(signature); out=[]
        for r in self.records:
            if r.environment!=str(environment) or r.confidence<float(min_confidence) or len(r.signature)!=len(sig): continue
            distance=max(abs(a-b) for a,b in zip(r.signature,sig))
            if distance<=self.tolerance: out.append(r)
        return tuple(out)

    def transfer(self, source_environment, target_environment, signature, min_confidence=0.5):
        candidates=self.retrieve(source_environment,signature,min_confidence)
        if not candidates: return None
        return max(candidates,key=lambda r:r.confidence)

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        payload=[asdict(r) for r in self.records]
        self.path.write_text(json.dumps(payload,separators=(",",":"),sort_keys=True),encoding="utf-8")

    def load(self):
        raw=json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw,list): raise ValueError("invalid memory file")
        self.records=[]
        for x in raw:
            self.add(x["environment"],x["signature"],x["state"],x["action"],x["result"],x["confidence"])
