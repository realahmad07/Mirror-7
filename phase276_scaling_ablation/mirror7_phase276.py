
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

@dataclass(frozen=True)
class ScalingPoint:
    size: int
    operations: int

class ScalingStudy:
    """Deterministic operation-count scaling rather than timing-sensitive benchmarks."""

    def run(self, sizes: Sequence[int], work: Callable[[int],int]) -> List[ScalingPoint]:
        if any(int(n)<1 for n in sizes):
            raise ValueError("sizes must be positive")
        return [ScalingPoint(int(n),int(work(int(n)))) for n in sizes]

class AblationHarness:
    def compare(self, task:Any, full:Callable[[Any],bool], ablated:Callable[[Any],bool])->Dict[str,bool]:
        return {"full":bool(full(task)),"ablated":bool(ablated(task))}

class ReproducibilityManifest:
    def digest(self, value:Any)->str:
        payload=json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()
        return hashlib.sha256(payload).hexdigest()
