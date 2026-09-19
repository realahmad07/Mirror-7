"""Phase 78: bounded cross-regime/environment transfer with rejection."""
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Mapping, Sequence, Tuple
import math

@dataclass(frozen=True)
class TransferResult:
    accepted: bool
    mapping: Dict[int, int]
    score: float
    reason: str

class CrossRegimeEnvironmentLearner:
    def __init__(self, tolerance: float = 0.05, min_score: float = 0.9):
        if tolerance < 0 or min_score <= 0 or min_score > 1:
            raise ValueError("invalid thresholds")
        self.tolerance, self.min_score = tolerance, min_score
        self.source: Dict[Hashable, Tuple[float, ...]] = {}

    def fit(self, signatures: Mapping[Hashable, Sequence[float]]) -> None:
        self.source.clear()
        for k, v in signatures.items():
            x = tuple(float(z) for z in v)
            if not x or not all(math.isfinite(z) for z in x):
                raise ValueError("invalid signature")
            self.source[k] = x

    @staticmethod
    def _norm(x: Sequence[float]) -> Tuple[float, ...]:
        lo, hi = min(x), max(x)
        if hi - lo == 0:
            return tuple(0.0 for _ in x)
        return tuple((z-lo)/(hi-lo) for z in x)

    def transfer(self, target: Mapping[Hashable, Sequence[float]]) -> TransferResult:
        if not self.source or not target:
            return TransferResult(False, {}, 0.0, "empty model")
        src = {k:self._norm(v) for k,v in self.source.items()}
        dst = {k:self._norm(v) for k,v in target.items()}
        if len(src) != len(dst):
            return TransferResult(False, {}, 0.0, "cardinality mismatch")
        candidates = []
        used = set()
        for sk, sv in src.items():
            best = None
            for dk, dv in dst.items():
                if dk in used or len(sv) != len(dv):
                    continue
                err = sum(abs(a-b) for a,b in zip(sv,dv)) / len(sv)
                score = max(0.0, 1.0-err)
                if best is None or score > best[0]: best=(score,dk)
            if best is None:
                return TransferResult(False, {}, 0.0, "shape mismatch")
            candidates.append((best[0], sk, best[1]))
            used.add(best[1])
        score = sum(x[0] for x in candidates)/len(candidates)
        if score < self.min_score:
            return TransferResult(False, {}, score, "unsupported transfer")
        return TransferResult(True, {sk:dk for _,sk,dk in candidates}, score, "accepted")
