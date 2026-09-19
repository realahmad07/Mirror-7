from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from math import isfinite
from typing import Sequence

@dataclass(frozen=True)
class TransferResult:
    accepted: bool
    mapping: tuple[int, ...]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    confidence: float
    reason: str

class StructuralTransfer:
    """Small-state optimal permutation matching with global affine normalization."""
    def __init__(self, max_dim: int = 7, min_confidence: float = .9):
        if max_dim < 1 or not 0 < min_confidence <= 1:
            raise ValueError("invalid bounds")
        self.max_dim = max_dim
        self.min_confidence = float(min_confidence)

    @staticmethod
    def _norm(v):
        x = tuple(float(a) for a in v)
        if not x or not all(isfinite(a) for a in x):
            raise ValueError("invalid vector")
        return x

    def match(self, source: Sequence[float], target: Sequence[float]) -> TransferResult:
        s = self._norm(source)
        t = self._norm(target)
        if len(s) != len(t):
            return TransferResult(False, (), (), (), 0.0, "dimension mismatch")
        if len(s) > self.max_dim:
            return TransferResult(False, (), (), (), 0.0, "dimension budget")
        n = len(s)
        mean_s = sum(s) / n
        var_s = sum((a - mean_s) ** 2 for a in s)
        best = None
        for p in permutations(range(n)):
            ts = [t[i] for i in p]
            mean_t = sum(ts) / n
            cov = sum((a - mean_s) * (b - mean_t) for a, b in zip(s, ts))
            scale = 1.0 if var_s < 1e-12 else cov / var_s
            if scale < 0:
                continue
            offset = mean_t - scale * mean_s
            residual = sum(abs(scale * a + offset - b) for a, b in zip(s, ts))
            norm_err = residual / (n * max(1.0, max(abs(v) for v in t)))
            conf = max(0.0, 1.0 - norm_err)
            candidate = (conf, p, scale, offset, residual)
            if best is None or (candidate[0], -candidate[4], tuple(-i for i in candidate[1])) > (best[0], -best[4], tuple(-i for i in best[1])):
                best = candidate
        if best is None:
            return TransferResult(False, (), (), (), 0.0, "no orientation-preserving match")
        conf, p, scale, offset, _ = best
        if conf < self.min_confidence:
            return TransferResult(False, tuple(p), (scale,), (offset,), conf, "low confidence")
        return TransferResult(True, tuple(p), (scale,), (offset,), conf, "accepted")
