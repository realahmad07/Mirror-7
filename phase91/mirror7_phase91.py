from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Mapping

@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: int
    signature: tuple[tuple[str, float], ...]
    support: int
    confidence: float

class HypothesisRevision:
    """Bounded create/merge/revise/abstain mechanism for unlabeled hypotheses."""
    def __init__(self, max_hypotheses: int = 32, merge_threshold: float = .15, revise_patience: int = 2):
        if max_hypotheses < 1 or merge_threshold < 0 or revise_patience < 1:
            raise ValueError("invalid bounds")
        self.max_hypotheses = max_hypotheses
        self.merge_threshold = float(merge_threshold)
        self.revise_patience = revise_patience
        self.hypotheses: list[Hypothesis] = []
        self._mismatch: dict[int, int] = {}

    @staticmethod
    def normalize(signature: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        if not signature:
            raise ValueError("empty signature")
        out = []
        for k, v in signature.items():
            x = float(v)
            if not k or not isfinite(x):
                raise ValueError("invalid signature")
            out.append((str(k), round(x, 8)))
        return tuple(sorted(out))

    @staticmethod
    def distance(a, b) -> float:
        da, db = dict(a), dict(b)
        keys = set(da) | set(db)
        if not keys:
            return 0.0
        return sum(abs(da.get(k, 0.0) - db.get(k, 0.0)) / max(1.0, abs(da.get(k, 0.0)), abs(db.get(k, 0.0))) for k in keys) / len(keys)

    def observe(self, signature: Mapping[str, float]) -> Hypothesis:
        sig = self.normalize(signature)
        if not self.hypotheses:
            h = Hypothesis(0, sig, 1, 1.0)
            self.hypotheses.append(h)
            return h
        ranked = sorted((self.distance(sig, h.signature), h) for h in self.hypotheses)
        d, h = ranked[0]
        if d <= self.merge_threshold:
            updated = Hypothesis(h.hypothesis_id, h.signature, h.support + 1, min(1.0, h.confidence + 0.05))
            self.hypotheses[h.hypothesis_id] = updated
            return updated
        count = self._mismatch.get(h.hypothesis_id, 0) + 1
        self._mismatch[h.hypothesis_id] = count
        if count >= self.revise_patience:
            if len(self.hypotheses) >= self.max_hypotheses:
                return h
            new = Hypothesis(len(self.hypotheses), sig, 1, 0.5)
            self.hypotheses.append(new)
            self._mismatch[h.hypothesis_id] = 0
            return new
        return h

    def infer(self, signature: Mapping[str, float], ambiguity_margin: float = .05) -> Hypothesis | None:
        sig = self.normalize(signature)
        if not self.hypotheses:
            return None
        ranked = sorted((self.distance(sig, h.signature), h) for h in self.hypotheses)
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < ambiguity_margin:
            return None
        return ranked[0][1]

    def contradiction(self, a: Mapping[str, float], b: Mapping[str, float], threshold: float = .5) -> bool:
        return self.distance(self.normalize(a), self.normalize(b)) >= threshold

    def fail_closed(self) -> dict[str, int]:
        return {"hypotheses": len(self.hypotheses)}
