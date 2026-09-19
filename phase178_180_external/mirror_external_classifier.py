from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence, Tuple, Hashable

Vector = Tuple[float, ...]

@dataclass(frozen=True)
class Example:
    features: Vector
    label: Hashable

class ExplicitPrototypeMemory:
    """Small non-neural external-task adapter using explicit state memory.

    It stores class prototypes and compares normalized feature states with a
    bounded k-nearest-neighbour vote. No learned weights or matrix products
    are used. This adapter is intentionally separate from Mirror 7's core
    runtime so external-task results cannot silently redefine core claims.
    """
    def __init__(self, k: int = 5) -> None:
        if k < 1 or k % 2 == 0:
            raise ValueError("k must be a positive odd integer")
        self.k = k
        self._train: list[Example] = []
        self._mins: Vector = ()
        self._spans: Vector = ()

    def fit(self, examples: Iterable[Example]) -> None:
        rows = list(examples)
        if not rows:
            raise ValueError("training set must not be empty")
        width = len(rows[0].features)
        if width == 0 or any(len(x.features) != width for x in rows):
            raise ValueError("inconsistent feature width")
        mins = tuple(min(x.features[j] for x in rows) for j in range(width))
        maxs = tuple(max(x.features[j] for x in rows) for j in range(width))
        spans = tuple((maxs[j] - mins[j]) or 1.0 for j in range(width))
        self._mins, self._spans = mins, spans
        self._train = [Example(self._norm(x.features), x.label) for x in rows]

    def _norm(self, features: Sequence[float]) -> Vector:
        if not self._mins or len(features) != len(self._mins):
            raise ValueError("classifier is not fitted or feature width differs")
        return tuple((float(v) - self._mins[j]) / self._spans[j] for j, v in enumerate(features))

    @staticmethod
    def _distance(a: Vector, b: Vector) -> float:
        return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict(self, features: Sequence[float]) -> Hashable:
        if not self._train:
            raise ValueError("classifier is not fitted")
        query = self._norm(features)
        ranked = sorted((self._distance(query, row.features), row.label) for row in self._train)
        votes: dict[Hashable, tuple[int, float]] = {}
        for distance, label in ranked[: min(self.k, len(ranked))]:
            count, total = votes.get(label, (0, 0.0))
            votes[label] = (count + 1, total + distance)
        return min(votes, key=lambda label: (-votes[label][0], votes[label][1], str(label)))

    def score(self, examples: Iterable[Example]) -> float:
        rows = list(examples)
        if not rows:
            return 0.0
        return sum(self.predict(x.features) == x.label for x in rows) / len(rows)
