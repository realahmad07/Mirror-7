from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

@dataclass(frozen=True)
class AlgorithmVariant:
    name: str
    operations: tuple[str, ...]

    def describe(self) -> str:
        return " -> ".join(self.operations)

class AlgorithmVariantSpace:
    """A finite declarative space of algorithm variants; no arbitrary code execution."""
    ALLOWED = frozenset({"last", "delta", "add_delta", "subtract_delta", "constant"})

    def __init__(self, operations: Sequence[str] | None = None):
        self.operations = tuple(operations or ("last", "delta", "add_delta", "subtract_delta", "constant"))
        unknown = set(self.operations) - self.ALLOWED
        if unknown:
            raise ValueError(f"unsupported operations: {sorted(unknown)}")

    def validate(self, variant: AlgorithmVariant) -> bool:
        return bool(variant.name) and bool(variant.operations) and set(variant.operations) <= self.ALLOWED

    def execute(self, variant: AlgorithmVariant, values: Sequence[float]) -> float:
        if not self.validate(variant):
            raise ValueError("invalid algorithm variant")
        if not values:
            raise ValueError("values must be non-empty")
        current = float(values[-1])
        delta = float(values[-1] - values[-2]) if len(values) >= 2 else 0.0
        for op in variant.operations:
            if op == "last":
                current = float(values[-1])
            elif op == "delta":
                current = delta
            elif op == "add_delta":
                current = current + delta
            elif op == "subtract_delta":
                current = current - delta
            elif op == "constant":
                current = 0.0
        return current

    def enumerate(self, max_depth: int = 3) -> List[AlgorithmVariant]:
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        results: List[AlgorithmVariant] = []
        frontier = [()] 
        for depth in range(1, max_depth + 1):
            next_frontier = []
            for prefix in frontier:
                for op in self.operations:
                    seq = prefix + (op,)
                    if seq not in [v.operations for v in results]:
                        results.append(AlgorithmVariant(f"v{len(results)+1}", seq))
                        next_frontier.append(seq)
            frontier = next_frontier
        return results
