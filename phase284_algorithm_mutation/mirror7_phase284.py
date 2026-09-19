from dataclasses import dataclass
from typing import List, Sequence

from phase283_algorithm_variant_space import AlgorithmVariant, AlgorithmVariantSpace

@dataclass(frozen=True)
class Mutation:
    parent: AlgorithmVariant
    child: AlgorithmVariant
    operation: str
    index: int

class AlgorithmMutator:
    """Generates small, finite algorithm edits without arbitrary source rewriting."""
    def __init__(self, space: AlgorithmVariantSpace):
        self.space = space

    def mutate(self, parent: AlgorithmVariant, *, max_children: int = 8) -> List[Mutation]:
        if max_children < 1:
            raise ValueError("max_children must be positive")
        if not self.space.validate(parent):
            raise ValueError("invalid parent")
        out: List[Mutation] = []
        for i, old in enumerate(parent.operations):
            for op in self.space.operations:
                if op == old:
                    continue
                ops = parent.operations[:i] + (op,) + parent.operations[i + 1:]
                child = AlgorithmVariant(f"{parent.name}.m{len(out)+1}", ops)
                out.append(Mutation(parent, child, f"replace[{i}] {old}->{op}", i))
                if len(out) >= max_children:
                    return out
        return out

    def mutate_by_append(self, parent: AlgorithmVariant, *, max_children: int = 8) -> List[Mutation]:
        if max_children < 1:
            raise ValueError("max_children must be positive")
        if not self.space.validate(parent):
            raise ValueError("invalid parent")
        out: List[Mutation] = []
        for op in self.space.operations:
            ops = parent.operations + (op,)
            if len(ops) > 5:
                continue
            child = AlgorithmVariant(f"{parent.name}.a{len(out)+1}", ops)
            out.append(Mutation(parent, child, f"append {op}", len(parent.operations)))
            if len(out) >= max_children:
                return out
        return out
