
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class Expr:
    kind: str
    value: object

class CompositionalSemantics:
    """Tiny inspectable grammar for composing learned grounded primitives."""

    def __init__(self):
        self.primitives: Dict[str, str] = {}
        self.last_reference: Optional[str] = None

    def learn_primitive(self, phrase: str, concept: str) -> None:
        words = tuple(w for w in phrase.lower().split() if w not in {"please","the","to","a"})
        if not words or not concept:
            raise ValueError("empty primitive")
        for w in words:
            self.primitives[w] = concept
        self.last_reference = concept

    def _primitive(self, token: str) -> Optional[str]:
        return self.primitives.get(token.lower())

    def parse(self, instruction: str) -> Expr:
        words = [w.strip(".,!?").lower() for w in instruction.split() if w.strip()]
        if not words:
            raise ValueError("empty instruction")
        if "then" in words:
            k = words.index("then")
            return Expr("sequence", (self.parse(" ".join(words[:k])),
                                     self.parse(" ".join(words[k+1:]))))
        if "and" in words:
            k = words.index("and")
            return Expr("parallel", (self.parse(" ".join(words[:k])),
                                     self.parse(" ".join(words[k+1:]))))
        if words[0] in {"this","that","it"} and self.last_reference:
            return Expr("reference", self.last_reference)
        concepts = [self._primitive(w) for w in words]
        concepts = [c for c in concepts if c is not None]
        unique = list(dict.fromkeys(concepts))
        if len(unique) != 1:
            raise ValueError("ambiguous or ungrounded instruction")
        self.last_reference = unique[0]
        return Expr("primitive", unique[0])

    def execute_plan(self, expr: Expr) -> Tuple[str, ...]:
        if expr.kind == "primitive":
            return (expr.value,)
        if expr.kind == "reference":
            return (expr.value,)
        if expr.kind in {"sequence","parallel"}:
            left, right = expr.value
            return self.execute_plan(left) + self.execute_plan(right)
        raise ValueError("unknown expression")
