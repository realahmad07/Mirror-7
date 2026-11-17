"""Result type returned by every native intelligence organ."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NativeAnswer:
    """A routed answer plus an honest description of how it was obtained.

    ``qualification`` values:

    * ``conversational``     - social reply; not a factual claim.
    * ``session_memory``     - repeats what the user said earlier in this session.
    * ``memory_stored``      - acknowledges a fact the user just stated.
    * ``verified_computation`` - exact result confirmed by two independent evaluators.
    * ``computed_result``    - exact deterministic computation (single evaluator).
    * ``computed_solution``  - word problem solved by a schema solver with shown steps.
    * ``deductive_proof``    - follows logically from premises stated in the prompt.
    * ``curated_knowledge``  - answered from a reviewed, source-tagged knowledge pack.
    """

    kind: str
    answer: str
    qualification: str
    confidence: float
    verified: bool = False
    steps: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "answer": self.answer,
            "qualification": self.qualification,
            "confidence": self.confidence,
            "verified": self.verified,
            "steps": list(self.steps),
            "metadata": dict(self.metadata),
        }
