from dataclasses import dataclass
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class Inquiry:
    question: str
    missing: tuple[str, ...]
    reason: str
    assumptions: tuple[str, ...]

class ContextInquiry:
    """Turns insufficient/conflicting context into a useful clarification request."""
    def __init__(self, required: Sequence[str] | None = None):
        self.required = tuple(required or ())

    def assess(self, context: Mapping[str, Any], required: Sequence[str] | None = None, conflicts: Sequence[str] = ()) -> Inquiry | None:
        req = tuple(required or self.required)
        missing = tuple(k for k in req if k not in context or context[k] in (None, ""))
        conflicts = tuple(conflicts)
        if not missing and not conflicts:
            return None
        reasons=[]
        if missing: reasons.append("missing context")
        if conflicts: reasons.append("conflicting information")
        parts = []
        if missing: parts.append("missing: " + ", ".join(missing))
        if conflicts: parts.append("conflicts: " + ", ".join(conflicts))
        return Inquiry(
            question="Can you provide more context about " + "; ".join(parts) + " so I can guide you accurately?",
            missing=missing,
            reason=" and ".join(reasons),
            assumptions=tuple(f"{k}=unknown" for k in missing),
        )
