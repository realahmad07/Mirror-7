from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class DialogueState:
    goal: str | None
    slots: dict[str, Any]
    unresolved: tuple[str, ...]

class GroundedDialogue:
    """Keeps only explicit context and asks targeted clarification for missing slots."""
    def __init__(self, required_slots=()):
        self.required_slots=tuple(required_slots)
        self.slots={}
        self.goal=None
    def ingest(self, text: str, *, goal: str|None=None, slots: dict[str,Any]|None=None):
        if goal is not None and goal.strip(): self.goal=goal.strip()
        if slots:
            for k,v in slots.items():
                if v not in (None, ""): self.slots[k]=v
        return self.state()
    def state(self):
        unresolved=tuple(k for k in self.required_slots if k not in self.slots)
        return DialogueState(self.goal,dict(self.slots),unresolved)
    def clarification(self):
        s=self.state()
        if not s.unresolved and s.goal: return None
        if not s.goal: return "What outcome are you trying to achieve?"
        return "Can you provide more context about " + ", ".join(s.unresolved) + " so I can guide you accurately?"
