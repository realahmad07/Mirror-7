"""Small, deterministic session constraints shared by Mirror Crystal engines."""
from __future__ import annotations
import re

_LIMIT_DECL_RE = re.compile(
    r"(?:limit|cap|budget|threshold|capacity|max(?:imum)?)\s*(?:is|=|:)\s*(\d+(?:\.\d+)?)", re.I
)
_BURST_CHECK_RE = re.compile(
    r"(\d+)\s*(?:bursts?|batches|groups|workers|packs|times|x)\s*(?:of|\*)\s*(\d+)", re.I
)

class BinarySessionMemory:
    """Retain explicitly declared numeric constraints between turns."""
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, float | str]] = {}

    def update_from_prompt(self, session_id: str, prompt: str) -> dict[str, float | str]:
        memory = self._sessions.setdefault(session_id, {})
        limit_match = _LIMIT_DECL_RE.search(prompt)
        if limit_match:
            memory["constraint_limit"] = float(limit_match.group(1))
        for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]{1,24})\s*=\s*(-?\d+(?:\.\d+)?)\b", prompt):
            memory[match.group(1).lower()] = float(match.group(2))
        return memory

    def evaluate_constraint_check(self, session_id: str, prompt: str) -> tuple[str, list[str]] | None:
        memory = self._sessions.get(session_id, {})
        if "constraint_limit" not in memory:
            return None
        burst_match = _BURST_CHECK_RE.search(prompt)
        if not burst_match:
            return None
        count, size = int(burst_match.group(1)), int(burst_match.group(2))
        total = count * size
        limit_value = float(memory["constraint_limit"])
        limit_display = int(limit_value) if limit_value.is_integer() else limit_value
        trace = [
            f"Recalled session constraint_limit = {limit_display}",
            f"Executed CPU product check: {count} * {size} = {total}",
        ]
        if total <= limit_value:
            margin = limit_display - total
            answer = (
                f"Safe: {count} * {size} = {total}, which is within your {limit_display} limit "
                f"(headroom: {margin})."
            )
        else:
            excess = total - limit_display
            answer = (
                f"Unsafe: {count} * {size} = {total}, which exceeds your {limit_display} limit by {excess}."
            )
            if "limiter" in prompt.lower() or "rate" in prompt.lower():
                answer += (
                    "\n\nclass TokenBucketLimiter:\n"
                    f"  def __init__(self, rate: float = {float(limit_value)}, capacity: float = {float(limit_value)}):\n"
                    "    self.rate, self.cap = float(rate), float(capacity)\n"
                    "    self.tokens, self.last_t = float(capacity), 0.0\n"
                    "  def allow(self, now: float, cost: float = 1.0) -> bool:\n"
                    "    self.tokens = min(self.cap, self.tokens + max(0.0, now - self.last_t) * self.rate)\n"
                    "    self.last_t = now\n"
                    "    if self.tokens >= cost:\n"
                    "      self.tokens -= cost\n"
                    "      return True\n"
                    "    return False"
                )
        return answer, trace
