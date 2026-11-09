"""Pure prompt-shaping helpers shared by Mirror Crystal runtimes."""
from __future__ import annotations
import re

_COMPOUND_SPLIT_RE = re.compile(
    r"\s+(?:and\s+also|also\s+please|and\s+separately)\s+|\s*;\s+|(?<=\?)\s+(?=[A-Z])",
    re.I,
)

def split_compound_intents(prompt: str) -> list[str]:
    """Split short, clearly delimited compound prompts into at most four tasks."""
    text = prompt.strip()
    if len(text) > 420 or "\n" in text:
        return [text]
    parts = [part.strip() for part in _COMPOUND_SPLIT_RE.split(text) if len(part.strip()) >= 12]
    return parts[:4] if len(parts) >= 2 else [text]
