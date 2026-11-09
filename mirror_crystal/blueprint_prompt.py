"""Indexed prompt-pattern matching for canonical blueprints."""
from __future__ import annotations

import re
from typing import Any


_SLOT_RE = re.compile(r"\{(v\d+)\}")
_PREFIX_STOP_WORDS = frozenset(
    "a an the this that these those use using used when whenever if while do does did "
    "can could should would please tell me how what which who is are was were be been being "
    "to of for from with in on at by as and or but problem task procedure rule method step "
    "steps way requires require need needs".split()
)


def normalize_blueprint_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        "".join(ch.casefold() if ch.isalnum() or ch.isspace() else " " for ch in str(value)),
    ).strip()


def _informative_prefix_tokens(value: str) -> list[str]:
    tokens = normalize_blueprint_text(value).split()
    return [token for token in tokens if token not in _PREFIX_STOP_WORDS]


def blueprint_prompt_fields(pattern: str) -> tuple[str, str]:
    key = normalize_blueprint_text(pattern)
    literal_prefix = _SLOT_RE.split(str(pattern), maxsplit=1)[0]
    raw_tokens = normalize_blueprint_text(literal_prefix).split()
    informative = _informative_prefix_tokens(literal_prefix)
    # Two informative words keep large families of generic "Use this..." prompts
    # from collapsing into one enormous prefix bucket. Patterns with an early slot
    # fall back to the strongest fixed word available before that slot.
    prefix = " ".join(informative[:2] or raw_tokens[:1])
    return key, prefix


def blueprint_query_prefixes(prompt: str) -> tuple[str, ...]:
    raw = normalize_blueprint_text(prompt).split()
    informative = _informative_prefix_tokens(prompt)
    candidates = []
    if len(informative) >= 2:
        candidates.append(" ".join(informative[:2]))
    if informative:
        candidates.append(informative[0])
    if raw:
        candidates.append(raw[0])
    return tuple(dict.fromkeys(item for item in candidates if item))


def blueprint_query_prefix(prompt: str) -> str:
    prefixes = blueprint_query_prefixes(prompt)
    return prefixes[0] if prefixes else ""


def match_blueprint_prompt(pattern: str, prompt: str) -> dict[str, Any] | None:
    """Match literal text plus named {vN} slots, returning the captured slot values."""
    pattern = str(pattern)
    prompt = str(prompt)
    slots = list(_SLOT_RE.finditer(pattern))
    if not slots:
        return {} if normalize_blueprint_text(pattern) == normalize_blueprint_text(prompt) else None

    pieces: list[str] = ["^"]
    cursor = 0
    seen: set[str] = set()
    for match in slots:
        literal = pattern[cursor:match.start()]
        literal_parts = re.split(r"(\s+)", literal)
        pieces.extend(r"\s+" if part.isspace() else re.escape(part) for part in literal_parts if part)
        name = match.group(1)
        if name in seen:
            return None
        seen.add(name)
        pieces.append(f"(?P<{name}>.+?)")
        cursor = match.end()
    tail = pattern[cursor:]
    tail_parts = re.split(r"(\s+)", tail)
    pieces.extend(r"\s+" if part.isspace() else re.escape(part) for part in tail_parts if part)
    pieces.append("$")
    found = re.fullmatch("".join(pieces), prompt.strip(), flags=re.IGNORECASE)
    if found is None:
        return None
    return {key: value.strip() for key, value in found.groupdict().items()}
