"""Indexed prompt-pattern matching for canonical blueprints."""
from __future__ import annotations

import re
from typing import Any


_SLOT_RE = re.compile(r"\{(v\d+)\}")


def normalize_blueprint_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        "".join(ch.casefold() if ch.isalnum() or ch.isspace() else " " for ch in str(value)),
    ).strip()


def blueprint_prompt_fields(pattern: str) -> tuple[str, str]:
    key = normalize_blueprint_text(pattern)
    literal_prefix = _SLOT_RE.split(str(pattern), maxsplit=1)[0]
    prefix_tokens = normalize_blueprint_text(literal_prefix).split()
    return key, " ".join(prefix_tokens[:2])


def blueprint_query_prefix(prompt: str) -> str:
    return " ".join(normalize_blueprint_text(prompt).split()[:2])


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
