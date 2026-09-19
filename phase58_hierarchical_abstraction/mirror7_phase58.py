"""Phase 58: hierarchical concept abstraction.

Discovers recurring contiguous compositions of already stable concepts and
recursively compresses them into higher-level concepts.  No hierarchy labels
are supplied.  Support is counted across independent episodes and candidates
are selected conservatively to avoid memorizing one-off substrings.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Hashable, Sequence

Atom = Hashable
Token = Hashable

@dataclass(frozen=True)
class ConceptLayer:
    depth: int
    children: tuple[Token, ...]
    support: int
    occurrences: int
    fingerprint: str

@dataclass(frozen=True)
class Hierarchy:
    levels: tuple[tuple[ConceptLayer, ...], ...]
    transformed_episodes: tuple[tuple[Token, ...], ...]

def _fp(children: tuple[Token, ...], depth: int) -> str:
    return sha256(repr((depth, children)).encode("utf-8")).hexdigest()

def _windows(seq: Sequence[Token], lo: int, hi: int):
    for n in range(lo, min(hi, len(seq)) + 1):
        for i in range(len(seq) - n + 1):
            yield i, n, tuple(seq[i:i+n])

def _select(
    episodes: Sequence[Sequence[Token]],
    depth: int,
    *,
    min_len: int,
    max_len: int,
    min_support: int,
) -> tuple[ConceptLayer, ...]:
    support: Counter[tuple[Token, ...]] = Counter()
    occ: Counter[tuple[Token, ...]] = Counter()
    for ep in episodes:
        seen = set()
        for _, _, w in _windows(ep, min_len, max_len):
            occ[w] += 1
            seen.add(w)
        for w in seen:
            support[w] += 1
    candidates = [w for w, s in support.items() if s >= min_support]
    candidates.sort(key=lambda w: (len(w), -support[w], -occ[w], repr(w)))
    selected: list[tuple[Token, ...]] = []
    for w in candidates:
        gain = occ[w] * (len(w) - 1) - len(w)
        if gain <= 0:
            continue
        if any(
            len(w) > len(x)
            and any(w[i:i+len(x)] == x for i in range(len(w)-len(x)+1))
            for x in selected
        ):
            continue
        selected.append(w)
    return tuple(
        ConceptLayer(depth, w, support[w], occ[w], _fp(w, depth))
        for w in selected
    )

def _replace_episode(
    ep: Sequence[Token], layers: Sequence[ConceptLayer]
) -> tuple[Token, ...]:
    by_children = {
        layer.children: ("H", layer.depth, layer.fingerprint)
        for layer in layers
    }
    if not by_children:
        return tuple(ep)
    max_len = max(len(x) for x in by_children)
    out: list[Token] = []
    i = 0
    while i < len(ep):
        match = None
        for n in range(min(max_len, len(ep)-i), 1, -1):
            w = tuple(ep[i:i+n])
            if w in by_children:
                match = w
                break
        if match is None:
            out.append(ep[i])
            i += 1
        else:
            out.append(by_children[match])
            i += len(match)
    return tuple(out)

def discover_hierarchy(
    episodes: Sequence[Sequence[Atom]],
    *,
    max_depth: int = 3,
    min_len: int = 2,
    max_len: int = 4,
    min_episode_support: int | None = None,
) -> Hierarchy:
    if not episodes:
        raise ValueError("episodes cannot be empty")
    if any(not ep for ep in episodes):
        raise ValueError("episodes cannot contain empty episodes")
    if max_depth < 1 or min_len < 2 or max_len < min_len:
        raise ValueError("invalid hierarchy bounds")
    if min_episode_support is None:
        min_episode_support = max(2, (len(episodes) + 1) // 2)
    current = tuple(tuple(ep) for ep in episodes)
    levels: list[tuple[ConceptLayer, ...]] = []
    for depth in range(1, max_depth + 1):
        layer = _select(
            current, depth,
            min_len=min_len,
            max_len=max_len,
            min_support=min_episode_support,
        )
        if not layer:
            break
        levels.append(layer)
        nxt = tuple(_replace_episode(ep, layer) for ep in current)
        if nxt == current:
            break
        current = nxt
    return Hierarchy(tuple(levels), current)
