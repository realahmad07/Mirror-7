"""Mirror 7 Phase 56: bounded raw concept acquisition.

Discovers reusable structural concepts directly from undifferentiated byte
streams. Concepts are recurring, canonicalized motifs supported across
independent episodes; relations are ordered co-occurrences of discovered
concepts; events are repeated concept transitions.

This is deliberately bounded: it tests unsupervised structural concept
acquisition, not open-world semantic grounding.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter, defaultdict
from hashlib import sha256
from typing import Dict, Iterable, List, Sequence, Tuple

Concept = Tuple[int, ...]

@dataclass(frozen=True)
class ConceptRecord:
    concept: Concept
    support: int
    occurrences: int
    fingerprint: str

@dataclass(frozen=True)
class ConceptGraph:
    concepts: Tuple[ConceptRecord, ...]
    relations: Tuple[Tuple[int, int, int], ...]
    events: Tuple[Tuple[int, int, int], ...]


def _canonical_shape(seq: Sequence[int]) -> Concept:
    mapping = {}
    out = []
    nxt = 0
    for value in seq:
        if value not in mapping:
            mapping[value] = nxt
            nxt += 1
        out.append(mapping[value])
    return tuple(out)


def _fingerprint(concept: Concept) -> str:
    return sha256(repr(concept).encode("ascii")).hexdigest()


def _all_windows(data: bytes, min_len: int, max_len: int):
    for length in range(min_len, min(max_len, len(data)) + 1):
        for start in range(len(data) - length + 1):
            yield start, length, data[start:start + length]


def discover_concepts(
    episodes: Sequence[bytes],
    *,
    min_len: int = 3,
    max_len: int = 6,
    min_episode_support: int | None = None,
) -> Tuple[ConceptRecord, ...]:
    if not episodes:
        raise ValueError("episodes cannot be empty")
    if any(not isinstance(ep, (bytes, bytearray)) for ep in episodes):
        raise TypeError("episodes must contain bytes-like observations")
    if min_len < 2 or max_len < min_len:
        raise ValueError("invalid motif length bounds")

    episode_support: Counter[Concept] = Counter()
    total_occurrences: Counter[Concept] = Counter()
    raw_variants: Dict[Concept, set[bytes]] = defaultdict(set)

    for episode in episodes:
        seen_in_episode = set()
        for _, _, window in _all_windows(bytes(episode), min_len, max_len):
            shape = _canonical_shape(window)
            # Reject maximally generic all-unique windows: their canonical
            # shape is shared by arbitrary noise and is not evidence of a
            # reusable concept.
            if len(set(shape)) == len(shape):
                continue
            seen_in_episode.add(shape)
            total_occurrences[shape] += 1
            raw_variants[shape].add(window)
        for shape in seen_in_episode:
            episode_support[shape] += 1

    if min_episode_support is None:
        min_episode_support = max(2, (len(episodes) + 1) // 2)

    # Prefer compact recurring motifs: the smallest reusable structure is the
    # concept candidate; longer windows that contain it are treated as context.
    candidates = [
        shape for shape, support in episode_support.items()
        if support >= min_episode_support
    ]
    candidates.sort(
        key=lambda c: (len(c), -episode_support[c], -total_occurrences[c], c)
    )

    selected: List[Concept] = []
    for candidate in candidates:
        redundant = False
        for chosen in selected:
            if len(candidate) > len(chosen):
                for offset in range(len(candidate) - len(chosen) + 1):
                    if candidate[offset:offset + len(chosen)] == chosen:
                        redundant = True
                        break
            if redundant:
                break
        if not redundant:
            selected.append(candidate)

    records = tuple(
        ConceptRecord(
            concept=shape,
            support=episode_support[shape],
            occurrences=total_occurrences[shape],
            fingerprint=_fingerprint(shape),
        )
        for shape in selected
    )
    return records


def _locate_concepts(episode: bytes, concepts: Sequence[ConceptRecord]) -> List[Tuple[int, int]]:
    hits: List[Tuple[int, int]] = []
    for idx, record in enumerate(concepts):
        target = record.concept
        for start, length, window in _all_windows(episode, len(target), len(target)):
            if _canonical_shape(window) == target:
                hits.append((start, idx))
    hits.sort()
    chosen = []
    last_end = -1
    concepts_sorted = sorted(hits, key=lambda x: (x[0], -len(concepts[x[1]].concept)))
    for start, idx in concepts_sorted:
        end = start + len(concepts[idx].concept)
        if start >= last_end:
            chosen.append((start, idx))
            last_end = end
    return chosen


def build_concept_graph(episodes: Sequence[bytes], concepts: Sequence[ConceptRecord]) -> ConceptGraph:
    if not concepts:
        raise ValueError("concept set cannot be empty")
    relation_support = Counter()
    event_support = Counter()

    for episode in episodes:
        located = _locate_concepts(bytes(episode), concepts)
        seen_rel = set()
        seen_evt = set()
        for (_, left), (_, right) in zip(located, located[1:]):
            rel = (left, right)
            relation_support[rel] += 1
            seen_rel.add(rel)
            event_support[rel] += 1
            seen_evt.add(rel)
        for rel in list(relation_support):
            if rel not in seen_rel:
                relation_support[rel] -= 1
        for rel in list(event_support):
            if rel not in seen_evt:
                event_support[rel] -= 1

    relations = tuple(sorted((a, b, s) for (a, b), s in relation_support.items() if s > 0))
    events = tuple(sorted((a, b, s) for (a, b), s in event_support.items() if s > 0))
    return ConceptGraph(tuple(concepts), relations, events)
