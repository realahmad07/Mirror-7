"""Mirror 7 Phase 57: cross-view raw concept acquisition.

Phase 57 removes the single-byte-stream boundary from Phase 56.  The learner
receives an episode as an unlabeled bundle of heterogeneous raw views.  A view
may be a 1-D sequence or a 2-D integer grid.  The system discovers recurring
relational patterns without semantic labels, view names, task-family labels,
or an expected concept graph.

Core idea:
    raw views
      -> modality-agnostic local relational atoms
      -> episode/view support statistics
      -> compact recurrent concepts
      -> cross-view concept graph
      -> view-invariant relations/events

The representation deliberately ignores raw symbol identity and view order.
It keeps only equality structure inside bounded local windows/patches.  This
makes the learned atom portable when the same latent pattern moves between
raw encodings or between 1-D and 2-D views.

This is still a bounded structural learner, not semantic perception or AGI.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import log2
import random
from typing import Iterable, Sequence, Tuple, Union

Raw1D = Union[bytes, bytearray, Sequence[int]]
RawView = Union[Raw1D, Sequence[Sequence[int]]]
Episode = Sequence[RawView]
Concept = Tuple[int, ...]


@dataclass(frozen=True)
class ConceptRecord:
    concept: Concept
    support: int                 # independent episode support
    view_support: int            # total supporting view occurrences
    multi_view_support: int      # episodes supporting the concept in >=2 views
    occurrences: int             # total local occurrences
    description_gain: float      # MDL-inspired repeated-structure gain
    fingerprint: str


@dataclass(frozen=True)
class MultiViewGraph:
    concepts: Tuple[ConceptRecord, ...]
    relations: Tuple[Tuple[int, int, int], ...]  # unordered co-occurrence
    events: Tuple[Tuple[int, int, int], ...]     # ordered within-view events


def _canonical_shape(values: Sequence[int]) -> Concept:
    """Canonicalize arbitrary integer values to first-appearance IDs."""
    mapping = {}
    out = []
    nxt = 0
    for value in values:
        if value not in mapping:
            mapping[value] = nxt
            nxt += 1
        out.append(mapping[value])
    return tuple(out)


def _fingerprint(concept: Concept) -> str:
    return sha256(repr(concept).encode("ascii")).hexdigest()


def _normalize_view(view: RawView) -> Tuple[str, Tuple]:
    """Return ('1d', tuple(int,...)) or ('2d', tuple(tuple(int,...),...))."""
    if isinstance(view, (bytes, bytearray)):
        data = tuple(int(x) for x in view)
        if not data:
            raise ValueError("raw view cannot be empty")
        return "1d", data

    try:
        outer = list(view)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("raw view must be bytes or an integer sequence") from exc
    if not outer:
        raise ValueError("raw view cannot be empty")

    # 1-D sequence of integers.
    if all(isinstance(x, int) for x in outer):
        return "1d", tuple(int(x) for x in outer)

    # 2-D rectangular integer grid.
    if not all(isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)) for row in outer):
        raise TypeError("raw view must be a 1-D integer sequence or 2-D integer grid")
    rows = [tuple(int(x) if isinstance(x, int) else (_ for _ in ()).throw(TypeError("grid cells must be integers")) for x in row) for row in outer]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("grid must be non-empty and rectangular")
    return "2d", tuple(rows)


def _iter_candidates(view: RawView, min_elements: int, max_elements: int):
    kind, data = _normalize_view(view)
    if kind == "1d":
        seq = data
        for length in range(min_elements, min(max_elements, len(seq)) + 1):
            for start in range(len(seq) - length + 1):
                yield start, (start + length,), seq[start:start + length]
        return

    grid = data
    height, width = len(grid), len(grid[0])
    for h in range(1, min(3, height) + 1):
        for w in range(1, min(3, width) + 1):
            n = h * w
            if n < min_elements or n > max_elements:
                continue
            for r in range(height - h + 1):
                for c in range(width - w + 1):
                    flat = tuple(grid[r + dr][c + dc] for dr in range(h) for dc in range(w))
                    # Use a stable linear position after the view is canonicalized.
                    yield r * width + c, (r * width + c + h * width,), flat


def _shuffled_view(view: RawView, seed: int) -> RawView:
    """Deterministic within-view permutation preserving shape/geometry."""
    kind, data = _normalize_view(view)
    rng = random.Random(seed)
    if kind == "1d":
        values = list(data)
        rng.shuffle(values)
        return tuple(values)
    height, width = len(data), len(data[0])
    values = [x for row in data for x in row]
    rng.shuffle(values)
    return tuple(tuple(values[r * width:(r + 1) * width]) for r in range(height))


def _shape_counts(views: Iterable[RawView], min_elements: int, max_elements: int) -> Counter[Concept]:
    counts: Counter[Concept] = Counter()
    for view in views:
        for _, _, values in _iter_candidates(view, min_elements, max_elements):
            shape = _canonical_shape(values)
            if _informative(shape):
                counts[shape] += 1
    return counts

def _informative(shape: Concept) -> bool:
    unique = len(set(shape))
    # All-unique and all-identical windows are high-frequency generic noise.
    return 1 < unique < len(shape)


def _description_gain(shape: Concept, occurrences: int) -> float:
    """MDL-inspired structural repetition gain.

    Exact compression depends on an external code book, so this is only a
    bounded deterministic ranking/filter: repeated internal equalities save
    symbol-description bits compared with independent symbols.
    """
    n = len(shape)
    unique = len(set(shape))
    redundancy = n - unique
    if redundancy <= 0:
        return 0.0
    literal_bits = occurrences * n * log2(max(2, unique + 1))
    structured_bits = unique * log2(max(2, n)) + occurrences * (n - redundancy) * log2(max(2, unique))
    return max(0.0, literal_bits - structured_bits)


def discover_concepts(
    episodes: Sequence[Episode],
    *,
    min_elements: int = 3,
    max_elements: int = 6,
    min_episode_support: int | None = None,
    min_description_gain: float = 1.0,
) -> Tuple[ConceptRecord, ...]:
    if not episodes:
        raise ValueError("episodes cannot be empty")
    if min_elements < 2 or max_elements < min_elements:
        raise ValueError("invalid candidate-size bounds")

    episode_support: Counter[Concept] = Counter()
    view_support: Counter[Concept] = Counter()
    multi_view_support: Counter[Concept] = Counter()
    occurrences: Counter[Concept] = Counter()
    null_occurrences: Counter[Concept] = Counter()

    for episode in episodes:
        if not episode:
            raise ValueError("episodes cannot contain empty episodes")
        seen_episode: set[Concept] = set()
        seen_views: set[Concept] = set()
        view_occurrence_sets: list[set[Concept]] = []
        for view in episode:
            seen_view: set[Concept] = set()
            for _, _, values in _iter_candidates(view, min_elements, max_elements):
                shape = _canonical_shape(values)
                if not _informative(shape):
                    continue
                occurrences[shape] += 1
                seen_episode.add(shape)
                seen_view.add(shape)
            seen_views.update(seen_view)
            view_occurrence_sets.append(seen_view)
        for shape in seen_episode:
            episode_support[shape] += 1
        for shape in seen_views:
            view_support[shape] += 1
        for shape in seen_episode:
            if sum(shape in seen_view for seen_view in view_occurrence_sets) >= 2:
                multi_view_support[shape] += 1

        # Build a deterministic local null from within-view permutations. This
        # preserves each raw view's marginal values/geometry while destroying
        # local equality structure. True concepts should beat this null by a
        # meaningful margin; chance motifs should not.
        for view_index, view in enumerate(episode):
            normalized = _normalize_view(view)[1]
            seed_material = sha256(repr(normalized).encode("utf-8")).digest()
            base_seed = int.from_bytes(seed_material[:8], "big")
            for replicate in range(3):
                null_view = _shuffled_view(view, base_seed + replicate)
                null_counts = _shape_counts((null_view,), min_elements, max_elements)
                null_occurrences.update(null_counts)

    if min_episode_support is None:
        min_episode_support = max(2, (len(episodes) + 1) // 2)

    candidates = []
    for shape, support in episode_support.items():
        if support < min_episode_support:
            continue
        # A cross-view concept must have independent evidence in at least two
        # raw views per supporting episode. This blocks chance motifs that only
        # arise from the birthday-collision statistics of one noisy stream.
        if multi_view_support[shape] < min_episode_support:
            continue
        # Three permutation-null replicates per raw view provide an empirical
        # background. Require both an absolute margin and a multiplicative
        # enrichment over chance.
        null_mean = null_occurrences[shape] / max(1, len(episodes) * 3)
        observed = occurrences[shape]
        if observed < null_mean + 2.0 or observed < 1.5 * null_mean + 2.0:
            continue
        gain = _description_gain(shape, occurrences[shape])
        if gain < min_description_gain:
            continue
        candidates.append(shape)

    # Prefer the shortest explanation first.  If equal length, prefer broad
    # support, cross-view support, then occurrence count, then lexical order.
    candidates.sort(
        key=lambda c: (
            len(c),
            -episode_support[c],
            -view_support[c],
            -multi_view_support[c],
            -occurrences[c],
            c,
        )
    )

    # Keep compact atoms and suppress candidates that contain an already
    # selected atom as an exact contiguous relational subsequence.
    selected: list[Concept] = []
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

    return tuple(
        ConceptRecord(
            concept=shape,
            support=episode_support[shape],
            view_support=view_support[shape],
            multi_view_support=multi_view_support[shape],
            occurrences=occurrences[shape],
            description_gain=_description_gain(shape, occurrences[shape]),
            fingerprint=_fingerprint(shape),
        )
        for shape in selected
    )


def _locate_concepts(view: RawView, concepts: Sequence[ConceptRecord]) -> list[tuple[int, int]]:
    concept_set = {record.concept: idx for idx, record in enumerate(concepts)}
    hits: list[tuple[int, int]] = []
    # Candidate extraction is intentionally bounded by the maximum learned atom size.
    max_n = max(len(record.concept) for record in concepts)
    for position, _, values in _iter_candidates(view, 2, max_n):
        shape = _canonical_shape(values)
        idx = concept_set.get(shape)
        if idx is not None:
            hits.append((position, idx))
    hits.sort(key=lambda item: (item[0], -len(concepts[item[1]].concept), item[1]))

    chosen: list[tuple[int, int]] = []
    occupied_until = -1
    for position, idx in hits:
        end = position + len(concepts[idx].concept)
        if position >= occupied_until:
            chosen.append((position, idx))
            occupied_until = end
    return chosen


def build_concept_graph(
    episodes: Sequence[Episode],
    concepts: Sequence[ConceptRecord],
) -> MultiViewGraph:
    if not concepts:
        raise ValueError("concept set cannot be empty")

    relation_episode_support: Counter[tuple[int, int]] = Counter()
    event_episode_support: Counter[tuple[int, int]] = Counter()

    for episode in episodes:
        episode_concepts: set[int] = set()
        event_pairs: set[tuple[int, int]] = set()

        for view in episode:
            located = _locate_concepts(view, concepts)
            ids = [idx for _, idx in located]
            episode_concepts.update(ids)
            for left, right in zip(ids, ids[1:]):
                if left != right:
                    event_pairs.add((left, right))

        # Relations are unordered so view permutation cannot change them.
        for left in episode_concepts:
            for right in episode_concepts:
                if left < right:
                    relation_episode_support[(left, right)] += 1
        for pair in event_pairs:
            event_episode_support[pair] += 1

    relations = tuple(
        (left, right, support)
        for (left, right), support in sorted(relation_episode_support.items())
        if support > 0
    )
    events = tuple(
        (left, right, support)
        for (left, right), support in sorted(event_episode_support.items())
        if support > 0
    )
    return MultiViewGraph(tuple(concepts), relations, events)
