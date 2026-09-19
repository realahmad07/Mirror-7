"""Mirror 7 Phase 55: representation-independent structural discovery.

Phase 55 tests whether the same underlying relational structure can be
recovered from substantially different surface encodings without giving the
discoverer semantic labels or task-family metadata.

Supported raw encodings are intentionally syntax-level and generic:
* binary adjacency matrices,
* delimited edge lists,
* delimited neighbor maps.

The learned artifact is a canonical unlabeled graph, not a task-specific label.
Exact canonicalization is bounded to <= 8 nodes so the acceptance tests can
provide an auditable, collision-free reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Set, Tuple

Graph = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class DiscoveredStructure:
    node_count: int
    edge_count: int
    canonical_graph: Graph
    fingerprint: str
    parser_family: str


def _graph_from_edges(edges: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
    graph: Dict[int, Set[int]] = {}
    for left, right in edges:
        if left == right:
            raise ValueError("self-loops are not supported")
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    if not graph:
        raise ValueError("empty graph")
    return graph


def _canonical_graph(graph: Dict[int, Set[int]]) -> Graph:
    nodes = sorted(graph)
    node_count = len(nodes)
    if node_count > 8:
        raise ValueError("exact canonicalization is bounded to at most 8 nodes")

    index = {node: i for i, node in enumerate(nodes)}
    matrix = [[0] * node_count for _ in range(node_count)]
    for left, neighbors in graph.items():
        for right in neighbors:
            matrix[index[left]][index[right]] = 1

    best_flat: Optional[Tuple[int, ...]] = None
    best_graph: Optional[Graph] = None

    for permutation in permutations(range(node_count)):
        flat: List[int] = []
        for row in range(node_count):
            for col in range(node_count):
                flat.append(matrix[permutation[row]][permutation[col]])
        candidate = tuple(flat)
        if best_flat is None or candidate < best_flat:
            best_flat = candidate
            best_graph = tuple(
                tuple(candidate[row * node_count:(row + 1) * node_count])
                for row in range(node_count)
            )

    assert best_graph is not None
    return best_graph


def _fingerprint(canonical_graph: Graph) -> str:
    return hashlib.sha256(repr(canonical_graph).encode("ascii")).hexdigest()


def _parse_edge_list(text: str) -> Dict[int, Set[int]]:
    if ";" not in text:
        raise ValueError("not an edge-list encoding")

    edges: List[Tuple[int, int]] = []
    for row in text.strip().split(";"):
        values = [value.strip() for value in row.split(",") if value.strip()]
        if len(values) != 2 or any(not re.fullmatch(r"-?\d+", value) for value in values):
            raise ValueError("invalid edge-list row")
        edges.append((int(values[0]), int(values[1])))

    return _graph_from_edges(edges)


def _parse_neighbor_map(text: str) -> Dict[int, Set[int]]:
    if ":" not in text:
        raise ValueError("not a neighbor-map encoding")

    edges: List[Tuple[int, int]] = []
    rows_seen = 0

    for row in text.strip().split(";"):
        if not row.strip():
            continue

        if ":" not in row:
            raise ValueError("invalid neighbor-map row")

        node_text, neighbor_text = row.split(":", 1)
        if not re.fullmatch(r"-?\d+", node_text.strip()):
            raise ValueError("invalid node id")

        node = int(node_text.strip())
        neighbors = [
            value.strip()
            for value in neighbor_text.split(",")
            if value.strip()
        ]
        if any(not re.fullmatch(r"-?\d+", value) for value in neighbors):
            raise ValueError("invalid neighbor id")

        for neighbor in neighbors:
            edges.append((node, int(neighbor)))
        rows_seen += 1

    if rows_seen == 0:
        raise ValueError("empty neighbor map")

    return _graph_from_edges(edges)


def _parse_binary_matrix(raw: bytes) -> Dict[int, Set[int]]:
    node_count = math.isqrt(len(raw))
    if node_count < 1 or node_count * node_count != len(raw):
        raise ValueError("not a square binary matrix")
    if any(value not in (0, 1) for value in raw):
        raise ValueError("matrix contains values other than 0/1")

    edges: List[Tuple[int, int]] = []

    for row in range(node_count):
        if raw[row * node_count + row] != 0:
            raise ValueError("matrix contains a self-loop")

        for col in range(row + 1, node_count):
            upper = raw[row * node_count + col]
            lower = raw[col * node_count + row]
            if upper != lower:
                raise ValueError("matrix is not symmetric")
            if upper:
                edges.append((row, col))

    return _graph_from_edges(edges)


def discover_structure(raw: bytes) -> DiscoveredStructure:
    """Discover an unlabeled graph from a raw byte representation.

    No semantic task name, graph family, or node meaning is supplied.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("raw observation must be bytes or bytearray")

    candidates = (
        ("matrix", lambda: _parse_binary_matrix(bytes(raw))),
        ("neighbor_map", lambda: _parse_neighbor_map(bytes(raw).decode("ascii"))),
        ("edge_list", lambda: _parse_edge_list(bytes(raw).decode("ascii"))),
    )

    errors = []

    for parser_family, parser in candidates:
        try:
            graph = parser()
            canonical = _canonical_graph(graph)
            edge_count = sum(len(values) for values in graph.values()) // 2
            return DiscoveredStructure(
                node_count=len(graph),
                edge_count=edge_count,
                canonical_graph=canonical,
                fingerprint=_fingerprint(canonical),
                parser_family=parser_family,
            )
        except (UnicodeDecodeError, ValueError) as error:
            errors.append(f"{parser_family}: {error}")

    raise ValueError("unsupported representation: " + " | ".join(errors))
