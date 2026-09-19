import random

import pytest

from .mirror7_phase55 import discover_structure


def path_graph(node_count):
    return [(index, index + 1) for index in range(node_count - 1)]


def star_graph(node_count):
    return [(0, index) for index in range(1, node_count)]


def cycle_graph(node_count):
    return [(index, (index + 1) % node_count) for index in range(node_count)]


def held_out_tree():
    return [(0, 1), (1, 2), (1, 3), (3, 4)]


def complete_graph(node_count):
    return [
        (left, right)
        for left in range(node_count)
        for right in range(left + 1, node_count)
    ]


def relabel_edges(edges, seed):
    rng = random.Random(seed)
    nodes = sorted({node for edge in edges for node in edge})
    shuffled = list(nodes)
    rng.shuffle(shuffled)
    mapping = dict(zip(nodes, shuffled))
    return [(mapping[left], mapping[right]) for left, right in edges]


def encode_edge_list(edges, seed):
    rng = random.Random(seed)
    node_ids = sorted({node for edge in edges for node in edge})
    opaque = {node: 100 + index * 17 for index, node in enumerate(node_ids)}
    rows = [f"{opaque[left]},{opaque[right]}" for left, right in edges]
    rng.shuffle(rows)
    return ";".join(rows).encode("ascii")


def encode_neighbor_map(edges, seed):
    rng = random.Random(seed)
    node_ids = sorted({node for edge in edges for node in edge})
    opaque = {node: 400 + index * 23 for index, node in enumerate(node_ids)}

    neighbors = {}
    for left, right in edges:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)

    rows = []
    for node in list(neighbors):
        values = [str(opaque[value]) for value in neighbors[node]]
        rng.shuffle(values)
        rows.append(f"{opaque[node]}:" + ",".join(values))
    rng.shuffle(rows)
    return ";".join(rows).encode("ascii")


def encode_matrix(edges, node_count):
    matrix = [[0] * node_count for _ in range(node_count)]
    for left, right in edges:
        matrix[left][right] = 1
        matrix[right][left] = 1
    return bytes(value for row in matrix for value in row)


def representation_triplet(edges, seed):
    relabeled = relabel_edges(edges, seed)
    node_ids = sorted({node for edge in relabeled for node in edge})
    node_count = len(node_ids)
    position = {node: index for index, node in enumerate(node_ids)}

    matrix_edges = [(position[left], position[right]) for left, right in relabeled]

    return (
        discover_structure(encode_edge_list(relabeled, seed + 1)),
        discover_structure(encode_neighbor_map(relabeled, seed + 2)),
        discover_structure(encode_matrix(matrix_edges, node_count)),
    )


def assert_cross_representation_equivalent(edges, seed):
    edge, neighbor, matrix = representation_triplet(edges, seed)
    assert {edge.fingerprint, neighbor.fingerprint, matrix.fingerprint} == {edge.fingerprint}
    assert edge.node_count == neighbor.node_count == matrix.node_count
    assert edge.edge_count == neighbor.edge_count == matrix.edge_count


@pytest.mark.parametrize(
    "family_builder, sizes",
    [
        (path_graph, (4, 5, 6)),
        (star_graph, (4, 5, 6)),
        (cycle_graph, (4, 5, 6)),
    ],
)
def test_progressive_representation_independence(family_builder, sizes):
    for size in sizes:
        for seed in range(3):
            assert_cross_representation_equivalent(
                family_builder(size),
                seed + size * 100,
            )


def test_held_out_unseen_families():
    for seed, edges in enumerate((held_out_tree(), complete_graph(4)), start=700):
        assert_cross_representation_equivalent(edges, seed)


def test_adversarial_structural_change_is_rejected():
    base = path_graph(5)
    changed = base[:-1] + [(0, 3)]

    base_rep = representation_triplet(base, 900)[0]
    changed_rep = representation_triplet(changed, 900)[0]

    assert base_rep.fingerprint != changed_rep.fingerprint


@pytest.mark.parametrize(
    "malformed",
    (
        b"",
        b"1,2;bad",
        b"0:1;1",
        bytes([1, 0, 0]),
        bytes([0, 1, 0, 0]),
    ),
)
def test_adversarial_malformed_input_rejected(malformed):
    with pytest.raises(ValueError):
        discover_structure(malformed)


def test_repeat_run_determinism_and_seed_invariance():
    target = held_out_tree()
    fingerprints = []

    for seed in range(20):
        fingerprints.append(
            representation_triplet(target, seed)[0].fingerprint
        )

    assert len(set(fingerprints)) == 1
    assert discover_structure(
        encode_matrix(target, 5)
    ).fingerprint == fingerprints[0]
