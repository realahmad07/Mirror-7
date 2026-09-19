import random
import pytest

from .mirror7_phase56 import discover_concepts, build_concept_graph


def encode_episode(concepts, seed, noise=2):
    rng = random.Random(seed)
    alphabet = list(range(20, 250))
    rng.shuffle(alphabet)
    mapping = {i: alphabet[i] for i in range(40)}
    out = bytearray()
    for concept_id in concepts:
        motif = {0: [1, 2, 1], 1: [3, 4, 5, 3], 2: [6, 7, 6, 7]}[concept_id]
        out.extend(mapping[x] for x in motif)
        out.extend(rng.randrange(0, 256) for _ in range(noise))
    return bytes(out)


def latent_to_bytes(labels, seed):
    return encode_episode(labels, seed)


def concept_fingerprints(graph):
    return {c.fingerprint for c in graph.concepts}


def run_family(seed, family):
    episodes = [latent_to_bytes(labels, seed + i) for i, labels in enumerate(family)]
    concepts = discover_concepts(episodes, min_len=3, max_len=6, min_episode_support=3)
    graph = build_concept_graph(episodes, concepts)
    return concepts, graph


def test_progressive_raw_concept_acquisition():
    families = [
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
        [[0, 2, 1, 0], [1, 0, 2, 1], [2, 1, 0, 2]],
        [[0, 1, 2, 1, 0], [2, 0, 1, 2, 0], [1, 2, 0, 1, 2]],
    ]
    for level, family in enumerate(families):
        for seed in range(3):
            concepts, graph = run_family(1000 + level * 100 + seed, family)
            assert len(concepts) >= 2
            assert len(graph.relations) >= 1
            assert all(c.support >= 3 for c in concepts)


def test_held_out_concept_recombination():
    family = [[0, 1, 2, 0], [2, 1, 0, 2], [1, 0, 2, 1], [0, 2, 1, 0]]
    concepts, graph = run_family(9000, family)
    assert len(concepts) >= 3
    held_out = latent_to_bytes([2, 0, 1, 2, 1], 9999)
    held = build_concept_graph([held_out], concepts)
    assert held.relations
    assert len(held.relations) >= 2


def test_held_out_raw_encoding_seed_change():
    train = [latent_to_bytes([0, 1, 2, 0], s) for s in range(3)]
    test = [latent_to_bytes([0, 2, 1, 0], s) for s in range(50, 53)]
    concepts = discover_concepts(train, min_len=3, max_len=6, min_episode_support=3)
    assert len(concepts) >= 3
    train_fp = concept_fingerprints(build_concept_graph(train, concepts))
    test_fp = concept_fingerprints(build_concept_graph(test, concepts))
    assert train_fp == test_fp


def test_adversarial_single_episode_noise_rejected():
    episodes = [latent_to_bytes([0, 1, 2, 0], s) for s in range(3)]
    discover_concepts(episodes, min_len=3, max_len=6, min_episode_support=3)
    single = bytes(range(3, 60))
    with pytest.raises(AssertionError):
        assert len(discover_concepts([single], min_len=3, max_len=6, min_episode_support=2)) >= 2


def test_adversarial_semantic_swap_changes_relation_graph():
    base = [latent_to_bytes([0, 1, 2, 0], s) for s in range(3)]
    swapped = [latent_to_bytes([0, 2, 1, 0], s) for s in range(3)]
    concepts = discover_concepts(base, min_len=3, max_len=6, min_episode_support=3)
    graph_a = build_concept_graph(base, concepts)
    graph_b = build_concept_graph(swapped, concepts)
    assert graph_a.relations != graph_b.relations


def test_malformed_inputs_rejected():
    with pytest.raises(TypeError):
        discover_concepts(["not bytes"], min_len=3, max_len=4)
    with pytest.raises(ValueError):
        discover_concepts([], min_len=3, max_len=4)


def test_repeat_determinism():
    episodes = [latent_to_bytes([0, 1, 2, 0], s) for s in range(3)]
    a = discover_concepts(episodes, min_len=3, max_len=6, min_episode_support=3)
    b = discover_concepts(episodes, min_len=3, max_len=6, min_episode_support=3)
    assert a == b
