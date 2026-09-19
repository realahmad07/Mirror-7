import random
import pytest

from .mirror7_phase57 import discover_concepts, build_concept_graph


def _map_values(values, mapping):
    return [mapping[v] for v in values]


def _motif(kind):
    # Same relational identities, different local encodings.
    return {
        0: [1, 2, 1],        # ABA
        1: [3, 4, 5, 3],     # ABCA
        2: [6, 7, 6, 8, 7],  # ABACB
    }[kind]


def make_episode(labels, seed, permute_views=False, noise=1, as_grid=False):
    rng = random.Random(seed)
    base = list(range(20, 240))
    rng.shuffle(base)
    mapping = {i: base[i] for i in range(60)}

    stream = []
    grid_blocks = []
    width = 6
    for kind in labels:
        motif = _map_values(_motif(kind), mapping)
        stream.extend(motif)
        stream.extend(rng.randrange(0, 256) for _ in range(noise))

        filler = lambda: rng.randrange(0, 256)
        block = [[filler() for _ in range(width)] for _ in range(2)]
        if kind == 0:
            # Horizontal 1x3 relational atom: ABA.
            block[0][0:3] = motif
        elif kind == 1:
            # 2x2 relational atom: AB / CA.
            block[0][0:2] = motif[0:2]
            block[1][0:2] = motif[2:4]
        else:
            # 2x3 relational atom: ABA / CBX; first five cells are ABACB.
            block[0][0:3] = motif[0:3]
            block[1][0:3] = [motif[3], motif[4], filler()]
        grid_blocks.extend(block)

    grid = tuple(tuple(row) for row in grid_blocks)
    views = [bytes(stream), grid]
    if permute_views:
        rng.shuffle(views)
    return tuple(views)


def fingerprints(concepts):
    return {c.fingerprint for c in concepts}


def train(seed, families):
    episodes = [make_episode(labels, seed + i, permute_views=(i % 2 == 1)) for i, labels in enumerate(families)]
    concepts = discover_concepts(episodes, min_elements=3, max_elements=6, min_episode_support=3)
    graph = build_concept_graph(episodes, concepts)
    return concepts, graph


def test_progressive_multiview_acquisition_3_seeds():
    families = [
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
        [[0, 2, 1, 0], [1, 0, 2, 1], [2, 1, 0, 2]],
        [[0, 1, 2, 1, 0], [2, 0, 1, 2, 0], [1, 2, 0, 1, 2]],
    ]
    for level, family in enumerate(families):
        for seed in range(3):
            concepts, graph = train(1000 + level * 100 + seed, family)
            assert len(concepts) >= 3
            assert all(c.support >= 3 for c in concepts)
            assert all(c.description_gain >= 1.0 for c in concepts)
            assert graph.relations
            assert graph.events


def test_held_out_view_permutation_and_reencoding():
    train_eps = [make_episode([0, 1, 2, 0], s, permute_views=(s % 2 == 0)) for s in range(4)]
    concepts = discover_concepts(train_eps, min_elements=3, max_elements=6, min_episode_support=3)
    held = [make_episode([2, 0, 1, 2, 1], 9000 + s, permute_views=True) for s in range(2)]
    a = build_concept_graph(held[:1], concepts)
    b = build_concept_graph(held[1:], concepts)
    assert fingerprints(a.concepts) == fingerprints(b.concepts)
    assert a.relations and a.events


def test_held_out_cross_geometry():
    train_eps = [make_episode([0, 1, 2, 0], s, permute_views=True) for s in range(3)]
    concepts = discover_concepts(train_eps, min_elements=3, max_elements=6, min_episode_support=3)
    # Keep only the 2-D view from a new episode. This is unseen geometry at
    # inference time for the corresponding latent occurrences.
    episode = make_episode([0, 2, 1, 0], 7777, permute_views=False)
    grid_only = (episode[1],)
    graph = build_concept_graph([grid_only], concepts)
    assert graph.concepts == tuple(concepts)
    assert graph.relations or graph.events


def test_adversarial_noise_only_rejected():
    rng = random.Random(42)
    noise_episodes = []
    for _ in range(3):
        noise = bytes(rng.randrange(0, 256) for _ in range(120))
        grid = tuple(tuple(rng.randrange(0, 256) for _ in range(10)) for _ in range(10))
        noise_episodes.append((noise, grid))
    concepts = discover_concepts(noise_episodes, min_elements=3, max_elements=6, min_episode_support=3)
    assert concepts == ()


def test_adversarial_semantic_relation_swap_changes_graph():
    base = [make_episode([0, 1, 2, 0], s) for s in range(3)]
    swapped = [make_episode([0, 2, 1, 0], s) for s in range(3)]
    concepts = discover_concepts(base, min_elements=3, max_elements=6, min_episode_support=3)
    graph_a = build_concept_graph(base, concepts)
    graph_b = build_concept_graph(swapped, concepts)
    assert graph_a.events != graph_b.events


def test_adversarial_view_dropout_preserves_surviving_structure():
    train_eps = [make_episode([0, 1, 2, 0], s, permute_views=True) for s in range(3)]
    concepts = discover_concepts(train_eps, min_elements=3, max_elements=6, min_episode_support=3)
    episode = make_episode([0, 1, 2, 0], 1234, permute_views=True)
    one_view = (episode[0],)
    graph = build_concept_graph([one_view], concepts)
    assert graph.events or graph.relations


def test_malformed_inputs_rejected():
    with pytest.raises(TypeError):
        discover_concepts([("bad",)], min_elements=3, max_elements=4)
    with pytest.raises(ValueError):
        discover_concepts([()], min_elements=3, max_elements=4)
    with pytest.raises(ValueError):
        discover_concepts([([1, 2, 3],)], min_elements=4, max_elements=3)


def test_repeat_determinism():
    episodes = [make_episode([0, 1, 2, 0], s, permute_views=True) for s in range(3)]
    a = discover_concepts(episodes, min_elements=3, max_elements=6, min_episode_support=3)
    b = discover_concepts(episodes, min_elements=3, max_elements=6, min_episode_support=3)
    assert a == b


def test_scaling_guard_small_linear_growth():
    import time
    small = [make_episode([0, 1, 2, 0], 2000 + i, permute_views=(i % 2 == 0)) for i in range(6)]
    large = [make_episode([0, 1, 2, 0] * 3, 3000 + i, permute_views=(i % 2 == 0), noise=2) for i in range(6)]
    t0 = time.perf_counter()
    discover_concepts(small, min_elements=3, max_elements=6, min_episode_support=4)
    t1 = time.perf_counter()
    discover_concepts(large, min_elements=3, max_elements=6, min_episode_support=4)
    t2 = time.perf_counter()
    small_t = max(t1 - t0, 1e-6)
    large_t = t2 - t1
    assert large_t < small_t * 15 + 0.02


def test_graph_view_order_invariance():
    episodes_a = [make_episode([0, 1, 2, 0], s, permute_views=False) for s in range(3)]
    episodes_b = [make_episode([0, 1, 2, 0], s, permute_views=True) for s in range(3)]
    ca = discover_concepts(episodes_a, min_elements=3, max_elements=6, min_episode_support=3)
    cb = discover_concepts(episodes_b, min_elements=3, max_elements=6, min_episode_support=3)
    assert fingerprints(ca) == fingerprints(cb)
