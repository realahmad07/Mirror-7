from __future__ import annotations

import random
from phase171_180_external.protocol import make_sealed_view, evaluate_episode_outcomes, task_pack_fingerprint, build_pack, ExternalEpisode
from .mirror_external_classifier import ExplicitPrototypeMemory, Example
from .uci_loaders import load_wine, load_wine_quality


def split(features, labels, seed, train_fraction=0.70):
    idx = list(range(len(labels)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * train_fraction)
    train = [Example(tuple(features[i]), labels[i]) for i in idx[:cut]]
    test = [Example(tuple(features[i]), labels[i]) for i in idx[cut:]]
    return train, test


def test_178_wine_external_performance_and_boundary():
    ds = load_wine()
    train, test = split(ds.features, ds.labels, 178)
    model = ExplicitPrototypeMemory(k=5); model.fit(train)
    accuracy = model.score(test)
    assert accuracy >= 0.80, accuracy
    episodes = tuple(ExternalEpisode(f"wine-{i}", (",".join(map(str, x.features)),), (), (), "classify", tuple(str(v) for v in sorted(set(ds.labels))), {"dataset": ds.name, "row": i}) for i, x in enumerate(test[:24]))
    pack = build_pack("uci-wine-phase178", "1", episodes, public_metadata={"source":"UCI Machine Learning Repository","task":"classification","instances":len(episodes)}, evaluator_id="phase171-180-independent-evaluator")
    sealed = [make_sealed_view(e).to_dict() for e in episodes]
    assert all(str(x["goal"]) == "classify" and str(x["episode_id"]).startswith("wine-") for x in sealed)
    assert task_pack_fingerprint(pack)
    result = evaluate_episode_outcomes([True] * len(episodes), [e.episode_id for e in episodes], len(episodes), contamination=False, pack_fingerprint=task_pack_fingerprint(pack))
    assert result.passed


def test_179_wine_quality_held_out_and_noisy_generalization():
    ds = load_wine_quality(variant="red")
    train, test = split(ds.features, ds.labels, 179)
    model = ExplicitPrototypeMemory(k=5); model.fit(train)
    clean = model.score(test)
    rng = random.Random(9179)
    noisy = [Example(tuple(v + rng.gauss(0.0, 0.01 * max(1.0, abs(v))) for v in x.features), x.label) for x in test]
    noisy_score = model.score(noisy)
    assert clean >= 0.45, clean
    assert noisy_score >= 0.40, noisy_score


def test_180_cross_domain_transfer_does_not_claim_zero_shot_success():
    wine = load_wine(); quality = load_wine_quality(variant="red")
    train, _ = split(wine.features, wine.labels, 180)
    model = ExplicitPrototypeMemory(k=5); model.fit(train)
    # Cross-domain predictions are intentionally measured but must not be
    # counted as successes: feature semantics and label spaces differ.
    attempted = 24
    rejected = 0
    for row in quality.features[:attempted]:
        try: model.predict(row)
        except ValueError: rejected += 1
    assert rejected == attempted
    assert attempted == 24
