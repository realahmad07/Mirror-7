import pytest

from .evaluate_experiment6 import run_multiseed_benchmark
from .experiment6_learned_operations import Config


def test_multiseed_runner_requires_three_unique_seeds():
    with pytest.raises(ValueError, match="at least three"):
        run_multiseed_benchmark((17, 29))
    with pytest.raises(ValueError, match="unique"):
        run_multiseed_benchmark((17, 29, 17))


def test_multiseed_runner_reports_fingerprints_and_blocks_promotion():
    cfg = Config(
        state_dim=12,
        op_dim=8,
        hidden_dim=20,
        concepts=6,
        operations=6,
        concept_dim=12,
        max_chain=4,
        epochs=1,
        batch_size=32,
    )
    report = run_multiseed_benchmark((17, 29, 43), config=cfg, device="cpu")
    assert report["seeds"] == [17, 29, 43]
    assert len({row["dataset_sha256"] for row in report["results"]}) == 3
    assert report["promotion_eligible"] is False
    assert len(report["promotion_blockers"]) >= 2
    assert set(report["aggregate"]) >= {"accuracy", "margin", "order_margin"}
