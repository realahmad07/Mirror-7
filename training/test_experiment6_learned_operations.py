import torch
import pytest
from dataclasses import replace

from .experiment6_learned_operations import (
    Config,
    Example,
    LearnedOperationBrain,
    _batch,
    evaluate,
    make_world,
    seed_everything,
    train_model,
    world_fingerprint,
)


def _example(concept: int, ops: tuple[int, ...]) -> Example:
    return Example(concept, ops, torch.nn.functional.normalize(torch.randn(64), dim=0), False)


def test_exp6_forward_and_world():
    cfg = Config()
    model = LearnedOperationBrain(cfg)
    rows = make_world(cfg, 42)[:4]
    concept_ids, operations, targets = _batch(rows, "cpu")
    output = model.compose(concept_ids, operations)
    assert output.shape == (4, cfg.state_dim)
    assert targets.shape == output.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(targets).all()


def test_exp6_has_learned_operation_parameters():
    cfg = Config()
    model = LearnedOperationBrain(cfg)
    assert model.operation.weight.shape == (cfg.operations, cfg.op_dim)
    assert sum(parameter.numel() for parameter in model.parameters()) < 500_000


def test_variable_length_padding_does_not_apply_operation_zero():
    cfg = Config()
    seed_everything(7)
    model = LearnedOperationBrain(cfg).eval()
    rows = [
        _example(0, (1,)),
        _example(1, (2, 3)),
        _example(2, (4, 5, 1)),
    ]
    concept_ids, operations, _ = _batch(rows, "cpu")
    assert operations.tolist() == [[1, -1, -1], [2, 3, -1], [4, 5, 1]]

    batched = model.compose(concept_ids, operations)
    individual = torch.cat([
        model.compose(
            torch.tensor([row.concept], dtype=torch.long),
            torch.tensor([row.ops], dtype=torch.long),
        )
        for row in rows
    ])
    assert torch.allclose(batched, individual, atol=1e-7, rtol=1e-6)


def test_padding_must_be_a_contiguous_right_suffix():
    cfg = Config()
    model = LearnedOperationBrain(cfg)
    concepts = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="right suffix"):
        model.compose(concepts, torch.tensor([[1, -1, 2]], dtype=torch.long))
    with pytest.raises(ValueError, match="sentinel"):
        model.compose(concepts, torch.tensor([[1, -2, -1]], dtype=torch.long))
    with pytest.raises(ValueError, match="vocabulary"):
        model.compose(concepts, torch.tensor([[cfg.operations]], dtype=torch.long))


def test_seeded_training_initialization_and_world_are_reproducible():
    cfg = Config(epochs=1)
    seed_everything(123)
    first = LearnedOperationBrain(cfg)
    first_world = make_world(cfg, 123)
    first_fingerprint = world_fingerprint(cfg, 123, first_world)

    seed_everything(123)
    second = LearnedOperationBrain(cfg)
    second_world = make_world(cfg, 123)
    second_fingerprint = world_fingerprint(cfg, 123, second_world)

    assert first_fingerprint == second_fingerprint
    training_only_change = replace(cfg, epochs=2, batch_size=64, lr=1e-3, state_dim=32, hidden_dim=64)
    assert world_fingerprint(training_only_change, 123, first_world) == first_fingerprint
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in first.state_dict())


def test_evaluator_reports_heldout_ranking_order_and_collapse_controls():
    cfg = Config()
    world = make_world(cfg, 29)
    model = LearnedOperationBrain(cfg)
    metrics = evaluate(model, world, cfg, "cpu")
    assert metrics["count"] > 1
    assert metrics["chance_accuracy"] == pytest.approx(1 / metrics["count"])
    assert metrics["order_control_count"] > 0
    assert metrics["order_violation_rate"] is not None
    assert metrics["collapse_similarity"] is not None
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_training_loop_runs_with_masked_variable_length_batches():
    cfg = Config(
        state_dim=16,
        op_dim=8,
        hidden_dim=24,
        concepts=6,
        operations=6,
        concept_dim=16,
        max_chain=4,
        epochs=2,
        batch_size=17,
    )
    model, world, history = train_model(cfg, 31, "cpu", log_every=1)
    assert len(history) == 2
    assert len([row for row in world if row.held_out]) > 1
    assert torch.isfinite(next(model.parameters())).all()
