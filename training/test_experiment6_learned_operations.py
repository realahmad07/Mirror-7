import torch
import pytest
from dataclasses import replace

from .experiment6_learned_operations import (
    Config,
    Example,
    LearnedOperationBrain,
    _batch,
    evaluate,
    derive_model_initialization_seed,
    loss_fn,
    _deduplicate_examples,
    make_world,
    seed_everything,
    train_model,
    transition_supervision_loss,
    world_fingerprint,
)


def _example(concept: int, ops: tuple[int, ...]) -> Example:
    return Example(concept, ops, torch.nn.functional.normalize(torch.randn(64), dim=0), False)



def test_duplicate_tasks_are_deduplicated_before_training():
    cfg = Config()
    world = make_world(cfg, 101)
    unique = _deduplicate_examples(world)
    assert len(unique) == len({(row.concept, row.ops) for row in world})
    assert len(unique) < len(world)


def test_duplicate_targets_are_not_used_as_contrastive_negatives():
    target = torch.nn.functional.normalize(torch.tensor([
        [1.0,  0.0],
        [1.0,  0.0],
        [0.0,  1.0],
        [0.0,  1.0],
    ]), dim=-1)
    prediction = target.clone().requires_grad_(True)

    loss = loss_fn(prediction, target, target, margin=0.20)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_loss_uses_hardest_distinct_negative_candidate():
    target = torch.tensor([[1.0, 0.0]])
    prediction = target.clone()
    candidates = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.0],   # The positive itself must be masked.
        [0.8, 0.6],   # Hardest valid negative: score 0.8.
        [0.0, 1.0],
    ]), dim=-1)

    loss = loss_fn(prediction, target, candidates, margin=0.30)
    assert loss.item() == pytest.approx(0.10, abs=1e-5)

    downweighted = loss_fn(
        prediction, target, candidates, margin=0.30, contrastive_weight=0.10,
    )
    assert downweighted.item() == pytest.approx(0.01, abs=1e-5)


def test_world_contains_a_complete_teacher_forced_transition_trace():
    cfg = Config(
        state_dim=12,
        concept_dim=12,
        concepts=6,
        operations=6,
        max_chain=4,
    )
    world = make_world(cfg, 17)
    assert world
    for row in world:
        assert len(row.intermediate_states) == len(row.ops)
        assert torch.allclose(row.intermediate_states[-1], row.target)
        assert all(torch.isfinite(state).all() for state in row.intermediate_states)


def test_intermediate_transition_supervision_produces_finite_gradients():
    cfg = Config(
        state_dim=12,
        concept_dim=12,
        concepts=6,
        operations=6,
        max_chain=4,
        transition_architecture="linear_operator",
    )
    model = LearnedOperationBrain(cfg)
    rows = [row for row in make_world(cfg, 23) if not row.held_out][:24]
    loss = transition_supervision_loss(model, rows, "cpu")
    assert torch.isfinite(loss)
    loss.backward()
    assert model.operation_matrices.grad is not None
    assert torch.isfinite(model.operation_matrices.grad).all()
    assert model.operation_matrices.grad.abs().sum() > 0


def test_transition_supervision_rejects_missing_intermediate_targets():
    cfg = Config()
    model = LearnedOperationBrain(cfg)
    with pytest.raises(ValueError, match="one intermediate target"):
        transition_supervision_loss(model, [_example(0, (1, 2))], "cpu")


def test_linear_operator_architecture_matches_declared_transition_contract():
    cfg = Config(
        state_dim=16,
        concept_dim=16,
        concepts=6,
        operations=3,
        max_chain=3,
        transition_architecture="linear_operator",
    )
    model = LearnedOperationBrain(cfg).eval()
    state = torch.nn.functional.normalize(torch.randn(2, cfg.state_dim), dim=-1)
    ops = torch.tensor([0, 1], dtype=torch.long)
    output = model.apply(state, ops)
    assert output.shape == state.shape
    assert torch.isfinite(output).all()
    assert model.operation_matrices.shape == (cfg.operations, cfg.state_dim, cfg.state_dim)


def test_operator_model_initialization_is_independent_of_hidden_world_matrices():
    cfg = Config(
        state_dim=12,
        concept_dim=12,
        concepts=5,
        operations=4,
        max_chain=3,
        epochs=1,
        transition_architecture="linear_operator",
    )
    seed = 17
    generator = torch.Generator().manual_seed(seed)
    torch.randn(cfg.concepts, cfg.concept_dim, generator=generator)
    hidden_matrices = []
    for operation_id in range(cfg.operations):
        matrix = torch.randn(
            cfg.concept_dim, cfg.concept_dim, generator=generator,
        ) * 0.35
        if operation_id % 2 == 0:
            matrix += torch.eye(cfg.concept_dim) * (0.15 + 0.03 * operation_id)
        hidden_matrices.append(matrix)

    model, _, _ = train_model(cfg, seed, "cpu")
    assert derive_model_initialization_seed(seed) != seed
    assert not torch.allclose(
        model.operation_matrices.detach(),
        torch.stack(hidden_matrices),
    )


def test_linear_operator_architecture_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="state_dim == concept_dim"):
        Config(
            state_dim=16,
            concept_dim=12,
            transition_architecture="linear_operator",
        )

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


def test_evaluator_can_measure_train_and_heldout_splits_separately():
    cfg = Config()
    world = make_world(cfg, 11)
    model = LearnedOperationBrain(cfg)
    train_metrics = evaluate(model, world, cfg, "cpu", split="train")
    heldout_metrics = evaluate(model, world, cfg, "cpu", split="held_out")
    assert train_metrics["split"] == "train"
    assert heldout_metrics["split"] == "held_out"
    assert train_metrics["count"] != heldout_metrics["count"]
    with pytest.raises(ValueError, match="split"):
        evaluate(model, world, cfg, "cpu", split="validation")


def test_evaluator_reports_heldout_ranking_order_and_collapse_controls():
    cfg = Config()
    world = make_world(cfg, 29)
    model = LearnedOperationBrain(cfg)
    metrics = evaluate(model, world, cfg, "cpu")
    first_held_out = next(row for row in world if row.held_out)
    duplicated_world = world + [
        Example(first_held_out.concept, first_held_out.ops, first_held_out.target.clone(), True)
    ]
    duplicate_metrics = evaluate(model, duplicated_world, cfg, "cpu")
    assert duplicate_metrics["count"] == metrics["count"]
    assert duplicate_metrics["accuracy"] == pytest.approx(metrics["accuracy"])
    assert duplicate_metrics["margin"] == pytest.approx(metrics["margin"])
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
