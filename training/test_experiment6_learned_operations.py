import torch
from .experiment6_learned_operations import Config, LearnedOperationBrain, make_world, _batch

def test_exp6_forward_and_world():
    cfg = Config()
    model = LearnedOperationBrain(cfg)
    rows = make_world(cfg, 42)[:4]
    c, ops, target = _batch(rows, "cpu")
    out = model.compose(c, ops)
    assert out.shape == (4, cfg.state_dim)
    assert target.shape == out.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(target).all()

def test_exp6_has_learned_operation_parameters():
    cfg = Config()
    model = LearnedOperationBrain(cfg)
    assert model.operation.weight.shape == (cfg.operations, cfg.op_dim)
    assert sum(p.numel() for p in model.parameters()) < 500_000
