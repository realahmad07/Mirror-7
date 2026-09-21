import torch
from .model import require_torch
from .native_brain import NativeBrainConfig, build_native_brain, _batch, _encode

def test_native_brain_forward_and_state_shape():
    model = build_native_brain(NativeBrainConfig())
    ids, mask = _batch([_encode("hello mirror",64), _encode("reason about this",64)],64)
    state, predicted = model(ids, mask)
    assert state.shape == (2,128)
    assert predicted.shape == (2,128)
    assert torch.isfinite(state).all()
    assert torch.isfinite(predicted).all()

def test_native_brain_is_small():
    model = build_native_brain(NativeBrainConfig())
    assert sum(p.numel() for p in model.parameters()) < 1_500_000
    assert model.embedding.num_embeddings == 257
