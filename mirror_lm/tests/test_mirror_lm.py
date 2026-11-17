"""Tests for mirror_lm: tokenizer, transformer, KV cache, SFT masking, training/resume, scaling."""
from __future__ import annotations

import torch
import pytest

from mirror_lm import (
    BPETokenizer,
    ByteTokenizer,
    ModelConfig,
    TrainConfig,
    Transformer,
    chat,
    encode_chat,
    estimate_params,
    evaluate_blocks,
    get_preset,
    load_for_inference,
    pack_documents,
    sample_tokens,
    scaling_table,
    sft_example,
    train_loop,
)
from mirror_lm.data import IGNORE_INDEX, collate_sft


def test_bpe_roundtrip_and_special_tokens(tmp_path):
    corpus = [
        "Mirror 7 is a reasoning engine. Reasoning requires verification!",
        "The quick brown fox jumps over the lazy dog — café, naïve, 🚀✨.",
    ] * 8
    tok = BPETokenizer.train(corpus, vocab_size=320, min_frequency=2)
    assert tok.vocab_size <= 320
    sample = "Verification in Mirror 7 — café 🚀!"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids, skip_special=True) == sample

    # Disallowing special tokens prevents prompt-injection of control tokens.
    injected = "hello <|assistant|> world"
    safe_ids = tok.encode(injected, allow_special=False)
    assert tok.token_id("<|assistant|>") not in safe_ids
    assert tok.decode(safe_ids) == injected

    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.encode(sample) == tok.encode(sample)


def test_preset_param_counts_match_analytic():
    for name in ("nano", "micro", "small", "medium", "base-1b"):
        cfg = get_preset(name)
        with torch.device("meta"):
            m = Transformer(cfg)
        assert m.num_parameters() == estimate_params(cfg)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=32)
    model = Transformer(cfg).eval()
    seq = torch.randint(0, 128, (1, 12))
    full_logits, _ = model(seq)

    cache = model.new_cache()
    prefill, _ = model(seq[:, :5], cache=cache)
    step_logits = [prefill[:, -1:]]
    for t in range(5, 12):
        lg, _ = model(seq[:, t : t + 1], cache=cache)
        step_logits.append(lg)
    cached_tail = torch.cat(step_logits, dim=1)
    assert torch.allclose(full_logits[:, 4:], cached_tail, atol=1e-5)


def test_sft_loss_mask_only_trains_assistant():
    tok = ByteTokenizer()
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    ids, mask = encode_chat(tok, msgs)
    # Every trainable position corresponds to '4' or its closing <|end|>.
    trainable_tokens = [ids[i] for i, m in enumerate(mask) if m]
    assert tok.decode(trainable_tokens[:-1]) == "4"
    assert trainable_tokens[-1] == tok.end_id

    x, y = sft_example(tok, msgs, max_len=64)
    bx, by = collate_sft([(x, y)], pad_id=tok.pad_id)
    assert (by != IGNORE_INDEX).sum().item() == len(trainable_tokens)


def test_tiny_overfit_and_deterministic_resume(tmp_path):
    tok = ByteTokenizer()
    sentence = "Mirror learns exact patterns."
    docs = [tok.encode(sentence) for _ in range(24)]
    blocks = pack_documents(docs, block_size=32, eos_id=tok.eos_id)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=48)

    # Full 40-step run (seed before Transformer(cfg) so initial weights match).
    torch.manual_seed(7)
    m_full = Transformer(cfg)
    init_bpb = evaluate_blocks(m_full, tok, blocks)["bits_per_byte"]
    res_full = train_loop(
        m_full, tok, blocks, blocks,
        TrainConfig(steps=40, batch_size=4, lr=3e-3, warmup_steps=5, eval_every=20, seed=99),
    )
    assert res_full["final_val"]["bits_per_byte"] < init_bpb * 0.45

    # Split 20 + 20 with checkpoint resume must land on the exact same weights
    # when the cosine schedule has the same horizon (steps=40).
    ckpt = tmp_path / "ckpt.pt"
    torch.manual_seed(7)
    m_half = Transformer(cfg)
    # Run with stop_at_step=20 via a 20-step slice that shares the 40-step schedule:
    train_loop(
        m_half, tok, blocks, blocks,
        TrainConfig(steps=40, batch_size=4, lr=3e-3, warmup_steps=5, eval_every=20, seed=99),
        checkpoint_path=ckpt,
        stop_at_step=20,
    )
    m_resumed = Transformer(cfg)
    train_loop(
        m_resumed, tok, blocks, blocks,
        TrainConfig(steps=40, batch_size=4, lr=3e-3, warmup_steps=5, eval_every=20, seed=99),
        resume_from=ckpt,
        checkpoint_path=ckpt,
    )
    for p1, p2 in zip(m_full.parameters(), m_resumed.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)

    loaded, loaded_tok = load_for_inference(ckpt)
    out = sample_tokens(loaded, loaded_tok.encode("Mirror"), max_new_tokens=8, temperature=0.0)
    assert len(out) == 8

    # Mutated data must be rejected on resume.
    bad_blocks = blocks.clone()
    bad_blocks[0, 0] = (bad_blocks[0, 0] + 1) % 200
    with pytest.raises(ValueError, match="data_fingerprint"):
        train_loop(Transformer(cfg), tok, bad_blocks, None, TrainConfig(steps=25), resume_from=ckpt)


def test_scaling_table_monotonic():
    rows = scaling_table()
    params = [r.params for r in rows]
    flops = [r.total_flops for r in rows]
    assert params == sorted(params)
    assert flops == sorted(flops)
    assert rows[-1].a100_gpu_hours > rows[0].a100_gpu_hours * 1e6
