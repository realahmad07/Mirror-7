"""Autoregressive decoding (with KV cache) and chat helper for mirror_lm."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .data import encode_chat
from .model import ModelConfig, Transformer
from .tokenizer import BPETokenizer


@torch.no_grad()
def sample_tokens(
    model: Transformer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Generate up to ``max_new_tokens`` new token ids using the KV cache."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    max_ctx = model.config.max_seq_len
    prompt_ids = list(prompt_ids)[-(max_ctx - 1) :]
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")

    cache = model.new_cache()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _ = model(idx, cache=cache)
    generated: list[int] = []
    seen = list(prompt_ids)

    for _ in range(max_new_tokens):
        if cache.seq_len >= max_ctx:
            break
        next_logits = logits[0, -1].float().clone()
        if repetition_penalty != 1.0 and seen:
            uniq = torch.tensor(sorted(set(seen)), dtype=torch.long, device=device)
            vals = next_logits[uniq]
            next_logits[uniq] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        if temperature <= 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            scaled = next_logits / temperature
            if top_k > 0:
                k = min(top_k, scaled.numel())
                cutoff = torch.topk(scaled, k).values[-1]
                scaled = torch.where(scaled < cutoff, torch.full_like(scaled, -float("inf")), scaled)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                drop = cum - probs > top_p
                sorted_logits = torch.where(drop, torch.full_like(sorted_logits, -float("inf")), sorted_logits)
                scaled = torch.empty_like(scaled).scatter_(0, sorted_idx, sorted_logits)
            probs = F.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        if stop_ids and next_id in stop_ids:
            break
        generated.append(next_id)
        seen.append(next_id)
        step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, _ = model(step, cache=cache)

    if was_training:
        model.train()
    return generated


def generate_text(
    model: Transformer,
    tok: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
) -> str:
    ids = tok.encode(prompt, add_bos=True)
    out = sample_tokens(
        model,
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_ids={tok.eos_id, tok.end_id},
    )
    return tok.decode(out, skip_special=True)


def chat(
    model: Transformer,
    tok: BPETokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    prompt_ids, _ = encode_chat(tok, messages, add_generation_prompt=True)
    out = sample_tokens(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_ids={tok.end_id, tok.eos_id},
    )
    return tok.decode(out, skip_special=True).strip()


def load_for_inference(checkpoint_path: str | Path, device: str = "cpu") -> tuple[Transformer, BPETokenizer]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.from_dict(ckpt["tokenizer"])
    return model, tok
