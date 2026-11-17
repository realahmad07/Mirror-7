"""Decoder-only transformer for mirror_lm (pure PyTorch).

Llama-style architecture: pre-norm RMSNorm, rotary position embeddings (RoPE),
SwiGLU MLP, grouped-query attention (GQA) via ``F.scaled_dot_product_attention``,
tied input/output embeddings, KV-cache for incremental decoding and
GPT-2/Llama-style scaled residual initialisation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    ffn_hidden: int = 352
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**d)


# Presets.  ``vocab_size`` here is the *intended* vocabulary; when training with
# a concrete tokenizer, override it with ``tokenizer.vocab_size``.
PRESETS: dict[str, ModelConfig] = {
    # ~1M params: comparable to the ~0.92-0.96M byte-level GRU V5 model.
    "nano": ModelConfig(vocab_size=2048, d_model=128, n_layers=4, n_heads=4,
                        n_kv_heads=2, ffn_hidden=352, max_seq_len=256),
    # ~10M params
    "micro": ModelConfig(vocab_size=8192, d_model=320, n_layers=6, n_heads=8,
                         n_kv_heads=4, ffn_hidden=864, max_seq_len=512),
    # ~125M params (GPT-2-small class; MHA, 12 x 768)
    "small": ModelConfig(vocab_size=32768, d_model=768, n_layers=12, n_heads=12,
                         n_kv_heads=12, ffn_hidden=2560, max_seq_len=1024),
    # ~350M params (GPT-2-medium class; 24 x 1024, GQA 16/8)
    "medium": ModelConfig(vocab_size=32768, d_model=1024, n_layers=24, n_heads=16,
                          n_kv_heads=8, ffn_hidden=3072, max_seq_len=2048),
    # ~1.1B params (TinyLlama-1.1B shape: 22 x 2048, GQA 32/4, SwiGLU 5632, untied head)
    "base-1b": ModelConfig(vocab_size=32000, d_model=2048, n_layers=22, n_heads=32,
                           n_kv_heads=4, ffn_hidden=5632, max_seq_len=2048,
                           tie_embeddings=False),
}


def get_preset(name: str, **overrides: Any) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(PRESETS[name], **overrides)


def estimate_params(cfg: ModelConfig) -> int:
    """Analytic parameter count (must equal ``Transformer(cfg).num_parameters()``)."""
    d, hd = cfg.d_model, cfg.head_dim
    attn = d * (cfg.n_heads * hd) + 2 * d * (cfg.n_kv_heads * hd) + (cfg.n_heads * hd) * d
    mlp = 3 * d * cfg.ffn_hidden
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = cfg.vocab_size * d
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * d
    return emb + cfg.n_layers * per_layer + d + head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_cache(seq_len: int, head_dim: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd); cos/sin: (T, hd/2). Rotate-half convention."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class KVCache:
    """Per-layer key/value cache: list of (k, v) tensors shaped (B, n_kv_heads, T, hd)."""

    def __init__(self, n_layers: int) -> None:
        self.layers: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    @property
    def seq_len(self) -> int:
        first = self.layers[0]
        return 0 if first is None else first[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev = self.layers[layer]
        if prev is not None:
            k = torch.cat([prev[0], k], dim=2)
            v = torch.cat([prev[1], v], dim=2)
        self.layers[layer] = (k, v)
        return k, v


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, layer_idx: int, cache: KVCache | None, start_pos: int):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        S = k.shape[2]
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        dp = self.dropout if self.training else 0.0
        if S == T:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        else:
            # queries at absolute positions S-T..S-1 attend to keys 0..q_pos
            q_pos = torch.arange(S - T, S, device=x.device).unsqueeze(1)
            k_pos = torch.arange(S, device=x.device).unsqueeze(0)
            mask = k_pos <= q_pos
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.hd)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, layer_idx, cache, start_pos):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, layer_idx, cache, start_pos))
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.config.init_std
        resid_std = std / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue  # RMSNorm weights stay at 1
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def new_cache(self) -> KVCache:
        return KVCache(self.config.n_layers)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        start_pos: int | None = None,
        reduction: str = "mean",
    ):
        """idx: (B, T) token ids.  targets: (B, T) next-token ids, -100 = ignore.

        Returns ``(logits, loss)``; loss is ``None`` without targets.  With
        ``reduction='none'`` the loss is per-token (B, T) (0 where ignored).
        """
        B, T = idx.shape
        if start_pos is None:
            start_pos = cache.seq_len if cache is not None else 0
        if start_pos + T > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {start_pos + T} exceeds max_seq_len {self.config.max_seq_len}"
            )
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        x = self.drop(self.tok_emb(idx))
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, i, cache, start_pos)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            if reduction == "none":
                loss = loss.view(B, T)
        return logits, loss


def preset_param_table(vocab_override: int | None = None) -> list[dict[str, Any]]:
    """Parameter counts for every preset (analytic, verified against meta-device instantiation in tests)."""
    rows = []
    for name, cfg in PRESETS.items():
        if vocab_override:
            cfg = replace(cfg, vocab_size=vocab_override)
        rows.append({"preset": name, "params": estimate_params(cfg), **cfg.to_dict()})
    return rows
