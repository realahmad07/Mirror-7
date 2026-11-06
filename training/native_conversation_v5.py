from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from training.dataset import TrainingExample, load_jsonl
from training.model import require_torch
from training.native_brain import PAD, VOCAB_SIZE, _batch, _encode

BOS = VOCAB_SIZE
EOS = VOCAB_SIZE + 1
TOTAL_VOCAB_SIZE = VOCAB_SIZE + 2


def _mask_fill_value(tensor: torch.Tensor) -> float:
    """Return a safe negative fill value that never overflows float16/bfloat16 under GPU or CPU AMP."""
    return -1e4 if tensor.dtype in (torch.float16, torch.bfloat16) else -1e9


@dataclass(frozen=True)
class NativeConversationV5Config:
    embedding_dim: int = 96
    hidden_dim: int = 256
    state_dim: int = 128
    attention_dim: int = 128
    dropout: float = 0.10
    max_bytes: int = 512
    bidirectional_encoder: bool = False
    num_heads: int = 1
    context_gating: bool = False
    scheduled_sampling_p: float = 0.0
    # CPU future-proofing & parameter-scaling extensions (default off for 100% V5 compatibility)
    checkpoint_segment_size: int = 0
    num_moe_experts: int = 0
    moe_top_k: int = 2
    byte_patch_size: int = 1
    memory_bank_slots: int = 0
    decoder_self_attn: bool = False

    def to_dict(self):
        return asdict(self)


def make_v5_enhanced_config(
    embedding_dim: int = 96,
    hidden_dim: int = 256,
    state_dim: int = 128,
    attention_dim: int = 128,
    dropout: float = 0.10,
    max_bytes: int = 512,
    num_heads: int = 4,
    scheduled_sampling_p: float = 0.0,
    checkpoint_segment_size: int = 0,
    num_moe_experts: int = 0,
    moe_top_k: int = 2,
    byte_patch_size: int = 1,
    memory_bank_slots: int = 0,
) -> NativeConversationV5Config:
    """Factory for the improved V5 architecture with BiGRU, MHA, and Context Gating."""
    return NativeConversationV5Config(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        max_bytes=max_bytes,
        bidirectional_encoder=True,
        num_heads=num_heads,
        context_gating=True,
        scheduled_sampling_p=scheduled_sampling_p,
        checkpoint_segment_size=checkpoint_segment_size,
        num_moe_experts=num_moe_experts,
        moe_top_k=moe_top_k,
        byte_patch_size=byte_patch_size,
        memory_bank_slots=memory_bank_slots,
    )


def make_v5_cpu_scaled_config(
    embedding_dim: int = 96,
    hidden_dim: int = 256,
    state_dim: int = 128,
    attention_dim: int = 128,
    dropout: float = 0.10,
    max_bytes: int = 1024,
    num_heads: int = 4,
    checkpoint_segment_size: int = 128,
    num_moe_experts: int = 8,
    moe_top_k: int = 2,
    byte_patch_size: int = 2,
    memory_bank_slots: int = 4096,
) -> NativeConversationV5Config:
    """Factory for scaling V5 to millions of parameters while keeping CPU FLOPs and RAM low."""
    return make_v5_enhanced_config(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        max_bytes=max_bytes,
        num_heads=num_heads,
        checkpoint_segment_size=checkpoint_segment_size,
        num_moe_experts=num_moe_experts,
        moe_top_k=moe_top_k,
        byte_patch_size=byte_patch_size,
        memory_bank_slots=memory_bank_slots,
    )


MODEL_SCALE_PRESETS: dict[str, dict[str, Any]] = {
    "base": {
        "embedding_dim": 96,
        "hidden_dim": 256,
        "state_dim": 128,
        "attention_dim": 128,
        "num_heads": 4,
        "num_moe_experts": 0,
        "moe_top_k": 2,
        "memory_bank_slots": 0,
        "byte_patch_size": 1,
        "checkpoint_segment_size": 0,
    },
    "cpu-scaled": {
        "embedding_dim": 96,
        "hidden_dim": 256,
        "state_dim": 128,
        "attention_dim": 128,
        "num_heads": 4,
        "num_moe_experts": 8,
        "moe_top_k": 2,
        "memory_bank_slots": 4096,
        "byte_patch_size": 2,
        "checkpoint_segment_size": 128,
    },
    "medium": {
        "embedding_dim": 160,
        "hidden_dim": 512,
        "state_dim": 256,
        "attention_dim": 256,
        "num_heads": 8,
        "num_moe_experts": 8,
        "moe_top_k": 2,
        "memory_bank_slots": 4096,
        "byte_patch_size": 1,
        "checkpoint_segment_size": 0,
    },
    "a100-pro": {
        "embedding_dim": 256,
        "hidden_dim": 768,
        "state_dim": 384,
        "attention_dim": 384,
        "num_heads": 8,
        "num_moe_experts": 8,
        "moe_top_k": 2,
        "memory_bank_slots": 8192,
        "byte_patch_size": 1,
        "checkpoint_segment_size": 0,
    },
    "a100-max": {
        "embedding_dim": 384,
        "hidden_dim": 1024,
        "state_dim": 512,
        "attention_dim": 512,
        "num_heads": 8,
        "num_moe_experts": 16,
        "moe_top_k": 2,
        "memory_bank_slots": 16384,
        "byte_patch_size": 1,
        "checkpoint_segment_size": 256,
        "decoder_self_attn": False,
    },
    "a100-ultra": {
        "embedding_dim": 512,
        "hidden_dim": 1536,
        "state_dim": 768,
        "attention_dim": 768,
        "num_heads": 12,
        "num_moe_experts": 16,
        "moe_top_k": 2,
        "memory_bank_slots": 32768,
        "byte_patch_size": 1,
        "checkpoint_segment_size": 256,
        "decoder_self_attn": True,
    },
}


def make_v5_preset_config(
    scale: str = "base",
    max_bytes: int = 1024,
    dropout: float = 0.10,
    bidirectional_encoder: bool = True,
    context_gating: bool = True,
    scheduled_sampling_p: float = 0.0,
    **overrides: Any,
) -> NativeConversationV5Config:
    """Create a NativeConversationV5Config from a named parameter-scaling preset (`base`, `medium`, `a100-pro`, `a100-max`)."""
    preset_key = scale.strip().lower()
    if preset_key not in MODEL_SCALE_PRESETS:
        valid = ", ".join(sorted(MODEL_SCALE_PRESETS))
        raise ValueError(f"Unknown model scale preset '{scale}'. Valid presets: {valid}")
    params = dict(MODEL_SCALE_PRESETS[preset_key])
    for k, v in overrides.items():
        if v is not None:
            params[k] = v
    return NativeConversationV5Config(
        embedding_dim=int(params["embedding_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        state_dim=int(params["state_dim"]),
        attention_dim=int(params["attention_dim"]),
        dropout=float(dropout),
        max_bytes=int(max_bytes),
        bidirectional_encoder=bool(bidirectional_encoder),
        num_heads=int(params["num_heads"]),
        context_gating=bool(context_gating),
        scheduled_sampling_p=float(scheduled_sampling_p),
        checkpoint_segment_size=int(params["checkpoint_segment_size"]),
        num_moe_experts=int(params["num_moe_experts"]),
        moe_top_k=int(params["moe_top_k"]),
        byte_patch_size=int(params["byte_patch_size"]),
        memory_bank_slots=int(params["memory_bank_slots"]),
        decoder_self_attn=bool(params.get("decoder_self_attn", False)),
    )


class KeyValueMemoryBank(nn.Module):
    """Constant-time O(1) lookup memory bank queried once per turn by the pooled state vector.

    Adds large factual parameter capacity using RAM embedding gathers rather than per-byte GEMMs.
    """

    def __init__(self, state_dim: int, num_slots: int, top_k: int = 8):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_slots = int(num_slots)
        self.top_k = min(max(1, int(top_k)), self.num_slots)
        self.mem_keys = nn.Parameter(
            torch.randn(self.num_slots, self.state_dim) * (1.0 / math.sqrt(self.state_dim))
        )
        self.mem_values = nn.Embedding(self.num_slots, self.state_dim)
        nn.init.normal_(self.mem_values.weight, mean=0.0, std=0.02)
        self.mem_gate = nn.Linear(self.state_dim * 2, self.state_dim)
        nn.init.constant_(self.mem_gate.bias, -2.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(state, self.mem_keys.t()) / math.sqrt(self.state_dim)
        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=-1)
        weights = torch.softmax(top_scores, dim=-1).unsqueeze(-1)
        retrieved_vals = self.mem_values(top_indices)
        mem_summary = (weights * retrieved_vals).sum(dim=1)
        gate = torch.sigmoid(self.mem_gate(torch.cat([state, mem_summary], dim=-1)))
        return torch.tanh(state + gate * mem_summary)


class SparseMoEBlock(nn.Module):
    """Sparse Top-K Mixture-of-Experts layer operating outside the sequential recurrence loop.

    Scales parameter capacity by 4x-16x while activating only ``top_k`` experts per token
    and dispatching only the tokens routed to each active expert.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int = 2, hidden_mult: int = 2):
        super().__init__()
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.top_k = min(max(1, int(top_k)), self.num_experts)
        inner_dim = self.dim * max(1, int(hidden_mult))
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim, inner_dim),
                    nn.GELU(),
                    nn.Linear(inner_dim, self.dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        for expert in self.experts:
            nn.init.normal_(expert[2].weight, mean=0.0, std=0.01)
            nn.init.zeros_(expert[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        router_logits = self.router(flat_x)
        top_vals, top_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_weights = torch.softmax(top_vals, dim=-1, dtype=flat_x.dtype)

        moe_out = torch.zeros_like(flat_x)
        for e_idx, expert in enumerate(self.experts):
            match = top_idx == e_idx
            if not bool(match.any()):
                continue
            token_indices, slot_indices = torch.where(match)
            selected_inp = flat_x.index_select(0, token_indices)
            expert_out = expert(selected_inp).to(dtype=flat_x.dtype)
            weighted_out = expert_out * top_weights[token_indices, slot_indices].unsqueeze(-1)
            moe_out.index_add_(0, token_indices, weighted_out)

        return (flat_x + moe_out).reshape(orig_shape)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear layers to enable fast, low-memory CPU fine-tuning."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(
            torch.empty((self.rank, self.in_features), device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((self.out_features, self.rank), device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return (self.base.weight.data + delta.to(dtype=self.base.weight.dtype)).clone()

    def merged_linear(self) -> nn.Linear:
        """Fold LoRA adapter into the underlying nn.Linear in-place and return it."""
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.data.add_(delta.to(dtype=self.base.weight.dtype))
        return self.base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = torch.nn.functional.linear(
            torch.nn.functional.linear(self.lora_dropout(x), self.lora_A),
            self.lora_B,
        )
        return base_out + lora_out * self.scaling


class DecoderSelfAttentionRoPE(nn.Module):
    """Causal Target Self-Attention with Rotary Position Embeddings (RoPE) and Zero-Init Residual.

    Enables the recurrent GRU decoder to attend directly over all previously generated target bytes
    with exact relative position awareness (fixing multi-line code indentation, balanced brackets,
    and long multi-step mathematical derivations).
    Zero-initializing ``out_proj`` allows existing V5 checkpoints to be upgraded in-place without
    any initial loss spike.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, max_seq_len: int = 4096):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = max(1, int(num_heads))
        while self.hidden_dim % self.num_heads != 0 or (self.hidden_dim // self.num_heads) % 2 != 0:
            self.num_heads = max(1, self.num_heads // 2)
            if self.num_heads == 1:
                break
        self.head_dim = self.hidden_dim // self.num_heads
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        half_dim = max(1, self.head_dim // 2)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_dim, dtype=torch.float32) / float(half_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embeddings to x of shape (B, H, T, D)."""
        half = x.shape[-1] // 2
        if half == 0:
            return x
        freqs = torch.outer(positions.float(), self.inv_freq.to(x.device))  # (T, half)
        cos = freqs.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
        sin = freqs.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        x1 = x[..., :half]
        x2 = x[..., half : 2 * half]
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        if x.shape[-1] > 2 * half:
            rotated = torch.cat([rotated, x[..., 2 * half :]], dim=-1)
        return rotated

    def forward(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        """Parallel causal RoPE self-attention over target sequence (B, T, H)."""
        B, T, H = hidden_seq.shape
        u = self.norm(hidden_seq)
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        pos = torch.arange(T, device=hidden_seq.device)
        q = self._apply_rope(q, pos)
        k = self._apply_rope(k, pos)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
        return hidden_seq + self.out_proj(attn_out)

    def step(
        self,
        hidden_step: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """Single-step autoregressive RoPE self-attention using pre-allocated KV cache."""
        B, H = hidden_step.shape
        u = self.norm(hidden_step)
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        q = q.view(B, self.num_heads, 1, self.head_dim)
        k = k.view(B, self.num_heads, 1, self.head_dim)
        v = v.view(B, self.num_heads, 1, self.head_dim)
        pos = torch.tensor([step_idx], device=hidden_step.device)
        q = self._apply_rope(q, pos)
        k = self._apply_rope(k, pos)
        k_cache[:, :, step_idx : step_idx + 1, :] = k
        v_cache[:, :, step_idx : step_idx + 1, :] = v
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k_cache[:, :, : step_idx + 1, :], v_cache[:, :, : step_idx + 1, :], is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, H)
        return hidden_step + self.out_proj(attn_out)



class NativeConversationV5(nn.Module):
    """V5 Input-Conditioned Recurrent Decoder.

    Key Architectural Advancement over V3:
    In V3, the decoder GRU received only target byte embeddings and its recurrent
    hidden state drifted from the encoder context after 15-25 steps.
    In V5, at EVERY decoding step:
      1. Attention context c_t is computed over encoder states using previous hidden state h_{t-1}.
      2. The token embedding e_t and attention context c_t are concatenated: [e_t; c_t].
      3. The combined representation is fed directly into the decoder GRUCell to produce h_t.
      4. The recurrent state h_t directly incorporates the encoder context at every time step.
      5. The output distribution is predicted from [h_t; c_t].
    """

    def __init__(self, config: NativeConversationV5Config):
        torch_mod, nn_mod = require_torch()
        super().__init__()
        self.config = config

        # Encoder
        self.embedding = nn_mod.Embedding(
            TOTAL_VOCAB_SIZE, config.embedding_dim, padding_idx=PAD
        )
        if getattr(config, "byte_patch_size", 1) > 1:
            p = int(config.byte_patch_size)
            self.byte_patcher = nn_mod.Conv1d(
                config.embedding_dim, config.embedding_dim, kernel_size=p, stride=p, padding=0
            )
        if config.bidirectional_encoder:
            self.encoder = nn_mod.GRU(
                config.embedding_dim,
                config.hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.encoder = nn_mod.GRU(
                config.embedding_dim,
                config.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        self.encoder_norm = nn_mod.LayerNorm(config.hidden_dim)
        self.state = nn_mod.Linear(config.hidden_dim, config.state_dim)
        if getattr(config, "memory_bank_slots", 0) > 0:
            self.memory_bank = KeyValueMemoryBank(
                config.state_dim, int(config.memory_bank_slots)
            )
        self.decoder_init = nn_mod.Linear(config.state_dim, config.hidden_dim)

        # Cross-attention projections
        attn_dim = config.attention_dim
        self.query = nn_mod.Linear(config.hidden_dim, attn_dim, bias=False)
        self.key = nn_mod.Linear(config.hidden_dim, attn_dim, bias=False)
        self.value = nn_mod.Linear(config.hidden_dim, attn_dim, bias=False)
        if config.num_heads > 1:
            self.attn_out = nn_mod.Linear(attn_dim, attn_dim)
        self.context_norm = nn_mod.LayerNorm(attn_dim)

        # Context gating
        if config.context_gating:
            self.gate_linear = nn_mod.Linear(config.hidden_dim + attn_dim, attn_dim)

        # Input-conditioned Recurrent Decoder Cell
        self.decoder_embedding = nn_mod.Embedding(
            TOTAL_VOCAB_SIZE, config.embedding_dim, padding_idx=PAD
        )
        self.cell = nn_mod.GRUCell(
            config.embedding_dim + attn_dim, config.hidden_dim
        )
        if getattr(config, "decoder_self_attn", False):
            self.decoder_self_attn = DecoderSelfAttentionRoPE(
                config.hidden_dim, num_heads=max(1, int(config.num_heads))
            )

        # Output projection head (with optional Sparse MoE capacity scaling)
        self.output_norm = nn_mod.LayerNorm(config.hidden_dim + attn_dim)
        if getattr(config, "num_moe_experts", 0) > 1:
            self.moe_block = SparseMoEBlock(
                config.hidden_dim + attn_dim,
                num_experts=int(config.num_moe_experts),
                top_k=int(getattr(config, "moe_top_k", 2)),
            )
        self.head = nn_mod.Linear(
            config.hidden_dim + attn_dim, TOTAL_VOCAB_SIZE
        )

    def init_decoder_hidden(self, state: torch.Tensor) -> torch.Tensor:
        """Map compact semantic state into initial decoder recurrent state (batch, hidden_dim)."""
        return torch.tanh(self.decoder_init(state))

    def encode(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ):
        """Encode input sequence, fuse optional working-memory prev_state, and precompute K/V."""
        emb = self.embedding(input_ids)
        mask_bool = input_mask.bool()
        if getattr(self, "byte_patcher", None) is not None:
            p = int(self.config.byte_patch_size)
            seq_len = emb.shape[1]
            rem = seq_len % p
            if rem != 0:
                pad_len = p - rem
                emb = torch.nn.functional.pad(emb, (0, 0, 0, pad_len))
                mask_bool = torch.nn.functional.pad(mask_bool, (0, pad_len), value=False)
            emb = emb * mask_bool.to(dtype=emb.dtype).unsqueeze(-1)
            emb = self.byte_patcher(emb.transpose(1, 2)).transpose(1, 2)
            mask_bool = mask_bool.view(mask_bool.shape[0], -1, p).any(dim=-1)
            mask_float = mask_bool.to(dtype=emb.dtype)
        else:
            mask_float = input_mask.to(dtype=emb.dtype)

        if (
            self.config.bidirectional_encoder
            and not bool(mask_bool.all())
            and bool(mask_bool.any())
        ):
            # Pack padded sequences so trailing PAD tokens never contaminate the backward GRU
            lengths = mask_bool.sum(dim=1).long().clamp_min(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.encoder(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=emb.shape[1]
            )
        else:
            x, _ = self.encoder(emb)
        x = self.encoder_norm(x)
        mask = mask_float.to(dtype=x.dtype).unsqueeze(-1)
        x = x * mask
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        state = torch.tanh(self.state(pooled))

        if getattr(self, "memory_bank", None) is not None:
            state = self.memory_bank(state)

        # Recurrent state carry-over across chunks/sub-goals (working memory register)
        if prev_state is not None:
            prev_s = prev_state.to(device=state.device, dtype=state.dtype)
            if prev_s.shape[0] == 1 and state.shape[0] > 1:
                prev_s = prev_s.expand(state.shape[0], -1)
            if prev_s.shape == state.shape:
                state = torch.tanh(0.70 * state + 0.30 * prev_s)

        if self.config.bidirectional_encoder:
            # Direct O(1) gradient highway from cross-attention at every step t into state & decoder_init
            state_ctx = torch.tanh(self.decoder_init(state)).unsqueeze(1)
            x = (x + 0.15 * state_ctx) * mask

        # Precompute keys and values for efficiency
        k = self.key(x)    # (batch, enc_len, attn_dim)
        v = self.value(x)  # (batch, enc_len, attn_dim)
        return x, mask_bool, state, k, v

    def _split_multiscale_heads(
        self, k: torch.Tensor, v: torch.Tensor, encoder_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split K/V into fine byte-level heads and coarse word/clause-pooled heads."""
        batch_size, enc_len, _ = k.shape
        num_heads = self.config.num_heads
        head_dim = self.config.attention_dim // num_heads
        k_h = k.view(batch_size, enc_len, num_heads, head_dim).transpose(1, 2).contiguous()
        v_h = v.view(batch_size, enc_len, num_heads, head_dim).transpose(1, 2).contiguous()

        if self.config.bidirectional_encoder and num_heads >= 2 and enc_len >= 3:
            half = num_heads // 2
            coarse_heads = num_heads - half
            mask_f = encoder_mask.to(dtype=k.dtype).unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
            k_coarse = k_h[:, half:] * mask_f
            v_coarse = v_h[:, half:] * mask_f
            bh = batch_size * coarse_heads

            k_pool = torch.nn.functional.avg_pool1d(
                k_coarse.reshape(bh, enc_len, head_dim).transpose(1, 2),
                kernel_size=5,
                stride=1,
                padding=2,
                count_include_pad=False,
            ).transpose(1, 2).reshape(batch_size, coarse_heads, enc_len, head_dim)

            v_pool = torch.nn.functional.avg_pool1d(
                v_coarse.reshape(bh, enc_len, head_dim).transpose(1, 2),
                kernel_size=5,
                stride=1,
                padding=2,
                count_include_pad=False,
            ).transpose(1, 2).reshape(batch_size, coarse_heads, enc_len, head_dim)

            denom = (
                torch.nn.functional.avg_pool1d(
                    mask_f.reshape(batch_size, 1, enc_len),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    count_include_pad=False,
                )
                .unsqueeze(-1)
                .clamp_min(0.20)
            )
            k_h = torch.cat([k_h[:, :half], (k_pool / denom) * mask_f], dim=1)
            v_h = torch.cat([v_h[:, :half], (v_pool / denom) * mask_f], dim=1)

        return k_h, v_h

    def _prepare_attention_cache(
        self, k: torch.Tensor, v: torch.Tensor, encoder_mask: torch.Tensor
    ) -> tuple:
        """Precompute reshaped multiscale K/V heads and masks once per sequence for fast unrolling."""
        if self.config.num_heads > 1:
            k_h, v_h = self._split_multiscale_heads(k, v, encoder_mask)
            mask_expanded = encoder_mask.unsqueeze(1).unsqueeze(2)
            has_valid = mask_expanded.any(dim=-1, keepdim=True)
            return (k_h, v_h, mask_expanded, has_valid)
        else:
            mask_expanded = encoder_mask.unsqueeze(1)
            has_valid = mask_expanded.any(dim=-1, keepdim=True)
            return (k, v, mask_expanded, has_valid)

    def _step_attention(
        self,
        query_state: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        encoder_mask: torch.Tensor,
        _precomputed: tuple | None = None,
        need_weights: bool = True,
        _sdpa_mask: torch.Tensor | None = None,
    ):
        """Compute single-step cross-attention (with dual-scale byte + word/clause heads when num_heads > 1).

        When ``need_weights=False`` and ``self.use_fused_attention`` is enabled, the fused
        ``F.scaled_dot_product_attention`` kernel is used and ``None`` is returned for the weights.
        """
        fused = (not need_weights) and getattr(self, "use_fused_attention", True)
        if self.config.num_heads > 1:
            batch_size = query_state.shape[0]
            num_heads = self.config.num_heads
            head_dim = self.config.attention_dim // num_heads

            # Normalize query_state to match encoder_norm variance and prevent query/key gradient attenuation
            q_in = torch.nn.functional.layer_norm(query_state, (query_state.shape[-1],))
            q = self.query(q_in).view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
            if _precomputed is not None:
                k_h, v_h, mask_expanded, has_valid = _precomputed
            else:
                k_h, v_h = self._split_multiscale_heads(k, v, encoder_mask)
                mask_expanded = encoder_mask.unsqueeze(1).unsqueeze(2)
                has_valid = mask_expanded.any(dim=-1, keepdim=True)

            if fused:
                sdpa_mask = _sdpa_mask if _sdpa_mask is not None else (mask_expanded | ~has_valid)
                context_h = torch.nn.functional.scaled_dot_product_attention(
                    q, k_h, v_h, attn_mask=sdpa_mask
                )
                context_h = context_h * has_valid.to(dtype=context_h.dtype)
                context = context_h.transpose(1, 2).reshape(batch_size, self.config.attention_dim)
                if hasattr(self, "attn_out"):
                    context = self.attn_out(context)
                return self.context_norm(context), None

            scores = torch.matmul(q, k_h.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores.masked_fill(~mask_expanded, _mask_fill_value(scores))
            weights = torch.softmax(scores, dim=-1)
            weights = torch.where(has_valid, weights, torch.zeros_like(weights))

            context_h = torch.matmul(weights, v_h)
            context = context_h.transpose(1, 2).contiguous().view(batch_size, self.config.attention_dim)
            if hasattr(self, "attn_out"):
                context = self.attn_out(context)
            mean_weights = weights.mean(dim=1).squeeze(1)
            return self.context_norm(context), mean_weights
        else:
            q = self.query(query_state).unsqueeze(1)  # (batch, 1, attn_dim)
            if _precomputed is not None:
                _, _, mask_expanded, has_valid = _precomputed
            else:
                mask_expanded = encoder_mask.unsqueeze(1)  # (batch, 1, enc_len)
                has_valid = mask_expanded.any(dim=-1, keepdim=True)  # (batch, 1, 1)

            if fused:
                sdpa_mask = _sdpa_mask if _sdpa_mask is not None else (mask_expanded | ~has_valid)
                context = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=sdpa_mask
                ).squeeze(1)
                context = context * has_valid.squeeze(1).to(dtype=context.dtype)
                return self.context_norm(context), None

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # (batch, 1, enc_len)
            scores = scores.masked_fill(~mask_expanded, _mask_fill_value(scores))
            weights = torch.softmax(scores, dim=-1)
            weights = torch.where(has_valid, weights, torch.zeros_like(weights))

            context = torch.matmul(weights, v).squeeze(1)  # (batch, attn_dim)
            return self.context_norm(context), weights.squeeze(1)

    def _can_hoist_input_projection(self) -> bool:
        """Hoisting needs a float nn.GRUCell (default False because fused C++ nn.GRUCell is faster on CPU)."""
        return getattr(self, "hoist_input_projection", False) and isinstance(
            getattr(self.cell, "weight_ih", None), torch.Tensor
        )

    def _gru_step_hoisted(
        self,
        gi_emb_t: torch.Tensor,
        context: torch.Tensor,
        hidden: torch.Tensor,
        w_ih_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """GRUCell step whose token-embedding input projection was precomputed outside the loop.

        Mathematically identical to ``self.cell(cat([emb, context]), hidden)`` (PyTorch GRU equations).
        """
        gi = gi_emb_t + torch.nn.functional.linear(context, w_ih_ctx)
        gh = torch.nn.functional.linear(hidden, self.cell.weight_hh, self.cell.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * hidden

    def _project_output(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Project recurrent state and attention context to vocabulary logits (supports 2D or 3D)."""
        if hidden.dim() == 3 and getattr(self, "decoder_self_attn", None) is not None:
            hidden = self.decoder_self_attn(hidden)
        if getattr(self, "gate_linear", None) is not None:
            gate = torch.sigmoid(self.gate_linear(torch.cat([hidden, context], dim=-1)))
            eff_context = context * (0.5 + 0.5 * gate)
        else:
            eff_context = context
        fused = self.output_norm(torch.cat([hidden, eff_context], dim=-1))
        if getattr(self, "moe_block", None) is not None:
            fused = self.moe_block(fused)
        return self.head(fused)

    def _unroll_decoder_segment(
        self,
        hidden: torch.Tensor,
        seg_target_emb: torch.Tensor,
        seg_boundary_all: torch.Tensor | None,
        anchor_term: torch.Tensor | None,
        k: torch.Tensor,
        v: torch.Tensor,
        encoder_mask: torch.Tensor,
        attn_cache: tuple,
        sdpa_mask: torch.Tensor | None,
        seg_boundary_steps: list[bool] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll a slice of decoder steps; suitable for ``torch.utils.checkpoint`` activation checkpointing."""
        seg_len = seg_target_emb.shape[1]
        use_anchor = seg_boundary_all is not None and anchor_term is not None and seg_boundary_steps is not None
        h_list = []
        c_list = []
        for t in range(seg_len):
            context, _ = self._step_attention(
                hidden,
                k,
                v,
                encoder_mask,
                _precomputed=attn_cache,
                need_weights=False,
                _sdpa_mask=sdpa_mask,
            )
            cur_emb = seg_target_emb[:, t, :]
            cell_input = torch.cat([cur_emb, context], dim=-1)
            hidden = self.cell(cell_input, hidden)
            if use_anchor and seg_boundary_steps[t]:
                hidden = torch.where(
                    seg_boundary_all[:, t : t + 1], 0.90 * hidden + anchor_term, hidden
                )
            h_list.append(hidden)
            c_list.append(context)
        return hidden, torch.stack(h_list, dim=1), torch.stack(c_list, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        target_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
        prev_state: torch.Tensor | None = None,
        return_attentions: bool = True,
        checkpoint_segment_size: int | None = None,
    ):
        """Teacher-forced forward pass unrolling input-conditioned recurrence with boundary anchor refresh.

        ``return_attentions=False`` is the training fast path: the fused attention kernel is used,
        no attention tensor is stored, and ``None`` is returned in the 4th slot.
        When ``checkpoint_segment_size > 0`` (or configured on ``self.config``), activation
        checkpointing is applied per segment to cap CPU/GPU RAM on long sequences or large models.
        """
        _, encoder_mask, state, k, v = self.encode(
            input_ids, input_mask, prev_state=prev_state
        )
        h0_anchor = self.init_decoder_hidden(state)
        if hidden is None:
            hidden = h0_anchor

        target_emb = self.decoder_embedding(target_ids)  # (batch, seq_len, emb_dim)
        batch_size, seq_len, _ = target_emb.shape
        if seq_len == 0:
            empty_logits = torch.empty(
                (batch_size, 0, TOTAL_VOCAB_SIZE), device=input_ids.device, dtype=h0_anchor.dtype
            )
            empty_attn = (
                torch.empty(
                    (batch_size, 0, encoder_mask.shape[1]),
                    device=input_ids.device,
                    dtype=h0_anchor.dtype,
                )
                if return_attentions
                else None
            )
            return empty_logits, state, hidden, empty_attn

        attn_cache = self._prepare_attention_cache(k, v, encoder_mask)
        sdpa_mask = None
        if not return_attentions:
            _, _, mask_expanded, has_valid = attn_cache
            sdpa_mask = mask_expanded | ~has_valid

        use_ss = bool(self.training and self.config.scheduled_sampling_p > 0.0)
        use_anchor_refresh = bool(self.config.bidirectional_encoder)

        # Hoist work that does not depend on the recurrence out of the sequential loop.
        boundary_all = None
        boundary_steps = None
        anchor_term = None
        if use_anchor_refresh:
            boundary_all = (
                (target_ids == 46) | (target_ids == 10) | (target_ids == 33) | (target_ids == 63)
            )
            boundary_steps = boundary_all.any(dim=0).tolist()  # one host sync per batch
            anchor_term = 0.10 * h0_anchor

        seg_size = (
            int(checkpoint_segment_size)
            if checkpoint_segment_size is not None
            else int(getattr(self.config, "checkpoint_segment_size", 0))
        )
        if (
            seg_size > 0
            and self.training
            and torch.is_grad_enabled()
            and not use_ss
            and not return_attentions
            and seq_len > 0
        ):
            seg_hiddens = []
            seg_contexts = []
            for start in range(0, seq_len, seg_size):
                end = min(seq_len, start + seg_size)
                seg_emb = target_emb[:, start:end]
                seg_b_all = boundary_all[:, start:end] if boundary_all is not None else None
                seg_b_steps = boundary_steps[start:end] if boundary_steps is not None else None
                hidden, sh, sc = torch.utils.checkpoint.checkpoint(
                    self._unroll_decoder_segment,
                    hidden,
                    seg_emb,
                    seg_b_all,
                    anchor_term,
                    k,
                    v,
                    encoder_mask,
                    attn_cache,
                    sdpa_mask,
                    seg_b_steps,
                    use_reentrant=False,
                )
                seg_hiddens.append(sh)
                seg_contexts.append(sc)
            all_hidden = torch.cat(seg_hiddens, dim=1)
            all_context = torch.cat(seg_contexts, dim=1)
            all_logits = self._project_output(all_hidden, all_context)
            return all_logits, state, hidden, None

        hoist = (not use_ss) and self._can_hoist_input_projection()
        if hoist:
            emb_dim = self.config.embedding_dim
            w_ih = self.cell.weight_ih
            gi_emb_all = torch.nn.functional.linear(
                target_emb, w_ih[:, :emb_dim], self.cell.bias_ih
            )  # (batch, seq_len, 3 * hidden)
            w_ih_ctx = w_ih[:, emb_dim:].contiguous()

        hidden_list = []
        context_list = []
        logits_list = []
        attentions = []
        prev_pred_tok = None
        prev_confident = None

        for t in range(seq_len):
            # 1. Step attention from previous recurrent state
            context, attn_weights = self._step_attention(
                hidden,
                k,
                v,
                encoder_mask,
                _precomputed=attn_cache,
                need_weights=return_attentions,
                _sdpa_mask=sdpa_mask,
            )
            if return_attentions:
                attentions.append(attn_weights)

            # 2. Input-conditioned recurrent step: [emb; context] -> GRUCell
            if hoist:
                hidden = self._gru_step_hoisted(gi_emb_all[:, t], context, hidden, w_ih_ctx)
            else:
                tok_t = target_ids[:, t : t + 1]
                if use_ss and t > 0 and prev_pred_tok is not None and prev_confident is not None:
                    rand_mask = (
                        torch.rand(batch_size, 1, device=input_ids.device)
                        < self.config.scheduled_sampling_p
                    ) & prev_confident & (tok_t != PAD)
                    gt_emb = target_emb[:, t, :]
                    pred_emb = self.decoder_embedding(prev_pred_tok).squeeze(1)
                    cur_emb = torch.where(rand_mask, pred_emb, gt_emb)
                else:
                    cur_emb = target_emb[:, t, :]
                cell_input = torch.cat([cur_emb, context], dim=-1)
                hidden = self.cell(cell_input, hidden)

            # Periodic sentence-boundary anchor refresh to prevent long-horizon goal drift
            if use_anchor_refresh and boundary_steps[t]:
                hidden = torch.where(
                    boundary_all[:, t : t + 1], 0.90 * hidden + anchor_term, hidden
                )

            if use_ss:
                logits = self._project_output(hidden, context)
                logits_list.append(logits)
                probs_max, prev_pred_tok = (
                    logits.detach().softmax(dim=-1).max(dim=-1, keepdim=True)
                )
                prev_confident = probs_max > 0.55
            else:
                hidden_list.append(hidden)
                context_list.append(context)

        if use_ss:
            all_logits = torch.stack(logits_list, dim=1)
        else:
            all_hidden = torch.stack(hidden_list, dim=1)
            all_context = torch.stack(context_list, dim=1)
            all_logits = self._project_output(all_hidden, all_context)

        all_attentions = (
            torch.stack(attentions, dim=1) if return_attentions else None
        )  # (batch, seq_len, enc_len)
        return all_logits, state, hidden, all_attentions

    def generate(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        max_new_tokens: int = 512,
        min_new_tokens: int = 1,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        prev_state: torch.Tensor | None = None,
        enforce_valid_utf8: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation using identical input-conditioned recurrence and anchor refresh.

        Repetition penalty, no-repeat n-gram blocking, and optional RFC-3629 UTF-8 grammar
        state-machine constraints are fully vectorized over the batch.
        """
        self.eval()
        with torch.no_grad():
            _, encoder_mask, state, k, v = self.encode(
                input_ids, input_mask, prev_state=prev_state
            )
            h0_anchor = self.init_decoder_hidden(state)
            hidden = h0_anchor
            batch_size = input_ids.shape[0]
            device = input_ids.device
            attn_cache = self._prepare_attention_cache(k, v, encoder_mask)
            _, _, mask_expanded, has_valid = attn_cache
            sdpa_mask = mask_expanded | ~has_valid
            use_anchor_refresh = bool(self.config.bidirectional_encoder)
            anchor_term = 0.10 * h0_anchor
            dsa = getattr(self, "decoder_self_attn", None)
            if dsa is not None and max_new_tokens > 0:
                self_k_cache = torch.empty(
                    (batch_size, dsa.num_heads, max_new_tokens, dsa.head_dim),
                    dtype=h0_anchor.dtype,
                    device=device,
                )
                self_v_cache = torch.empty(
                    (batch_size, dsa.num_heads, max_new_tokens, dsa.head_dim),
                    dtype=h0_anchor.dtype,
                    device=device,
                )
            else:
                self_k_cache = None
                self_v_cache = None

            token = torch.full(
                (batch_size, 1),
                BOS,
                dtype=torch.long,
                device=device,
            )
            gen_buf = torch.full(
                (batch_size, max(0, max_new_tokens)), PAD, dtype=torch.long, device=device
            )
            n_gen = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            n = int(no_repeat_ngram_size)
            rep_excluded = torch.tensor([PAD, EOS, 32, 10], dtype=torch.long, device=device)

            if enforce_valid_utf8:
                utf8_rem = torch.zeros(batch_size, dtype=torch.long, device=device)
                utf8_next_min = torch.full((batch_size,), 128, dtype=torch.long, device=device)
                utf8_next_max = torch.full((batch_size,), 191, dtype=torch.long, device=device)
                vocab_idx = torch.arange(TOTAL_VOCAB_SIZE, device=device).unsqueeze(0)
                base_utf8_start = torch.zeros(TOTAL_VOCAB_SIZE, dtype=torch.bool, device=device)
                base_utf8_start[9] = True
                base_utf8_start[10] = True
                base_utf8_start[32:127] = True
                base_utf8_start[194:245] = True
                base_utf8_start[EOS] = True

            for step in range(max_new_tokens):
                # 1. Attention from previous recurrent state
                context, _ = self._step_attention(
                    hidden,
                    k,
                    v,
                    encoder_mask,
                    _precomputed=attn_cache,
                    need_weights=False,
                    _sdpa_mask=sdpa_mask,
                )

                # 2. Input-conditioned recurrent step: [emb; context] -> GRUCell
                emb = self.decoder_embedding(token).squeeze(1)  # (batch, emb_dim)
                cell_input = torch.cat([emb, context], dim=-1)
                hidden = self.cell(cell_input, hidden)

                if use_anchor_refresh:
                    is_boundary = (
                        (token == 46) | (token == 10) | (token == 33) | (token == 63)
                    )
                    hidden = torch.where(is_boundary, 0.90 * hidden + anchor_term, hidden)

                # 3. Predict logits (with optional RoPE causal target self-attention)
                if dsa is not None and self_k_cache is not None and self_v_cache is not None:
                    h_proj = dsa.step(hidden, self_k_cache, self_v_cache, step)
                else:
                    h_proj = hidden
                logits = self._project_output(h_proj, context)
                neg_inf = _mask_fill_value(logits)

                # Disallow PAD and BOS
                logits[:, PAD] = neg_inf
                logits[:, BOS] = neg_inf

                # Disallow early EOS if min_new_tokens requested
                if step < min_new_tokens:
                    logits[:, EOS] = neg_inf

                # Prevent runaway space-byte indentation loops (cap consecutive spaces at 6)
                if n_gen >= 6:
                    space_run = (gen_buf[:, n_gen - 6 : n_gen] == 32).all(dim=-1)
                    logits[space_run, 32] = neg_inf

                # Repetition penalty over the last 64 generated tokens (spelling-preserving)
                if repetition_penalty != 1.0 and n_gen > 0:
                    window = gen_buf[:, max(0, n_gen - 64) : n_gen]
                    seen = torch.zeros(logits.shape, dtype=torch.bool, device=device)
                    seen.scatter_(1, window, True)
                    seen[:, rep_excluded] = False
                    penalized = torch.where(
                        logits > 0, logits / repetition_penalty, logits * repetition_penalty
                    )
                    logits = torch.where(seen, penalized, logits)

                # No-repeat n-gram blocking
                if n > 1 and n_gen >= n:
                    hist = gen_buf[:, :n_gen]
                    windows = hist.unfold(1, n, 1)  # (batch, n_gen - n + 1, n)
                    prefix = hist[:, n_gen - (n - 1) :].unsqueeze(1)  # (batch, 1, n - 1)
                    match = (windows[:, :, : n - 1] == prefix).all(dim=-1)
                    ban_counts = torch.zeros(
                        logits.shape, dtype=torch.long, device=device
                    ).scatter_add_(1, windows[:, :, n - 1], match.long())
                    logits = logits.masked_fill(ban_counts > 0, neg_inf)

                # Vectorized RFC-3629 UTF-8 state machine grammar mask
                if enforce_valid_utf8:
                    rem_steps = max_new_tokens - 1 - step
                    allowed_start = base_utf8_start.clone()
                    if rem_steps < 3:
                        allowed_start[240:245] = False
                    if rem_steps < 2:
                        allowed_start[224:240] = False
                    if rem_steps < 1:
                        allowed_start[194:224] = False
                    if step < min_new_tokens:
                        allowed_start[EOS] = False

                    in_cont = (utf8_rem > 0).unsqueeze(1)
                    cont_mask = (vocab_idx >= utf8_next_min.unsqueeze(1)) & (
                        vocab_idx <= utf8_next_max.unsqueeze(1)
                    )
                    valid_tok_mask = torch.where(in_cont, cont_mask, allowed_start.unsqueeze(0))
                    logits = logits.masked_fill(~valid_tok_mask, neg_inf)

                    # Safety fallback if n-gram ban masked all valid options for any row
                    all_masked = (logits <= neg_inf).all(dim=-1, keepdim=True)
                    if bool(all_masked.any()):
                        fallback_logits = torch.where(valid_tok_mask, torch.zeros_like(logits), neg_inf)
                        logits = torch.where(all_masked, fallback_logits, logits)

                if temperature <= 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k and 0 < top_k < logits.shape[-1]:
                        values, indices = torch.topk(logits, top_k, dim=-1)
                        filtered = torch.full_like(logits, neg_inf)
                        filtered.scatter_(1, indices, values)
                        logits = filtered
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits = logits.masked_fill(indices_to_remove, neg_inf)
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                # Pad finished sequences
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, PAD),
                    next_token,
                )
                tok_1d = next_token.squeeze(-1)
                gen_buf[:, n_gen] = tok_1d
                n_gen += 1

                if enforce_valid_utf8:
                    was_cont = utf8_rem > 0
                    new_rem = torch.where(was_cont, utf8_rem - 1, torch.zeros_like(utf8_rem))
                    new_min = torch.full_like(utf8_next_min, 128)
                    new_max = torch.full_like(utf8_next_max, 191)
                    is_2byte = (~was_cont) & (tok_1d >= 194) & (tok_1d <= 223)
                    is_3byte = (~was_cont) & (tok_1d >= 224) & (tok_1d <= 239)
                    is_4byte = (~was_cont) & (tok_1d >= 240) & (tok_1d <= 244)
                    new_rem = torch.where(is_2byte, torch.ones_like(new_rem), new_rem)
                    new_rem = torch.where(is_3byte, torch.full_like(new_rem, 2), new_rem)
                    new_rem = torch.where(is_4byte, torch.full_like(new_rem, 3), new_rem)
                    new_min = torch.where((~was_cont) & (tok_1d == 224), torch.full_like(new_min, 160), new_min)
                    new_max = torch.where((~was_cont) & (tok_1d == 237), torch.full_like(new_max, 159), new_max)
                    new_min = torch.where((~was_cont) & (tok_1d == 240), torch.full_like(new_min, 144), new_min)
                    new_max = torch.where((~was_cont) & (tok_1d == 244), torch.full_like(new_max, 143), new_max)
                    utf8_rem, utf8_next_min, utf8_next_max = new_rem, new_min, new_max

                is_eos = (tok_1d == EOS)
                finished = finished | is_eos

                if finished.all():
                    break
                token = next_token

            return gen_buf[:, :n_gen]

    def generate_text(
        self,
        user_text: str,
        max_new_tokens: int = 512,
        min_new_tokens: int = 1,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        prev_state: torch.Tensor | None = None,
        enforce_valid_utf8: bool = True,
    ) -> str:
        device = next(self.parameters()).device
        raw = _encode_prompt(user_text, self.config.max_bytes)
        if len(raw) == 0:
            input_ids = torch.tensor([[PAD]], dtype=torch.long, device=device)
            input_mask = torch.zeros((1, 1), dtype=torch.float32, device=device)
        else:
            input_ids = torch.tensor([raw], dtype=torch.long, device=device)
            input_mask = torch.ones((1, len(raw)), dtype=torch.float32, device=device)

        output_tokens = self.generate(
            input_ids,
            input_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            prev_state=prev_state,
            enforce_valid_utf8=enforce_valid_utf8,
        )
        token_list = output_tokens[0].tolist()
        byte_values = []
        for tok in token_list:
            if tok == EOS:
                break
            if 0 <= tok <= 255:
                byte_values.append(tok)
        return bytes(byte_values).decode("utf-8", errors="replace")

    def generate_best_of_n(
        self,
        user_text: str,
        n: int = 4,
        max_new_tokens: int = 512,
        min_new_tokens: int = 1,
        temperature: float = 0.45,
        top_k: int = 40,
        top_p: float = 0.92,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 24,
        prev_state: torch.Tensor | None = None,
        enforce_valid_utf8: bool = True,
    ) -> dict[str, Any]:
        """Generate N candidate responses and select the highest-scoring one via native verification."""
        n = max(1, int(n))
        candidates: list[str] = []
        # Slot 0: deterministic greedy decode
        candidates.append(
            self.generate_text(
                user_text,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=0.0,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                prev_state=prev_state,
                enforce_valid_utf8=enforce_valid_utf8,
            )
        )
        # Slots 1..n-1: diverse nucleus-sampled candidates in a single batched call
        if n > 1:
            device = next(self.parameters()).device
            raw = _encode_prompt(user_text, self.config.max_bytes)
            if len(raw) == 0:
                input_ids = torch.full((n - 1, 1), PAD, dtype=torch.long, device=device)
                input_mask = torch.zeros((n - 1, 1), dtype=torch.float32, device=device)
            else:
                input_ids = torch.tensor([raw] * (n - 1), dtype=torch.long, device=device)
                input_mask = torch.ones((n - 1, len(raw)), dtype=torch.float32, device=device)

            sampled_tokens = self.generate(
                input_ids,
                input_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=max(0.1, temperature),
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                prev_state=prev_state,
                enforce_valid_utf8=enforce_valid_utf8,
            )
            for row in sampled_tokens.tolist():
                b_vals = []
                for tok in row:
                    if tok == EOS:
                        break
                    if 0 <= tok <= 255:
                        b_vals.append(tok)
                candidates.append(bytes(b_vals).decode("utf-8", errors="replace"))

        scores = [_score_candidate_response(user_text, c) for c in candidates]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        return {
            "best_text": candidates[best_idx],
            "best_score": scores[best_idx],
            "candidates": candidates,
            "scores": scores,
            "selected_index": best_idx,
        }

    def generate_with_thinking(
        self,
        user_text: str,
        goal: str | None = None,
        state: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        actions: Sequence[str] | None = None,
        max_plan_tokens: int = 96,
        max_answer_tokens: int = 512,
        num_candidates: int = 4,
        temperature: float = 0.65,
        prev_state: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Two-pass 'Think-Then-Answer' deliberative generation with structured cognitive state."""
        device = next(self.parameters()).device
        subgoals = decompose_long_instruction(user_text, max_subgoal_bytes=256)
        effective_goal = goal or (subgoals[0] if len(subgoals) > 1 else None)
        effective_actions = list(actions or (subgoals if len(subgoals) > 1 else []))

        # Pass 1: Deliberation / Internal Scratchpad encoding & plan generation
        plan_prompt = format_cognitive_prompt(
            f"Plan steps to answer: {user_text}",
            goal=effective_goal,
            state=state,
            evidence=evidence,
            actions=effective_actions,
        )
        raw_plan = _encode_prompt(plan_prompt, self.config.max_bytes)
        if len(raw_plan) == 0:
            p_ids = torch.tensor([[PAD]], dtype=torch.long, device=device)
            p_mask = torch.zeros((1, 1), dtype=torch.float32, device=device)
        else:
            p_ids = torch.tensor([raw_plan], dtype=torch.long, device=device)
            p_mask = torch.ones((1, len(raw_plan)), dtype=torch.float32, device=device)

        self.eval()
        with torch.no_grad():
            _, _, plan_state, _, _ = self.encode(p_ids, p_mask, prev_state=prev_state)

        thought_draft = self.generate_text(
            plan_prompt,
            max_new_tokens=max_plan_tokens,
            temperature=0.0,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            prev_state=prev_state,
        ).strip()

        # Pass 2: Verified response generation conditioned on cognitive context + plan_state
        answer_prompt = format_cognitive_prompt(
            user_text,
            goal=effective_goal,
            state=state,
            evidence=evidence,
            actions=effective_actions,
        )
        best_res = self.generate_best_of_n(
            answer_prompt,
            n=num_candidates,
            max_new_tokens=max_answer_tokens,
            temperature=temperature,
            prev_state=plan_state,
        )
        return {
            "answer": best_res["best_text"],
            "thought_plan": thought_draft,
            "subgoals": subgoals,
            "cognitive_prompt": answer_prompt,
            "plan_state": plan_state,
            "candidates": best_res["candidates"],
            "scores": best_res["scores"],
            "selected_index": best_res["selected_index"],
        }

    def execute_long_goal(
        self,
        instruction: str,
        goal: str | None = None,
        initial_state: torch.Tensor | None = None,
        max_subgoal_bytes: int = 384,
        max_tokens_per_step: int = 256,
        num_candidates: int = 2,
    ) -> dict[str, Any]:
        """Divide-and-conquer execution for long instructions and multi-step goals with state carry-over."""
        device = next(self.parameters()).device
        subgoals = decompose_long_instruction(instruction, max_subgoal_bytes=max_subgoal_bytes)
        if not subgoals:
            subgoals = [instruction]

        current_state = initial_state
        step_outputs: list[str] = []
        completed_summary: list[str] = []

        for idx, subgoal in enumerate(subgoals):
            step_context = format_cognitive_prompt(
                subgoal,
                goal=goal or f"Step {idx + 1} of {len(subgoals)}",
                state={"step": idx + 1, "total_steps": len(subgoals)},
                actions=completed_summary[-2:] if completed_summary else None,
            )
            raw = _encode_prompt(step_context, self.config.max_bytes)
            if len(raw) == 0:
                s_ids = torch.tensor([[PAD]], dtype=torch.long, device=device)
                s_mask = torch.zeros((1, 1), dtype=torch.float32, device=device)
            else:
                s_ids = torch.tensor([raw], dtype=torch.long, device=device)
                s_mask = torch.ones((1, len(raw)), dtype=torch.float32, device=device)

            with torch.no_grad():
                _, _, current_state, _, _ = self.encode(
                    s_ids, s_mask, prev_state=current_state
                )

            if num_candidates > 1:
                step_res = self.generate_best_of_n(
                    step_context,
                    n=num_candidates,
                    max_new_tokens=max_tokens_per_step,
                    prev_state=current_state,
                )["best_text"].strip()
            else:
                step_res = self.generate_text(
                    step_context,
                    max_new_tokens=max_tokens_per_step,
                    temperature=0.0,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=4,
                    prev_state=current_state,
                ).strip()

            step_outputs.append(step_res)
            completed_summary.append(f"Done({idx + 1}): {subgoal[:48]}")

        combined = "\n\n".join(s for s in step_outputs if s)
        return {
            "combined_response": combined,
            "subgoals": subgoals,
            "step_outputs": step_outputs,
            "final_state": current_state,
        }


def build_native_conversation_v5(config: NativeConversationV5Config):
    return NativeConversationV5(config)


def _detect_physical_cpu_cores() -> int:
    """Estimate physical CPU core count to avoid hyperthread L1/L2 cache thrashing."""
    import os

    logical = os.cpu_count() or 1
    if logical >= 4 and logical % 2 == 0:
        return max(1, logical // 2)
    return max(1, logical)


def configure_cpu_runtime(num_threads: int | None = None) -> dict[str, Any]:
    """Configure PyTorch CPU threads, OneDNN (MKL-DNN), and matmul precision for large-model CPU efficiency."""
    import os

    logical_cores = os.cpu_count() or 1
    target_threads = (
        int(num_threads)
        if num_threads is not None and int(num_threads) > 0
        else _detect_physical_cpu_cores()
    )
    try:
        torch.set_num_threads(target_threads)
    except RuntimeError:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True
    return {
        "logical_cores": logical_cores,
        "active_cpu_threads": int(torch.get_num_threads()),
        "mkldnn_enabled": bool(getattr(getattr(torch.backends, "mkldnn", None), "enabled", False)),
        "matmul_precision": str(torch.get_float32_matmul_precision()),
    }


_DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "state",
    "decoder_init",
    "query",
    "key",
    "value",
    "attn_out",
    "gate_linear",
    "head",
)


def apply_lora_to_model(
    model: NativeConversationV5,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
    target_modules: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Attach low-rank adapters (LoRA) to linear projections for fast, low-RAM CPU fine-tuning."""
    targets = tuple(target_modules) if target_modules is not None else _DEFAULT_LORA_TARGETS
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    adapted_layers: list[str] = []
    for attr_name in targets:
        layer = getattr(model, attr_name, None)
        if isinstance(layer, nn.Linear):
            setattr(
                model,
                attr_name,
                LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout),
            )
            adapted_layers.append(attr_name)
        elif isinstance(layer, LoRALinear):
            layer.lora_A.requires_grad = True
            layer.lora_B.requires_grad = True
            adapted_layers.append(attr_name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "adapted_layers": adapted_layers,
        "rank": int(rank),
        "alpha": float(alpha),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": round(trainable_params / max(1, total_params), 4),
    }


def merge_lora_weights(model: NativeConversationV5) -> NativeConversationV5:
    """Fold all LoRALinear adapters back into standard nn.Linear weights with zero inference overhead."""
    for name, module in list(model.named_children()):
        if isinstance(module, LoRALinear):
            setattr(model, name, module.merged_linear())
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_lora_state_dict(model: NativeConversationV5) -> dict[str, torch.Tensor]:
    """Extract only the lightweight LoRA adapter parameters (lora_A, lora_B) from a model."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def _export_clean_state_dict(model: NativeConversationV5) -> dict[str, torch.Tensor]:
    """Export a standard NativeConversationV5 state_dict, transparently merging any active LoRALinear layers."""
    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    if not has_lora:
        return model.state_dict()

    clean_sd: dict[str, torch.Tensor] = {}
    lora_prefixes: dict[str, LoRALinear] = {
        name: mod for name, mod in model.named_modules() if isinstance(mod, LoRALinear)
    }
    raw_sd = model.state_dict()
    handled_prefixes: set[str] = set()
    for prefix, lora_mod in lora_prefixes.items():
        clean_sd[f"{prefix}.weight"] = lora_mod.merged_weight()
        if lora_mod.bias is not None:
            clean_sd[f"{prefix}.bias"] = lora_mod.bias.data.clone()
        handled_prefixes.add(f"{prefix}.")

    for key, val in raw_sd.items():
        if any(key.startswith(p) for p in handled_prefixes):
            continue
        clean_sd[key] = val
    return clean_sd


def quantize_for_cpu_inference(model: NativeConversationV5) -> NativeConversationV5:
    """Apply dynamic INT8 quantization to Linear, GRU, and GRUCell layers for fast local CPU inference."""
    import copy
    import warnings

    cpu_model = copy.deepcopy(model).cpu().eval()
    merge_lora_weights(cpu_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.ao.quantization.quantize_dynamic(
            cpu_model,
            {nn.Linear, nn.GRU, nn.GRUCell},
            dtype=torch.qint8,
        )


def _score_candidate_response(user_text: str, candidate: str) -> float:
    """Fast native coherence, diversity, and relevance critic for Best-of-N selection."""
    text = candidate.strip()
    if not text:
        return -10.0

    # 1. Penalize UTF-8 replacement characters or non-printable noise
    replacement_count = text.count("\ufffd")
    printable_ratio = sum(1 for ch in text if ch.isprintable() or ch in "\n\t") / max(1, len(text))
    score = 2.0 * printable_ratio - 0.5 * replacement_count

    # 2. Character & 3-gram diversity (penalize repetitive loops)
    if len(text) >= 6:
        trigrams = [text[i : i + 3] for i in range(len(text) - 2)]
        unique_trigram_ratio = len(set(trigrams)) / max(1, len(trigrams))
        score += 2.0 * unique_trigram_ratio

    # 3. Word structure & natural spacing
    words = text.split()
    if words:
        unique_word_ratio = len(set(w.lower() for w in words)) / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        word_len_ok = 1.0 if 2.5 <= avg_word_len <= 10.0 else 0.2
        score += 1.5 * unique_word_ratio + word_len_ok

        # 4. Prompt keyword alignment bonus
        prompt_words = {
            w.lower().strip(".,!?;:\"'()[]")
            for w in user_text.split()
            if len(w) >= 4
        }
        cand_words = {w.lower().strip(".,!?;:\"'()[]") for w in words}
        if prompt_words:
            overlap = len(prompt_words & cand_words) / len(prompt_words)
            score += 0.8 * overlap

    # 5. Linguistic / alphanumeric balance and symbol-noise suppression
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    alpha_ratio = alpha_chars / max(1, len(text))
    if alpha_ratio >= 0.55:
        score += 0.6
    elif alpha_ratio < 0.25:
        score -= 0.6

    if alpha_chars >= 4:
        vowels = sum(1 for ch in text.lower() if ch in "aeiou")
        vowel_ratio = vowels / max(1, alpha_chars)
        if 0.20 <= vowel_ratio <= 0.55:
            score += 0.4

    # 6. Clean sentence termination bonus
    if text[-1] in ".!?\"')":
        score += 0.4

    return float(score)


def decompose_long_instruction(
    instruction: str, max_subgoal_bytes: int = 384
) -> list[str]:
    """Decompose a long or multi-step instruction into an ordered queue of focused sub-goals."""
    clean = instruction.strip()
    if not clean:
        return []

    # 1. Check for explicit numbered steps or bullet points
    numbered = re.split(r"(?:^|\n)\s*(?:\d+[\.\)]|[-*•])\s+", clean)
    numbered = [seg.strip() for seg in numbered if seg.strip()]
    if len(numbered) > 1:
        return numbered

    # 2. If within byte budget, check for multi-clause sequence markers ("First,", "Second,", "Finally,")
    raw = clean.encode("utf-8")
    if len(raw) <= max_subgoal_bytes:
        clause_split = re.split(
            r"\b(?:first,|second,|third,|next,|then,|finally,)\b",
            clean,
            flags=re.IGNORECASE,
        )
        clause_split = [seg.strip(" ,;:") for seg in clause_split if seg.strip(" ,;:")]
        if len(clause_split) > 1:
            return clause_split
        return [clean]

    # 3. Split long paragraph instructions along sentence boundaries into <= max_subgoal_bytes chunks
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if s.strip()]
    subgoals: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for sent in sentences:
        s_len = len(sent.encode("utf-8"))
        if current and current_bytes + 1 + s_len > max_subgoal_bytes:
            subgoals.append(" ".join(current))
            current = [sent]
            current_bytes = s_len
        else:
            current.append(sent)
            current_bytes += s_len + (1 if current else 0)
    if current:
        subgoals.append(" ".join(current))
    return subgoals if subgoals else [clean]


def format_cognitive_prompt(
    user_text: str,
    goal: str | None = None,
    state: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    actions: Sequence[str] | None = None,
) -> str:
    """Format structured cognitive workspace context (goal, state, evidence, actions) with user query."""
    parts: list[str] = []
    if goal and str(goal).strip():
        parts.append(f"[GOAL] {str(goal).strip()}")
    if state:
        compact_state = ", ".join(f"{k}={v}" for k, v in list(state.items())[:4])
        if compact_state:
            parts.append(f"[STATE] {compact_state}")
    if evidence:
        compact_ev = ", ".join(f"{k}={v}" for k, v in list(evidence.items())[:4])
        if compact_ev:
            parts.append(f"[FACTS] {compact_ev}")
    if actions:
        compact_act = " -> ".join(str(a).strip() for a in list(actions)[:4] if str(a).strip())
        if compact_act:
            parts.append(f"[PLAN] {compact_act}")
    if not parts:
        return user_text
    parts.append(f"[USER] {user_text}")
    return "\n".join(parts)


def _encode_example_prompt(ex: TrainingExample, max_bytes: int) -> list[int]:
    """Encode a TrainingExample incorporating its structured cognitive context when present."""
    if ex.goal or ex.state or ex.evidence or ex.actions:
        formatted = format_cognitive_prompt(
            ex.user_text,
            goal=ex.goal,
            state=ex.state,
            evidence=ex.evidence,
            actions=ex.actions,
        )
        return _encode_prompt(formatted, max_bytes)
    return _encode_prompt(ex.user_text, max_bytes)


def _valid_utf8_prefix(raw: bytes) -> bytes:
    while raw:
        try:
            raw.decode("utf-8")
            return raw
        except UnicodeDecodeError:
            raw = raw[:-1]
    return b""


def _valid_utf8_suffix(raw: bytes) -> bytes:
    while raw:
        try:
            raw.decode("utf-8")
            return raw
        except UnicodeDecodeError:
            raw = raw[1:]
    return b""


def _encode_prompt(text: str, max_bytes: int) -> list[int]:
    """Encode prompt preserving both opening context and final question if > max_bytes."""
    if max_bytes <= 0:
        return []
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return list(raw)
    if max_bytes < 16:
        return list(_valid_utf8_prefix(raw[:max_bytes]))
    sep = b" ... "
    avail = max_bytes - len(sep)
    head_len = avail // 2
    tail_len = avail - head_len
    head = _valid_utf8_prefix(raw[:head_len])
    tail = _valid_utf8_suffix(raw[-tail_len:])
    return list(head + sep + tail)


def _clean_truncate(text: str, max_payload: int) -> list[int]:
    """Truncate text cleanly at sentence boundaries first, then word boundaries without trailing space."""
    if max_payload <= 0:
        return []
    raw = text.encode("utf-8")
    if len(raw) <= max_payload:
        return list(raw)
    sliced = raw[:max_payload]
    # 1. Prefer complete sentence boundary (., !, ?, \n) in the last 160 bytes
    min_sentence_pos = max(max_payload // 2, len(sliced) - 160)
    for i in range(len(sliced) - 1, min_sentence_pos - 1, -1):
        if sliced[i] in (10, 46, 33, 63):
            return list(sliced[: i + 1])
    # 2. Fall back to clause/word boundary and strip trailing space so EOS never follows a space
    min_word_pos = max(0, len(sliced) - 64)
    for i in range(len(sliced) - 1, min_word_pos - 1, -1):
        if sliced[i] in (32, 10, 44, 59, 58):
            cleaned = sliced[:i].rstrip(b" ")
            if cleaned:
                return list(cleaned)
    return list(_valid_utf8_prefix(sliced))


def _prepare_targets(texts, max_bytes):
    rows = []
    max_payload = max(0, max_bytes - 2)
    for text in texts:
        payload = _clean_truncate(text, max_payload)
        rows.append([BOS] + payload + [EOS])
    return rows


def expand_examples_into_chunks(
    examples: list[TrainingExample], max_bytes: int
) -> list[TrainingExample]:
    """Split long target responses into clean sentence/clause continuation chunks for zero data loss."""
    max_payload = max(32, max_bytes - 2)
    expanded: list[TrainingExample] = []
    for ex in examples:
        raw_target = ex.target_text.encode("utf-8")
        if len(raw_target) <= max_payload:
            expanded.append(ex)
            continue

        remaining = ex.target_text.strip()
        prev_tail = ""
        chunk_idx = 0
        while remaining:
            raw_rem = remaining.encode("utf-8")
            if len(raw_rem) <= max_payload:
                seg_bytes = raw_rem
            else:
                seg_bytes = bytes(_clean_truncate(remaining, max_payload))
                if not seg_bytes:
                    seg_bytes = _valid_utf8_prefix(raw_rem[:max_payload])
            seg_text = seg_bytes.decode("utf-8", errors="ignore").strip()
            if not seg_text:
                break

            if chunk_idx == 0:
                prompt_text = ex.user_text
            else:
                prompt_text = f"{ex.user_text}\n[Continue: ...{prev_tail}]"

            expanded.append(
                replace(
                    ex,
                    example_id=f"{ex.example_id}#chunk{chunk_idx}" if chunk_idx > 0 else ex.example_id,
                    user_text=prompt_text,
                    target_text=seg_text,
                )
            )
            # Advance remaining text past the consumed segment
            consumed_len = len(seg_bytes)
            remaining = raw_rem[consumed_len:].decode("utf-8", errors="ignore").lstrip()
            prev_tail = seg_text[-96:]
            chunk_idx += 1

    return expanded


def save_checkpoint(
    path: str | Path,
    model: NativeConversationV5,
    config: NativeConversationV5Config,
    metadata: dict | None = None,
    history: list | None = None,
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
) -> None:
    torch_mod, _ = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_sd = _export_clean_state_dict(model)
    parameter_count = sum(v.numel() for v in clean_sd.values())
    payload = {
        "format": "mirror7-native-conversation-v5-input-conditioned",
        "config": config.to_dict(),
        "special_tokens": {"PAD": PAD, "BOS": BOS, "EOS": EOS},
        "vocab_size": VOCAB_SIZE,
        "total_vocab_size": TOTAL_VOCAB_SIZE,
        "parameter_count": parameter_count,
        "metadata": dict(metadata or {}),
        "history": list(history or []),
        "state_dict": clean_sd,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    torch_mod.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    from dataclasses import fields

    torch_mod, _ = require_torch()
    path = Path(path)
    payload = torch_mod.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "mirror7-native-conversation-v5-input-conditioned":
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format')!r}"
        )
    raw_config = payload.get("config", {})
    valid_keys = {f.name for f in fields(NativeConversationV5Config)}
    filtered_config = {k: v for k, v in raw_config.items() if k in valid_keys}
    config = NativeConversationV5Config(**filtered_config)
    model = build_native_conversation_v5(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, payload


def train(args):
    import random
    torch_mod, _ = require_torch()
    examples = load_jsonl(args.dataset)
    train_items = [x for x in examples if x.split == "train"]
    val_items = [x for x in examples if x.split == "validation"]
    if not train_items or not val_items:
        raise ValueError("dataset must contain train and validation splits")

    device = torch_mod.device(
        args.device or ("cuda" if torch_mod.cuda.is_available() else "cpu")
    )

    history = []
    loaded_payload = None
    if getattr(args, "resume", None):
        model, loaded_payload = load_checkpoint(args.resume, device=str(device))
        config = model.config
        history = list(loaded_payload.get("history", []))
    else:
        config = NativeConversationV5Config(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            state_dim=args.state_dim,
            attention_dim=args.attention_dim,
            dropout=args.dropout,
            max_bytes=args.max_bytes,
            bidirectional_encoder=getattr(args, "bidirectional_encoder", False),
            num_heads=getattr(args, "num_heads", 1),
            context_gating=getattr(args, "context_gating", False),
            scheduled_sampling_p=getattr(args, "scheduled_sampling_p", 0.0),
        )
        model = build_native_conversation_v5(config)
        model.to(device)

    if getattr(args, "chunk_long_examples", True):
        train_items = expand_examples_into_chunks(train_items, config.max_bytes)

    optimizer = torch_mod.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if getattr(args, "use_scheduler", False):
        scheduler = torch_mod.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1
        )

    if loaded_payload is not None:
        if "optimizer_state_dict" in loaded_payload:
            try:
                optimizer.load_state_dict(loaded_payload["optimizer_state_dict"])
            except Exception:
                pass
        if scheduler is not None and "scheduler_state_dict" in loaded_payload:
            try:
                scheduler.load_state_dict(loaded_payload["scheduler_state_dict"])
            except Exception:
                pass

    label_smoothing = getattr(args, "label_smoothing", 0.0)
    should_shuffle = getattr(args, "shuffle", True)

    def run_epoch(items, training):
        model.train(training)
        total = 0.0
        batches = 0
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            ui, um = _batch(
                [_encode_example_prompt(x, config.max_bytes) for x in batch],
                config.max_bytes,
            )
            target_rows = _prepare_targets(
                [x.target_text for x in batch],
                config.max_bytes,
            )
            ti, _ = _batch(target_rows, config.max_bytes)
            ui, um, ti = ui.to(device), um.to(device), ti.to(device)

            with torch_mod.set_grad_enabled(training):
                logits, _, _, _ = model(ui, um, ti[:, :-1], return_attentions=False)
                labels = ti[:, 1:]
                loss = torch_mod.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=PAD,
                    label_smoothing=label_smoothing if training else 0.0,
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch_mod.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                    optimizer.step()

            total += float(loss.detach().cpu())
            batches += 1

        return total / max(1, batches)

    start_epoch = len(history) + 1
    best_val_loss = float("inf")
    if loaded_payload is not None:
        meta_best = loaded_payload.get("metadata", {}).get("best_val_loss")
        hist_best = min(
            (r.get("validation_loss", float("inf")) for r in history),
            default=float("inf"),
        )
        best_val_loss = min(
            float(meta_best) if meta_best is not None else float("inf"),
            float(hist_best),
        )

    output = Path(args.output)
    best_output = output.with_name(f"{output.stem}.best{output.suffix}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_items = list(train_items)
        if should_shuffle:
            random.shuffle(epoch_items)

        train_loss = run_epoch(epoch_items, True)
        val_loss = run_epoch(val_items, False)
        if scheduler:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_output,
                model,
                config,
                metadata={"epochs": epoch, "lr": args.lr, "best_val_loss": best_val_loss},
                history=history,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict() if scheduler else None,
            )

    save_checkpoint(
        output,
        model,
        config,
        metadata={
            "epochs": start_epoch + args.epochs - 1,
            "lr": args.lr,
            "best_val_loss": best_val_loss,
        },
        history=history,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict() if scheduler else None,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {"output": str(output), "parameters": parameter_count}
        )
    )
    return model, history


def main():
    p = argparse.ArgumentParser(
        description="Train Mirror 7 native conversation v5 (Input-Conditioned Recurrent Decoder)."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--output",
        default="artifacts/mirror7_native_conversation_v5.pt",
    )
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--use-scheduler", action="store_true", default=False)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--embedding-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--state-dim", type=int, default=128)
    p.add_argument("--attention-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--bidirectional-encoder", action="store_true", default=False)
    p.add_argument("--num-heads", type=int, default=1)
    p.add_argument("--context-gating", action="store_true", default=False)
    p.add_argument("--scheduled-sampling-p", type=float, default=0.0)
    p.add_argument("--device", default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
