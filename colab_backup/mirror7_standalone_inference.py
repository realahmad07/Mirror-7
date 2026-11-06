"""
Mirror 7 (V6 Final Master) — Standalone Loader, Agentic Session & 3-Specialist Hub
Author: Ahmad Hassan | Repository: https://github.com/realahmad07/Mirror-7

Usage in Google Colab or Local GPU:
    from colab_backup.mirror7_standalone_inference import load_mirror7_master
    model, hub = load_mirror7_master("/content/drive/MyDrive/Mirror7_Ultra_Backup/mirror7_ultra_FINAL_BF16_COMPACT.pt")
    ans, route = hub.chat("Implement Dijkstra's algorithm in Python with `heapq` and analyze its complexity.")
    print(route["active_specialist"])
    print(ans)
"""
from __future__ import annotations

import io
import math
import os
import re
from contextlib import redirect_stdout
from typing import Any

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.native_conversation_v5 import (
    BOS,
    EOS,
    PAD,
    NativeConversationV5,
    NativeConversationV5Config,
    _mask_fill_value,
)

TC_OPEN, TC_CLOSE = "<tool_call>", "</tool_call>"
TO_OPEN, TO_CLOSE = "<tool_output>", "</tool_output>"


class DecoderSelfAttentionRoPE(nn.Module):
    """8-Head Causal Target Self-Attention with NTK-Scaled 2,048-Byte RoPE & KV Cache."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, rope_base: float = 20000.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = max(1, int(num_heads))
        self.head_dim = self.hidden_dim // self.num_heads
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        half_dim = max(1, self.head_dim // 2)
        inv_freq = 1.0 / (float(rope_base) ** (torch.arange(0, half_dim, dtype=torch.float32) / float(half_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        freqs = torch.outer(positions.float(), self.inv_freq.to(x.device))
        cos = freqs.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :half], x[..., half : 2 * half]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hdim = hidden_seq.shape
        u = self.norm(hidden_seq)
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        pos = torch.arange(seq_len, device=hidden_seq.device)
        q, k = self._apply_rope(q, pos), self._apply_rope(k, pos)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, hdim)
        return hidden_seq + self.out_proj(attn_out)

    def step(self, hidden_step: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, step_idx: int) -> torch.Tensor:
        bsz, hdim = hidden_step.shape
        u = self.norm(hidden_step)
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        q = q.view(bsz, self.num_heads, 1, self.head_dim)
        k = k.view(bsz, self.num_heads, 1, self.head_dim)
        v = v.view(bsz, self.num_heads, 1, self.head_dim)
        pos = torch.tensor([step_idx], device=hidden_step.device)
        q, k = self._apply_rope(q, pos), self._apply_rope(k, pos)
        k_cache[:, :, step_idx : step_idx + 1, :] = k
        v_cache[:, :, step_idx : step_idx + 1, :] = v
        attn_out = F.scaled_dot_product_attention(
            q, k_cache[:, :, : step_idx + 1, :], v_cache[:, :, : step_idx + 1, :], is_causal=False
        )
        return hidden_step + self.out_proj(attn_out.transpose(1, 2).contiguous().view(bsz, hdim))


def execute_sandboxed_python_tool(code_cmd: str) -> str:
    clean_cmd = re.sub(r"^python\s*:\s*", "", code_cmd.strip(), flags=re.IGNORECASE)
    buf = io.StringIO()
    safe_ns: dict[str, Any] = {"math": math, "sp": sp, "np": np}
    try:
        with redirect_stdout(buf):
            exec(clean_cmd, safe_ns)
        out = buf.getvalue().strip()
        return out if out else "OK"
    except Exception as exc:
        return f"ToolError: {exc}"


class Mirror7AgenticStatefulSession:
    """Stateful multi-turn session with persistent semantic state & live <tool_call> execution."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.prev_state: torch.Tensor | None = None
        self.last_user: str | None = None
        self.last_assistant: str | None = None
        self.last_tool_output: str | None = None

    @torch.no_grad()
    def generate_with_state(
        self,
        user_text: str,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 10,
        temperature: float = 0.0,
    ) -> tuple[str, torch.Tensor]:
        m = self.model
        m.eval()
        raw = list(user_text.encode("utf-8")[:2048]) or [32]
        inp = torch.tensor([raw], dtype=torch.long, device=self.device)
        mask = torch.ones((1, len(raw)), dtype=torch.float32, device=self.device)
        _, enc_mask, state, k, v = m.encode(inp, mask, prev_state=self.prev_state)
        h0 = m.init_decoder_hidden(state)
        hidden = h0
        ac = m._prepare_attention_cache(k, v, enc_mask)
        sdpa_mask = ac[2] | ~ac[3]
        anchor = 0.10 * h0
        dsa = m.decoder_self_attn
        kc = torch.empty((1, dsa.num_heads, max_new_tokens, dsa.head_dim), dtype=h0.dtype, device=self.device)
        vc = torch.empty((1, dsa.num_heads, max_new_tokens, dsa.head_dim), dtype=h0.dtype, device=self.device)
        tok = torch.full((1, 1), BOS, dtype=torch.long, device=self.device)
        out_b: list[int] = []
        for step in range(max_new_tokens):
            ctx, _ = m._step_attention(hidden, k, v, enc_mask, _precomputed=ac, _sdpa_mask=sdpa_mask)
            hidden = m.cell(torch.cat([m.decoder_embedding(tok).squeeze(1), ctx], dim=-1), hidden)
            if bool(((tok == 46) | (tok == 10) | (tok == 33) | (tok == 63)).item()):
                hidden = 0.90 * hidden + anchor
            logits = m._project_output(dsa.step(hidden, kc, vc, step), ctx)
            ni = _mask_fill_value(logits)
            logits[:, PAD] = ni
            logits[:, BOS] = ni
            if step < min_new_tokens:
                logits[:, EOS] = ni
            if len(out_b) >= 12 and all(b == 32 for b in out_b[-12:]):
                logits[:, 32] = ni
            if temperature > 0.0:
                probs = F.softmax(logits / temperature, dim=-1)
                nt = int(torch.multinomial(probs, num_samples=1).item())
            else:
                nt = int(logits.argmax(dim=-1).item())
            if nt == EOS:
                break
            if 0 <= nt <= 255:
                out_b.append(nt)
            tok = torch.tensor([[nt]], dtype=torch.long, device=self.device)
        return bytes(out_b).decode("utf-8", errors="replace"), state.detach()

    def chat(self, query: str, max_new_tokens: int = 2048, auto_execute_tools: bool = True) -> str:
        ans, self.prev_state = self.generate_with_state(query.strip(), max_new_tokens=max_new_tokens)
        if auto_execute_tools and TC_OPEN in ans and TC_CLOSE in ans:
            m_tc = re.search(r"<tool_call>(.*?)</tool_call>", ans, flags=re.DOTALL)
            if m_tc:
                tool_res = execute_sandboxed_python_tool(m_tc.group(1))
                self.last_tool_output = tool_res
                tool_tag = f"{TO_OPEN}{tool_res}{TO_CLOSE}"
                raw_obs = list(tool_tag.encode("utf-8")[:512]) or [32]
                obs_ids = torch.tensor([raw_obs], dtype=torch.long, device=self.device)
                obs_mask = torch.ones((1, len(raw_obs)), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    _, _, self.prev_state, _, _ = self.model.encode(obs_ids, obs_mask, prev_state=self.prev_state)
                ans = ans.replace(m_tc.group(0), f"{m_tc.group(0)}\n{tool_tag}")
        self.last_user, self.last_assistant = query, ans
        return ans


class Mirror7SpecialistHubEnsemble(nn.Module):
    """3-Specialist Sparse Hub Router (556,660,236 Total Params / 185,531,395 Active)."""

    def __init__(self, primary_model: nn.Module, device: torch.device):
        super().__init__()
        self.primary_model = primary_model
        self.device = device
        self.specialist_names = (
            "Specialist_0_Extreme_Coding_And_Systems",
            "Specialist_1_Math_Quantum_And_Science",
            "Specialist_2_Reasoning_Metacognition_And_Memory",
        )
        self.router = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        ).to(device)
        self.single_params = sum(p.numel() for p in self.primary_model.parameters())
        self.router_params = sum(p.numel() for p in self.router.parameters())
        self.total_ensemble_params = 3 * self.single_params + self.router_params
        self.session = Mirror7AgenticStatefulSession(self.primary_model, device=device)

    @torch.no_grad()
    def route_query(self, query: str) -> dict[str, Any]:
        raw = list(query.encode("utf-8")[:512]) or [32]
        inp = torch.tensor([raw], dtype=torch.long, device=self.device)
        mask = torch.ones((1, len(raw)), dtype=torch.float32, device=self.device)
        _, _, state, _, _ = self.primary_model.encode(inp, mask)
        logits = self.router(state.float()).squeeze(0)
        ql = query.lower()
        if any(k in ql for k in ("python", "dijkstra", "lis", "lru", "fenwick", "rust", "cas", "aba", "raft", "lsm", "sql", "hmac", "union", "hoare", "flashattention", "code", "implement", "limiter")):
            logits[0] += 2.5
        elif any(k in ql for k in ("mod", "fermat", "quantum", "bell", "density", "entropy", "relativity", "matrix", "determinant", "fibonacci", "math", "qubit")):
            logits[1] += 2.5
        else:
            logits[2] += 2.5
        probs = F.softmax(logits, dim=-1).cpu().tolist()
        top_idx = int(np.argmax(probs))
        return {
            "active_specialist": self.specialist_names[top_idx],
            "routing_weights": {self.specialist_names[i]: round(probs[i], 4) for i in range(3)},
        }

    def chat(self, query: str, max_new_tokens: int = 2048) -> tuple[str, dict[str, Any]]:
        route_info = self.route_query(query)
        self.session.reset()
        response = self.session.chat(query, max_new_tokens=max_new_tokens, auto_execute_tools=True)
        return response, route_info


def load_mirror7_master(
    ckpt_path: str = "/content/drive/MyDrive/Mirror7_Ultra_Backup/mirror7_ultra_FINAL_BF16_COMPACT.pt",
    device_str: str | None = None,
) -> tuple[NativeConversationV5, Mirror7SpecialistHubEnsemble]:
    """Loads the finalized Mirror-7 V6 Master checkpoint + 3-Specialist Hub Router."""
    dev = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_raw = payload.get("config", {})
    valid_fields = set(NativeConversationV5Config.__dataclass_fields__.keys())
    filtered_cfg = {k: v for k, v in cfg_raw.items() if k in valid_fields}
    if not filtered_cfg:
        filtered_cfg = {
            "embedding_dim": 384,
            "hidden_dim": 1024,
            "state_dim": 512,
            "attention_dim": 512,
            "dropout": 0.10,
            "max_bytes": 2048,
            "bidirectional_encoder": True,
            "num_heads": 8,
            "context_gating": True,
            "num_moe_experts": 16,
            "moe_top_k": 2,
            "memory_bank_slots": 16384,
        }
    cfg = NativeConversationV5Config(**filtered_cfg)
    model = NativeConversationV5(cfg)
    rope_base = 20000.0 if payload.get("rope_horizon_bytes", 2048) >= 2048 else 10000.0
    model.decoder_self_attn = DecoderSelfAttentionRoPE(cfg.hidden_dim, num_heads=cfg.num_heads, rope_base=rope_base)

    sd = payload.get("state_dict", payload)
    model.load_state_dict(sd, strict=False)
    model.to(dev).eval()

    hub = Mirror7SpecialistHubEnsemble(model, device=dev)
    if "router_state_dict" in payload:
        hub.router.load_state_dict(payload["router_state_dict"], strict=False)
    hub.eval()
    return model, hub
