from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math


SPECIAL_TOKENS = {
    "PAD": 256,
    "BOS": 257,
    "EOS": 258,
    "CONTEXT": 259,
    "GOAL": 260,
    "USER": 261,
    "RESPONSE": 262,
}
VOCAB_SIZE = 263


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    embedding_dim: int = 192
    hidden_dim: int = 384
    num_layers: int = 2
    dropout: float = 0.10
    max_sequence_bytes: int = 4096

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for training. Install requirements-training.txt."
        ) from exc
    return torch, nn


def build_model(config: ModelConfig):
    torch, nn = require_torch()

    class ByteGRULanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
            self.gru = nn.GRU(
                config.embedding_dim,
                config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.head = nn.Linear(config.hidden_dim, config.vocab_size)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, input_ids, hidden=None):
            x = self.embedding(input_ids)
            x, hidden = self.gru(x, hidden)
            x = self.norm(x)
            return self.head(x), hidden

        @torch.no_grad()
        def generate(self, prefix, *, max_new_bytes=768, temperature=0.8, top_k=40):
            self.eval()
            generated = prefix.clone()

            # Prime the recurrent state with the entire prompt before generating.
            # The previous implementation only fed the final RESPONSE token on the
            # first step, so every prompt effectively started from the same hidden state.
            _, hidden = self(prefix)

            for _ in range(max_new_bytes):
                logits, hidden = self(generated[:, -1:], hidden)
                next_logits = logits[:, -1, :]
                next_logits[:, SPECIAL_TOKENS["PAD"]] = -math.inf
                next_logits[:, SPECIAL_TOKENS["BOS"]] = -math.inf
                next_logits[:, SPECIAL_TOKENS["CONTEXT"]] = -math.inf
                next_logits[:, SPECIAL_TOKENS["GOAL"]] = -math.inf
                next_logits[:, SPECIAL_TOKENS["USER"]] = -math.inf
                next_logits[:, SPECIAL_TOKENS["RESPONSE"]] = -math.inf
                if temperature <= 0:
                    next_id = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k and top_k < next_logits.shape[-1]:
                        values, indices = torch.topk(next_logits, top_k)
                        filtered = torch.full_like(next_logits, -math.inf)
                        filtered.scatter_(1, indices, values)
                        next_logits = filtered
                    probs = torch.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_id], dim=1)
                if int(next_id.item()) == SPECIAL_TOKENS["EOS"]:
                    break
            return generated

    return ByteGRULanguageModel()


def save_checkpoint(path, model, config: ModelConfig, metadata: dict[str, Any]) -> None:
    torch, _ = require_torch()
    payload = {
        "format": "mirror7-byte-gru-v1",
        "config": config.to_dict(),
        "special_tokens": dict(SPECIAL_TOKENS),
        "metadata": dict(metadata),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path):
    torch, _ = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "mirror7-byte-gru-v1":
        raise ValueError("unsupported Mirror 7 training checkpoint")
    config = ModelConfig(**payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["state_dict"])
    return model, payload
