from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from training.dataset import load_jsonl
from training.model import require_torch
from training.native_brain import PAD, VOCAB_SIZE, _batch, _encode

BOS = VOCAB_SIZE
EOS = VOCAB_SIZE + 1
TOTAL_VOCAB_SIZE = VOCAB_SIZE + 2


@dataclass(frozen=True)
class NativeConversationConfig:
    embedding_dim: int = 96
    hidden_dim: int = 256
    state_dim: int = 128
    layers: int = 1
    dropout: float = 0.10
    max_bytes: int = 512
    attention_dim: int = 256

    def to_dict(self):
        return asdict(self)


class NativeConversation(nn.Module):
    """Byte-level encoder/decoder with sequence cross-attention.

    The encoder retains every input position. The decoder is causal (GRU) and
    attends to the full encoded input at every generated step, avoiding the
    lossy single-vector bottleneck of the previous model.
    """

    def __init__(self, config: NativeConversationConfig):
        torch_mod, nn_mod = require_torch()
        super().__init__()
        self.config = config

        self.embedding = nn_mod.Embedding(
            VOCAB_SIZE, config.embedding_dim, padding_idx=PAD
        )
        self.encoder = nn_mod.GRU(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.encoder_norm = nn_mod.LayerNorm(config.hidden_dim)
        self.state = nn_mod.Linear(config.hidden_dim, config.state_dim)

        self.decoder_embedding = nn_mod.Embedding(
            TOTAL_VOCAB_SIZE, config.embedding_dim
        )
        self.decoder = nn_mod.GRU(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )

        attn_dim = config.attention_dim
        self.query = nn_mod.Linear(config.hidden_dim, attn_dim, bias=False)
        self.key = nn_mod.Linear(config.hidden_dim, attn_dim, bias=False)
        self.value = nn_mod.Linear(config.hidden_dim, attn_dim, bias=False)
        self.context_norm = nn_mod.LayerNorm(attn_dim)
        self.output_norm = nn_mod.LayerNorm(config.hidden_dim + attn_dim)
        self.head = nn_mod.Linear(
            config.hidden_dim + attn_dim, TOTAL_VOCAB_SIZE
        )

    def encode(self, input_ids, input_mask):
        x, hidden = self.encoder(self.embedding(input_ids))
        x = self.encoder_norm(x)
        mask = input_mask.to(dtype=x.dtype)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / (
            mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        )
        state = torch.tanh(self.state(pooled))
        return x, input_mask.bool(), state, hidden

    def _cross_attention(self, decoder_states, encoder_states, encoder_mask):
        import math

        q = self.query(decoder_states)
        k = self.key(encoder_states)
        v = self.value(encoder_states)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(
            ~encoder_mask.unsqueeze(1),
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        return self.context_norm(context), weights

    def forward(self, input_ids, input_mask, target_ids, hidden=None):
        encoder_states, encoder_mask, state, _ = self.encode(
            input_ids, input_mask
        )
        decoder_emb = self.decoder_embedding(target_ids)
        decoder_states, hidden = self.decoder(decoder_emb, hidden)
        context, attention = self._cross_attention(
            decoder_states, encoder_states, encoder_mask
        )
        fused = self.output_norm(
            torch.cat([decoder_states, context], dim=-1)
        )
        return self.head(fused), state, hidden, attention

    def generate(
        self,
        input_ids,
        input_mask,
        max_new_tokens=512,
        temperature=0.8,
        top_k=40,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        with torch.no_grad():
            encoder_states, encoder_mask, _, hidden = self.encode(
                input_ids, input_mask
            )
            token = torch.full(
                (input_ids.shape[0], 1),
                BOS,
                dtype=torch.long,
                device=input_ids.device,
            )
            generated = []
            for _ in range(max_new_tokens):
                decoder_states, hidden = self.decoder(
                    self.decoder_embedding(token), hidden
                )
                context, _ = self._cross_attention(
                    decoder_states, encoder_states, encoder_mask
                )
                fused = self.output_norm(
                    torch.cat([decoder_states, context], dim=-1)
                )
                logits = self.head(fused[:, -1, :]) / temperature

                if top_k and top_k < logits.shape[-1]:
                    values, indices = torch.topk(logits, top_k, dim=-1)
                    filtered = torch.full_like(
                        logits, torch.finfo(logits.dtype).min
                    )
                    filtered.scatter_(1, indices, values)
                    logits = filtered

                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), 1
                )
                generated.append(next_token)
                token = next_token

                if torch.all(next_token.squeeze(-1) == EOS):
                    break

            if not generated:
                return torch.empty(
                    (input_ids.shape[0], 0),
                    dtype=torch.long,
                    device=input_ids.device,
                )
            return torch.cat(generated, dim=1)


def build_native_conversation(config: NativeConversationConfig):
    return NativeConversation(config)


def _prepare_targets(texts, max_bytes):
    rows = []
    for text in texts:
        payload = list(text.encode("utf-8")[:max_bytes])
        rows.append([BOS] + payload + [EOS])
    return rows


def train(args):
    torch_mod, _ = require_torch()
    examples = load_jsonl(args.dataset)
    train_items = [x for x in examples if x.split == "train"]
    val_items = [x for x in examples if x.split == "validation"]
    if not train_items or not val_items:
        raise ValueError("dataset must contain train and validation splits")

    config = NativeConversationConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        layers=args.layers,
        dropout=args.dropout,
        max_bytes=args.max_bytes,
        attention_dim=args.attention_dim,
    )
    model = build_native_conversation(config)
    device = torch_mod.device(
        args.device or ("cuda" if torch_mod.cuda.is_available() else "cpu")
    )
    model.to(device)
    optimizer = torch_mod.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def run_epoch(items, training):
        model.train(training)
        total = 0.0
        batches = 0
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            ui, um = _batch(
                [_encode(x.user_text, config.max_bytes) for x in batch],
                config.max_bytes,
            )
            target_rows = _prepare_targets(
                [x.target_text for x in batch],
                config.max_bytes,
            )
            ti, _ = _batch(target_rows, config.max_bytes)
            ui, um, ti = ui.to(device), um.to(device), ti.to(device)

            with torch_mod.set_grad_enabled(training):
                logits, _, _, _ = model(ui, um, ti[:, :-1])
                labels = ti[:, 1:]
                loss = torch_mod.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=PAD,
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

    history = []
    for epoch in range(1, args.epochs + 1):
        record = {
            "epoch": epoch,
            "train_loss": run_epoch(train_items, True),
            "validation_loss": run_epoch(val_items, False),
        }
        history.append(record)
        print(json.dumps(record))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    torch_mod.save(
        {
            "format": "mirror7-native-conversation-v3-cross-attention",
            "config": config.to_dict(),
            "special_tokens": {"PAD": PAD, "BOS": BOS, "EOS": EOS},
            "vocab_size": VOCAB_SIZE,
            "total_vocab_size": TOTAL_VOCAB_SIZE,
            "parameter_count": parameter_count,
            "history": history,
            "state_dict": model.state_dict(),
        },
        output,
    )
    print(
        json.dumps(
            {"output": str(output), "parameters": parameter_count}
        )
    )


def main():
    p = argparse.ArgumentParser(
        description="Train Mirror 7 native conversation v3."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--output",
        default="artifacts/mirror7_native_conversation_v3.pt",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--state-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--attention-dim", type=int, default=256)
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--device", default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
