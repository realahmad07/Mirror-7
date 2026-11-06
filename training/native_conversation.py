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


def _mask_fill_value(tensor: torch.Tensor) -> float:
    """Return a safe negative fill value that does not overflow float16 under AMP."""
    return -1e4 if tensor.dtype == torch.float16 else -1e9


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
            TOTAL_VOCAB_SIZE, config.embedding_dim, padding_idx=PAD
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
        self.decoder_init = nn_mod.Linear(
            config.state_dim, config.layers * config.hidden_dim
        )

        self.decoder_embedding = nn_mod.Embedding(
            TOTAL_VOCAB_SIZE, config.embedding_dim, padding_idx=PAD
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

    def init_decoder_hidden(self, state: torch.Tensor) -> torch.Tensor:
        """Map compact semantic state into initial decoder recurrent state.

        Args:
            state: Tensor of shape (batch_size, state_dim).
        Returns:
            Tensor of shape (layers, batch_size, hidden_dim).
        """
        batch_size = state.shape[0]
        h0 = torch.tanh(self.decoder_init(state))
        return (
            h0.view(batch_size, self.config.layers, self.config.hidden_dim)
            .permute(1, 0, 2)
            .contiguous()
        )

    def encode(self, input_ids: torch.Tensor, input_mask: torch.Tensor):
        x, hidden = self.encoder(self.embedding(input_ids))
        x = self.encoder_norm(x)
        mask = input_mask.to(dtype=x.dtype)
        # Zero out padding representations in encoder states
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(dim=1) / (
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

        mask_expanded = encoder_mask.unsqueeze(1)  # (batch, 1, enc_len)
        has_valid = mask_expanded.any(dim=-1, keepdim=True)  # (batch, 1, 1)

        scores = scores.masked_fill(~mask_expanded, _mask_fill_value(scores))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))

        context = torch.matmul(weights, v)
        return self.context_norm(context), weights

    def forward(self, input_ids, input_mask, target_ids, hidden=None):
        encoder_states, encoder_mask, state, _ = self.encode(
            input_ids, input_mask
        )
        if hidden is None:
            hidden = self.init_decoder_hidden(state)
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
        min_new_tokens=1,
        temperature=0.8,
        top_k=40,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    ):
        self.eval()
        with torch.no_grad():
            encoder_states, encoder_mask, state, _ = self.encode(
                input_ids, input_mask
            )
            hidden = self.init_decoder_hidden(state)
            batch_size = input_ids.shape[0]
            token = torch.full(
                (batch_size, 1),
                BOS,
                dtype=torch.long,
                device=input_ids.device,
            )
            generated = []
            finished = torch.zeros(
                batch_size, dtype=torch.bool, device=input_ids.device
            )

            for step in range(max_new_tokens):
                decoder_states, hidden = self.decoder(
                    self.decoder_embedding(token), hidden
                )
                context, _ = self._cross_attention(
                    decoder_states, encoder_states, encoder_mask
                )
                fused = self.output_norm(
                    torch.cat([decoder_states, context], dim=-1)
                )
                logits = self.head(fused[:, -1, :])

                # Disallow generating PAD and BOS
                neg_inf = _mask_fill_value(logits)
                logits[:, PAD] = neg_inf
                logits[:, BOS] = neg_inf

                # Disallow early EOS if min_new_tokens requested
                if step < min_new_tokens:
                    logits[:, EOS] = neg_inf

                # Apply repetition penalty
                if repetition_penalty != 1.0 and generated:
                    for b in range(batch_size):
                        seen = set(
                            generated[t][b, 0].item()
                            for t in range(max(0, len(generated) - 64), len(generated))
                        )
                        seen.discard(PAD)
                        seen.discard(EOS)
                        for tok in seen:
                            if logits[b, tok] > 0:
                                logits[b, tok] /= repetition_penalty
                            else:
                                logits[b, tok] *= repetition_penalty

                # Apply no-repeat n-gram blocking
                if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size - 1:
                    for b in range(batch_size):
                        gen_b = [generated[t][b, 0].item() for t in range(len(generated))]
                        prefix = tuple(gen_b[-(no_repeat_ngram_size - 1):])
                        for i in range(len(gen_b) - no_repeat_ngram_size + 1):
                            if tuple(gen_b[i : i + no_repeat_ngram_size - 1]) == prefix:
                                banned = gen_b[i + no_repeat_ngram_size - 1]
                                logits[b, banned] = neg_inf

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

                # For sequences that already finished, emit PAD
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, PAD),
                    next_token,
                )
                generated.append(next_token)

                # Update finished status
                is_eos = (next_token.squeeze(-1) == EOS)
                finished = finished | is_eos

                if finished.all():
                    break
                token = next_token

            if not generated:
                return torch.empty(
                    (batch_size, 0),
                    dtype=torch.long,
                    device=input_ids.device,
                )
            return torch.cat(generated, dim=1)

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
    ) -> str:
        device = next(self.parameters()).device
        raw = list(user_text.encode("utf-8")[:self.config.max_bytes])
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
        )
        token_list = output_tokens[0].tolist()
        byte_values = []
        for tok in token_list:
            if tok == EOS:
                break
            if 0 <= tok <= 255:
                byte_values.append(tok)
        return bytes(byte_values).decode("utf-8", errors="replace")


def build_native_conversation(config: NativeConversationConfig):
    return NativeConversation(config)


def _clean_truncate(text: str, max_payload: int) -> list[int]:
    raw = text.encode("utf-8")
    if len(raw) <= max_payload:
        return list(raw)
    sliced = raw[:max_payload]
    for i in range(len(sliced) - 1, max(0, len(sliced) - 64), -1):
        if sliced[i] in (32, 10, 46, 33, 63):
            return list(sliced[: i + 1])
    while sliced:
        try:
            sliced.decode("utf-8")
            return list(sliced)
        except UnicodeDecodeError:
            sliced = sliced[:-1]
    return []


def _prepare_targets(texts, max_bytes):
    """Encode target texts into [BOS, bytes..., EOS] preserving special tokens.

    To guarantee BOS and EOS are never truncated by _batch(..., max_bytes),
    the raw payload is clamped to at most max_bytes - 2 and clean-truncated
    at word boundaries to avoid training on mid-word cutoffs.
    """
    rows = []
    max_payload = max(0, max_bytes - 2)
    for text in texts:
        payload = _clean_truncate(text, max_payload)
        rows.append([BOS] + payload + [EOS])
    return rows


def save_checkpoint(
    path: str | Path,
    model: NativeConversation,
    config: NativeConversationConfig,
    metadata: dict | None = None,
    history: list | None = None,
) -> None:
    torch_mod, _ = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    payload = {
        "format": "mirror7-native-conversation-v3-cross-attention",
        "config": config.to_dict(),
        "special_tokens": {"PAD": PAD, "BOS": BOS, "EOS": EOS},
        "vocab_size": VOCAB_SIZE,
        "total_vocab_size": TOTAL_VOCAB_SIZE,
        "parameter_count": parameter_count,
        "metadata": dict(metadata or {}),
        "history": list(history or []),
        "state_dict": model.state_dict(),
    }
    torch_mod.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    torch_mod, _ = require_torch()
    path = Path(path)
    payload = torch_mod.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "mirror7-native-conversation-v3-cross-attention":
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format')!r}"
        )
    raw_config = payload.get("config", {})
    config = NativeConversationConfig(**raw_config)
    model = build_native_conversation(config)
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
    if getattr(args, "resume", None):
        model, loaded_payload = load_checkpoint(args.resume, device=str(device))
        config = model.config
        history = list(loaded_payload.get("history", []))
    else:
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
        model.to(device)

    optimizer = torch_mod.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if getattr(args, "use_scheduler", False):
        scheduler = torch_mod.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )

    label_smoothing = getattr(args, "label_smoothing", 0.0)
    should_shuffle = getattr(args, "shuffle", True)

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
            )

    save_checkpoint(
        output,
        model,
        config,
        metadata={"epochs": start_epoch + args.epochs - 1, "lr": args.lr},
        history=history,
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
        description="Train Mirror 7 native conversation v3."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--output",
        default="artifacts/mirror7_native_conversation_v3.pt",
    )
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--use-scheduler", action="store_true", default=False)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no-shuffle", action="store_false", dest="shuffle")
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
