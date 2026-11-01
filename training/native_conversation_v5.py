from __future__ import annotations

import argparse
import json
import math
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
    # Kept false by default so checkpoints created before this option remain
    # strict-load compatible. The enhanced factory opts in explicitly.
    recurrent_context_gating: bool = False

    def __post_init__(self):
        if self.embedding_dim <= 0 or self.hidden_dim <= 0 or self.state_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.attention_dim <= 0 or self.num_heads <= 0:
            raise ValueError("attention_dim and num_heads must be positive")
        if self.attention_dim % self.num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        if self.bidirectional_encoder and self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even for a bidirectional encoder")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.scheduled_sampling_p <= 1.0:
            raise ValueError("scheduled_sampling_p must be in [0, 1]")
        if self.max_bytes < 2:
            raise ValueError("max_bytes must be at least 2")

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
    scheduled_sampling_p: float = 0.15,
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
        recurrent_context_gating=True,
    )


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
        self.embedding_dropout = nn_mod.Dropout(config.dropout)
        self.decoder_embedding_dropout = nn_mod.Dropout(config.dropout)

        # Encoder
        self.embedding = nn_mod.Embedding(
            TOTAL_VOCAB_SIZE, config.embedding_dim, padding_idx=PAD
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
        # This separate gate conditions the recurrent input itself. It is
        # optional to retain strict loading of pre-gate V5 checkpoints.
        if config.recurrent_context_gating:
            self.recurrent_gate_linear = nn_mod.Linear(
                config.hidden_dim + attn_dim, attn_dim
            )

        # Input-conditioned Recurrent Decoder Cell
        self.decoder_embedding = nn_mod.Embedding(
            TOTAL_VOCAB_SIZE, config.embedding_dim, padding_idx=PAD
        )
        self.cell = nn_mod.GRUCell(
            config.embedding_dim + attn_dim, config.hidden_dim
        )

        # Output projection head
        self.output_norm = nn_mod.LayerNorm(config.hidden_dim + attn_dim)
        self.head = nn_mod.Linear(
            config.hidden_dim + attn_dim, TOTAL_VOCAB_SIZE
        )

    def init_decoder_hidden(self, state: torch.Tensor) -> torch.Tensor:
        """Map compact semantic state into initial decoder recurrent state (batch, hidden_dim)."""
        return torch.tanh(self.decoder_init(state))

    def encode(self, input_ids: torch.Tensor, input_mask: torch.Tensor):
        """Encode input sequence and precompute key/value attention projections."""
        x, _ = self.encoder(self.embedding_dropout(self.embedding(input_ids)))
        x = self.encoder_norm(x)
        mask = input_mask.to(dtype=x.dtype).unsqueeze(-1)
        x = x * mask
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        state = torch.tanh(self.state(pooled))

        # Precompute keys and values for efficiency
        k = self.key(x)    # (batch, enc_len, attn_dim)
        v = self.value(x)  # (batch, enc_len, attn_dim)
        return x, input_mask.bool(), state, k, v

    def _step_attention(
        self,
        query_state: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        encoder_mask: torch.Tensor,
    ):
        """Compute single-step cross-attention.

        Args:
            query_state: (batch, hidden_dim)
            k: (batch, enc_len, attn_dim)
            v: (batch, enc_len, attn_dim)
            encoder_mask: (batch, enc_len) bool
        Returns:
            context: (batch, attn_dim)
            weights: (batch, enc_len)
        """
        if self.config.num_heads > 1:
            batch_size = query_state.shape[0]
            enc_len = k.shape[1]
            num_heads = self.config.num_heads
            head_dim = self.config.attention_dim // num_heads

            q = self.query(query_state).view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
            k_h = k.view(batch_size, enc_len, num_heads, head_dim).transpose(1, 2)
            v_h = v.view(batch_size, enc_len, num_heads, head_dim).transpose(1, 2)

            scores = torch.matmul(q, k_h.transpose(-2, -1)) / math.sqrt(head_dim)
            mask_expanded = encoder_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, -1e9)
            weights = torch.softmax(scores, dim=-1)
            has_valid = mask_expanded.any(dim=-1, keepdim=True)
            weights = torch.where(has_valid, weights, torch.zeros_like(weights))

            context_h = torch.matmul(weights, v_h)
            context = context_h.transpose(1, 2).contiguous().view(batch_size, self.config.attention_dim)
            if hasattr(self, "attn_out"):
                context = self.attn_out(context)
            mean_weights = weights.mean(dim=1).squeeze(1)
            return self.context_norm(context), mean_weights
        else:
            q = self.query(query_state).unsqueeze(1)  # (batch, 1, attn_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # (batch, 1, enc_len)

            mask_expanded = encoder_mask.unsqueeze(1)  # (batch, 1, enc_len)
            has_valid = mask_expanded.any(dim=-1, keepdim=True)  # (batch, 1, 1)

            scores = scores.masked_fill(~mask_expanded, -1e9)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.where(has_valid, weights, torch.zeros_like(weights))

            context = torch.matmul(weights, v).squeeze(1)  # (batch, attn_dim)
            return self.context_norm(context), weights.squeeze(1)

    def _gate_recurrent_context(
        self, hidden: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        gate_layer = getattr(self, "recurrent_gate_linear", None)
        if gate_layer is None:
            return context
        gate = torch.sigmoid(gate_layer(torch.cat([hidden, context], dim=-1)))
        return context * gate

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        target_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ):
        """Teacher-forced forward pass unrolling input-conditioned recurrence."""
        _, encoder_mask, state, k, v = self.encode(input_ids, input_mask)
        if hidden is None:
            hidden = self.init_decoder_hidden(state)

        target_emb = self.decoder_embedding_dropout(
            self.decoder_embedding(target_ids)
        )  # (batch, seq_len, emb_dim)
        batch_size, seq_len, _ = target_emb.shape

        logits_list = []
        attentions = []
        prev_pred_tok = None

        for t in range(seq_len):
            # 1. Step attention from previous recurrent state
            context, attn_weights = self._step_attention(hidden, k, v, encoder_mask)
            attentions.append(attn_weights)

            # 2. Input-conditioned recurrent step: [emb; context] -> GRUCell
            if (
                self.training
                and self.config.scheduled_sampling_p > 0
                and t > 0
                and prev_pred_tok is not None
            ):
                use_pred = torch.rand(batch_size, 1, device=input_ids.device) < self.config.scheduled_sampling_p
                gt_emb = target_emb[:, t, :]
                pred_emb = self.decoder_embedding_dropout(
                    self.decoder_embedding(prev_pred_tok)
                ).squeeze(1)
                cur_emb = torch.where(use_pred, pred_emb, gt_emb)
            else:
                cur_emb = target_emb[:, t, :]

            recurrent_context = self._gate_recurrent_context(hidden, context)
            cell_input = torch.cat([cur_emb, recurrent_context], dim=-1)
            hidden = self.cell(cell_input, hidden)

            # 3. Output prediction from updated recurrent state and context
            if getattr(self, "gate_linear", None) is not None:
                gate = torch.sigmoid(self.gate_linear(torch.cat([hidden, context], dim=-1)))
                eff_context = context * gate
            else:
                eff_context = context

            fused = self.output_norm(torch.cat([hidden, eff_context], dim=-1))
            logits = self.head(fused)
            logits_list.append(logits)

            if self.training and self.config.scheduled_sampling_p > 0:
                prev_pred_tok = logits.detach().argmax(dim=-1, keepdim=True)

        all_logits = torch.stack(logits_list, dim=1)  # (batch, seq_len, TOTAL_VOCAB_SIZE)
        all_attentions = torch.stack(attentions, dim=1)  # (batch, seq_len, enc_len)
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
    ) -> torch.Tensor:
        """Autoregressive generation using identical input-conditioned recurrence."""
        self.eval()
        with torch.no_grad():
            _, encoder_mask, state, k, v = self.encode(input_ids, input_mask)
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
                # 1. Attention from previous recurrent state
                context, _ = self._step_attention(hidden, k, v, encoder_mask)

                # 2. Input-conditioned recurrent step: [emb; context] -> GRUCell
                emb = self.decoder_embedding_dropout(
                    self.decoder_embedding(token)
                ).squeeze(1)  # (batch, emb_dim)
                recurrent_context = self._gate_recurrent_context(hidden, context)
                cell_input = torch.cat([emb, recurrent_context], dim=-1)
                hidden = self.cell(cell_input, hidden)

                # 3. Predict logits
                if getattr(self, "gate_linear", None) is not None:
                    gate = torch.sigmoid(self.gate_linear(torch.cat([hidden, context], dim=-1)))
                    eff_context = context * gate
                else:
                    eff_context = context

                fused = self.output_norm(torch.cat([hidden, eff_context], dim=-1))
                logits = self.head(fused)

                # Disallow PAD and BOS
                logits[:, PAD] = -1e9
                logits[:, BOS] = -1e9

                # Disallow early EOS if min_new_tokens requested
                if step < min_new_tokens:
                    logits[:, EOS] = -1e9

                # Repetition penalty (spelling-preserving)
                if repetition_penalty != 1.0 and generated:
                    for b in range(batch_size):
                        seen = set(
                            generated[t][b, 0].item()
                            for t in range(max(0, len(generated) - 64), len(generated))
                        )
                        seen.discard(PAD)
                        seen.discard(EOS)
                        seen.discard(32)  # space
                        seen.discard(10)  # newline
                        for tok in seen:
                            if logits[b, tok] > 0:
                                logits[b, tok] /= repetition_penalty
                            else:
                                logits[b, tok] *= repetition_penalty

                # No-repeat n-gram blocking
                if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size - 1:
                    for b in range(batch_size):
                        gen_b = [generated[t][b, 0].item() for t in range(len(generated))]
                        prefix = tuple(gen_b[-(no_repeat_ngram_size - 1):])
                        for i in range(len(gen_b) - no_repeat_ngram_size + 1):
                            if tuple(gen_b[i : i + no_repeat_ngram_size - 1]) == prefix:
                                banned = gen_b[i + no_repeat_ngram_size - 1]
                                logits[b, banned] = -1e9

                if temperature <= 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k and 0 < top_k < logits.shape[-1]:
                        values, indices = torch.topk(logits, top_k, dim=-1)
                        filtered = torch.full_like(logits, -1e9)
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
                        logits = logits.masked_fill(indices_to_remove, -1e9)
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                # Pad finished sequences
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, PAD),
                    next_token,
                )
                generated.append(next_token)

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
        raw = list(user_text.encode("utf-8")[: self.config.max_bytes])
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


def build_native_conversation_v5(config: NativeConversationV5Config):
    return NativeConversationV5(config)


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
    rows = []
    max_payload = max(0, max_bytes - 2)
    for text in texts:
        payload = _clean_truncate(text, max_payload)
        rows.append([BOS] + payload + [EOS])
    return rows


def save_checkpoint(
    path: str | Path,
    model: NativeConversationV5,
    config: NativeConversationV5Config,
    metadata: dict | None = None,
    history: list | None = None,
    training_state: dict | None = None,
) -> None:
    torch_mod, _ = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    payload = {
        "format": "mirror7-native-conversation-v5-input-conditioned",
        "config": config.to_dict(),
        "special_tokens": {"PAD": PAD, "BOS": BOS, "EOS": EOS},
        "vocab_size": VOCAB_SIZE,
        "total_vocab_size": TOTAL_VOCAB_SIZE,
        "parameter_count": parameter_count,
        "metadata": dict(metadata or {}),
        "history": list(history or []),
        "training_state": dict(training_state or {}),
        "state_dict": model.state_dict(),
    }
    torch_mod.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    torch_mod, _ = require_torch()
    path = Path(path)
    payload = torch_mod.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "mirror7-native-conversation-v5-input-conditioned":
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format')!r}"
        )
    raw_config = payload.get("config", {})
    config = NativeConversationV5Config(**raw_config)
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
    p.add_argument("--lr", type=float, default=3e-4)
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
