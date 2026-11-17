"""Training, evaluation (loss / perplexity / bits-per-byte) and atomic checkpointing."""
from __future__ import annotations

import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .data import IGNORE_INDEX, BlockBatcher, SFTBatcher, fingerprint_tensor
from .model import ModelConfig, Transformer
from .tokenizer import BPETokenizer

LN2 = math.log(2.0)


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 50
    seed: int = 1337
    device: str = "cpu"
    use_bf16: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: Transformer, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "tok_emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.steps:
        return cfg.lr * cfg.min_lr_ratio
    span = max(cfg.steps - cfg.warmup_steps, 1)
    prog = (step - cfg.warmup_steps) / span
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * coeff)


@torch.no_grad()
def evaluate_blocks(
    model: Transformer,
    tok: BPETokenizer,
    blocks: torch.Tensor,
    batch_size: int = 16,
) -> dict[str, float]:
    """Compute exact mean token NLL, perplexity, and tokenizer-independent bits-per-byte."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    byte_lens = torch.tensor(tok.token_byte_lengths(), dtype=torch.float64, device=device)

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0.0
    for i in range(0, len(blocks), batch_size):
        chunk = blocks[i : i + batch_size].to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        _, loss_tok = model(x, y, reduction="none")
        mask = y != IGNORE_INDEX
        total_nll += float(loss_tok[mask].double().sum().item())
        total_tokens += int(mask.sum().item())
        valid_y = y[mask]
        total_bytes += float(byte_lens[valid_y].sum().item())

    if was_training:
        model.train()
    mean_loss = total_nll / max(total_tokens, 1)
    bpb = (total_nll / LN2) / max(total_bytes, 1.0)
    return {
        "loss": round(mean_loss, 5),
        "perplexity": round(math.exp(min(mean_loss, 20.0)), 3),
        "bits_per_byte": round(bpb, 4),
        "tokens": total_tokens,
        "bytes": int(total_bytes),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    tok: BPETokenizer,
    step: int,
    train_cfg: TrainConfig,
    batcher_state: dict[str, Any],
    data_fingerprint: str,
    metrics: list[dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokenizer": tok.to_dict(),
        "step": step,
        "train_config": asdict(train_cfg),
        "batcher_state": batcher_state,
        "data_fingerprint": data_fingerprint,
        "rng_torch": torch.get_rng_state(),
        "rng_py": random.getstate(),
        "metrics": metrics,
    }
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def train_loop(
    model: Transformer,
    tok: BPETokenizer,
    train_data: torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]],
    val_blocks: torch.Tensor | None,
    cfg: TrainConfig,
    *,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    max_wall_seconds: float | None = None,
    stop_at_step: int | None = None,
) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)

    if isinstance(train_data, torch.Tensor):
        batcher: BlockBatcher | SFTBatcher = BlockBatcher(train_data, cfg.batch_size, seed=cfg.seed)
        fp = fingerprint_tensor(train_data)
    else:
        batcher = SFTBatcher(train_data, cfg.batch_size, tok.pad_id, seed=cfg.seed)
        fp = f"sft:{len(train_data)}"

    start_step = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if ckpt.get("data_fingerprint") != fp:
            raise ValueError("checkpoint data_fingerprint does not match current training data")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        batcher.load_state_dict(ckpt["batcher_state"])
        torch.set_rng_state(ckpt["rng_torch"])
        random.setstate(ckpt["rng_py"])
        start_step = int(ckpt["step"])
        history = list(ckpt.get("metrics", []))

    model.train()
    t0 = time.monotonic()
    tokens_seen = 0
    step = start_step
    target_step = min(cfg.steps, stop_at_step) if stop_at_step is not None else cfg.steps
    use_amp = cfg.use_bf16 and device.type == "cuda"

    while step < target_step:
        if max_wall_seconds is not None and (time.monotonic() - t0) >= max_wall_seconds and step > start_step:
            break
        lr = cosine_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = batcher.next()
            x, y = x.to(device), y.to(device)
            tokens_seen += int((y != IGNORE_INDEX).sum().item())
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            loss_accum += float(loss.item()) / cfg.grad_accum
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1

        if (cfg.eval_every and step % cfg.eval_every == 0) or step == cfg.steps:
            rec: dict[str, Any] = {
                "step": step,
                "train_loss": round(loss_accum, 5),
                "lr": round(lr, 7),
                "tokens_seen": tokens_seen,
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
            if val_blocks is not None:
                rec["val"] = evaluate_blocks(model, tok, val_blocks)
            history.append(rec)

    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            tok=tok,
            step=step,
            train_cfg=cfg,
            batcher_state=batcher.state_dict(),
            data_fingerprint=fp,
            metrics=history,
        )
    final_val = evaluate_blocks(model, tok, val_blocks) if val_blocks is not None else None
    return {
        "steps_completed": step,
        "tokens_seen": tokens_seen,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "final_val": final_val,
        "history": history,
    }
