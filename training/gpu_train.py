"""
Mirror 7 Portable GPU/CPU Training Pipeline.

Designed for Google Colab (T4, V100, A100) and local environments.
Supports:
- Automatic CUDA/MPS/CPU detection
- Mixed precision (FP16/BF16 AMP)
- Gradient accumulation and clipping
- Dynamic sequence length bucketed batching (up to 2048 bytes)
- Cosine Annealing LR scheduling
- Dataset fingerprint verification
- Best-checkpoint selection and resume support
- Memory (VRAM) and throughput (bytes/sec, tokens/sec) metrics
- Post-training generation evaluation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from .dataset import load_jsonl, validate_split_isolation, summarize
from .schema import TrainingExample
from .native_conversation_v5 import (
    NativeConversationV5,
    NativeConversationV5Config,
    build_native_conversation_v5,
    save_checkpoint,
    load_checkpoint,
    _batch,
    _encode,
    _prepare_targets,
    PAD,
    BOS,
    EOS,
)

logger = logging.getLogger("mirror7_gpu_train")


def accumulation_group_size(total_batches: int, batch_index: int, grad_accum_steps: int) -> int:
    """Return actual microbatch count in the accumulation group containing index."""
    if total_batches < 0 or grad_accum_steps <= 0:
        raise ValueError("total_batches must be non-negative and grad_accum_steps positive")
    if not 0 <= batch_index < total_batches:
        raise IndexError("batch_index is outside the epoch")
    group_start = (batch_index // grad_accum_steps) * grad_accum_steps
    return min(grad_accum_steps, total_batches - group_start)


def verify_resume_fingerprint(
    checkpoint_fingerprint: str | None,
    dataset_fingerprint: str,
    *,
    allow_unverified: bool = False,
) -> None:
    """Fail closed on changed data; require explicit opt-in for legacy checkpoints."""
    if checkpoint_fingerprint is None:
        if allow_unverified:
            return
        raise ValueError(
            "resume checkpoint has no dataset fingerprint; pass "
            "allow_unverified_resume=True only after manually verifying its dataset"
        )
    if checkpoint_fingerprint != dataset_fingerprint:
        raise ValueError(
            "resume dataset fingerprint mismatch: checkpoint was trained on "
            f"{checkpoint_fingerprint}, current dataset is {dataset_fingerprint}"
        )


class DynamicBucketSampler:
    """Groups examples of similar target length into batches to minimize padding waste."""

    def __init__(self, examples: Sequence[TrainingExample], batch_size: int, max_bytes: int):
        self.examples = list(examples)
        self.batch_size = batch_size
        self.max_bytes = max_bytes

    def get_batches(self, shuffle: bool = True) -> list[list[TrainingExample]]:
        # Sort by target length for bucketing
        sorted_examples = sorted(
            self.examples,
            key=lambda x: max(
                len(x.user_text.encode("utf-8")),
                len(x.target_text.encode("utf-8")),
            ),
        )
        batches: list[list[TrainingExample]] = []
        for i in range(0, len(sorted_examples), self.batch_size):
            batches.append(sorted_examples[i : i + self.batch_size])

        if shuffle:
            random.shuffle(batches)
        return batches


def get_hardware_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["device_name"] = props.name
        info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
        info["cuda_capability"] = f"{props.major}.{props.minor}"
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    else:
        info["device_name"] = "CPU / Host RAM"
        info["total_vram_gb"] = 0.0
        info["bf16_supported"] = False
    return info


def evaluate_model(
    model: NativeConversationV5,
    examples: Sequence[TrainingExample],
    device: torch.device,
    max_eval_examples: int = 50,
) -> dict[str, float]:
    model.eval()
    config = model.config
    eval_subset = list(examples)[:max_eval_examples]
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for i in range(0, len(eval_subset), 16):
            batch = eval_subset[i : i + 16]
            ui, um = _batch(
                [_encode(x.user_text, config.max_bytes) for x in batch],
                config.max_bytes,
            )
            target_rows = _prepare_targets([x.target_text for x in batch], config.max_bytes)
            ti, _ = _batch(target_rows, config.max_bytes)
            ui, um, ti = ui.to(device), um.to(device), ti.to(device)

            logits, _, _, _ = model(ui, um, ti[:, :-1])
            labels = ti[:, 1:]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=PAD,
            )
            total_loss += float(loss.item())
            total_batches += 1

    return {"eval_loss": total_loss / max(1, total_batches)}


def train_gpu(
    dataset_path: str | Path,
    output_path: str | Path,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_bytes: int = 1024,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    device_str: str | None = None,
    resume_path: str | Path | None = None,
    smoke_test_steps: int = 0,
    bidirectional_encoder: bool = True,
    num_heads: int = 4,
    context_gating: bool = True,
    scheduled_sampling_p: float = 0.15,
    recurrent_context_gating: bool = True,
    allow_unverified_resume: bool = False,
) -> dict[str, Any]:
    """Main training routine optimized for GPU execution."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_output = output.with_name(f"{output.stem}.best{output.suffix}")

    # 1. Device Selection
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    hw_info = get_hardware_info(device)
    print(f"Hardware Config: {json.dumps(hw_info, indent=2)}")

    # 2. Dataset Loading & Fingerprint Verification
    print(f"Loading dataset: {dataset_path}")
    all_examples = load_jsonl(dataset_path)
    validate_split_isolation(all_examples)
    summary = summarize(all_examples)
    print(f"Dataset Verified. Fingerprint: {summary['fingerprint']}")
    print(f"Total: {summary['total']}, Splits: {summary['splits']}")

    train_examples = [x for x in all_examples if x.split == "train"]
    val_examples = [x for x in all_examples if x.split == "validation"]
    if not train_examples or not val_examples:
        raise ValueError("dataset must contain non-empty train and validation splits")
    dataset_fingerprint = str(summary["fingerprint"])

    # 3. Model Initialization or Resume
    history: list[dict[str, Any]] = []
    resume_payload: dict[str, Any] | None = None
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
        model, resume_payload = load_checkpoint(resume_path, device=str(device))
        config = model.config
        history = list(resume_payload.get("history", []))
        checkpoint_fingerprint = resume_payload.get("metadata", {}).get("dataset_fingerprint")
        verify_resume_fingerprint(
            checkpoint_fingerprint,
            dataset_fingerprint,
            allow_unverified=allow_unverified_resume,
        )
    else:
        config = NativeConversationV5Config(
            embedding_dim=96,
            hidden_dim=256,
            state_dim=128,
            attention_dim=128,
            dropout=0.10,
            max_bytes=max_bytes,
            bidirectional_encoder=bidirectional_encoder,
            num_heads=num_heads,
            context_gating=context_gating,
            scheduled_sampling_p=scheduled_sampling_p,
            recurrent_context_gating=recurrent_context_gating,
        )
        model = build_native_conversation_v5(config)
        model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {param_count:,} parameters (Architecture: V5 Enhanced [BiGRU={config.bidirectional_encoder}, MHA={config.num_heads}, Gated={config.context_gating}])")

    if batch_size <= 0 or grad_accum_steps <= 0 or epochs <= 0:
        raise ValueError("epochs, batch_size and grad_accum_steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    resume_training_state = (resume_payload or {}).get("training_state", {})
    resume_metadata = (resume_payload or {}).get("metadata", {})
    scheduled_total_epochs = int(
        resume_metadata.get("scheduled_total_epochs", len(history) + epochs)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, scheduled_total_epochs), eta_min=lr * 0.05
    )

    # Mixed precision setup
    use_cuda_amp = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if hw_info.get("bf16_supported", False) else torch.float16
    scaler = GradScaler("cuda", enabled=use_cuda_amp and amp_dtype == torch.float16)

    if resume_payload:
        if resume_training_state.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_training_state["optimizer_state_dict"])
        if resume_training_state.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_training_state["scheduler_state_dict"])
        if scaler.is_enabled() and resume_training_state.get("scaler_state_dict"):
            scaler.load_state_dict(resume_training_state["scaler_state_dict"])

    sampler = DynamicBucketSampler(train_examples, batch_size=batch_size, max_bytes=config.max_bytes)
    start_epoch = len(history) + 1
    best_val_loss = float(resume_metadata.get("best_val_loss", float("inf")))
    start_time = time.perf_counter()

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0.0
        batches = sampler.get_batches(shuffle=True)
        effective_batch_count = min(
            len(batches), smoke_test_steps if smoke_test_steps > 0 else len(batches)
        )
        batches = batches[:effective_batch_count]
        step_count = 0
        epoch_bytes_processed = 0
        epoch_start_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        for b_idx, batch in enumerate(batches):
            ui, um = _batch(
                [_encode(x.user_text, config.max_bytes) for x in batch],
                config.max_bytes,
            )
            target_rows = _prepare_targets([x.target_text for x in batch], config.max_bytes)
            ti, _ = _batch(target_rows, config.max_bytes)
            ui, um, ti = ui.to(device), um.to(device), ti.to(device)

            epoch_bytes_processed += int(ui.numel() + ti.numel())

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda_amp):
                logits, _, _, _ = model(ui, um, ti[:, :-1])
                labels = ti[:, 1:]
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=PAD,
                )
                raw_loss_value = float(loss.detach().item())
                # Normalize a partial final accumulation group by its actual
                # number of microbatches, not the configured full group size.
                group_size = accumulation_group_size(
                    len(batches), b_idx, grad_accum_steps
                )
                loss = loss / max(1, group_size)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(batches):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += raw_loss_value
            step_count += 1

            if smoke_test_steps > 0 and step_count >= smoke_test_steps:
                print(f"Smoke test complete at step {step_count}.")
                break

        scheduler.step()
        train_loss = epoch_loss / max(1, step_count)
        val_metrics = evaluate_model(model, val_examples, device=device)
        val_loss = val_metrics["eval_loss"]

        epoch_duration = time.perf_counter() - epoch_start_time
        throughput_bytes = epoch_bytes_processed / max(0.001, epoch_duration)

        peak_vram_mb = 0.0
        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "validation_loss": round(val_loss, 4),
            "duration_seconds": round(epoch_duration, 2),
            "throughput_bytes_sec": round(throughput_bytes, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        training_state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else {},
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_output,
                model,
                config,
                metadata={
                    "epochs": epoch,
                    "best_val_loss": best_val_loss,
                    "hw": hw_info,
                    "dataset_fingerprint": dataset_fingerprint,
                    "scheduled_total_epochs": scheduled_total_epochs,
                },
                history=history,
                training_state=training_state,
            )

        if smoke_test_steps > 0:
            break

    # Save final checkpoint
    save_checkpoint(
        output,
        model,
        config,
        metadata={
            "epochs": epoch,
            "final_val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "hw": hw_info,
            "dataset_fingerprint": dataset_fingerprint,
            "scheduled_total_epochs": scheduled_total_epochs,
        },
        history=history,
        training_state={
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else {},
        },
    )

    # Quick generation sample check
    model.eval()
    test_prompt = "Explain how cross-attention works in 2 sentences."
    sample_out = model.generate_text(test_prompt, max_new_tokens=128, temperature=0.7)
    safe_out = sample_out.encode("ascii", errors="backslashreplace").decode("ascii")
    print("\n--- Generation Check ---")
    print(f"Prompt: {test_prompt}")
    print(f"Response: {safe_out}\n")

    return {
        "status": "completed",
        "output_checkpoint": str(output),
        "best_checkpoint": str(best_output),
        "history": history,
        "parameters": param_count,
        "sample_output": sample_out,
    }


def main():
    p = argparse.ArgumentParser(description="Mirror 7 GPU/CPU Training Pipeline")
    p.add_argument("--dataset", default="training/data/mirror_oasst2_curated_v1.jsonl")
    p.add_argument("--output", default="artifacts/mirror7_gpu_model.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-bytes", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--no-amp", action="store_true", default=False)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke-test-steps", type=int, default=0)
    p.add_argument("--no-bidirectional", action="store_true", default=False)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--no-context-gating", action="store_true", default=False)
    p.add_argument("--scheduled-sampling-p", type=float, default=0.15)
    p.add_argument("--no-recurrent-context-gating", action="store_true", default=False)
    p.add_argument("--allow-unverified-resume", action="store_true", default=False)
    args = p.parse_args()

    train_gpu(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_bytes=args.max_bytes,
        grad_accum_steps=args.grad_accum,
        use_amp=not args.no_amp,
        device_str=args.device,
        resume_path=args.resume,
        smoke_test_steps=args.smoke_test_steps,
        bidirectional_encoder=not args.no_bidirectional,
        num_heads=args.num_heads,
        context_gating=not args.no_context_gating,
        scheduled_sampling_p=args.scheduled_sampling_p,
        recurrent_context_gating=not args.no_recurrent_context_gating,
        allow_unverified_resume=args.allow_unverified_resume,
    )


if __name__ == "__main__":
    main()
