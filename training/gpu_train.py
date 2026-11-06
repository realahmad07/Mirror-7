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
import math
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
    MODEL_SCALE_PRESETS,
    make_v5_preset_config,
    build_native_conversation_v5,
    configure_cpu_runtime,
    apply_lora_to_model,
    merge_lora_weights,
    save_checkpoint,
    load_checkpoint,
    expand_examples_into_chunks,
    _batch,
    _encode,
    _encode_prompt,
    _encode_example_prompt,
    _prepare_targets,
    PAD,
    BOS,
    EOS,
)

logger = logging.getLogger("mirror7_gpu_train")


class DynamicBucketSampler:
    """Groups examples of similar target length into batches and supports progressive curriculum ordering.

    If ``max_tokens_per_batch`` is set, batches are formed by a padded-token budget instead of a fixed
    example count: buckets of short examples get more rows, buckets of long examples get fewer
    (capped at ``max_batch_size``). With the default (``None``), behavior is the fixed ``batch_size``.
    """

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        batch_size: int,
        max_bytes: int,
        max_tokens_per_batch: int | None = None,
        max_batch_size: int = 256,
        lengths: dict[int, tuple[int, int]] | None = None,
    ):
        self.examples = list(examples)
        self.batch_size = batch_size
        self.max_bytes = max_bytes
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_batch_size = max(1, max_batch_size)
        self._lengths = lengths

    def _lens(self, ex: TrainingExample) -> tuple[int, int]:
        """(prompt_len, target_len) in padded tokens, capped at max_bytes."""
        if self._lengths is not None and id(ex) in self._lengths:
            return self._lengths[id(ex)]
        p = min(self.max_bytes, len(ex.user_text))
        t = min(self.max_bytes, len(ex.target_text) + 2)
        return max(1, p), t

    def _target_len(self, ex: TrainingExample) -> int:
        if self._lengths is not None and id(ex) in self._lengths:
            return self._lengths[id(ex)][1]
        return len(ex.target_text)

    def _form_batches(self, sorted_examples: list[TrainingExample]) -> list[list[TrainingExample]]:
        if not self.max_tokens_per_batch:
            return [
                sorted_examples[i : i + self.batch_size]
                for i in range(0, len(sorted_examples), self.batch_size)
            ]
        budget = int(self.max_tokens_per_batch)
        batches: list[list[TrainingExample]] = []
        current: list[TrainingExample] = []
        max_p = max_t = 0
        for ex in sorted_examples:
            p, t = self._lens(ex)
            new_p, new_t = max(max_p, p), max(max_t, t)
            if current and (
                (len(current) + 1) * (new_p + new_t) > budget
                or len(current) >= self.max_batch_size
            ):
                batches.append(current)
                current, max_p, max_t = [], 0, 0
                new_p, new_t = p, t
            current.append(ex)
            max_p, max_t = new_p, new_t
        if current:
            batches.append(current)
        return batches

    def get_batches(
        self,
        shuffle: bool = True,
        epoch: int = 1,
        total_epochs: int = 5,
        curriculum: bool = True,
    ) -> list[list[TrainingExample]]:
        if shuffle:
            # Add slight length jitter so similar-length examples mix into fresh combinations each epoch
            sorted_examples = sorted(
                self.examples,
                key=lambda x: self._target_len(x) + random.randint(-24, 24),
            )
        else:
            sorted_examples = sorted(self.examples, key=self._target_len)
        batches = self._form_batches(sorted_examples)

        if shuffle:
            if curriculum and epoch == 1 and len(batches) >= 4:
                # Curriculum Stage (Epoch 1): progress through 3 macro-tiers (short/simple -> medium -> long/multi-step)
                # while shuffling batches within each macro-tier so gradients remain well-conditioned.
                tier_size = max(1, len(batches) // 3)
                tier1 = batches[:tier_size]
                tier2 = batches[tier_size : 2 * tier_size]
                tier3 = batches[2 * tier_size :]
                random.shuffle(tier1)
                random.shuffle(tier2)
                random.shuffle(tier3)
                batches = tier1 + tier2 + tier3
            else:
                random.shuffle(batches)
        return batches


class EncodedExampleCache:
    """Caches encoded byte-id tensors across epochs so UTF-8/truncation work runs at most once per example.

    Keys are ``id(example)``; the cache keeps references to the examples so ids stay valid.
    """

    def __init__(self, examples: Sequence[TrainingExample], max_bytes: int, eager: bool = False):
        self.max_bytes = max_bytes
        self._examples = list(examples)
        self._prompt: dict[int, torch.Tensor] = {}
        self._target: dict[int, torch.Tensor] = {}
        if eager:
            for ex in self._examples:
                self._add(ex)

    def _add(self, ex: TrainingExample) -> None:
        p = _encode_example_prompt(ex, self.max_bytes)
        t = _prepare_targets([ex.target_text], self.max_bytes)[0]
        self._prompt[id(ex)] = torch.tensor(p, dtype=torch.long)
        self._target[id(ex)] = torch.tensor(t, dtype=torch.long)

    def lengths(self) -> dict[int, tuple[int, int]]:
        return {
            key: (max(1, int(self._prompt[key].numel())), int(self._target[key].numel()))
            for key in self._prompt
        }

    def get(self, ex: TrainingExample) -> tuple[torch.Tensor, torch.Tensor]:
        key = id(ex)
        if key not in self._prompt:
            self._examples.append(ex)
            self._add(ex)
        return self._prompt[key], self._target[key]


def _pad_rows(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad 1-D id tensors with PAD; returns (ids, float mask). Matches ``_batch`` exactly."""
    max_len = max(1, max(int(r.numel()) for r in rows))
    ids = torch.full((len(rows), max_len), PAD, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.float32)
    for i, r in enumerate(rows):
        n = int(r.numel())
        if n:
            ids[i, :n] = r
            mask[i, :n] = 1.0
    return ids, mask


def collate_batch(
    batch: Sequence[TrainingExample],
    cache: EncodedExampleCache | None,
    max_bytes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (prompt_ids, prompt_mask, target_ids) for a batch and move them to ``device``.

    On CUDA, host tensors are pinned and copied asynchronously (``non_blocking=True``).
    """
    if cache is not None:
        pairs = [cache.get(ex) for ex in batch]
        ui, um = _pad_rows([p for p, _ in pairs])
        ti, _ = _pad_rows([t for _, t in pairs])
    else:
        ui, um = _batch([_encode_example_prompt(x, max_bytes) for x in batch], max_bytes)
        ti, _ = _batch(_prepare_targets([x.target_text for x in batch], max_bytes), max_bytes)
    if device.type == "cuda":
        ui, um, ti = ui.pin_memory(), um.pin_memory(), ti.pin_memory()
        return (
            ui.to(device, non_blocking=True),
            um.to(device, non_blocking=True),
            ti.to(device, non_blocking=True),
        )
    return ui.to(device), um.to(device), ti.to(device)


def build_optimizer(model: nn.Module, lr: float, device: torch.device) -> torch.optim.Optimizer:
    """AdamW over trainable parameters using the fused CUDA kernel when available."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        trainable_params = list(model.parameters())
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2, fused=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            print(f"Note: fused AdamW unavailable ({exc}); using standard AdamW.")
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)


def get_hardware_info(device: torch.device, cpu_threads: int | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["device_name"] = props.name
        info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
        info["cuda_capability"] = f"{props.major}.{props.minor}"
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        # Enable Ampere+ (A100 / L4 / H100) TF32 Tensor Cores & cuDNN autotuner automatically
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            info["tf32_tensor_cores_enabled"] = True
        else:
            info["tf32_tensor_cores_enabled"] = False
        torch.backends.cudnn.benchmark = True
    else:
        cpu_cfg = configure_cpu_runtime(num_threads=cpu_threads)
        info["device_name"] = "CPU / Host RAM"
        info["total_vram_gb"] = 0.0
        info["bf16_supported"] = False
        info["cpu_bf16_supported"] = True
        info.update(cpu_cfg)
    return info


def evaluate_model(
    model: NativeConversationV5,
    examples: Sequence[TrainingExample],
    device: torch.device,
    max_eval_examples: int = 50,
    cache: EncodedExampleCache | None = None,
) -> dict[str, float]:
    model.eval()
    config = model.config
    eval_subset = list(examples)[:max_eval_examples]
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for i in range(0, len(eval_subset), 16):
            batch = eval_subset[i : i + 16]
            ui, um, ti = collate_batch(batch, cache, config.max_bytes, device)

            logits, _, _, _ = model(ui, um, ti[:, :-1], return_attentions=False)
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
    lr: float = 1e-3,
    max_bytes: int = 1024,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    device_str: str | None = None,
    resume_path: str | Path | None = None,
    smoke_test_steps: int = 0,
    bidirectional_encoder: bool = True,
    num_heads: int | None = None,
    context_gating: bool = True,
    scheduled_sampling_p: float = 0.0,
    chunk_long_examples: bool = True,
    save_every_steps: int = 100,
    max_tokens_per_batch: int | None = None,
    use_cpu_amp: bool = False,
    cpu_threads: int | None = None,
    lora_rank: int = 0,
    checkpoint_segment_size: int | None = None,
    num_moe_experts: int | None = None,
    moe_top_k: int | None = None,
    byte_patch_size: int | None = None,
    memory_bank_slots: int | None = None,
    model_scale: str = "base",
    embedding_dim: int | None = None,
    hidden_dim: int | None = None,
    state_dim: int | None = None,
    attention_dim: int | None = None,
    val_subset_size: int = 50,
) -> dict[str, Any]:
    """Main training routine optimized for GPU and CPU execution with zero data loss and warmup-cosine LR.

    ``val_subset_size`` controls how many validation examples are scored each epoch. The
    default (50) keeps the historical behaviour; use 1000+ for a statistically meaningful
    generalization signal on large mixed-domain datasets.
    """
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

    hw_info = get_hardware_info(device, cpu_threads=cpu_threads)
    print(f"Hardware Config: {json.dumps(hw_info, indent=2)}")

    # 2. Dataset Loading & Fingerprint Verification
    print(f"Loading dataset: {dataset_path}")
    all_examples = load_jsonl(dataset_path)
    validate_split_isolation(all_examples)
    summary = summarize(all_examples)
    print(f"Dataset Verified. Fingerprint: {summary['fingerprint']}")
    print(f"Total: {summary['total']}, Splits: {summary['splits']}")

    raw_train_examples = [x for x in all_examples if x.split == "train"]
    val_examples = [x for x in all_examples if x.split == "validation"]

    # 3. Model Initialization or Resume
    history: list[dict[str, Any]] = []
    loaded_payload: dict[str, Any] | None = None
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        model, loaded_payload = load_checkpoint(resume_path, device=str(device))
        config = model.config
        history = list(loaded_payload.get("history", []))
    else:
        config = make_v5_preset_config(
            scale=model_scale,
            max_bytes=max_bytes,
            dropout=0.10,
            bidirectional_encoder=bidirectional_encoder,
            context_gating=context_gating,
            scheduled_sampling_p=scheduled_sampling_p,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            checkpoint_segment_size=checkpoint_segment_size,
            num_moe_experts=num_moe_experts,
            moe_top_k=moe_top_k,
            byte_patch_size=byte_patch_size,
            memory_bank_slots=memory_bank_slots,
        )
        model = build_native_conversation_v5(config)
        model.to(device)

    lora_info: dict[str, Any] | None = None
    if lora_rank > 0:
        lora_info = apply_lora_to_model(model, rank=lora_rank, alpha=float(lora_rank * 2))
        print(
            f"LoRA CPU/GPU Adapter Active: rank={lora_rank}, "
            f"trainable={lora_info['trainable_params']:,}/{lora_info['total_params']:,} "
            f"({lora_info['trainable_ratio'] * 100:.1f}%)"
        )

    if chunk_long_examples:
        train_examples = expand_examples_into_chunks(raw_train_examples, config.max_bytes)
        print(
            f"Zero-Data-Loss Chunking: {len(raw_train_examples):,} raw train examples -> "
            f"{len(train_examples):,} continuation-preserving segments (max_bytes={config.max_bytes})"
        )
    else:
        train_examples = raw_train_examples

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Model initialized: {param_count:,} parameters "
        f"(Architecture: V5 Enhanced [BiGRU={config.bidirectional_encoder}, MHA={config.num_heads}, Gated={config.context_gating}])"
    )

    start_epoch = len(history) + 1
    optimizer = build_optimizer(model, lr=lr, device=device)
    steps_per_epoch = max(1, math.ceil(len(train_examples) / (batch_size * max(1, grad_accum_steps))))
    total_steps = max(1, (start_epoch - 1 + epochs if loaded_payload else epochs) * steps_per_epoch)
    warmup_steps = min(50, max(1, int(0.02 * total_steps)))

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return 0.50 + 0.50 * (float(current_step + 1) / float(warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.15 + 0.85 * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    if loaded_payload is not None and lora_rank <= 0:
        if "optimizer_state_dict" in loaded_payload:
            try:
                optimizer.load_state_dict(loaded_payload["optimizer_state_dict"])
                for pg in optimizer.param_groups:
                    pg.setdefault("initial_lr", lr)
                print("Restored optimizer momentum/variance state from checkpoint.")
            except Exception as exc:
                print(f"Note: Starting fresh optimizer state ({exc})")
        if "scheduler_state_dict" in loaded_payload:
            try:
                scheduler.load_state_dict(loaded_payload["scheduler_state_dict"])
                print("Restored LR scheduler state from checkpoint.")
            except Exception as exc:
                print(f"Note: Starting fresh LR scheduler state ({exc})")

    # Mixed precision setup (supports both CUDA FP16/BF16 and CPU BF16)
    use_cuda_amp = use_amp and device.type == "cuda"
    enable_cpu_bf16 = bool(use_cpu_amp and device.type == "cpu")
    amp_enabled = use_cuda_amp or enable_cpu_bf16
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if hw_info.get("bf16_supported", False) else torch.float16
    else:
        amp_dtype = torch.bfloat16
    scaler = GradScaler("cuda", enabled=use_cuda_amp and amp_dtype == torch.float16)

    encode_cache = EncodedExampleCache(train_examples, max_bytes=config.max_bytes, eager=False)
    val_subset_size = max(1, int(val_subset_size))
    if len(val_examples) > val_subset_size:
        # Deterministic stride sample so every source/domain in the file is represented,
        # instead of only the first N rows (which usually come from a single source).
        stride = len(val_examples) / float(val_subset_size)
        val_examples = [val_examples[int(i * stride)] for i in range(val_subset_size)]
    val_cache = EncodedExampleCache(val_examples, max_bytes=config.max_bytes, eager=False)
    sampler = DynamicBucketSampler(
        train_examples,
        batch_size=batch_size,
        max_bytes=config.max_bytes,
        max_tokens_per_batch=max_tokens_per_batch,
    )
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
        if math.isfinite(best_val_loss):
            print(f"Preserved previous best_val_loss={best_val_loss:.4f} across resume.")

    epoch = start_epoch
    val_loss = best_val_loss
    global_opt_steps = 0

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            model.train()
            epoch_loss = 0.0
            batches = sampler.get_batches(
                shuffle=True, epoch=epoch, total_epochs=start_epoch + epochs - 1, curriculum=True
            )
            step_count = 0
            epoch_bytes_processed = 0
            epoch_start_time = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            for b_idx, batch in enumerate(batches):
                ui, um, ti = collate_batch(batch, encode_cache, config.max_bytes, device)

                epoch_bytes_processed += int(ui.numel() + ti.numel())

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits, _, _, _ = model(
                        ui,
                        um,
                        ti[:, :-1],
                        return_attentions=False,
                        checkpoint_segment_size=checkpoint_segment_size or None,
                    )
                    labels = ti[:, 1:]
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=PAD,
                    )
                    loss = loss / grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(batches):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_opt_steps += 1

                epoch_loss += float(loss.item() * grad_accum_steps)
                step_count += 1

                if (
                    save_every_steps > 0
                    and step_count % save_every_steps == 0
                    and smoke_test_steps == 0
                ):
                    save_checkpoint(
                        output,
                        model,
                        config,
                        metadata={
                            "epochs": epoch - 1,
                            "in_progress_epoch": epoch,
                            "in_progress_step": step_count,
                            "best_val_loss": best_val_loss,
                            "hw": hw_info,
                        },
                        history=history,
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                    )

                if smoke_test_steps > 0 and step_count >= smoke_test_steps:
                    print(f"Smoke test complete at step {step_count}.")
                    break

            train_loss = epoch_loss / max(1, step_count)
            val_metrics = evaluate_model(
                model, val_examples, device=device, max_eval_examples=len(val_examples), cache=val_cache
            )
            val_loss = val_metrics["eval_loss"]

            epoch_duration = time.perf_counter() - epoch_start_time
            throughput_bytes = epoch_bytes_processed / max(0.001, epoch_duration)

            peak_vram_mb = 0.0
            if device.type == "cuda":
                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "validation_loss": round(val_loss, 4),
                "lr": round(current_lr, 6),
                "duration_seconds": round(epoch_duration, 2),
                "throughput_bytes_sec": round(throughput_bytes, 1),
                "peak_vram_mb": round(peak_vram_mb, 1),
            }
            history.append(epoch_record)
            print(json.dumps(epoch_record))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_output,
                    model,
                    config,
                    metadata={"epochs": epoch, "best_val_loss": best_val_loss, "hw": hw_info},
                    history=history,
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                )

            # Always persist latest completed epoch immediately so progress is never lost
            save_checkpoint(
                output,
                model,
                config,
                metadata={
                    "epochs": epoch,
                    "final_val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "hw": hw_info,
                },
                history=history,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
            )

            if smoke_test_steps > 0:
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving emergency recovery checkpoint...")
        save_checkpoint(
            output,
            model,
            config,
            metadata={
                "epochs": len(history),
                "interrupted_epoch": epoch,
                "best_val_loss": best_val_loss,
                "hw": hw_info,
            },
            history=history,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
        )
        raise

    if lora_rank > 0:
        merge_lora_weights(model)

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
            "lora": lora_info,
        },
        history=history,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
    )

    # Quick generation sample check (verified Best-of-N with RFC-3629 UTF-8 grammar constraints)
    model.eval()
    test_prompt = "Explain how cross-attention works in 2 sentences."
    gen_check = model.generate_best_of_n(
        test_prompt,
        n=4,
        max_new_tokens=256,
        temperature=0.30,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        enforce_valid_utf8=True,
    )
    sample_out = gen_check["best_text"]
    safe_out = sample_out.encode("ascii", errors="backslashreplace").decode("ascii")
    print("\n--- Generation Check ---")
    print(f"Prompt: {test_prompt}")
    print(f"Response (Best-of-4 score={gen_check['best_score']:.2f}): {safe_out}\n")

    return {
        "status": "completed",
        "output_checkpoint": str(output),
        "best_checkpoint": str(best_output),
        "history": history,
        "parameters": param_count,
        "lora": lora_info,
        "sample_output": sample_out,
    }


def main():
    default_dataset = (
        "training/data/mirror_extreme_quality_master_v1.jsonl"
        if os.path.exists("training/data/mirror_extreme_quality_master_v1.jsonl")
        else "training/data/mirror_oasst2_curated_v1.jsonl"
    )
    p = argparse.ArgumentParser(description="Mirror 7 GPU/CPU Training Pipeline")
    p.add_argument("--dataset", default=default_dataset)
    p.add_argument("--output", default="artifacts/mirror7_gpu_model.pt")
    p.add_argument(
        "--model-scale",
        default="base",
        choices=sorted(MODEL_SCALE_PRESETS.keys()),
        help="Parameter scaling preset: base (0.92M), cpu-scaled (3.85M), medium (13.8M), a100-pro (36.7M), a100-max (103.5M)",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-bytes", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--no-amp", action="store_true", default=False)
    p.add_argument("--cpu-amp", action="store_true", default=False)
    p.add_argument("--cpu-threads", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke-test-steps", type=int, default=0)
    p.add_argument("--no-bidirectional", action="store_true", default=False)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--no-context-gating", action="store_true", default=False)
    p.add_argument("--scheduled-sampling-p", type=float, default=0.0)
    p.add_argument("--no-chunking", action="store_true", default=False)
    p.add_argument("--save-every-steps", type=int, default=100)
    p.add_argument("--max-tokens-per-batch", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--grad-checkpoint", type=int, default=None)
    p.add_argument("--num-moe-experts", type=int, default=None)
    p.add_argument("--moe-top-k", type=int, default=None)
    p.add_argument("--byte-patch-size", type=int, default=None)
    p.add_argument("--memory-bank-slots", type=int, default=None)
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--state-dim", type=int, default=None)
    p.add_argument("--attention-dim", type=int, default=None)
    p.add_argument("--val-subset-size", type=int, default=50)
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
        chunk_long_examples=not args.no_chunking,
        save_every_steps=args.save_every_steps,
        max_tokens_per_batch=args.max_tokens_per_batch,
        use_cpu_amp=args.cpu_amp,
        cpu_threads=args.cpu_threads,
        lora_rank=args.lora_rank,
        checkpoint_segment_size=args.grad_checkpoint,
        num_moe_experts=args.num_moe_experts,
        moe_top_k=args.moe_top_k,
        byte_patch_size=args.byte_patch_size,
        memory_bank_slots=args.memory_bank_slots,
        model_scale=args.model_scale,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        attention_dim=args.attention_dim,
        val_subset_size=args.val_subset_size,
    )


if __name__ == "__main__":
    main()
