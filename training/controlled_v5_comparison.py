"""
Controlled small-scale experimental comparison between:
1. Baseline V5 Architecture (Unidirectional GRU, Single-Head Attention, No Gating, 100% Teacher Forcing)
2. Enhanced V5 Architecture (BiGRU Encoder, 4-Head Cross-Attention, Context Gating, Scheduled Sampling)

Trained on identical subset, identical seed, identical optimization parameters.
Measures: loss curve, parameter count, attention sharpness/entropy, generation outputs on held-out prompts.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from training.dataset import load_jsonl
from training.native_brain import PAD, _batch, _encode
from training.native_conversation_v5 import (
    BOS,
    EOS,
    NativeConversationV5,
    NativeConversationV5Config,
    _prepare_targets,
    build_native_conversation_v5,
    make_v5_enhanced_config,
)


def compute_attention_entropy(model: NativeConversationV5, prompt: str) -> dict[str, float]:
    """Compute average attention entropy and peak attention weight on a prompt."""
    model.eval()
    device = next(model.parameters()).device
    raw = list(prompt.encode("utf-8")[: model.config.max_bytes])
    input_ids = torch.tensor([raw], dtype=torch.long, device=device)
    input_mask = torch.ones((1, len(raw)), dtype=torch.float32, device=device)

    with torch.no_grad():
        _, encoder_mask, state, k, v = model.encode(input_ids, input_mask)
        hidden = model.init_decoder_hidden(state)
        token = torch.full((1, 1), BOS, dtype=torch.long, device=device)

        entropies = []
        max_weights = []

        for _ in range(16):
            context, attn_weights = model._step_attention(hidden, k, v, encoder_mask)
            emb = model.decoder_embedding(token).squeeze(1)
            cell_input = torch.cat([emb, context], dim=-1)
            hidden = model.cell(cell_input, hidden)

            ent = -float((attn_weights * torch.log(attn_weights + 1e-9)).sum().item())
            entropies.append(ent)
            max_weights.append(float(attn_weights.max().item()))

            logits = model._project_output(hidden, context)
            token = logits.argmax(dim=-1, keepdim=True)

    max_possible_entropy = math.log(len(raw))
    return {
        "mean_entropy": sum(entropies) / len(entropies),
        "max_possible_entropy": max_possible_entropy,
        "entropy_ratio": (sum(entropies) / len(entropies)) / max_possible_entropy,
        "mean_max_weight": sum(max_weights) / len(max_weights),
    }


def run_controlled_experiment(
    dataset_path: str = "training/data/mirror_oasst2_curated_v1.jsonl",
    num_steps: int = 30,
    batch_size: int = 16,
    lr: float = 0.003,
    max_bytes: int = 256,
):
    print("Loading dataset for controlled small-scale experiment...")
    examples = load_jsonl(dataset_path)
    train_pool = [x for x in examples if x.split == "train"][:128]
    val_pool = [x for x in examples if x.split == "validation"][:32]

    # Benchmark test prompts
    test_prompts = [
        "Hi, how are you doing today?",
        "What is the chemical formula for water?",
        "What is 15 multiplied by 8?",
    ]

    configs = {
        "Baseline_V5": NativeConversationV5Config(
            embedding_dim=96,
            hidden_dim=256,
            state_dim=128,
            attention_dim=128,
            dropout=0.0,
            max_bytes=max_bytes,
            bidirectional_encoder=False,
            num_heads=1,
            context_gating=False,
            scheduled_sampling_p=0.0,
        ),
        "Enhanced_V5": make_v5_enhanced_config(
            embedding_dim=96,
            hidden_dim=256,
            state_dim=128,
            attention_dim=128,
            dropout=0.0,
            max_bytes=max_bytes,
            num_heads=4,
            scheduled_sampling_p=0.0,
        ),
    }

    results = {}

    for name, config in configs.items():
        print(f"\n==================================================")
        print(f"Running Experiment: {name}")
        print(f"==================================================")
        torch.manual_seed(42)
        random.seed(42)

        model = build_native_conversation_v5(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        # Pre-training attention diagnostic
        pre_attn = compute_attention_entropy(model, "What is the chemical formula for water?")
        print(f"Pre-training Attention Entropy Ratio: {pre_attn['entropy_ratio']:.3f} (mean wt: {pre_attn['mean_max_weight']:.3f})")

        train_losses = []
        t0 = time.perf_counter()

        # Run controlled training steps
        model.train()
        for step in range(num_steps):
            # Select batch deterministically
            start_idx = (step * batch_size) % (len(train_pool) - batch_size)
            batch = train_pool[start_idx : start_idx + batch_size]

            ui, um = _batch([_encode(x.user_text, max_bytes) for x in batch], max_bytes)
            ti, _ = _batch(_prepare_targets([x.target_text for x in batch], max_bytes), max_bytes)

            logits, _, _, _ = model(ui, um, ti[:, :-1])
            labels = ti[:, 1:]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=PAD,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(float(loss.detach()))
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Step {step + 1:2d}/{num_steps}: loss = {loss.item():.4f}")

        duration = time.perf_counter() - t0

        # Post-training validation loss
        model.eval()
        with torch.no_grad():
            v_ui, v_um = _batch([_encode(x.user_text, max_bytes) for x in val_pool], max_bytes)
            v_ti, _ = _batch(_prepare_targets([x.target_text for x in val_pool], max_bytes), max_bytes)
            v_logits, _, _, _ = model(v_ui, v_um, v_ti[:, :-1])
            val_loss = float(
                nn.functional.cross_entropy(
                    v_logits.reshape(-1, v_logits.shape[-1]),
                    v_ti[:, 1:].reshape(-1),
                    ignore_index=PAD,
                ).item()
            )

        # Post-training attention diagnostic
        post_attn = compute_attention_entropy(model, "What is the chemical formula for water?")
        print(f"Post-training Attention Entropy Ratio: {post_attn['entropy_ratio']:.3f} (mean wt: {post_attn['mean_max_weight']:.3f})")
        print(f"Validation Loss: {val_loss:.4f} (duration: {duration:.2f}s)")

        # Generation samples
        samples = {}
        for p in test_prompts:
            gen_text = model.generate_text(p, max_new_tokens=48, temperature=0.0)
            samples[p] = gen_text
            print(f"  Prompt: '{p}' -> Output: {repr(gen_text)}")

        results[name] = {
            "parameters": param_count,
            "initial_loss": train_losses[0],
            "final_train_loss": train_losses[-1],
            "val_loss": val_loss,
            "duration_seconds": duration,
            "pre_attn_entropy_ratio": pre_attn["entropy_ratio"],
            "post_attn_entropy_ratio": post_attn["entropy_ratio"],
            "post_attn_max_weight": post_attn["mean_max_weight"],
            "samples": samples,
        }

    return results


if __name__ == "__main__":
    res = run_controlled_experiment(num_steps=30, batch_size=16)
    out_file = Path("artifacts/controlled_v5_comparison_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nResults successfully written to {out_file}")
