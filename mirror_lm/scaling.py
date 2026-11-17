"""Compute-budget calculator for pretraining a language model from scratch.

Uses the Chinchilla compute-optimal rule of thumb (Hoffmann et al., 2022):

* tokens  ``D ≈ 20 · N``
* FLOPs   ``C ≈ 6 · N · D``  (forward ~2ND + backward ~4ND)

and converts FLOPs into wall-clock time on real hardware at a stated model FLOPs
utilization (MFU, default 40% on GPU; 2-core CPU throughput estimated at ~40
GFLOP/s sustained fp32). Numbers are order-of-magnitude planning figures.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import PRESETS, estimate_params

# Peak dense fp16/bf16 tensor-core TFLOP/s (without 2:4 sparsity).
GPU_PEAK_TFLOPS: dict[str, float] = {
    "T4 (16 GB)": 65.0,
    "A100 (80 GB)": 312.0,
    "H100 (80 GB)": 989.0,
}

CPU_2CORE_GFLOPS = 40.0  # realistic sustained fp32 matmul on a 2-vCPU VM


@dataclass(frozen=True)
class ScalingEstimate:
    name: str
    params: int
    chinchilla_tokens: int
    total_flops: float
    cpu_days_2core: float
    t4_gpu_hours: float
    a100_gpu_hours: float
    h100_gpu_hours: float


def estimate_for_params(
    name: str,
    params: int,
    tokens_per_param: float = 20.0,
    gpu_mfu: float = 0.40,
) -> ScalingEstimate:
    tokens = int(round(params * tokens_per_param))
    flops = 6.0 * params * tokens

    def gpu_hours(peak_tflops: float) -> float:
        eff = peak_tflops * 1e12 * gpu_mfu
        return flops / eff / 3600.0

    cpu_days = flops / (CPU_2CORE_GFLOPS * 1e9) / 86400.0
    return ScalingEstimate(
        name=name,
        params=params,
        chinchilla_tokens=tokens,
        total_flops=flops,
        cpu_days_2core=cpu_days,
        t4_gpu_hours=gpu_hours(GPU_PEAK_TFLOPS["T4 (16 GB)"]),
        a100_gpu_hours=gpu_hours(GPU_PEAK_TFLOPS["A100 (80 GB)"]),
        h100_gpu_hours=gpu_hours(GPU_PEAK_TFLOPS["H100 (80 GB)"]),
    )


def scaling_table(gpu_mfu: float = 0.40) -> list[ScalingEstimate]:
    rows = [estimate_for_params(name, estimate_params(cfg), gpu_mfu=gpu_mfu) for name, cfg in PRESETS.items()]
    rows.append(estimate_for_params("7B-reference", 7_000_000_000, gpu_mfu=gpu_mfu))
    rows.append(estimate_for_params("70B-chat-class", 70_000_000_000, gpu_mfu=gpu_mfu))
    return rows


def _fmt_num(n: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.0f}"


def markdown_scaling_table(gpu_mfu: float = 0.40) -> str:
    lines = [
        f"| Model | Params | Chinchilla Tokens (20N) | Pretrain FLOPs (6ND) | 2-vCPU Days | 1×T4 GPU-h | 1×A100 GPU-h | 1×H100 GPU-h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in scaling_table(gpu_mfu=gpu_mfu):
        lines.append(
            f"| `{r.name}` | {_fmt_num(r.params)} | {_fmt_num(r.chinchilla_tokens)} | "
            f"{r.total_flops:.2e} | {r.cpu_days_2core:,.1f} | {r.t4_gpu_hours:,.1f} | "
            f"{r.a100_gpu_hours:,.2f} | {r.h100_gpu_hours:,.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown_scaling_table())
