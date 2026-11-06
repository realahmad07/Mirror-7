# PHASE 0 — CPU EXPERIMENT CLOSURE RECORD

## Date: 2026-11-01 00:48 +05:00

## Training Process Status

| Field | Value |
|-------|-------|
| **Training script** | `training/native_conversation_v5.py` |
| **Architecture** | V5 Input-Conditioned Recurrent Decoder (`NativeConversationV5`) |
| **Parameters** | 957,251 |
| **Dataset** | `training/data/mirror_oasst2_curated_v1.jsonl` (14,627 examples) |
| **Process ID** | PID 12480 (started 2026-10-31 23:06:53) |
| **Task ID** | task-1030 |
| **Status at closure** | Killed (CANCELED) |
| **Command** | `--resume artifacts/mirror7_native_conversation_v5_ep1.pt --output artifacts/mirror7_native_conversation_v5_ep5.pt --epochs 4 --batch-size 32 --lr 3e-4 --shuffle --use-scheduler --device cpu` |
| **Device** | CPU only |

## Completed Epochs (V5 run)

| Epoch | Train Loss | Validation Loss | Status |
|-------|-----------|-----------------|--------|
| 1 | 2.4604 | 2.0732 | ✅ Completed (saved in ep1 checkpoint) |
| 2 | 1.9254 | 1.8436 | ✅ Completed (saved in best checkpoint) |
| 3 | Unknown | Unknown | ❌ In progress when killed — NOT completed |
| 4 | — | — | ❌ Not reached |
| 5 | — | — | ❌ Not reached |

**The best V5 checkpoint reflects epoch 2 only.** Epoch 3 was in progress and its weights were NOT saved to any checkpoint file.

## Checkpoint Inventory (Frozen — Do Not Overwrite)

| Checkpoint | Architecture | Epochs | Val Loss | Params | Date | Status |
|------------|-------------|--------|----------|--------|------|--------|
| `baseline_v3_5ep.pt` | V3 cross-attention | 5 | 1.6956 | 990,915 | 2026-10-29 | Frozen baseline |
| `mirror7_native_conversation_v3_full.pt` | V3 cross-attention | 5 | 1.6956 | 990,915 | 2026-10-29 | Original V3 (identical to baseline) |
| `mirror7_native_conversation_v3_v4.pt` | V3 cross-attention | 6 | 1.6994 | 990,915 | 2026-10-29 | V4 continuation |
| `mirror7_native_conversation_v3_v4.best.pt` | V3 cross-attention | 5 | 1.6928 | 990,915 | 2026-10-29 | V4 best (epoch 5) |
| `mirror7_native_conversation_v5_ep1.pt` | V5 input-conditioned | 1 | 2.0732 | 957,251 | 2026-10-29 | V5 epoch 1 final |
| `mirror7_native_conversation_v5_ep1.best.pt` | V5 input-conditioned | 1 | 2.0732 | 957,251 | 2026-10-29 | V5 epoch 1 best |
| `mirror7_native_conversation_v5_ep5.best.pt` | V5 input-conditioned | 2 | 1.8436 | 957,251 | 2026-10-31 | V5 best through epoch 2 (misleading filename) |
| `smoke_test_cpu.pt` | V3 cross-attention | — | — | 990,915 | 2026-10-28 | Smoke test artifact |

> **Note:** `mirror7_native_conversation_v5_ep5.best.pt` contains only 2 completed epochs despite its filename suggesting 5. Training was interrupted during epoch 3. The checkpoint's internal metadata correctly records `epochs: 2`.

## Benchmark Files Preserved

| File | Description |
|------|-------------|
| `benchmark_baseline_untouched.json` | V3 baseline on untouched test split (100 examples) |
| `benchmark_v5_ep1_dev.json` | V5 epoch 1 on dev split (100 examples) |
| `benchmark_100_v4.json` | V4 100-example benchmark |
| `benchmark_baseline_greedy.json` | V3 baseline greedy decoding |
| `benchmark_baseline_penalized.json` | V3 baseline with penalties |
| `benchmark_v4_epoch6.json` | V4 epoch 6 benchmark |
| `model_comparison_100.json` | 4-way V3/V4 comparison |
| `diagnostic_v3_baseline.json` | V3 conditioning diagnostic |
| `diagnostic_v5_ep1.json` | V5 epoch 1 conditioning diagnostic |
| `eval_test.json` | Initial evaluation |

## Key Findings From CPU Experiments

1. **V3 baseline (5 epochs):** val_loss=1.6956, but outputs are incoherent gibberish with near-zero prompt relevance (Jaccard ~0.04, 4-gram F1 ~0.09).
2. **V5 (2 completed epochs):** val_loss=1.8436 (still higher than V3's 5-epoch result), outputs also incoherent.
3. **Root cause identified:** A ~1M parameter byte-level GRU trained from scratch on 14K examples cannot learn spelling, grammar, and semantics simultaneously. The V5 architecture (input-conditioned recurrence) is architecturally sound but the fundamental capacity and data limitations prevent coherent generation.
4. **CPU training speed:** ~8 minutes per epoch — impractical for the scale of experimentation needed.

## Test Suite Status

**36/36 tests passing** as of 2026-11-01 00:49:02 +05:00.

## Decision

No further long CPU training runs will be launched. Future training will use GPU (Colab) with improved data pipeline and knowledge transfer.
