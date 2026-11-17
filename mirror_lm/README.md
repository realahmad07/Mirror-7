# `mirror_lm` — Independent Decoder-Only Transformer Stack

`mirror_lm` replaces Mirror 7's byte-level GRU realization prototype (`training/native_conversation_v5.py`) with a self-contained, teacher-independent decoder-only transformer stack implemented in pure PyTorch + the Python standard library.

## Why `NativeConversationV5` could not reach fluent language generation

| Dimension | `NativeConversationV5` (`training/native_conversation_v5.py`) | `mirror_lm` (`mirror_lm/`) |
|---|---|---|
| Tokenization | Raw UTF-8 bytes (`VOCAB_SIZE = 259`, 1 byte/step) | Byte-level BPE (`BPETokenizer`) with lossless byte fallback + chat control tokens (`~3.6` bytes/token on English) |
| Architecture | Single-layer BiGRU encoder + GRU decoder | Pre-norm decoder-only Transformer: RMSNorm, RoPE, SwiGLU MLP, Grouped-Query Attention (`F.scaled_dot_product_attention`), tied embeddings |
| Context efficiency | 512 bytes (~80–100 English words) unrolled sequentially | Subword packing + exact KV-cache incremental decoding |
| SFT objective | Ad-hoc prompt/target byte concatenation | Role-delimited `<|system|>…<|end|><|user|>…<|end|><|assistant|>…<|end|>` with `IGNORE_INDEX = -100` on non-assistant tokens |
| Cross-tokenizer metric | Per-byte cross-entropy only | Tokenizer-independent **bits-per-byte (BPB)** alongside token NLL and perplexity |
| Checkpointing | Scheduler timeline reset on resume | Atomic `tmp + os.replace` checkpointing of weights, AdamW state, batcher RNG, torch/python RNG, and dataset fingerprint |

## Modules

- `tokenizer.py` — `BPETokenizer` (GPT-2-style pre-tokenization regex, heap-accelerated merge training, JSON save/load, special-token injection guard) and `ByteTokenizer`.
- `model.py` — `ModelConfig`, `Transformer`, `KVCache`, `estimate_params`, and scale presets (`nano` ~1.0M, `micro` ~9.6M, `small` ~124M, `medium` ~354M, `base-1b` ~1.10B).
- `data.py` — `pack_documents`, deterministic hash split/dedup, `encode_chat`, `sft_example` with assistant-only loss masking, `BlockBatcher`, `SFTBatcher`.
- `train.py` — `train_loop`, AdamW decay/no-decay parameter grouping, warmup + cosine LR, `evaluate_blocks` (loss, perplexity, bits-per-byte), atomic `save_checkpoint` / `resume_from`.
- `generate.py` — KV-cached `sample_tokens` (greedy, temperature, top-k, top-p, repetition penalty), `generate_text`, `chat`, `load_for_inference`.
- `scaling.py` — Chinchilla (`D = 20N`, `C = 6ND`) compute-budget calculator comparing 2-vCPU days against T4 / A100 / H100 GPU-hours.
- `bench_vs_v5.py` — Controlled equal-wall-clock CPU experiment comparing byte-level vs subword BPE at ~1M parameters on held-out documentation.

## Quick start

```bash
# Run unit tests (tokenizer round-trip, KV-cache equivalence, SFT mask, deterministic resume)
pytest mirror_lm -q

# Print Chinchilla compute-scaling table from nano (~1M) to 70B
python -m mirror_lm.scaling

# Run the controlled byte-vs-BPE CPU benchmark and write mirror_lm/results/BENCH_REPORT.md
python -m mirror_lm.bench_vs_v5 --wall-seconds 25
```
