# Controlled CPU Benchmark: Byte-Level vs Subword BPE Transformer (~1M Params)

Corpus: 747 train paragraphs (329,335 UTF-8 bytes), 100 held-out validation paragraphs. BPE compression on train: **2.43 bytes/token**.

| Variant | Params | Vocab | Steps | Wall (s) | Init Val BPB | Final Val BPB | Final Val PPL (token) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Byte-level (`vocab=263`) | 951,552 | 263 | 200 | 26.0 | 8.038 | **3.237** | 9.38 |
| Subword BPE (`vocab=1024`) | 869,504 | 1024 | 197 | 25.3 | 4.216 | **2.386** | 50.78 |

## Greedy Continuations (tiny ~1M model trained for seconds on repo docs)

### `byte_level`
- **Mirror 7 is a** → `Mirror 7 is and and `mirror_cores and and and and and and and`
- **Every stage requires** → `Every stage requires and and and and and and and contration and and `
- **Verification ensures that** → `Verification ensures thate and and and and and and and and and and and an`

### `bpe_1024`
- **Mirror 7 is a** → `Mirror 7 is a semantic-.`
- **Every stage requires** → `Every stage requires.`
- **Verification ensures that** → `Verification ensures that the service.`

## Compute Scaling Reality Check (Chinchilla `D = 20N`, `C = 6ND`, 40% GPU MFU)

| Model | Params | Chinchilla Tokens (20N) | Pretrain FLOPs (6ND) | 2-vCPU Days | 1×T4 GPU-h | 1×A100 GPU-h | 1×H100 GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nano` | 1.00M | 20.01M | 1.20e+14 | 0.0 | 0.0 | 0.00 | 0.00 |
| `micro` | 9.45M | 188.91M | 1.07e+16 | 3.1 | 0.1 | 0.02 | 0.01 |
| `small` | 124.28M | 2.49B | 1.85e+18 | 536.3 | 19.8 | 4.13 | 1.30 |
| `medium` | 335.59M | 6.71B | 1.35e+19 | 3,910.5 | 144.4 | 30.08 | 9.49 |
| `base-1b` | 1.10B | 22.00B | 1.45e+20 | 42,017.6 | 1,551.4 | 323.21 | 101.96 |
| `7B-reference` | 7.00B | 140.00B | 5.88e+21 | 1,701,388.9 | 62,820.5 | 13,087.61 | 4,128.75 |
| `70B-chat-class` | 70.00B | 1.40T | 5.88e+23 | 170,138,888.9 | 6,282,051.3 | 1,308,760.68 | 412,874.96 |

> **Scope note:** A ~1M-parameter model trained for under a minute on a small documentation corpus memorizes
> local phrase statistics; it is **not** a general conversational model. The table above quantifies why:
> reaching GPT-3.5/ChatGPT-class fluency and world knowledge requires ~10⁹–10¹¹ parameters, ~10¹⁰–10¹² tokens,
> and hundreds to tens of thousands of A100/H100 GPU-hours.
