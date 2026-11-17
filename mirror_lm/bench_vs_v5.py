"""Controlled CPU experiment comparing BPE vs byte-level tokenization at matched ~1M parameters.

Mirror's previous native model (`NativeConversationV5`, ~974K parameters) encoded
raw UTF-8 bytes (`VOCAB_SIZE = 259`) with a GRU. At the same ~1M parameter scale
and under an identical wall-clock CPU budget on the same held-out English markdown
documents, this script compares:

1. **byte-level transformer** — `ByteTokenizer` (`vocab_size = 263`), matched
   parameter count (~0.97M) and sequence window (`block_size = 128` bytes);
2. **BPE transformer (`nano`)** — `BPETokenizer` trained on the training split
   only (`vocab_size = 1024`), ~0.95M parameters (`block_size = 128` subword
   tokens, covering ~3.5× more text per forward pass).

Because a BPE token spans multiple bytes, raw per-token cross-entropy is not
comparable across tokenizers. Both runs are therefore evaluated in
**bits-per-byte (BPB)** on the exact same held-out text:

    BPB = (total negative log-likelihood in nats / ln 2) / (UTF-8 byte count)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import dedupe_documents, pack_documents, split_documents_by_hash
from .generate import generate_text
from .model import ModelConfig, Transformer
from .scaling import markdown_scaling_table
from .tokenizer import BPETokenizer, ByteTokenizer
from .train import TrainConfig, evaluate_blocks, train_loop

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_markdown_corpus() -> tuple[list[str], list[str]]:
    paths = sorted((REPO_ROOT / "docs").glob("*.md")) + [REPO_ROOT / "README.md", REPO_ROOT / "STATUS.md"]
    paras: list[str] = []
    for p in paths:
        if not p.is_file():
            continue
        for block in p.read_text(encoding="utf-8", errors="ignore").split("\n\n"):
            clean = " ".join(block.split())
            if len(clean) >= 80:
                paras.append(clean)
    docs = dedupe_documents(paras)
    return split_documents_by_hash(docs, val_fraction=0.12, salt="mirror_bench_v1")


def run_benchmark(wall_seconds: float = 45.0, block_size: int = 128) -> dict:
    train_docs, val_docs = load_markdown_corpus()

    # 1. Byte-level tokenizer + ~0.97M-param transformer (d_model=144, 4 layers).
    byte_tok = ByteTokenizer()
    byte_train = pack_documents([byte_tok.encode(d) for d in train_docs], block_size=block_size, eos_id=byte_tok.eos_id)
    byte_val = pack_documents([byte_tok.encode(d) for d in val_docs], block_size=block_size, eos_id=byte_tok.eos_id)
    byte_cfg = ModelConfig(
        vocab_size=byte_tok.vocab_size,
        d_model=144,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=384,
        max_seq_len=block_size + 16,
    )
    byte_model = Transformer(byte_cfg)
    byte_init = evaluate_blocks(byte_model, byte_tok, byte_val)
    byte_run = train_loop(
        byte_model,
        byte_tok,
        byte_train,
        byte_val,
        TrainConfig(steps=10_000, batch_size=16, lr=2e-3, warmup_steps=25, eval_every=100, seed=42),
        max_wall_seconds=wall_seconds,
    )

    # 2. Subword BPE tokenizer (trained on train_docs only) + ~0.95M-param transformer.
    bpe_tok = BPETokenizer.train(train_docs, vocab_size=1024, min_frequency=2)
    bpe_train = pack_documents([bpe_tok.encode(d) for d in train_docs], block_size=block_size, eos_id=bpe_tok.eos_id)
    bpe_val = pack_documents([bpe_tok.encode(d) for d in val_docs], block_size=block_size, eos_id=bpe_tok.eos_id)
    bpe_cfg = ModelConfig(
        vocab_size=bpe_tok.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=352,
        max_seq_len=block_size + 16,
    )
    bpe_model = Transformer(bpe_cfg)
    bpe_init = evaluate_blocks(bpe_model, bpe_tok, bpe_val)
    bpe_run = train_loop(
        bpe_model,
        bpe_tok,
        bpe_train,
        bpe_val,
        TrainConfig(steps=10_000, batch_size=16, lr=2e-3, warmup_steps=25, eval_every=100, seed=42),
        max_wall_seconds=wall_seconds,
    )

    prompts = ["Mirror 7 is a", "Every stage requires", "Verification ensures that"]
    samples = {
        "byte_level": {p: generate_text(byte_model, byte_tok, p, max_new_tokens=48, temperature=0.0) for p in prompts},
        "bpe_1024": {p: generate_text(bpe_model, bpe_tok, p, max_new_tokens=32, temperature=0.0) for p in prompts},
    }
    train_bytes = sum(len(d.encode("utf-8")) for d in train_docs)
    bpe_tokens = sum(len(bpe_tok.encode(d)) for d in train_docs)
    return {
        "corpus": {
            "train_docs": len(train_docs),
            "val_docs": len(val_docs),
            "train_utf8_bytes": train_bytes,
            "bpe_compression_bytes_per_token": round(train_bytes / max(bpe_tokens, 1), 3),
        },
        "byte_level_transformer": {
            "params": byte_model.num_parameters(),
            "vocab_size": byte_tok.vocab_size,
            "init_val": byte_init,
            "steps": byte_run["steps_completed"],
            "elapsed_s": byte_run["elapsed_s"],
            "final_val": byte_run["final_val"],
        },
        "bpe_transformer": {
            "params": bpe_model.num_parameters(),
            "vocab_size": bpe_tok.vocab_size,
            "init_val": bpe_init,
            "steps": bpe_run["steps_completed"],
            "elapsed_s": bpe_run["elapsed_s"],
            "final_val": bpe_run["final_val"],
        },
        "samples_greedy": samples,
    }


def render_report(res: dict) -> str:
    b = res["byte_level_transformer"]
    s = res["bpe_transformer"]
    c = res["corpus"]
    lines = [
        "# Controlled CPU Benchmark: Byte-Level vs Subword BPE Transformer (~1M Params)",
        "",
        f"Corpus: {c['train_docs']} train paragraphs ({c['train_utf8_bytes']:,} UTF-8 bytes), "
        f"{c['val_docs']} held-out validation paragraphs. BPE compression on train: "
        f"**{c['bpe_compression_bytes_per_token']} bytes/token**.",
        "",
        "| Variant | Params | Vocab | Steps | Wall (s) | Init Val BPB | Final Val BPB | Final Val PPL (token) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Byte-level (`vocab=263`) | {b['params']:,} | {b['vocab_size']} | {b['steps']} | {b['elapsed_s']:.1f} | "
        f"{b['init_val']['bits_per_byte']:.3f} | **{b['final_val']['bits_per_byte']:.3f}** | {b['final_val']['perplexity']:.2f} |",
        f"| Subword BPE (`vocab=1024`) | {s['params']:,} | {s['vocab_size']} | {s['steps']} | {s['elapsed_s']:.1f} | "
        f"{s['init_val']['bits_per_byte']:.3f} | **{s['final_val']['bits_per_byte']:.3f}** | {s['final_val']['perplexity']:.2f} |",
        "",
        "## Greedy Continuations (tiny ~1M model trained for seconds on repo docs)",
        "",
    ]
    for variant, mapping in res["samples_greedy"].items():
        lines.append(f"### `{variant}`")
        for p, out in mapping.items():
            lines.append(f"- **{p}** → `{p}{out}`")
        lines.append("")
    lines += [
        "## Compute Scaling Reality Check (Chinchilla `D = 20N`, `C = 6ND`, 40% GPU MFU)",
        "",
        markdown_scaling_table(),
        "",
        "> **Scope note:** A ~1M-parameter model trained for under a minute on a small documentation corpus memorizes",
        "> local phrase statistics; it is **not** a general conversational model. The table above quantifies why:",
        "> reaching GPT-3.5/ChatGPT-class fluency and world knowledge requires ~10⁹–10¹¹ parameters, ~10¹⁰–10¹² tokens,",
        "> and hundreds to tens of thousands of A100/H100 GPU-hours.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-seconds", type=float, default=35.0)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    res = run_benchmark(wall_seconds=args.wall_seconds)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_results.json").write_text(json.dumps(res, indent=2) + "\n")
    report = render_report(res)
    (out / "BENCH_REPORT.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
