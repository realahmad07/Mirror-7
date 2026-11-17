"""Data utilities: pretraining document packing, chat template + SFT loss masking,
and JSONL loaders for the repo's common record formats."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .tokenizer import BPETokenizer

IGNORE_INDEX = -100
ROLES = ("system", "user", "assistant")


# ----------------------------------------------------------------- pretraining
def pack_documents(
    token_docs: Iterable[Sequence[int]],
    block_size: int,
    eos_id: int | None,
    stride: int | None = None,
) -> torch.Tensor:
    """Concatenate documents (separated by ``eos_id``) and cut into rows of
    ``block_size + 1`` tokens so that ``x = row[:-1]``, ``y = row[1:]``.

    ``stride`` defaults to ``block_size`` (rows overlap by exactly one token, so
    every token except the very first is a prediction target exactly once).
    The trailing remainder shorter than a full row is dropped.
    """
    stride = stride or block_size
    stream: list[int] = []
    for doc in token_docs:
        stream.extend(doc)
        if eos_id is not None:
            stream.append(eos_id)
    rows = []
    for start in range(0, len(stream) - block_size, stride):
        rows.append(stream[start : start + block_size + 1])
    if not rows:
        raise ValueError(
            f"not enough tokens ({len(stream)}) for one block of {block_size + 1}"
        )
    return torch.tensor(rows, dtype=torch.long)


def split_documents_by_hash(
    docs: Sequence[str], val_fraction: float = 0.1, salt: str = "mirror_lm"
) -> tuple[list[str], list[str]]:
    """Deterministic train/val split by content hash (stable across runs/machines)."""
    train, val = [], []
    buckets = 10_000
    cut = int(round(val_fraction * buckets))
    for d in docs:
        h = int(hashlib.sha256((salt + d).encode("utf-8")).hexdigest(), 16) % buckets
        (val if h < cut else train).append(d)
    if not val and train:
        val.append(train.pop())
    return train, val


def dedupe_documents(docs: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for d in docs:
        key = hashlib.sha256(" ".join(d.split()).lower().encode("utf-8")).hexdigest()
        if key in seen or not d.strip():
            continue
        seen.add(key)
        out.append(d)
    return out


# ------------------------------------------------------------------------- SFT
def format_chat(messages: Sequence[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render ``<|system|>..<|end|><|user|>..<|end|><|assistant|>..<|end|>`` as text."""
    parts = ["<|bos|>"]
    for m in messages:
        role = m["role"]
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        parts.append(f"<|{role}|>{m['content']}<|end|>")
    if add_generation_prompt:
        parts.append("<|assistant|>")
    return "".join(parts)


def encode_chat(
    tok: BPETokenizer,
    messages: Sequence[dict[str, str]],
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[bool]]:
    """Encode a conversation into token ids and a per-token "train on this" mask.

    Message contents are encoded with ``allow_special=False`` so user text can
    never inject control tokens.  Only assistant content tokens and the
    assistant's closing ``<|end|>`` are marked trainable.
    """
    ids: list[int] = [tok.bos_id]
    mask: list[bool] = [False]
    for m in messages:
        role = m["role"]
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        ids.append(tok.token_id(f"<|{role}|>"))
        mask.append(False)
        content = tok.encode(m["content"], allow_special=False)
        trainable = role == "assistant"
        ids.extend(content)
        mask.extend([trainable] * len(content))
        ids.append(tok.end_id)
        mask.append(trainable)
    if add_generation_prompt:
        ids.append(tok.token_id("<|assistant|>"))
        mask.append(False)
    return ids, mask


def sft_example(
    tok: BPETokenizer, messages: Sequence[dict[str, str]], max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(input_ids, labels)`` of equal length <= max_len.

    ``labels[t]`` is the target for position t (i.e. token t+1) or -100 when that
    token is not an assistant token.
    """
    ids, mask = encode_chat(tok, messages)
    ids, mask = ids[: max_len + 1], mask[: max_len + 1]
    x = ids[:-1]
    y = [t if m else IGNORE_INDEX for t, m in zip(ids[1:], mask[1:])]
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_sft(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(x.numel() for x, _ in batch)
    xs = torch.full((len(batch), T), pad_id, dtype=torch.long)
    ys = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, : x.numel()] = x
        ys[i, : y.numel()] = y
    return xs, ys


# ---------------------------------------------------------------------- JSONL
def record_to_messages(rec: dict[str, Any]) -> list[dict[str, str]] | None:
    """Map common record shapes to a chat message list (None if not conversational).

    Supported: ``messages``/``conversations`` lists (role/content or from/value),
    ``prompt``/``response``, ``user_text``/``target_text`` (Mirror 7 TrainingExample),
    ``instruction``(+``input``)/``output``, ``question``/``answer``.
    """
    msgs = rec.get("messages") or rec.get("conversations")
    if isinstance(msgs, list):
        out = []
        role_map = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant"}
        for m in msgs:
            if not isinstance(m, dict):
                return None
            role = m.get("role", m.get("from"))
            role = role_map.get(role, role)
            content = m.get("content", m.get("value"))
            if role not in ROLES or not isinstance(content, str):
                return None
            out.append({"role": role, "content": content})
        return out or None
    pairs = (
        ("prompt", "response"),
        ("user_text", "target_text"),
        ("question", "answer"),
        ("instruction", "output"),
        ("input", "output"),
    )
    for pk, rk in pairs:
        if isinstance(rec.get(pk), str) and isinstance(rec.get(rk), str):
            prompt = rec[pk]
            if pk == "instruction" and isinstance(rec.get("input"), str) and rec["input"].strip():
                prompt = f"{prompt}\n\n{rec['input']}"
            out = []
            if isinstance(rec.get("system"), str) and rec["system"].strip():
                out.append({"role": "system", "content": rec["system"]})
            out += [{"role": "user", "content": prompt}, {"role": "assistant", "content": rec[rk]}]
            return out
    return None


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_conversations(path: str | Path) -> list[list[dict[str, str]]]:
    convs = []
    for rec in load_jsonl_records(path):
        m = record_to_messages(rec)
        if m:
            convs.append(m)
    return convs


def load_text_documents(paths: Iterable[str | Path]) -> list[str]:
    """Plain-text documents from .txt/.md files (one doc per file) or JSONL
    (``text`` field, else conversations flattened to plain text)."""
    docs: list[str] = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".jsonl":
            for rec in load_jsonl_records(p):
                if isinstance(rec.get("text"), str):
                    docs.append(rec["text"])
                else:
                    m = record_to_messages(rec)
                    if m:
                        docs.append("\n\n".join(x["content"] for x in m))
        else:
            docs.append(p.read_text(encoding="utf-8", errors="replace"))
    return docs


class BlockBatcher:
    """Random batches of packed blocks driven by a dedicated, checkpointable generator."""

    def __init__(self, blocks: torch.Tensor, batch_size: int, seed: int = 0) -> None:
        self.blocks = blocks
        self.batch_size = batch_size
        self.gen = torch.Generator().manual_seed(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, self.blocks.shape[0], (self.batch_size,), generator=self.gen)
        rows = self.blocks[ix]
        return rows[:, :-1], rows[:, 1:]

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, s: dict) -> None:
        self.gen.set_state(s["gen"])


class SFTBatcher:
    def __init__(self, examples: list[tuple[torch.Tensor, torch.Tensor]], batch_size: int,
                 pad_id: int, seed: int = 0) -> None:
        self.examples = examples
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.gen = torch.Generator().manual_seed(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, len(self.examples), (self.batch_size,), generator=self.gen)
        return collate_sft([self.examples[i] for i in ix.tolist()], self.pad_id)

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, s: dict) -> None:
        self.gen.set_state(s["gen"])


def fingerprint_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16] + f"-{tuple(t.shape)}"


def shuffled(seq: Sequence[Any], seed: int) -> list[Any]:
    out = list(seq)
    random.Random(seed).shuffle(out)
    return out
