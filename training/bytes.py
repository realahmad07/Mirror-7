from __future__ import annotations

from dataclasses import dataclass

from .schema import TrainingExample


PAD = 256
BOS = 257
EOS = 258
CONTEXT = 259
GOAL = 260
USER = 261
RESPONSE = 262

VOCAB_SIZE = 263


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    loss_start: int
    loss_end: int
    quality: float


def encode(example: TrainingExample, max_bytes: int = 4096) -> EncodedExample:
    example.validate()
    context = example.canonical_context().decode("utf-8")
    prompt = [BOS, CONTEXT, *context.encode("utf-8"), GOAL, USER]
    prompt.extend(example.user_text.encode("utf-8"))
    response_prefix = [RESPONSE]
    response = list(example.target_text.encode("utf-8"))
    ids = prompt + response_prefix + response + [EOS]

    if len(ids) > max_bytes:
        ids = ids[: max_bytes - 1] + [EOS]

    loss_start = len(prompt) + 1
    loss_end = len(ids) - 1
    return EncodedExample(tuple(ids), loss_start, loss_end, float(example.quality))
