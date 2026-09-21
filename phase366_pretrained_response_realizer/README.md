# Phase 366 — Pretrained Response Realizer

This phase adds an optional pretrained language model as the **surface realization layer**. Mirror 7 remains responsible for semantic state, goals, reasoning, planning, observations, and actions.

Default model: `Qwen/Qwen2.5-0.5B-Instruct`.

The realizer receives only the existing backend realization contract and returns response text. It does not modify Mirror 7 state or execute actions.

The dependency on `transformers` is optional for the core backend and is loaded only when this realizer is explicitly configured.
