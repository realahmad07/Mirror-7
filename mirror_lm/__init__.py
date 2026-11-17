"""Independent decoder-only transformer language-modeling stack for Mirror 7."""
from .data import encode_chat, format_chat, pack_documents, sft_example
from .generate import chat, generate_text, load_for_inference, sample_tokens
from .model import ModelConfig, Transformer, estimate_params, get_preset
from .scaling import markdown_scaling_table, scaling_table
from .tokenizer import BPETokenizer, ByteTokenizer
from .train import TrainConfig, evaluate_blocks, save_checkpoint, train_loop

__all__ = [
    "BPETokenizer",
    "ByteTokenizer",
    "ModelConfig",
    "Transformer",
    "estimate_params",
    "get_preset",
    "pack_documents",
    "format_chat",
    "encode_chat",
    "sft_example",
    "TrainConfig",
    "train_loop",
    "evaluate_blocks",
    "save_checkpoint",
    "sample_tokens",
    "generate_text",
    "chat",
    "load_for_inference",
    "scaling_table",
    "markdown_scaling_table",
]
