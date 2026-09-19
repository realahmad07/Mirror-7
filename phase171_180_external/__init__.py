"""External evaluation foundation for phases 171-180."""
from .protocol import (
    ExternalEpisode, ExternalTaskPack, EvaluatorResult, SealedTaskView,
    build_pack, load_pack, save_pack, task_pack_fingerprint,
)
from .independent_evaluator import evaluate_pack_result

__all__ = [
    "ExternalEpisode", "ExternalTaskPack", "EvaluatorResult",
    "SealedTaskView", "build_pack", "load_pack", "save_pack",
    "task_pack_fingerprint", "evaluate_pack_result",
]
