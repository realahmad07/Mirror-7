"""Strictly public learner-facing contracts and deterministic stdlib utilities for Stage S11.

ISOLATION INVARIANT:
- This module NEVER imports ``evaluator.hidden_eval`` or ``evaluator.oracle``.
- ``OpenWorldObservation`` contains ONLY raw public inputs available at inference time:
  ZERO answers, ZERO labels, ZERO ``is_impossible`` flags, ZERO ``has_answer`` flags,
  ZERO ``answer_session_ids``, ZERO ``mirrored`` flags, ZERO gold event structures.
- Pure Python 3.11 standard library ONLY (no Python 3.12+ APIs, no GPU, no external LLMs).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Sequence
import zlib

# Public stopwords shared across deterministic CPU-native baselines (pure lexical, domain-agnostic).
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
        "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
        "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
        "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
        "same", "she", "should", "so", "some", "such", "t", "than", "that", "the",
        "their", "theirs", "them", "themselves", "then", "there", "these", "they",
        "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
        "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
        "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class SolverFrozenError(RuntimeError):
    """Raised when a frozen solver attempts to fit, calibrate, or mutate persistent memory."""


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic UTF-8 canonical JSON encoding for hashing and bit-budget audits."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compressed_bit_length(data: bytes) -> int:
    if not data:
        return 0
    return len(zlib.compress(data, level=9)) * 8


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/articles, and collapse whitespace for fair lexical matching."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.casefold())
    tokens = [tok for tok in cleaned.split() if tok not in {"a", "an", "the"}]
    return " ".join(tokens)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(m.group(0).casefold() for m in _TOKEN_RE.finditer(text))


def content_tokens(text: str) -> tuple[str, ...]:
    return tuple(tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1)


def split_sentences_with_offsets(text: str) -> tuple[tuple[int, int, str], ...]:
    """Deterministically segment raw text into (start_char, end_char, sentence_text) spans."""
    stripped = text.strip()
    if not stripped:
        return ()
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        end = match.start()
        seg = text[cursor:end].strip()
        if seg:
            seg_start = text.find(seg, cursor, end)
            spans.append((seg_start, seg_start + len(seg), seg))
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        tail_start = text.find(tail, cursor)
        spans.append((tail_start, tail_start + len(tail), tail))
    return tuple(spans)


@dataclass(frozen=True)
class EvidenceSentence:
    """One public text segment (sentence or paragraph unit) with an opaque citation ID."""

    evidence_id: str
    text: str
    char_start: int = 0
    char_end: int = 0


@dataclass(frozen=True)
class ConversationTurn:
    """One strictly public conversational turn (NEVER contains gold ``has_answer`` flags)."""

    turn_id: str
    session_id: str
    turn_index: int
    role: str
    content: str


@dataclass(frozen=True)
class ConversationSession:
    """One strictly public conversational session in a long-term memory haystack."""

    session_id: str
    session_date: str
    turns: tuple[ConversationTurn, ...]


@dataclass(frozen=True)
class OpenWorldObservation:
    """Strictly public learner-facing task observation.

    SECURITY / ISOLATION CONTRACT:
    Contains ZERO gold answers, ZERO choice labels, ZERO ``is_impossible`` flags,
    ZERO ``has_answer`` flags, ZERO ``answer_session_ids``, ZERO ``mirrored`` flags.
    """

    task_id: str
    benchmark_id: str
    split: str
    task_kind: str  # "causal_choice" | "extractive_qa" | "long_memory_qa"
    prompt_text: str
    question_text: str
    candidate_choices: tuple[str, ...] = ()
    evidence_units: tuple[EvidenceSentence, ...] = ()
    sessions: tuple[ConversationSession, ...] = ()
    question_date: str = ""
    relation_direction: str = ""  # public prompt cue e.g. "cause" | "effect" for COPA
    public_metadata: tuple[tuple[str, str], ...] = ()

    def public_meta_dict(self) -> dict[str, str]:
        return dict(self.public_metadata)


@dataclass(frozen=True)
class OpenWorldTrainingExample:
    """Public training-split supervision example.

    Only ever constructed for items in ``split == "train"`` (or validation threshold tuning
    explicitly passed to ``calibrate``). Never contains hidden evaluation items or gold
    latent event schemas.
    """

    observation: OpenWorldObservation
    target_choice_index: int | None = None
    target_answer: str | None = None
    is_unanswerable: bool = False
    supporting_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenWorldPrediction:
    """Standardized prediction emitted by every baseline and oracle."""

    task_id: str
    benchmark_id: str
    predicted_answer: str | None = None
    predicted_choice_index: int | None = None
    abstained: bool = False
    abstain_reason: str = ""
    confidence: float = 0.0  # P(prediction is correct) in [0.0, 1.0]
    uncertainty: float = 1.0  # 1.0 - confidence (explicitly recorded)
    cited_evidence_ids: tuple[str, ...] = ()
    cited_evidence_spans: tuple[str, ...] = ()
    memory_reads: int = 0
    memory_writes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        conf = max(0.0, min(1.0, float(self.confidence)))
        unc = max(0.0, min(1.0, float(self.uncertainty)))
        object.__setattr__(self, "confidence", conf)
        object.__setattr__(self, "uncertainty", unc)


@dataclass(frozen=True)
class SolverMemoryFootprint:
    """Explicit persistent memory & MDL footprint reported by a solver after fit/freeze."""

    serialized_bytes: int
    compressed_mdl_bits: int
    active_entry_count: int
    external_readonly_pack_entries: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class OpenWorldSolver(ABC):
    """Common freeze-enforced interface for all Stage S11.0 baselines."""

    solver_id: str = "base"
    category: str = "baseline"
    uses_gpu: bool = False
    uses_pretrained_llm: bool = False
    uses_external_api: bool = False
    uses_neural_training: bool = False
    evaluator_only: bool = False

    def __init__(self, seed: int = 101) -> None:
        self.seed = int(seed)
        self._frozen: bool = False
        self._post_freeze_writes: int = 0

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def post_freeze_writes(self) -> int:
        return self._post_freeze_writes

    def freeze(self) -> None:
        """Freeze learned state/memory before hidden evaluation."""
        self._frozen = True

    def _require_unfrozen(self) -> None:
        if self._frozen:
            self._post_freeze_writes += 1
            raise SolverFrozenError(
                f"Solver {self.solver_id!r} is frozen and cannot fit or mutate persistent memory"
            )

    @abstractmethod
    def fit(self, benchmark_id: str, train_examples: Sequence[OpenWorldTrainingExample]) -> None:
        """Fit strictly on source-disjoint training examples."""

    def calibrate(
        self, benchmark_id: str, val_examples: Sequence[OpenWorldTrainingExample]
    ) -> None:
        """Optional validation-only abstention/confidence threshold calibration."""
        self._require_unfrozen()

    @abstractmethod
    def predict(self, observation: OpenWorldObservation) -> OpenWorldPrediction:
        """Predict or abstain strictly from public observation."""

    @abstractmethod
    def memory_footprint(self) -> SolverMemoryFootprint:
        """Return explicit serialized bytes, MDL bits, and stored entry count."""
