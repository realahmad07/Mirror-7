"""Micro-Embedding Semantic Search Engine for fast CPU similarity ranking.

Provides sub-millisecond semantic vector projection and cosine similarity
ranking without heavy PyTorch or ONNX dependencies. Uses deterministic
n-gram hashing and synonym expansion to map queries and candidate rules
into dense normalized vectors (d=128), enabling semantic matching across the vault.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

from .text import content_words, normalize, stem

DIMENSION = 128

# Semantic synonym thesaurus for query expansion
SYNONYM_CLUSTERS = (
    {"capital", "capital city", "seat of government"},
    {"currency", "money", "legal tender", "coinage"},
    {"author", "writer", "wrote", "written by", "playwright", "novelist"},
    {"inventor", "invented", "creator", "created", "discovered", "discoverer", "founder"},
    {"price", "cost", "costs", "spend", "spends", "fee", "rate"},
    {"speed", "velocity", "pace", "rate of travel", "fast"},
    {"calculate", "compute", "evaluate", "solve", "work out", "find"},
    {"function", "method", "procedure", "program", "routine", "code"},
    {"reverse", "invert", "flip", "backwards"},
    {"prime", "prime number", "divisible"},
    {"large", "big", "great", "huge", "enormous", "giant"},
    {"small", "tiny", "little", "miniature"},
    {"explain", "describe", "define", "clarify", "detail"},
)


def _hash_token_to_vector(token: str, dim: int = DIMENSION) -> list[float]:
    """Generates a deterministic pseudo-random sparse unit vector for a token."""
    vec = [0.0] * dim
    # 4 hash indices per token
    h = hashlib.md5(token.encode("utf-8")).digest()
    for i in range(4):
        idx = int.from_bytes(h[i * 2 : i * 2 + 2], "big") % dim
        sign = 1.0 if (h[8 + i] % 2 == 0) else -1.0
        vec[idx] += sign
    return vec


def _extract_ngrams_and_subwords(text: str) -> list[str]:
    tokens = [stem(w.lower()) for w in re.findall(r"[a-zA-Z0-9]+", normalize(text))]
    subwords = list(tokens)
    for tok in tokens:
        if len(tok) >= 4:
            # Add character trigrams for typo-tolerance and morphological similarity
            for i in range(len(tok) - 2):
                subwords.append(f"#{tok[i:i+3]}")
    return subwords


class MicroEmbeddingEngine:
    """Zero-dependency, sub-millisecond semantic embedding and similarity ranker."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension

    def expand_query_synonyms(self, text: str) -> str:
        """Appends semantic synonyms to the query string for expanded recall."""
        text_norm = normalize(text).casefold()
        expanded_parts = [text_norm]
        for cluster in SYNONYM_CLUSTERS:
            for term in cluster:
                if re.search(r"\b" + re.escape(term) + r"\b", text_norm):
                    # Add remaining terms in the synonym cluster
                    remaining = [t for t in cluster if t != term]
                    expanded_parts.append(" ".join(remaining))
                    break
        return " ".join(expanded_parts)

    def embed(self, text: str, expand: bool = True) -> list[float]:
        """Maps arbitrary text into a dense normalized L2 vector."""
        effective_text = self.expand_query_synonyms(text) if expand else text
        subwords = _extract_ngrams_and_subwords(effective_text)
        if not subwords:
            return [0.0] * self.dimension

        dense = [0.0] * self.dimension
        for sw in subwords:
            # Subwords starting with # have slightly lower weight than full tokens
            weight = 0.4 if sw.startswith("#") else 1.0
            vec = _hash_token_to_vector(sw, self.dimension)
            for d in range(self.dimension):
                dense[d] += vec[d] * weight

        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in dense))
        if norm > 1e-12:
            return [x / norm for x in dense]
        return [0.0] * self.dimension

    @staticmethod
    def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
        """Calculates cosine similarity between two unit vectors."""
        if len(v1) != len(v2) or not v1:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(-1.0, min(1.0, dot))

    def rank_candidates(
        self,
        query: str,
        candidates: Sequence[tuple[str, Any]],
        top_k: int = 5,
    ) -> list[tuple[float, str, Any]]:
        """Ranks candidates by semantic similarity score against the query."""
        q_vec = self.embed(query, expand=True)
        scored = []
        for text, payload in candidates:
            c_vec = self.embed(text, expand=False)
            sim = self.cosine_similarity(q_vec, c_vec)
            scored.append((sim, text, payload))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
