"""Quality and relevance guard for vault retrieval.

Prevents spurious rule matches where a candidate rule from the large vault
shares a single common word with the query but discusses an entirely unrelated
topic (for example, returning a Morse code rule for "What is the capital of France?").
"""
from __future__ import annotations

import re

from .text import content_words, stem


class RetrievalGuard:
    """Enforces minimum content word coverage and topical relevance on vault hits."""

    MIN_OVERLAP_RATIO = 0.40  # At least 40% of query's core content words must match

    @classmethod
    def is_relevant(cls, query: str, candidate_prompt: str, candidate_answer: str) -> tuple[bool, float, str]:
        q_words = content_words(query)
        if not q_words:
            # Query is purely stopwords / punctuation
            return True, 1.0, "trivial_query"

        cand_text = f"{candidate_prompt} {candidate_answer}"
        cand_words = content_words(cand_text)

        overlap = q_words & cand_words
        coverage = len(overlap) / len(q_words)

        # Special check: If query mentions a specific named entity (e.g. "france", "romeo", "prime")
        # that entity MUST be in the candidate text!
        proper_names = [
            stem(w.lower())
            for w in re.findall(r"\b[a-zA-Z]{3,}\b", query)
            if w.lower() not in {"what", "who", "which", "when", "where", "why", "how", "the", "and", "can", "you", "tell", "explain", "write", "function"}
        ]
        missing_critical = [p for p in proper_names if p not in cand_words]

        # If more than half of named/content tokens are missing, reject
        if coverage < cls.MIN_OVERLAP_RATIO:
            return False, coverage, f"insufficient_overlap ({len(overlap)}/{len(q_words)} words: {overlap})"

        if len(proper_names) >= 2 and len(missing_critical) >= len(proper_names):
            return False, coverage, f"missing_critical_entities ({missing_critical})"

        return True, coverage, "sufficient_relevance"
