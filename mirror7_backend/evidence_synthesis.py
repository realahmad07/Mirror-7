"""Claim-level synthesis that only emits claims supported by verified evidence.

The optional language generator may reorder or format supported claims, but any
novel or paraphrased sentence is rejected until a semantic verifier is configured.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any, Callable, Mapping

_NEGATIONS = {"no", "not", "never", "without", "neither", "cannot", "can't"}
_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def normalize_claim(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text).casefold()))


def _polarity_and_signature(text: str) -> tuple[str, bool]:
    tokens = normalize_claim(text).split()
    negative = any(token in _NEGATIONS for token in tokens)
    signature = " ".join(token for token in tokens if token not in _NEGATIONS)
    return signature, negative


def split_claims(answer: str) -> list[str]:
    """Split plain prose into sentence candidates."""
    raw = str(answer or "").strip()
    if not raw:
        return []
    pieces = re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z0-9*#])", raw)
    claims: list[str] = []
    for piece in pieces:
        item = piece.strip()
        item = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", item).strip()
        if item and normalize_claim(item):
            claims.append(item)
    return claims


class EvidenceSynthesisEngine:
    """Merge verified source claims, detect polarity conflicts, and reject unsupported output."""

    def synthesize(
        self,
        prompt: str,
        evidence: list[dict[str, Any]],
        *,
        memory_context: str | None = None,
        generator: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        verified = [item for item in evidence if item.get("verified") is True]
        grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
        polarity: dict[str, bool] = {}
        conflicts: list[dict[str, Any]] = []

        for item in verified:
            for claim in split_claims(item.get("answer", "")):
                norm = normalize_claim(claim)
                signature, negative = _polarity_and_signature(claim)
                if not signature:
                    continue
                previous_polarity = polarity.get(signature)
                if previous_polarity is not None and previous_polarity != negative:
                    conflicts.append({
                        "signature": signature,
                        "sources": [str(item.get("source", "unknown"))],
                    })
                    continue
                polarity[signature] = negative
                existing = grouped.get(norm)
                if existing is None:
                    grouped[norm] = {
                        "claim_id": hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16],
                        "text": claim,
                        "sources": [str(item.get("source", "unknown"))],
                        "verified": True,
                    }
                elif item.get("source") not in existing["sources"]:
                    existing["sources"].append(str(item.get("source", "unknown")))

        if conflicts:
            return {
                "answer": "", "verified": False, "conflict": True,
                "claims": list(grouped.values()), "conflicts": conflicts,
                "generator_used": False, "generator_rejected": False,
                "memory_used_as_evidence": False,
            }

        claims = list(grouped.values())
        if not claims:
            return {
                "answer": "", "verified": False, "conflict": False,
                "claims": [], "conflicts": [], "generator_used": False,
                "generator_rejected": False, "memory_used_as_evidence": False,
            }

        deterministic_answer = " ".join(item["text"] for item in claims)
        answer = deterministic_answer
        generator_used = False
        generator_rejected = False
        if generator is not None:
            payload = {
                "prompt": prompt,
                "verified_claims": claims,
                "memory_context": memory_context,
                "instruction": (
                    "Synthesize only the supplied verified claims. Return only factual "
                    "sentences copied exactly from supplied claim text; do not add facts."
                ),
            }
            try:
                generated = generator(payload)
                if isinstance(generated, Mapping):
                    generated = generated.get("answer") or generated.get("text") or ""
                generated_text = str(generated or "").strip()
                generated_claims = split_claims(generated_text)
                allowed = {normalize_claim(item["text"]) for item in claims}
                generated_norms = {normalize_claim(item) for item in generated_claims}
                if generated_claims and generated_norms == allowed:
                    answer = generated_text
                    generator_used = True
                else:
                    generator_rejected = True
            except Exception:
                generator_rejected = True

        return {
            "answer": answer, "verified": True, "conflict": False,
            "claims": claims, "conflicts": [], "generator_used": generator_used,
            "generator_rejected": generator_rejected,
            "memory_used_as_evidence": False,
        }
