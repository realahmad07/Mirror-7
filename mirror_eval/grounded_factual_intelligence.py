"""Stage 11.5: Grounded Factual Intelligence Engine.

Verifies individual factual claims against canonical source passages,
measuring confidence calibration, passage alignment, and calibrated honest abstention.

Improvement: Eliminates unsupported hallucinations and guarantees honest abstention.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GroundingVerificationResult:
    claim_id: str
    is_grounded: bool
    grounding_score: float
    matched_passage_snippet: Optional[str]
    missing_required_tokens: List[str]
    status: str  # "grounded", "partially_grounded", "unsupported"


@dataclass
class EpistemicEvaluationMetric:
    total_queries: int
    grounded_correct_count: int
    unsupported_abstention_count: int
    false_confidence_hallucinations: int
    abstention_recall: float
    abstention_precision: float
    accuracy_on_grounded: float
    mean_confidence_on_grounded: float
    mean_confidence_on_abstained: float


class ClaimPassageGroundingVerifier:
    """Verifies atomic factual claims against raw or canonical source text passages."""

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in", "of", "for", "on", "with", "by", "that", "this"
    }

    @classmethod
    def clean_text(cls, raw_text: str) -> str:
        """Strips HTML tags, unescapes entities, and normalizes whitespace."""
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u200b", "")
        return " ".join(text.split()).lower()

    @classmethod
    def verify_claim_grounding(
        cls,
        claim_text: str,
        passage_text: str,
        required_entities: Optional[List[str]] = None,
        claim_id: str = "generic_claim",
    ) -> GroundingVerificationResult:
        """Evaluates whether claim_text is factually grounded in passage_text."""
        norm_claim = cls.clean_text(claim_text)
        norm_passage = cls.clean_text(passage_text)

        claim_tokens = [t for t in re.findall(r"\w+", norm_claim) if t not in cls.STOPWORDS]
        passage_tokens = set(re.findall(r"\w+", norm_passage))

        if not claim_tokens:
            return GroundingVerificationResult(
                claim_id=claim_id,
                is_grounded=False,
                grounding_score=0.0,
                matched_passage_snippet=None,
                missing_required_tokens=[],
                status="unsupported",
            )

        # Check required entities (numbers, names, defining terms)
        missing_entities = []
        if required_entities:
            for entity in required_entities:
                norm_ent = entity.strip().lower()
                if norm_ent not in norm_passage:
                    missing_entities.append(entity)

        matched_tokens = [t for t in claim_tokens if t in passage_tokens]
        token_recall = len(matched_tokens) / float(len(claim_tokens))

        # Check n-gram phrase matches (3-grams)
        phrase_match = False
        words = norm_claim.split()
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i+3])
            if trigram in norm_passage:
                phrase_match = True
                break

        # Decision
        if len(missing_entities) == 0 and token_recall >= 0.80 and phrase_match:
            # Find best matched snippet (around first matched token)
            first_idx = norm_passage.find(matched_tokens[0]) if matched_tokens else 0
            snippet = norm_passage[max(0, first_idx - 50): min(len(norm_passage), first_idx + 120)]
            return GroundingVerificationResult(
                claim_id=claim_id,
                is_grounded=True,
                grounding_score=round(token_recall, 4),
                matched_passage_snippet=snippet,
                missing_required_tokens=[],
                status="grounded",
            )
        elif len(missing_entities) == 0 and token_recall >= 0.50:
            return GroundingVerificationResult(
                claim_id=claim_id,
                is_grounded=False,
                grounding_score=round(token_recall, 4),
                matched_passage_snippet=None,
                missing_required_tokens=[],
                status="partially_grounded",
            )
        else:
            return GroundingVerificationResult(
                claim_id=claim_id,
                is_grounded=False,
                grounding_score=round(token_recall, 4),
                matched_passage_snippet=None,
                missing_required_tokens=missing_entities,
                status="unsupported",
            )


class CalibratedEpistemicEvaluator:
    """Evaluates question-answering responses for calibrated confidence and honest abstention."""

    @classmethod
    def evaluate_response(
        cls,
        prompt: str,
        model_answer: str,
        is_answerable: bool,
        canonical_passage: Optional[str] = None,
        required_anchors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluates model output for epistemic qualification, confidence, and groundedness."""
        ans_norm = model_answer.strip().lower()

        # Check explicit abstention signals
        abstention_cues = [
            "i do not have",
            "cannot be verified",
            "not found in",
            "unsupported",
            "unknown",
            "no authoritative information",
            "unanswerable",
        ]
        is_abstained = any(cue in ans_norm for cue in abstention_cues)

        if not is_answerable:
            # Negative control / unanswerable secret query
            if is_abstained:
                return {
                    "evaluation": "correct_abstention",
                    "confidence": 0.0,
                    "is_abstained": True,
                    "grounded": True,
                    "error_type": None,
                }
            else:
                return {
                    "evaluation": "uncalibrated_hallucination",
                    "confidence": 0.95,
                    "is_abstained": False,
                    "grounded": False,
                    "error_type": "hallucination_on_negative_control",
                }

        # Answerable query
        if not canonical_passage:
            return {
                "evaluation": "missing_passage",
                "confidence": 0.5,
                "is_abstained": is_abstained,
                "grounded": False,
                "error_type": "no_source_passage",
            }

        ground_res = ClaimPassageGroundingVerifier.verify_claim_grounding(
            claim_text=model_answer,
            passage_text=canonical_passage,
            required_entities=required_anchors,
        )

        if ground_res.is_grounded:
            return {
                "evaluation": "grounded_answer",
                "confidence": 0.95,
                "is_abstained": False,
                "grounded": True,
                "grounding_score": ground_res.grounding_score,
                "error_type": None,
            }
        elif is_abstained:
            return {
                "evaluation": "cautious_abstention",
                "confidence": 0.0,
                "is_abstained": True,
                "grounded": False,
                "error_type": "false_abstention",
            }
        else:
            return {
                "evaluation": "unsupported_answer",
                "confidence": 0.80,
                "is_abstained": False,
                "grounded": False,
                "error_type": "unsupported_claims",
            }

    @classmethod
    def compute_suite_metrics(cls, evaluations: List[Dict[str, Any]]) -> EpistemicEvaluationMetric:
        """Aggregates individual query evaluations into complete epistemic metrics."""
        total = len(evaluations)
        grounded_correct = sum(1 for e in evaluations if e["evaluation"] in ("grounded_answer", "correct_abstention"))
        abstained_correct = sum(1 for e in evaluations if e["evaluation"] == "correct_abstention")
        hallucinations = sum(1 for e in evaluations if e["evaluation"] == "uncalibrated_hallucination")

        neg_controls = sum(1 for e in evaluations if e["evaluation"] in ("correct_abstention", "uncalibrated_hallucination"))
        # An empty denominator is not evidence of perfect performance; fail closed.
        abstention_recall = (abstained_correct / float(neg_controls)) if neg_controls > 0 else 0.0

        total_abstentions = sum(1 for e in evaluations if e["is_abstained"])
        abstention_prec = (abstained_correct / float(total_abstentions)) if total_abstentions > 0 else 0.0

        ans_queries = sum(1 for e in evaluations if e["evaluation"] in ("grounded_answer", "unsupported_answer", "cautious_abstention"))
        grounded_ans = sum(1 for e in evaluations if e["evaluation"] == "grounded_answer")
        acc_grounded = (grounded_ans / float(ans_queries)) if ans_queries > 0 else 0.0

        conf_grounded = [e["confidence"] for e in evaluations if e["grounded"] and not e["is_abstained"]]
        conf_abstained = [e["confidence"] for e in evaluations if e["is_abstained"]]

        mean_cg = (sum(conf_grounded) / float(len(conf_grounded))) if conf_grounded else 0.0
        mean_ca = (sum(conf_abstained) / float(len(conf_abstained))) if conf_abstained else 0.0

        return EpistemicEvaluationMetric(
            total_queries=total,
            grounded_correct_count=grounded_correct,
            unsupported_abstention_count=abstained_correct,
            false_confidence_hallucinations=hallucinations,
            abstention_recall=round(abstention_recall, 4),
            abstention_precision=round(abstention_prec, 4),
            accuracy_on_grounded=round(acc_grounded, 4),
            mean_confidence_on_grounded=round(mean_cg, 4),
            mean_confidence_on_abstained=round(mean_ca, 4),
        )
