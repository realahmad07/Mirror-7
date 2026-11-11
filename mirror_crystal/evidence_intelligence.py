"""Stage 14: Evidence-Based Intelligence Subsystem.

Provides:
- Atomic claim extraction, verification, and grounding against source passages.
- Cryptographic source provenance tracking (SHA-256, HTTPS URIs, SPDX licensing, locators).
- Multi-dimensional contradiction detection (negation, numerical disparity, antonym polarity, attribute collision).
- Epistemic calibration: calibrated confidence scores and honest abstention contracts.
- Explicit segregation of Verified Facts, Plausible Inferences, Assumptions, and Conjectures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import html
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .production_readiness import StructuredLogger


class EpistemicCategory(str, Enum):
    """Epistemic status of a claim distinguishing verified reality from speculation."""
    VERIFIED_FACT = "verified_fact"
    PLAUSIBLE_INFERENCE = "plausible_inference"
    EXPLICIT_ASSUMPTION = "explicit_assumption"
    UNVERIFIED_CONJECTURE = "unverified_conjecture"
    CONTRADICTION = "contradiction"
    ABSTENTION = "abstention"


class ContradictionType(str, Enum):
    """Taxonomy of logical and factual contradictions."""
    DIRECT_NEGATION = "direct_negation"
    NUMERICAL_DISPARITY = "numerical_disparity"
    ANTONYM_POLARITY = "antonym_polarity"
    ATTRIBUTE_COLLISION = "attribute_collision"


class EvidenceError(ValueError):
    """Raised when evidence verification, provenance, or calibration contract fails."""


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    """Strips HTML tags, unescapes entities, normalizes punctuation, and normalizes whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = cleaned.replace("\u00a0", " ").replace("\u202f", " ").replace("\u200b", "")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return " ".join(cleaned.split()).lower()


def _compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceProvenance:
    """Cryptographically verifiable provenance record for an external knowledge source."""
    source_id: str
    title: str
    url: str
    publisher: str
    license_spdx: str
    retrieved_at: str
    sha256_hash: str
    content: str = ""

    SPDX_LICENSE_IDS = {
        "0BSD", "AFL-3.0", "AGPL-3.0-only", "AGPL-3.0-or-later", "Apache-2.0",
        "Artistic-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSL-1.0", "CC-BY-3.0",
        "CC-BY-3.0-IGO", "CC-BY-4.0", "CC0-1.0", "CDDL-1.0", "EPL-2.0",
        "EUPL-1.2", "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only",
        "GPL-3.0-or-later", "ISC", "LGPL-2.1-only", "LGPL-2.1-or-later",
        "LGPL-3.0-only", "MIT", "MPL-2.0", "MS-PL", "NCSA", "OFL-1.1",
        "OpenSSL", "PostgreSQL", "PSF-2.0", "Unlicense", "WTFPL", "Zlib",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise EvidenceError("source_id must not be empty")
        if not isinstance(self.title, str) or not self.title.strip():
            raise EvidenceError("title must not be empty")
        if not isinstance(self.url, str) or any(ord(ch) < 32 for ch in self.url):
            raise EvidenceError("url must be valid HTTPS or URN URI")
        try:
            parsed = urlparse(self.url)
            scheme = parsed.scheme.lower()
            if scheme == "https":
                # Accessing hostname/port also validates malformed authority/port syntax.
                if not parsed.hostname or parsed.username is not None or parsed.password is not None:
                    raise EvidenceError("url must be valid HTTPS or URN URI")
                _ = parsed.port
            elif scheme == "urn":
                if not re.fullmatch(r"urn:[A-Za-z0-9][A-Za-z0-9-]{0,31}:.+", self.url, re.IGNORECASE):
                    raise EvidenceError("url must be valid HTTPS or URN URI")
            else:
                raise EvidenceError("url must be valid HTTPS or URN URI")
        except ValueError as exc:
            raise EvidenceError("url must be valid HTTPS or URN URI") from exc
        if not isinstance(self.publisher, str) or not self.publisher.strip():
            raise EvidenceError("publisher must not be empty")
        if not isinstance(self.license_spdx, str) or self.license_spdx not in self.SPDX_LICENSE_IDS:
            if not (isinstance(self.license_spdx, str) and
                    re.fullmatch(r"LicenseRef-[A-Za-z0-9.-]+", self.license_spdx)):
                raise EvidenceError("license_spdx must be a recognized SPDX license identifier or LicenseRef")
        if not isinstance(self.retrieved_at, str):
            raise EvidenceError("retrieved_at must be an ISO-8601 timestamp with an explicit timezone")
        try:
            timestamp = datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EvidenceError("retrieved_at must be an ISO-8601 timestamp with an explicit timezone") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise EvidenceError("retrieved_at must include an explicit timezone")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256_hash or ""):
            raise EvidenceError("sha256_hash must be a 64-character lowercase hex digest")
        if not isinstance(self.content, str) or not self.content.strip():
            raise EvidenceError("source content is required so its SHA-256 digest can be verified")
        actual = _compute_sha256(self.content)
        if actual != self.sha256_hash:
            raise EvidenceError(f"source content SHA-256 digest mismatch: expected {self.sha256_hash}, got {actual}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidencePassage:
    """Granular passage excerpt from a verified source document."""
    passage_id: str
    source_id: str
    locator: str  # e.g., "Section 4.1, Paragraph 2"
    text: str
    sha256_hash: str
    authoritativeness_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.passage_id.strip():
            raise EvidenceError("passage_id must not be empty")
        if not self.source_id.strip():
            raise EvidenceError("source_id must not be empty")
        if not self.text.strip():
            raise EvidenceError("text must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256_hash):
            raise EvidenceError("sha256_hash must be a 64-character lowercase hex digest")
        actual = _compute_sha256(self.text)
        if actual != self.sha256_hash:
            raise EvidenceError("passage text SHA-256 digest mismatch")
        if not (0.0 <= self.authoritativeness_weight <= 1.0):
            raise EvidenceError("authoritativeness_weight must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AtomicClaim:
    """Atomic, decomposed factual claim or premise with epistemic categorization."""
    claim_id: str
    statement: str
    subject: str = ""
    predicate: str = ""
    object_val: str = ""
    is_assumption: bool = False
    epistemic_category: EpistemicCategory = EpistemicCategory.UNVERIFIED_CONJECTURE
    calibrated_confidence: float = 0.0
    grounding_score: float = 0.0
    grounding_passage_ids: List[str] = field(default_factory=list)
    contradicting_passage_ids: List[str] = field(default_factory=list)
    contradicting_claim_ids: List[str] = field(default_factory=list)
    matched_snippet: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.is_assumption and self.statement:
            assumption_cues = ["assuming", "suppose", "assume", "hypothetically", "if we assume", "premise:"]
            if any(cue in self.statement.lower() for cue in assumption_cues):
                self.is_assumption = True

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["epistemic_category"] = self.epistemic_category.value
        return res


@dataclass(frozen=True)
class ContradictionConflict:
    """Formal contradiction report between statements or evidence."""
    conflict_type: ContradictionType
    claim_a_id: str
    claim_b_id: str
    statement_a: str
    statement_b: str
    explanation: str
    severity: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["conflict_type"] = self.conflict_type.value
        return res


class ContradictionDetector:
    """Detects multi-dimensional factual and logical contradictions."""

    # Lexical pairs with opposing polarity
    ANTONYM_PAIRS: List[Tuple[str, str]] = [
        ("increase", "decrease"),
        ("increased", "decreased"),
        ("increasing", "decreasing"),
        ("rise", "fall"),
        ("risen", "fallen"),
        ("expand", "contract"),
        ("expanded", "contracted"),
        ("accelerate", "decelerate"),
        ("accelerated", "decelerated"),
        ("enable", "disable"),
        ("enabled", "disabled"),
        ("permit", "forbid"),
        ("permitted", "forbidden"),
        ("allow", "disallow"),
        ("allowed", "disallowed"),
        ("valid", "invalid"),
        ("true", "false"),
        ("success", "failure"),
        ("succeeded", "failed"),
        ("converge", "diverge"),
        ("converged", "diverged"),
        ("benign", "malignant"),
        ("accept", "reject"),
        ("accepted", "rejected"),
        ("stable", "unstable"),
        ("optimal", "suboptimal"),
        ("compatible", "incompatible"),
        ("present", "absent"),
        ("active", "inactive"),
    ]

    NEGATION_WORDS = {"not", "never", "cannot", "no", "neither", "nor", "hardly", "scarcely", "without"}

    @classmethod
    def _has_negation(cls, text: str) -> bool:
        tokens = set(re.findall(r"\b\w+\b", _normalize_text(text)))
        return bool(tokens.intersection(cls.NEGATION_WORDS))

    @classmethod
    def _strip_negation(cls, text: str) -> str:
        norm = _normalize_text(text)
        for neg in cls.NEGATION_WORDS:
            norm = re.sub(rf"\b{neg}\b", " ", norm)
        return " ".join(norm.split())

    @classmethod
    def _extract_numbers_with_units(cls, text: str) -> List[Tuple[float, str]]:
        """Extracts numerical quantities and associated units (e.g. 300000 km/s)."""
        pattern = r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z/%]*)"
        matches = []
        for val_str, unit in re.findall(pattern, text):
            try:
                matches.append((float(val_str), unit.lower().strip()))
            except ValueError:
                continue
        return matches

    @classmethod
    def detect_conflicts(
        cls,
        claims: List[AtomicClaim],
        passages: Optional[List[EvidencePassage]] = None,
    ) -> List[ContradictionConflict]:
        """Scans claims against each other and against evidence passages for contradictions."""
        conflicts: List[ContradictionConflict] = []
        passages = passages or []

        # 1. Check claim-to-claim mutual contradictions
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                c1 = claims[i]
                c2 = claims[j]
                conflict = cls._evaluate_pair(
                    id_a=c1.claim_id,
                    text_a=c1.statement,
                    subj_a=c1.subject,
                    pred_a=c1.predicate,
                    id_b=c2.claim_id,
                    text_b=c2.statement,
                    subj_b=c2.subject,
                    pred_b=c2.predicate,
                )
                if conflict:
                    conflicts.append(conflict)
                    c1.contradicting_claim_ids.append(c2.claim_id)
                    c2.contradicting_claim_ids.append(c1.claim_id)

        # 2. Check claim-to-passage contradictions
        for c in claims:
            for p in passages:
                conflict = cls._evaluate_pair(
                    id_a=c.claim_id,
                    text_a=c.statement,
                    subj_a=c.subject,
                    pred_a=c.predicate,
                    id_b=p.passage_id,
                    text_b=p.text,
                    subj_b="",
                    pred_b="",
                )
                if conflict:
                    conflicts.append(conflict)
                    c.contradicting_passage_ids.append(p.passage_id)

        return conflicts

    @classmethod
    def _evaluate_pair(
        cls,
        id_a: str,
        text_a: str,
        subj_a: str,
        pred_a: str,
        id_b: str,
        text_b: str,
        subj_b: str,
        pred_b: str,
    ) -> Optional[ContradictionConflict]:
        norm_a = _normalize_text(text_a)
        norm_b = _normalize_text(text_b)

        # A. Direct Negation Check
        has_neg_a = cls._has_negation(norm_a)
        has_neg_b = cls._has_negation(norm_b)
        if has_neg_a != has_neg_b:
            stripped_a = cls._strip_negation(norm_a)
            stripped_b = cls._strip_negation(norm_b)
            # Check overlap after removing negation
            tokens_a = set(re.findall(r"\b\w+\b", stripped_a))
            tokens_b = set(re.findall(r"\b\w+\b", stripped_b))
            if tokens_a and tokens_b:
                overlap = len(tokens_a.intersection(tokens_b)) / float(min(len(tokens_a), len(tokens_b)))
                if overlap >= 0.75:
                    return ContradictionConflict(
                        conflict_type=ContradictionType.DIRECT_NEGATION,
                        claim_a_id=id_a,
                        claim_b_id=id_b,
                        statement_a=text_a,
                        statement_b=text_b,
                        explanation=f"Direct negation polarity conflict between '{text_a}' and '{text_b}' (overlap: {overlap:.2f})",
                        severity=1.0,
                    )

        # B. Antonym Polarity Check
        for w1, w2 in cls.ANTONYM_PAIRS:
            in_a1 = f" {w1} " in f" {norm_a} "
            in_a2 = f" {w2} " in f" {norm_a} "
            in_b1 = f" {w1} " in f" {norm_b} "
            in_b2 = f" {w2} " in f" {norm_b} "

            if (in_a1 and in_b2) or (in_a2 and in_b1):
                # Verify that they share common context/subject
                tokens_a = set(re.findall(r"\b\w+\b", norm_a))
                tokens_b = set(re.findall(r"\b\w+\b", norm_b))
                overlap = len(tokens_a.intersection(tokens_b)) / float(max(1, min(len(tokens_a), len(tokens_b))))
                if overlap >= 0.40:
                    return ContradictionConflict(
                        conflict_type=ContradictionType.ANTONYM_POLARITY,
                        claim_a_id=id_a,
                        claim_b_id=id_b,
                        statement_a=text_a,
                        statement_b=text_b,
                        explanation=f"Antonym polarity conflict between '{w1}' and '{w2}' regarding shared subject",
                        severity=0.90,
                    )

        # C. Numerical Disparity Check
        nums_a = cls._extract_numbers_with_units(text_a)
        nums_b = cls._extract_numbers_with_units(text_b)
        if nums_a and nums_b:
            vals_in_b = {val for val, _ in nums_b}
            vals_in_a = {val for val, _ in nums_a}
            unmatched_a = [(val, unit) for val, unit in nums_a if val not in vals_in_b]
            unmatched_b = [(val, unit) for val, unit in nums_b if val not in vals_in_a]

            for val_a, unit_a in unmatched_a:
                for val_b, unit_b in unmatched_b:
                    if unit_a == unit_b and abs(val_a - val_b) > 1e-4:
                        # Check shared context
                        tokens_a = set(re.findall(r"\b[a-zA-Z]{3,}\b", norm_a))
                        tokens_b = set(re.findall(r"\b[a-zA-Z]{3,}\b", norm_b))
                        overlap = tokens_a.intersection(tokens_b)
                        if len(overlap) >= 2:
                            return ContradictionConflict(
                                conflict_type=ContradictionType.NUMERICAL_DISPARITY,
                                claim_a_id=id_a,
                                claim_b_id=id_b,
                                statement_a=text_a,
                                statement_b=text_b,
                                explanation=f"Numerical value disparity: {val_a} {unit_a} vs {val_b} {unit_b}",
                                severity=0.85,
                            )

        # D. Explicit Subject/Predicate Collision
        if subj_a and subj_b and pred_a and pred_b:
            if _normalize_text(subj_a) == _normalize_text(subj_b) and _normalize_text(pred_a) == _normalize_text(pred_b):
                if norm_a != norm_b:
                    return ContradictionConflict(
                        conflict_type=ContradictionType.ATTRIBUTE_COLLISION,
                        claim_a_id=id_a,
                        claim_b_id=id_b,
                        statement_a=text_a,
                        statement_b=text_b,
                        explanation=f"Mutually exclusive attribute collision on subject '{subj_a}' with predicate '{pred_a}'",
                        severity=0.80,
                    )

        return None


class ClaimGroundingVerifier:
    """Verifies atomic factual claims against granular evidence passages."""

    STOPWORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in", "of",
        "for", "on", "with", "by", "that", "this", "it", "at", "as", "from"
    }

    @classmethod
    def verify_grounding(
        cls,
        claim: AtomicClaim,
        passages: List[EvidencePassage],
    ) -> Tuple[float, List[str], Optional[str]]:
        """Evaluates token recall, n-gram containment, and authoritativeness across passages."""
        norm_claim = _normalize_text(claim.statement)
        claim_tokens = [t for t in re.findall(r"\b\w+\b", norm_claim) if t not in cls.STOPWORDS]
        if not claim_tokens:
            return 0.0, [], None

        best_score = 0.0
        best_passage_id: Optional[str] = None
        best_snippet: Optional[str] = None
        matched_passage_ids: List[str] = []

        for passage in passages:
            norm_pass = _normalize_text(passage.text)
            pass_tokens = set(re.findall(r"\b\w+\b", norm_pass))

            # Token recall
            matched = [t for t in claim_tokens if t in pass_tokens]
            recall = len(matched) / float(len(claim_tokens))

            # Phrase 3-gram match
            words = norm_claim.split()
            phrase_matched = False
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    trigram = " ".join(words[i:i+3])
                    if trigram in norm_pass:
                        phrase_matched = True
                        break
            elif len(words) > 0 and norm_claim in norm_pass:
                phrase_matched = True

            # Weighted score incorporating authoritativeness
            passage_score = recall * (1.15 if phrase_matched else 0.85) * passage.authoritativeness_weight
            passage_score = min(1.0, round(passage_score, 4))

            if passage_score >= 0.50:
                matched_passage_ids.append(passage.passage_id)
                if passage_score > best_score:
                    best_score = passage_score
                    best_passage_id = passage.passage_id
                    # Locate snippet around first matched token
                    first_idx = norm_pass.find(matched[0]) if matched else 0
                    start = max(0, first_idx - 30)
                    end = min(len(norm_pass), first_idx + 120)
                    best_snippet = passage.text[start:end]

        return best_score, matched_passage_ids, best_snippet


class ConfidenceCalibrator:
    """Computes calibrated epistemic confidence and enforces honest abstention."""

    @classmethod
    def calibrate_claim(
        cls,
        claim: AtomicClaim,
        grounding_score: float,
        has_contradiction: bool,
        num_sources: int,
    ) -> Tuple[float, EpistemicCategory]:
        """Calibrates confidence and assigns epistemic category."""
        # 1. Contradictions fail closed
        if has_contradiction:
            return 0.0, EpistemicCategory.CONTRADICTION

        # 2. Explicit assumptions are capped
        if claim.is_assumption:
            conf = min(0.40, max(0.10, round(grounding_score * 0.40, 3)))
            return conf, EpistemicCategory.EXPLICIT_ASSUMPTION

        # 3. Grounded facts with source corroboration
        if grounding_score >= 0.80 and num_sources >= 1:
            multi_boost = min(0.05, (num_sources - 1) * 0.02)
            conf = min(0.99, round(0.90 + (grounding_score - 0.80) * 0.45 + multi_boost, 3))
            return conf, EpistemicCategory.VERIFIED_FACT

        # 4. Plausible inference
        if grounding_score >= 0.50:
            conf = round(0.50 + (grounding_score - 0.50) * 0.60, 3)
            return conf, EpistemicCategory.PLAUSIBLE_INFERENCE

        # 5. Unverified conjecture
        conf = round(max(0.05, grounding_score * 0.50), 3)
        return conf, EpistemicCategory.UNVERIFIED_CONJECTURE

    @classmethod
    def evaluate_abstention(
        cls,
        claims: List[AtomicClaim],
        conflicts: List[ContradictionConflict],
        confidence_threshold: float = 0.55,
    ) -> Tuple[bool, str]:
        """Determines if the system must honestly abstain due to lack of evidence or contradictions."""
        if not claims:
            return True, "No verifiable claims provided for evaluation."

        # High-severity contradictions require abstention
        severe_conflicts = [c for c in conflicts if c.severity >= 0.80]
        if severe_conflicts:
            reasons = "; ".join(f"[{c.conflict_type.value}] {c.explanation}" for c in severe_conflicts[:2])
            return True, f"Abstention required due to unresolved contradictions: {reasons}"

        # If zero facts are verified and no assumptions are declared
        verified_count = sum(1 for c in claims if c.epistemic_category == EpistemicCategory.VERIFIED_FACT)
        inference_count = sum(1 for c in claims if c.epistemic_category == EpistemicCategory.PLAUSIBLE_INFERENCE)
        assumption_count = sum(1 for c in claims if c.epistemic_category == EpistemicCategory.EXPLICIT_ASSUMPTION)

        if (verified_count + inference_count + assumption_count) == 0:
            return True, "Abstention required: All proposed claims lack authoritative source grounding."

        # Mean confidence across non-assumption claims
        substantive = [c for c in claims if not c.is_assumption]
        if substantive:
            mean_conf = sum(c.calibrated_confidence for c in substantive) / float(len(substantive))
            if mean_conf < confidence_threshold:
                return True, (
                    f"Abstention required: Calibrated confidence ({mean_conf:.2f}) "
                    f"falls below threshold ({confidence_threshold:.2f})."
                )

        return False, ""


class EvidenceProvenanceGraph:
    """Directed Acyclic Graph tracking evidence provenance and cryptographic integrity."""

    def __init__(self) -> None:
        self.sources: Dict[str, SourceProvenance] = {}
        self.passages: Dict[str, EvidencePassage] = {}
        self.claims: Dict[str, AtomicClaim] = {}

    def add_source(self, source: SourceProvenance) -> None:
        if source.source_id in self.sources:
            raise EvidenceError(f"Duplicate source_id '{source.source_id}' is not allowed")
        self.sources[source.source_id] = source

    def add_passage(self, passage: EvidencePassage) -> None:
        if passage.passage_id in self.passages:
            raise EvidenceError(f"Duplicate passage_id '{passage.passage_id}' is not allowed")
        source = self.sources.get(passage.source_id)
        if source is None:
            raise EvidenceError(f"Cannot add passage: referenced source_id '{passage.source_id}' does not exist")
        if _normalize_text(passage.text) not in _normalize_text(source.content):
            raise EvidenceError(
                f"Passage '{passage.passage_id}' is not contained in the verified content of source '{source.source_id}'"
            )
        self.passages[passage.passage_id] = passage

    def add_claim(self, claim: AtomicClaim) -> None:
        if claim.claim_id in self.claims:
            raise EvidenceError(f"Duplicate claim_id '{claim.claim_id}' is not allowed")
        self.claims[claim.claim_id] = claim

    def get_provenance_trail(self, claim_id: str) -> Dict[str, Any]:
        """Traverses the graph to return the complete cryptographic provenance for a claim."""
        claim = self.claims.get(claim_id)
        if not claim:
            raise EvidenceError(f"Claim '{claim_id}' not found in provenance graph")

        passages_info = []
        sources_info = []
        seen_sources: Set[str] = set()

        for pid in claim.grounding_passage_ids:
            passage = self.passages.get(pid)
            if passage:
                passages_info.append({
                    "passage_id": passage.passage_id,
                    "locator": passage.locator,
                    "sha256_hash": passage.sha256_hash,
                    "weight": passage.authoritativeness_weight,
                })
                if passage.source_id not in seen_sources:
                    seen_sources.add(passage.source_id)
                    source = self.sources.get(passage.source_id)
                    if source:
                        sources_info.append({
                            "source_id": source.source_id,
                            "title": source.title,
                            "url": source.url,
                            "publisher": source.publisher,
                            "license_spdx": source.license_spdx,
                            "retrieved_at": source.retrieved_at,
                            "sha256_hash": source.sha256_hash,
                        })

        return {
            "claim_id": claim.claim_id,
            "statement": claim.statement,
            "epistemic_category": claim.epistemic_category.value,
            "calibrated_confidence": claim.calibrated_confidence,
            "grounding_passages": passages_info,
            "source_provenance": sources_info,
        }

    def verify_graph_integrity(self) -> Tuple[bool, List[str]]:
        """Verifies cryptographic hashes and structural consistency across the entire graph."""
        issues: List[str] = []

        # 1. Verify sources
        for s_id, source in self.sources.items():
            if not source.content:
                issues.append(f"Source '{s_id}' has no verifiable source content")
            else:
                digest = _compute_sha256(source.content)
                if digest != source.sha256_hash:
                    issues.append(f"Source '{s_id}' content hash mismatch")
            if s_id != source.source_id:
                issues.append(f"Source key '{s_id}' does not match record ID '{source.source_id}'")

        # 2. Verify passages
        for p_id, passage in self.passages.items():
            if passage.source_id not in self.sources:
                issues.append(f"Passage '{p_id}' has dangling source_id '{passage.source_id}'")
            elif _normalize_text(passage.text) not in _normalize_text(self.sources[passage.source_id].content):
                issues.append(f"Passage '{p_id}' is not contained in its source content")
            if p_id != passage.passage_id:
                issues.append(f"Passage key '{p_id}' does not match record ID '{passage.passage_id}'")
            digest = _compute_sha256(passage.text)
            if digest != passage.sha256_hash:
                issues.append(f"Passage '{p_id}' text hash mismatch")

        # 3. Verify claims
        for c_id, claim in self.claims.items():
            if c_id != claim.claim_id:
                issues.append(f"Claim key '{c_id}' does not match record ID '{claim.claim_id}'")
            for pid in claim.grounding_passage_ids:
                if pid not in self.passages:
                    issues.append(f"Claim '{c_id}' references unknown passage '{pid}'")

        return len(issues) == 0, issues

    def export_graph(self) -> Dict[str, Any]:
        return {
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
            "passages": {k: v.to_dict() for k, v in self.passages.items()},
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
        }


@dataclass
class GroundedSynthesisReport:
    """Synthesis result segregating verified facts, assumptions, and inferences."""
    query: str
    abstained: bool
    abstention_reason: str
    overall_confidence: float
    verified_facts: List[str]
    plausible_inferences: List[str]
    explicit_assumptions: List[str]
    unverified_conjectures: List[str]
    contradictions_detected: List[Dict[str, Any]]
    formatted_response: str
    provenance_summary: List[Dict[str, Any]]
    evaluated_at: str = field(default_factory=_iso_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceBasedIntelligenceEngine:
    """Unified Orchestrator for Stage 14 Evidence-Based Intelligence."""

    def __init__(self, logger: Optional[StructuredLogger] = None) -> None:
        self.logger = logger or StructuredLogger(name="mirror-evidence-intelligence")
        self.contradiction_detector = ContradictionDetector()
        self.grounding_verifier = ClaimGroundingVerifier()
        self.calibrator = ConfidenceCalibrator()

    @classmethod
    def decompose_statements(cls, text: str) -> List[AtomicClaim]:
        """Decomposes structured or prose text into atomic claims."""
        # Split by sentences or newline bullets
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        claims: List[AtomicClaim] = []
        idx = 1
        for s in raw_sentences:
            s_clean = s.strip().lstrip("-*1234567890. ")
            if len(s_clean) < 5:
                continue

            # Detect explicit assumption markers
            assumption_cues = ["assuming", "suppose", "assume", "hypothetically", "if we assume", "premise:"]
            is_assump = any(cue in s_clean.lower() for cue in assumption_cues)

            # Extract naive subject/predicate
            words = s_clean.split()
            subject = words[0] if words else ""
            predicate = words[1] if len(words) > 1 else ""

            claim = AtomicClaim(
                claim_id=f"claim_{idx:03d}",
                statement=s_clean,
                subject=subject,
                predicate=predicate,
                is_assumption=is_assump,
            )
            claims.append(claim)
            idx += 1
        return claims

    def process_and_synthesize(
        self,
        query: str,
        claims: List[AtomicClaim],
        sources: List[SourceProvenance],
        passages: List[EvidencePassage],
        confidence_threshold: float = 0.55,
    ) -> GroundedSynthesisReport:
        """Executes the full Stage 14 evidence pipeline."""
        graph = EvidenceProvenanceGraph()
        for s in sources:
            graph.add_source(s)
        for p in passages:
            graph.add_passage(p)

        # 1. Contradiction Detection across claims and passages
        conflicts = self.contradiction_detector.detect_conflicts(claims, passages)

        # 2. Grounding Verification & Confidence Calibration for each claim
        for claim in claims:
            score, matched_pids, snippet = ClaimGroundingVerifier.verify_grounding(claim, passages)
            claim.grounding_score = score
            claim.grounding_passage_ids = matched_pids
            claim.matched_snippet = snippet

            has_contra = len(claim.contradicting_claim_ids) > 0 or len(claim.contradicting_passage_ids) > 0
            num_sources = len({graph.passages[pid].source_id for pid in matched_pids if pid in graph.passages})

            calibrated_conf, category = self.calibrator.calibrate_claim(
                claim=claim,
                grounding_score=score,
                has_contradiction=has_contra,
                num_sources=num_sources,
            )
            claim.calibrated_confidence = calibrated_conf
            claim.epistemic_category = category
            graph.add_claim(claim)

        # 3. Honest Abstention Evaluation
        should_abstain, abstention_reason = self.calibrator.evaluate_abstention(
            claims=claims,
            conflicts=conflicts,
            confidence_threshold=confidence_threshold,
        )

        # 4. Group Claims by Epistemic Category
        facts = [c.statement for c in claims if c.epistemic_category == EpistemicCategory.VERIFIED_FACT]
        inferences = [c.statement for c in claims if c.epistemic_category == EpistemicCategory.PLAUSIBLE_INFERENCE]
        assumptions = [c.statement for c in claims if c.epistemic_category == EpistemicCategory.EXPLICIT_ASSUMPTION]
        conjectures = [c.statement for c in claims if c.epistemic_category == EpistemicCategory.UNVERIFIED_CONJECTURE]

        # 5. Overall Confidence Calculation
        if should_abstain:
            overall_conf = 0.0
        else:
            confs = [c.calibrated_confidence for c in claims if not c.is_assumption]
            overall_conf = round(sum(confs) / float(len(confs)), 3) if confs else 0.40

        # 6. Formatted Response Generation
        if should_abstain:
            formatted = f"**Abstention Notice:** {abstention_reason}"
        else:
            sections = []
            if facts:
                sections.append("### Verified Facts\n" + "\n".join(f"- {f}" for f in facts))
            if inferences:
                sections.append("### Plausible Inferences\n" + "\n".join(f"- {inf}" for inf in inferences))
            if assumptions:
                sections.append("### Explicit Assumptions\n" + "\n".join(f"- {a}" for a in assumptions))
            if conjectures:
                sections.append("### Unverified Conjectures (Requires Verification)\n" + "\n".join(f"- {c}" for c in conjectures))
            formatted = "\n\n".join(sections)

        # 7. Provenance Summaries
        provenance_summaries = []
        for c in claims:
            if c.grounding_passage_ids:
                trail = graph.get_provenance_trail(c.claim_id)
                provenance_summaries.append(trail)

        self.logger.info(
            "Evidence-based intelligence synthesis complete",
            extra={
                "query": query,
                "abstained": should_abstain,
                "overall_confidence": overall_conf,
                "num_facts": len(facts),
                "num_assumptions": len(assumptions),
                "num_conflicts": len(conflicts),
            },
        )

        return GroundedSynthesisReport(
            query=query,
            abstained=should_abstain,
            abstention_reason=abstention_reason,
            overall_confidence=overall_conf,
            verified_facts=facts,
            plausible_inferences=inferences,
            explicit_assumptions=assumptions,
            unverified_conjectures=conjectures,
            contradictions_detected=[c.to_dict() for c in conflicts],
            formatted_response=formatted,
            provenance_summary=provenance_summaries,
        )
