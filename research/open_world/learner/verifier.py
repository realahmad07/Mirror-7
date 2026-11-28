"""Stage S11.1 Stage 8: Independent Verification (Zero Self-Certification!).

STRICT ISOLATION & ZERO MANUAL DOMAIN SCHEMAS:
- Pure Python 3.11 standard library ONLY.
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from research.open_world.contracts import STOPWORDS, content_tokens, tokenize
from research.open_world.learner.evidence_ledger import RawEvidenceUnit, parse_timestamp_date
from research.open_world.learner.extraction import (
    _DATE_SPAN_RE,
    _PERCENT_FRAC_RE,
    _QUANTITY_SPAN_RE,
    detect_polarity_and_modality,
    morph_stem,
    token_grounded_in_set,
)
from research.open_world.learner.temporal_graph import BiTemporalGraph, resolve_event_date

_RAW_CONTRAST_PAIRS: tuple[tuple[str, str], ...] = (
    ("smallest", "largest"), ("small", "large"), ("smaller", "larger"),
    ("minimum", "maximum"), ("min", "max"), ("minimal", "maximal"),
    ("short", "long"), ("short", "lengthy"), ("shorter", "longer"), ("shortest", "longest"),
    ("first", "last"), ("before", "after"), ("earlier", "later"), ("early", "late"),
    ("most", "least"), ("more", "less"), ("more", "fewer"), ("many", "few"),
    ("highest", "lowest"), ("higher", "lower"), ("high", "low"),
    ("oldest", "newest"), ("older", "newer"), ("old", "new"),
    ("oldest", "youngest"), ("older", "younger"), ("old", "young"),
    ("increase", "decrease"), ("increased", "decreased"), ("increasing", "decreasing"),
    ("start", "end"), ("started", "ended"), ("begin", "end"), ("began", "ended"),
    ("inside", "outside"), ("internal", "external"), ("inner", "outer"),
    ("above", "below"), ("over", "under"), ("top", "bottom"), ("up", "down"),
    ("north", "south"), ("northern", "southern"), ("east", "west"), ("eastern", "western"),
    ("left", "right"), ("major", "minor"), ("public", "private"),
    ("legal", "illegal"), ("active", "passive"), ("true", "false"),
    ("never", "always"), ("only", "never"), ("all", "none"),
    ("best", "worst"), ("better", "worse"), ("good", "bad"),
    ("win", "lose"), ("won", "lost"), ("gain", "lose"), ("gained", "lost"),
    ("open", "close"), ("opened", "closed"), ("add", "remove"), ("added", "removed"),
    ("keep", "remove"), ("retain", "relinquish"), ("retain", "give"),
    ("create", "destroy"), ("build", "destroy"), ("allow", "forbid"),
    ("accept", "reject"), ("include", "exclude"), ("positive", "negative"),
    ("former", "current"), ("previous", "next"), ("prior", "subsequent"),
    ("modern", "ancient"), ("domestic", "foreign"), ("urban", "rural"),
    ("rich", "poor"), ("strong", "weak"), ("strongest", "weakest"),
    ("fast", "slow"), ("fastest", "slowest"), ("heavy", "light"), ("heaviest", "lightest"),
    ("hot", "cold"), ("warm", "cool"), ("big", "small"), ("biggest", "smallest"),
    ("buy", "sell"), ("bought", "sold"), ("enter", "exit"), ("arrive", "depart"),
    ("expand", "shrink"), ("rise", "fall"), ("rose", "fell"),
    ("give", "take"), ("single", "multiple"), ("common", "rare"),
    ("narrow", "wide"), ("deep", "shallow"), ("thick", "thin"),
    ("simple", "complex"), ("easy", "difficult"), ("safe", "dangerous"),
    ("full", "empty"), ("alive", "dead"), ("present", "absent"),
)

CONTRAST_MAP: dict[str, set[str]] = {}
for _a, _b in _RAW_CONTRAST_PAIRS:
    CONTRAST_MAP.setdefault(_a, set()).add(_b)
    CONTRAST_MAP.setdefault(_b, set()).add(_a)
    _sa, _sb = morph_stem(_a), morph_stem(_b)
    if _sa != _sb:
        CONTRAST_MAP.setdefault(_sa, set()).add(_sb)
        CONTRAST_MAP.setdefault(_sb, set()).add(_sa)

_PROPER_Q_RE = re.compile(
    r"(?<!\w)(?:(?:Dr|Mr|Mrs|Ms|Prof|St|Mt)\.\s+)?[A-ZÀ-ÖØ-Þ][^\W_]*(?:['’+-][^\W_]+)*(?:\s+[A-ZÀ-ÖØ-Þ][^\W_]*(?:['’+-][^\W_]+)*){0,4}(?!\w)"
)
_QUOTED_Q_RE = re.compile(r"(?:\"([^\"\n]{2,80})\"|'([^'\n]{2,80})'|“([^”\n]{2,80})”)")
_YEAR_OR_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_QTY_EXTRACT_RE = re.compile(
    r"(?:([$€£])\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*(%|percent|dollars?|euros?|pounds?|days?|weeks?|months?|years?|hours?|hrs?|minutes?|mins?|miles?|km|kilometers?|kilometres?|pages?|books?|times?|items?|people|subjects|tickets?|postcards?|mugs?|cups?|sessions?)?)",
    re.I,
)
_QUESTION_META_WORDS: frozenset[str] = frozenset(
    {
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "name", "kind", "type", "sort", "form", "many", "much", "long", "far",
        "often", "old", "year", "years", "date", "month", "months", "day", "days",
        "week", "weeks", "time", "times", "number", "percentage", "percent",
        "did", "does", "do", "was", "were", "is", "are", "has", "have", "had",
        "can", "could", "would", "should", "will", "might", "may",
        "tell", "remind", "mention", "mentioned", "previous", "conversation",
        "talked", "said", "asked", "looking", "back", "follow", "wondering",
        "currently", "initially", "originally", "previously", "recently", "latest",
        "total", "altogether", "combined", "difference", "between", "passed", "since",
        "suggest", "suggestions", "recommend", "recommendations", "tips", "ideas",
    }
)


_CALENDAR_NON_ENTITIES: frozenset[str] = frozenset(
    {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "i", "i'm", "i've", "i'd", "i'll", "hi", "hey", "hello",
    }
)


def detect_required_answer_type(question: str) -> str:
    """Detect strict grammatical answer-type requirement (ZERO domain schemas)."""
    ql = question.strip().casefold()
    if re.search(r"\b(?:what\s+percentage|what\s+percent|what\s+proportion|what\s+fraction)\b", ql):
        return "percentage"
    if re.search(r"\b(?:how\s+many|how\s+much|what\s+number|how\s+old|how\s+far|how\s+tall|how\s+long|how\s+fast|what\s+distance|what\s+price|what\s+cost)\b", ql):
        return "numeric"
    if re.search(
        r"(?:(?:^|[.?!;,]\s*)when\s+(?:did|do|does|was|were|is|are|will|would|had|has|have)\b"
        r"|\b(?:what\s+year|which\s+year|in\s+what\s+year|in\s+what\s+years|what\s+date|what\s+century|in\s+what\s+century|what\s+month)\b)",
        ql,
    ):
        return "temporal"
    return "general"


def unit_satisfies_answer_type(text: str, question: str, req_type: str) -> bool:
    """Check whether ``text`` contains a NOVEL expression matching ``req_type``."""
    if req_type == "general":
        return True
    q_nums = set(_YEAR_OR_NUM_RE.findall(question))
    if req_type == "temporal":
        for m in _DATE_SPAN_RE.finditer(text):
            val = m.group(0).strip()
            if val not in q_nums and val.casefold() not in question.casefold():
                return True
        if re.search(r"\b(?:century|decade|era|period|spring|summer|autumn|winter)\b", text, re.I):
            return True
        return False
    if req_type == "percentage":
        for m in _PERCENT_FRAC_RE.finditer(text):
            val = m.group(0).strip()
            if val.casefold() not in question.casefold():
                return True
        return False
    if req_type == "numeric":
        for m in _QUANTITY_SPAN_RE.finditer(text):
            val = m.group(0).strip()
            if re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", val):
                continue
            if val not in q_nums and val.casefold() not in question.casefold():
                return True
        return False
    return True


def answer_satisfies_type(answer: str, req_type: str) -> bool:
    if req_type == "general":
        return True
    if req_type == "temporal":
        return bool(_DATE_SPAN_RE.search(answer) or re.search(r"\b(?:century|decade|year|month|day|week|ago|before|after|since|until|into)\b", answer, re.I))
    if req_type == "percentage":
        return bool(_PERCENT_FRAC_RE.search(answer))
    if req_type == "numeric":
        return bool(_QUANTITY_SPAN_RE.search(answer))
    return True


@dataclass(frozen=True)
class VerificationResult:
    """Structured independent verification report returned by Stage 8 verifier."""

    passed: bool
    support_score: float
    span_grounded: bool
    answer_type_compatible: bool
    entity_sentence_cov: float
    entity_passage_cov: float
    number_grounding_cov: float
    content_sentence_cov: float
    content_passage_cov: float
    focal_missing_frac: float
    rare_token_grounded_frac: float
    polarity_consistent: bool
    antonym_contradiction: bool
    bitemporal_superseded: bool
    bitemporal_contested: bool
    missing_entities: tuple[str, ...]
    missing_numbers: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "support_score": round(self.support_score, 4),
            "span_grounded": self.span_grounded,
            "answer_type_compatible": self.answer_type_compatible,
            "entity_sentence_cov": round(self.entity_sentence_cov, 4),
            "entity_passage_cov": round(self.entity_passage_cov, 4),
            "number_grounding_cov": round(self.number_grounding_cov, 4),
            "content_sentence_cov": round(self.content_sentence_cov, 4),
            "content_passage_cov": round(self.content_passage_cov, 4),
            "focal_missing_frac": round(self.focal_missing_frac, 4),
            "rare_token_grounded_frac": round(self.rare_token_grounded_frac, 4),
            "polarity_consistent": self.polarity_consistent,
            "antonym_contradiction": self.antonym_contradiction,
            "bitemporal_superseded": self.bitemporal_superseded,
            "bitemporal_contested": self.bitemporal_contested,
            "missing_entities": list(self.missing_entities),
            "missing_numbers": list(self.missing_numbers),
            "rejection_reasons": list(self.rejection_reasons),
        }


def _extract_question_proper_entities(question: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _QUOTED_Q_RE.finditer(question):
        g = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if g and g.casefold() not in seen:
            seen.add(g.casefold())
            out.append(g)
    for m in _PROPER_Q_RE.finditer(question):
        surf = m.group(0).strip()
        if not surf:
            continue
        if m.start() == 0:
            words = surf.split()
            if len(words) == 1:
                if not (len(surf) >= 2 and sum(1 for c in surf if c.isupper()) >= 2):
                    continue
            else:
                if words[0].casefold() in STOPWORDS or words[0].casefold() in _QUESTION_META_WORDS:
                    surf = " ".join(words[1:]).strip()
                    if not surf or not surf[0].isupper():
                        continue
        if surf.casefold() in STOPWORDS or surf.casefold() in _QUESTION_META_WORDS or surf.casefold() in _CALENDAR_NON_ENTITIES:
            continue
        if surf.casefold() not in seen:
            seen.add(surf.casefold())
            out.append(surf)
    return tuple(out)


def _entity_in_text(entity: str, text_low: str, text_stems: set[str], text_raw: set[str]) -> bool:
    el = entity.casefold()
    if el in text_low:
        return True
    e_toks = [t for t in content_tokens(entity) if len(t) >= 2]
    if not e_toks:
        return True
    hits = sum(1 for t in e_toks if token_grounded_in_set(t, text_stems, text_raw))
    if len(e_toks) == 1:
        return hits == 1
    return hits >= max(1, (len(e_toks) + 1) // 2)


def compute_local_idf(units: Sequence[RawEvidenceUnit]) -> dict[str, float]:
    n = len(units)
    if n <= 1:
        return {}
    df: Counter[str] = Counter()
    for u in units:
        for st in {morph_stem(t) for t in content_tokens(u.text)}:
            df[st] += 1
    return {st: math.log(1.0 + (n + 0.5) / (c + 0.5)) for st, c in df.items()}

def verify_evidence_grounding(
    *,
    question: str,
    proposed_answer: str,
    cited_units: Sequence[RawEvidenceUnit],
    all_units: Sequence[RawEvidenceUnit],
    idf_lookup: Mapping[str, float],
    is_computed_answer: bool = False,
    bitemporal_graph: BiTemporalGraph | None = None,
    temporal_mode: str = "latest",
) -> VerificationResult:
    """Independently verify that ``proposed_answer`` and every critical constraint in ``question`` are entailed."""
    if not cited_units:
        return VerificationResult(
            passed=False, support_score=0.0, span_grounded=False, answer_type_compatible=False,
            entity_sentence_cov=0.0, entity_passage_cov=0.0, number_grounding_cov=0.0,
            content_sentence_cov=0.0, content_passage_cov=0.0, focal_missing_frac=1.0,
            rare_token_grounded_frac=0.0, polarity_consistent=False, antonym_contradiction=False,
            bitemporal_superseded=False, bitemporal_contested=False,
            missing_entities=(), missing_numbers=(), rejection_reasons=("no_cited_evidence_units",),
        )

    cited_text = " ".join(u.text for u in cited_units)
    cited_low = cited_text.casefold()
    cited_raw = set(tokenize(cited_text))
    cited_stems = {morph_stem(t) for t in cited_raw}

    if len(all_units) <= 40:
        all_text = " ".join(u.text for u in all_units) if all_units else cited_text
        all_low = all_text.casefold()
        all_raw = set(tokenize(all_text))
    else:
        all_raw = set(cited_raw)
        for u in all_units:
            all_raw.update(tokenize(u.text))
        all_low = cited_low
    all_stems = {morph_stem(t) for t in all_raw}

    # 1. Exact span grounding in raw UTF-8 text
    if is_computed_answer:
        span_grounded = bool(proposed_answer.strip())
    else:
        ans_norm = proposed_answer.strip().casefold()
        if not ans_norm:
            span_grounded = False
        elif ans_norm in cited_low:
            span_grounded = True
        else:
            a_toks = set(content_tokens(proposed_answer))
            span_grounded = bool(a_toks) and (len(a_toks & cited_raw) / len(a_toks) >= 0.75)

    # 2. Strict grammatical answer-type presence check
    req_type = detect_required_answer_type(question)
    if is_computed_answer:
        type_ok = True
    else:
        type_ok = unit_satisfies_answer_type(cited_text, question, req_type) and answer_satisfies_type(
            proposed_answer, req_type
        )

    # 3. Proper entity & quoted-term grounding
    q_entities = _extract_question_proper_entities(question)
    missing_ents: list[str] = []
    if q_entities:
        in_sent = 0
        in_pass = 0
        for ent in q_entities:
            ok_s = _entity_in_text(ent, cited_low, cited_stems, cited_raw)
            if ok_s:
                ok_p = True
            elif len(all_units) <= 40:
                ok_p = _entity_in_text(ent, all_low, all_stems, all_raw)
            else:
                e_toks = [t for t in content_tokens(ent) if len(t) >= 2]
                ok_p = all(token_grounded_in_set(t, all_stems, all_raw, allow_fuzzy=False) for t in e_toks) if e_toks else True
            if ok_s:
                in_sent += 1
            if ok_p:
                in_pass += 1
            else:
                missing_ents.append(ent)
        ent_sent_cov = in_sent / len(q_entities)
        ent_pass_cov = in_pass / len(q_entities)
    else:
        ent_sent_cov = 1.0
        ent_pass_cov = 1.0

    # 4. Numeric / year literal grounding
    q_nums = [n for n in _YEAR_OR_NUM_RE.findall(question) if len(n) >= 2 or n.isdigit()]
    missing_nums: list[str] = []
    if q_nums:
        n_hits = 0.0
        for num in q_nums:
            if num in cited_raw or num in cited_low:
                n_hits += 1.0
            elif num in all_raw:
                n_hits += 0.35
                missing_nums.append(num)
            else:
                missing_nums.append(num)
        num_cov = n_hits / len(q_nums)
    else:
        num_cov = 1.0

    # 5. Polarity & negation consistency
    q_pol, _ = detect_polarity_and_modality(question)
    ev_pol, _ = detect_polarity_and_modality(cited_text)
    polarity_consistent = not (q_pol == "NEGATED" and ev_pol == "AFFIRMED")

    # 6. Domain-agnostic scalar/antonym contradiction detection
    q_raw_toks = set(tokenize(question))
    antonym_contradiction = False
    for qt in q_raw_toks:
        if qt in STOPWORDS or qt in _QUESTION_META_WORDS:
            continue
        qst = morph_stem(qt)
        opposites = CONTRAST_MAP.get(qt) or CONTRAST_MAP.get(qst)
        if not opposites:
            continue
        if not token_grounded_in_set(qt, cited_stems, cited_raw):
            if any(opp in cited_raw or opp in cited_stems for opp in opposites):
                antonym_contradiction = True
                break

    # 7. Local-IDF + Corpus-IDF weighted focal content grounding
    local_idf = compute_local_idf(all_units) if len(all_units) <= 40 else {}
    q_content = [t for t in content_tokens(question) if t not in _QUESTION_META_WORDS]
    if q_content:
        total_w = 0.0
        sent_w = 0.0
        pass_w = 0.0
        focal_missing_w = 0.0
        rare_total = 0
        rare_hit = 0
        uniq_q = list(dict.fromkeys(q_content))
        weights: list[tuple[str, float]] = []
        for tok in uniq_q:
            st = morph_stem(tok)
            c_idf = float(idf_lookup.get(tok, idf_lookup.get(st, 2.5)))
            l_idf = float(local_idf.get(st, math.log(1.0 + max(2, len(all_units))))) if local_idf else c_idf
            w = 0.55 * l_idf + 0.45 * c_idf
            weights.append((tok, w))
        max_w = max(w for _, w in weights) if weights else 1.0
        rare_cut = max(1.6, 0.72 * max_w)
        use_fuzzy_all = len(all_units) <= 40
        for tok, w in weights:
            total_w += w
            hit_s = token_grounded_in_set(tok, cited_stems, cited_raw, allow_fuzzy=True)
            hit_p = hit_s or token_grounded_in_set(tok, all_stems, all_raw, allow_fuzzy=use_fuzzy_all)
            if hit_s:
                sent_w += w
            else:
                focal_missing_w += w * (1.0 if not hit_p else 0.45)
            if hit_p:
                pass_w += w
            if w >= rare_cut:
                rare_total += 1
                if hit_p:
                    rare_hit += 1
        content_sent_cov = sent_w / max(1e-6, total_w)
        content_pass_cov = pass_w / max(1e-6, total_w)
        focal_missing_frac = min(1.0, focal_missing_w / max(1e-6, total_w))
        rare_frac = (rare_hit / rare_total) if rare_total > 0 else content_pass_cov
    else:
        content_sent_cov = 0.5
        content_pass_cov = 0.5
        focal_missing_frac = 0.5
        rare_frac = 0.5

    # 8. Bi-temporal contradiction & supersession audit
    bitemp_superseded = False
    bitemp_contested = False
    if bitemporal_graph is not None and bitemporal_graph.relations:
        cited_ev_ids = {u.evidence_id for u in cited_units}
        for rid, rel in bitemporal_graph.relations.items():
            if rel.evidence_id in cited_ev_ids:
                if temporal_mode != "earliest" and bitemporal_graph.has_later_supersession(rid):
                    bitemp_superseded = True
                if bitemporal_graph.is_contested(rid):
                    bitemp_contested = True

    # Structural SQuAD/open-world contradiction signals (zero domain schemas):
    # (a) Cross-sentence entity split (entities appear in passage but NOT in same cited sentence)
    ent_split_gap = max(0.0, ent_pass_cov - ent_sent_cov)
    # (b) Main final non-entity predicate/verb grounding in cited sentence
    proper_tok_set: set[str] = set()
    for _e in q_entities:
        proper_tok_set.update(tokenize(_e))
    non_ent_q = [t for t in q_content if t not in proper_tok_set and not t.isdigit()]
    last_pred_ok = (
        True
        if not non_ent_q
        else token_grounded_in_set(non_ent_q[-1], cited_stems, cited_raw, allow_fuzzy=True)
    )
    # (c) Ungrounded superlative/extreme scalar modifier in question
    ungrounded_super = False
    for m_sup in re.finditer(
        r"\b(smallest|largest|biggest|highest|lowest|oldest|newest|shortest|longest|maximum|minimum|most|least|first|last|only|never|short|small)\b",
        question.casefold(),
    ):
        if not token_grounded_in_set(m_sup.group(1), cited_stems, cited_raw, allow_fuzzy=True):
            ungrounded_super = True
            break

    reasons: list[str] = []
    score = (
        0.42 * content_sent_cov
        + 0.22 * ent_sent_cov
        + 0.12 * num_cov
        + 0.12 * rare_frac
        + (0.12 if last_pred_ok else 0.0)
        - 0.26 * focal_missing_frac
        - 0.30 * ent_split_gap
    )
    if not last_pred_ok:
        reasons.append(f"ungrounded_focal_predicate:{non_ent_q[-1]}")
    if ent_split_gap >= 0.35:
        reasons.append("cross_sentence_entity_split")
    if ungrounded_super:
        score -= 0.16
        reasons.append("ungrounded_scalar_superlative_modifier")
    if not span_grounded:
        score -= 0.40
        reasons.append("span_not_grounded_in_raw_utf8_bytes")
    if not type_ok:
        score -= 0.45
        reasons.append(f"missing_required_{req_type}_answer_in_evidence")
    if missing_ents:
        score -= 0.36 * min(1.0, len(missing_ents) / max(1, len(q_entities)))
        reasons.append(f"ungrounded_entities:{','.join(missing_ents[:3])}")
    if missing_nums:
        score -= 0.38 * min(1.0, len(missing_nums) / max(1, len(q_nums)))
        reasons.append(f"ungrounded_numbers:{','.join(missing_nums[:3])}")
    if not polarity_consistent:
        score -= 0.38
        reasons.append("polarity_negation_mismatch")
    if antonym_contradiction:
        score -= 0.42
        reasons.append("scalar_antonym_contradiction")
    if bitemp_superseded:
        score -= 0.10
        reasons.append("bitemporal_superseded_by_later_observation")
    if bitemp_contested:
        score -= 0.22
        reasons.append("unresolved_contested_observation")

    score = max(0.0, min(1.0, score))
    hard_fail = (
        not span_grounded
        or not type_ok
        or antonym_contradiction
        or (not polarity_consistent)
        or (bool(missing_nums) and num_cov < 0.5)
        or (bool(missing_ents) and ent_pass_cov < 0.5)
    )
    passed = (not hard_fail) and score >= 0.35

    return VerificationResult(
        passed=passed, support_score=score, span_grounded=span_grounded,
        answer_type_compatible=type_ok, entity_sentence_cov=ent_sent_cov,
        entity_passage_cov=ent_pass_cov, number_grounding_cov=num_cov,
        content_sentence_cov=content_sent_cov, content_passage_cov=content_pass_cov,
        focal_missing_frac=focal_missing_frac, rare_token_grounded_frac=rare_frac,
        polarity_consistent=polarity_consistent, antonym_contradiction=antonym_contradiction,
        bitemporal_superseded=bitemp_superseded, bitemporal_contested=bitemp_contested,
        missing_entities=tuple(missing_ents), missing_numbers=tuple(missing_nums),
        rejection_reasons=tuple(reasons),
    )


def _score_unit_against_clause(
    clause_text: str,
    unit: RawEvidenceUnit,
    idf_lookup: Mapping[str, float],
) -> float:
    ctoks = [t for t in content_tokens(clause_text) if t not in _QUESTION_META_WORDS]
    if not ctoks:
        return 0.0
    u_raw = set(tokenize(unit.text))
    u_stems = {morph_stem(t) for t in u_raw}
    num = 0.0
    den = 0.0
    for t in dict.fromkeys(ctoks):
        w = float(idf_lookup.get(t, idf_lookup.get(morph_stem(t), 2.0)))
        den += w
        if token_grounded_in_set(t, u_stems, u_raw, allow_fuzzy=True):
            num += w
    return num / max(1e-6, den)


def execute_temporal_or_numeric_verifier(
    *,
    question: str,
    question_date: str,
    ranked_units: Sequence[tuple[float, RawEvidenceUnit]],
    idf_lookup: Mapping[str, float],
    enable_bitemporal: bool = True,
) -> dict[str, Any] | None:
    """Executable Stage 8 verifier for exact calendar date deltas, temporal ordering, and numeric aggregation."""
    if not ranked_units:
        return None
    ql = question.strip().casefold()

    m_dur = re.search(
        r"\bhow\s+many\s+(days?|weeks?|months?|years?)\s+(?:have|has|had|were|did)?\s*(?:passed|elapsed|been|between|ago|separated)",
        ql,
    )
    if not m_dur:
        m_dur = re.search(r"\bhow\s+long\s+(?:ago|has\s+it\s+been|had\s+passed|between)\b", ql)
    if m_dur and enable_bitemporal:
        req_unit = m_dur.group(1).casefold() if m_dur.lastindex else "days"
        q_dt = parse_timestamp_date(question_date)

        split_m = re.search(
            r"between\s+(.{6,}?)\s+and\s+(.{6,})",
            question,
            re.I,
        ) or re.search(
            r"since\s+(.{8,}?)\s+(?:when|before|until|to\s+when)\s+(.{8,})",
            question,
            re.I,
        )
        if split_m:
            part_a, part_b = split_m.group(1), split_m.group(2)
            dated_units = [u for _, u in ranked_units if u.observed_ordinal > 0]
            if dated_units:
                best_a = max(dated_units, key=lambda u: _score_unit_against_clause(part_a, u, idf_lookup))
                sc_a = _score_unit_against_clause(part_a, best_a, idf_lookup)
                other_units = [u for u in dated_units if u.source_id != best_a.source_id] or dated_units
                best_b = max(other_units, key=lambda u: _score_unit_against_clause(part_b, u, idf_lookup))
                sc_b = _score_unit_against_clause(part_b, best_b, idf_lookup)
                dt_a = resolve_event_date(best_a.text, best_a.observed_timestamp)
                dt_b = resolve_event_date(best_b.text, best_b.observed_timestamp)
                if dt_a and dt_b and sc_a >= 0.30 and sc_b >= 0.30:
                    delta_days = abs((dt_b - dt_a).days)
                    return {
                        "answer": _format_duration_delta(delta_days, req_unit),
                        "cited_units": (best_a, best_b),
                        "cited_ids": tuple(dict.fromkeys((best_a.citation_id, best_b.citation_id))),
                        "operation": f"date_diff_two_events({dt_a.isoformat()}..{dt_b.isoformat()}={delta_days}d)",
                        "confidence_boost": 0.18,
                    }

        if q_dt is not None:
            dated_ranked = [(sc, u) for sc, u in ranked_units if u.observed_ordinal > 0]
            if dated_ranked:
                top_sc = dated_ranked[0][0]
                strong = [(sc, u) for sc, u in dated_ranked if sc >= max(0.25, 0.80 * top_sc)]
                if "first" in ql or "earliest" in ql:
                    chosen_u = min(strong, key=lambda x: x[1].observed_ordinal)[1]
                elif "last" in ql or "recent" in ql:
                    chosen_u = max(strong, key=lambda x: x[1].observed_ordinal)[1]
                else:
                    chosen_u = dated_ranked[0][1]
                ev_dt = resolve_event_date(chosen_u.text, chosen_u.observed_timestamp)
                if ev_dt is not None:
                    delta_days = abs((q_dt - ev_dt).days)
                    return {
                        "answer": _format_duration_delta(delta_days, req_unit),
                        "cited_units": (chosen_u,),
                        "cited_ids": (chosen_u.citation_id,),
                        "operation": f"date_diff_to_qdate({ev_dt.isoformat()}..{q_dt.isoformat()}={delta_days}d)",
                        "confidence_boost": 0.16,
                    }

    if enable_bitemporal and re.search(r"\b(?:first|earlier|last|later|more recently|most recently|before)\b", ql):
        m_or = re.search(r"(.{4,60}?)\s+or\s+(.{4,60}?)\??$", question, re.I)
        if m_or:
            opt_a = re.sub(
                r"^.*?\b(?:did i|happened|came|was|were|first|earlier|last|later|get|buy|visit|attend|read|finish|start|adopt)\b\s*",
                "",
                m_or.group(1),
                flags=re.I,
            ).strip(" ,?\"'")
            opt_b = m_or.group(2).strip(" ,?\"'")
            dated_units = [u for _, u in ranked_units if u.observed_ordinal > 0]
            if opt_a and opt_b and dated_units:
                ua = max(dated_units, key=lambda u: _score_unit_against_clause(opt_a, u, idf_lookup))
                ub = max(dated_units, key=lambda u: _score_unit_against_clause(opt_b, u, idf_lookup))
                sa = _score_unit_against_clause(opt_a, ua, idf_lookup)
                sb = _score_unit_against_clause(opt_b, ub, idf_lookup)
                da = resolve_event_date(ua.text, ua.observed_timestamp)
                db = resolve_event_date(ub.text, ub.observed_timestamp)
                if da and db and da != db and sa >= 0.30 and sb >= 0.30:
                    wants_first = bool(re.search(r"\b(?:first|earlier|before)\b", ql))
                    win = opt_a if ((da < db) if wants_first else (da > db)) else opt_b
                    return {
                        "answer": win,
                        "cited_units": (ua, ub),
                        "cited_ids": tuple(dict.fromkeys((ua.citation_id, ub.citation_id))),
                        "operation": f"temporal_order({da.isoformat()} vs {db.isoformat()})",
                        "confidence_boost": 0.15,
                    }

    wants_diff = bool(re.search(r"\b(?:how much more|how many more|difference|fewer|less than|more than)\b", ql))
    wants_sum = bool(re.search(r"\b(?:total|altogether|combined|in total|all\s+together|across\s+all)\b", ql))
    if wants_diff or wants_sum:
        top_sc = ranked_units[0][0]
        if top_sc >= 0.28:
            seen_src: set[str] = set()
            nums: list[tuple[float, str, RawEvidenceUnit]] = []
            strong_distinct: list[RawEvidenceUnit] = []
            for sc, u in ranked_units:
                if sc < max(0.24, 0.65 * top_sc):
                    continue
                if u.source_id in seen_src:
                    continue
                seen_src.add(u.source_id)
                strong_distinct.append(u)
                m_q = _QTY_EXTRACT_RE.search(u.text)
                if not m_q:
                    continue
                cur_sym = m_q.group(1) or ""
                raw_val = (m_q.group(2) or m_q.group(3) or "").replace(",", "")
                unit_lbl = (m_q.group(4) or "").strip()
                try:
                    val = float(raw_val)
                except ValueError:
                    continue
                if 1900 <= val <= 2035 and not cur_sym and not unit_lbl:
                    continue
                nums.append((val, cur_sym or unit_lbl, u))
                if len(nums) >= 4:
                    break
            if len(nums) >= 2:
                unit_tag = next((lbl for _, lbl, _ in nums if lbl), "")
                if wants_diff:
                    res = abs(nums[0][0] - nums[1][0])
                    used = nums[:2]
                else:
                    res = sum(v for v, _, _ in nums)
                    used = nums
                res_str = str(int(res)) if abs(res - round(res)) < 1e-6 else f"{res:.2f}"
                if unit_tag in {"$", "€", "£"}:
                    ans = f"{unit_tag}{res_str}"
                elif unit_tag:
                    ans = f"{res_str} {unit_tag}"
                else:
                    ans = res_str
                return {
                    "answer": ans,
                    "cited_units": tuple(u for _, _, u in used),
                    "cited_ids": tuple(dict.fromkeys(u.citation_id for _, _, u in used)),
                    "operation": f"numeric_{'diff' if wants_diff else 'sum'}({[v for v, _, _ in used]})",
                    "confidence_boost": 0.12,
                }
            if re.search(r"\bhow many\s+(?:times|days|sessions)\b", ql) and len(strong_distinct) >= 2:
                return {
                    "answer": str(len(strong_distinct)),
                    "cited_units": tuple(strong_distinct[:4]),
                    "cited_ids": tuple(dict.fromkeys(u.citation_id for u in strong_distinct[:4])),
                    "operation": f"distinct_session_count({len(strong_distinct)})",
                    "confidence_boost": 0.10,
                }
    return None


def _format_duration_delta(delta_days: int, req_unit: str) -> str:
    if req_unit.startswith("month"):
        months = max(0, int(round(delta_days / 30.4375)))
        return f"{months} {'month' if months == 1 else 'months'}"
    if req_unit.startswith("week"):
        weeks = max(0, int(round(delta_days / 7.0)))
        return f"{weeks} {'week' if weeks == 1 else 'weeks'}"
    if req_unit.startswith("year"):
        years = max(0, int(round(delta_days / 365.25)))
        return f"{years} {'year' if years == 1 else 'years'}"
    return f"{delta_days} {'day' if delta_days == 1 else 'days'}"