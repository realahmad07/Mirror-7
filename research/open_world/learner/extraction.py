"""Stage S11.1 Stage 3: Schema-Free Candidate Entity & Candidate Event Discovery.

STRICT ISOLATION & ZERO MANUAL DOMAIN SCHEMAS:
- Pure Python 3.11 stdlib + Mirror 7's pure-CPU 10,240-bit BSC primitives
  (``mirror_crystal.quotient_egraph_resonator``).
- NEVER imports ``evaluator``, ``hidden_eval``, ``oracle``, or ``datasets.loader``.
- ZERO manually supplied domain action lists, domain predicate lists, or gazetteers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Iterable, Sequence

from mirror_crystal.quotient_egraph_resonator import (
    BitSlicedBundle10240,
    UNIFIED_MASK,
    UNIFIED_ODD_MASK,
    hamming_sim10240,
    hv10240_of,
    permute10240,
)
from research.open_world.contracts import STOPWORDS, content_tokens, tokenize
from research.open_world.learner.evidence_ledger import RawEvidenceUnit

# Purely grammatical closed-class English syntax markers (ZERO domain action lists!)
NEGATION_TOKENS: frozenset[str] = frozenset(
    {
        "not", "no", "never", "none", "neither", "nor", "nothing", "nobody",
        "nowhere", "without", "cannot", "cant", "can't", "didn't", "didnt",
        "doesn't", "doesnt", "don't", "dont", "isn't", "isnt", "wasn't", "wasnt",
        "weren't", "werent", "haven't", "havent", "hasn't", "hasnt", "hadn't",
        "won't", "wont", "wouldn't", "couldn't", "shouldn't", "aren't",
        "hardly", "barely", "scarcely", "seldom", "rarely",
    }
)
UNCERTAIN_MODALITY_TOKENS: frozenset[str] = frozenset(
    {
        "might", "may", "maybe", "perhaps", "possibly", "probably", "allegedly",
        "supposedly", "uncertain", "unsure", "unknown", "rumored", "could",
    }
)
HYPOTHETICAL_TOKENS: frozenset[str] = frozenset(
    {"if", "suppose", "assuming", "hypothetical", "would", "unless", "whether"}
)
PREFERENCE_TOKENS: frozenset[str] = frozenset(
    {"prefer", "favorite", "favourite", "recommend", "suggest", "suggestions", "tips", "ideas", "advice"}
)
CLAUSE_PIVOTS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her",
        "its", "our", "their", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
        "can", "could", "may", "might", "must", "in", "on", "at", "by", "for", "with",
        "from", "to", "of", "into", "onto", "over", "under", "between", "through",
        "during", "before", "after", "since", "until", "about", "against", "among",
        "and", "or", "but", "so", "yet", "nor", "because", "although", "though",
        "while", "when", "where", "which", "who", "whom", "whose", "what", "how",
        "if", "than", "as", "there", "here", "then", "also", "still", "very", "last", "next", "yesterday", "today", "tonight", "recently", "currently", "initially", "originally", "previously", "actually",
    }
)
_SENT_START_NON_PROPER: frozenset[str] = STOPWORDS | frozenset(
    {
        "along", "following", "despite", "within", "beyond", "between", "among",
        "throughout", "however", "moreover", "furthermore", "therefore", "meanwhile",
        "instead", "earlier", "later", "recently", "currently", "usually", "often",
        "sometimes", "generally", "typically", "especially", "particularly", "mainly",
        "mostly", "nearly", "almost", "several", "many", "various", "numerous",
        "other", "others", "another", "either", "neither", "whether", "unlike",
        "although", "though", "whereas", "while", "since", "because", "according",
        "around", "across", "behind", "inside", "outside", "towards", "toward",
        "former", "latter", "smaller", "larger", "higher", "lower", "older", "newer",
        "escalating", "expanding", "increasing", "decreasing", "starting", "ending",
    }
)

_FOOTNOTE_RE = re.compile(r"\[\s*(?:citation needed|\d+|[a-z]|note\s*\d*)\s*\]", re.I)
# Unicode-aware word/symbol matcher (handles accented European names, co-NP, 28.5°E, $50)
_TOKEN_SPAN_RE = re.compile(r"[^\W_]+(?:['’.-][^\W_]+)*|[%$€£°]")
_PROPER_SPAN_RE = re.compile(
    r"(?<!\w)(?:(?:Dr|Mr|Mrs|Ms|Prof|St|Mt|Rev|Gen|Sen|Rep)\.\s+)?"
    r"[A-ZÀ-ÖØ-Þ][^\W_]*(?:['’+-][^\W_]+)*"
    r"(?:\s+(?:(?:of|the|and|for|de|del|van|von|al|la|le|in|on|to)\s+)?[A-ZÀ-ÖØ-Þ][^\W_]*(?:['’+-][^\W_]+)*){0,5}(?!\w)"
)
_QUOTED_SPAN_RE = re.compile(r"(?:\"([^\"\n]{1,90})\"|'([^'\n]{2,80})'|“([^”\n]{1,90})”)")
_MONTH_NAMES_RE = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_ORD_CENTURY_RE = (
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
    r"thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty-first|\d{1,2}(?:st|nd|rd|th))"
)
_DATE_SPAN_RE = re.compile(
    rf"\b(?:{_MONTH_NAMES_RE}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:1[0-9]{{3}}|20[0-9]{{2}})"
    rf"|{_MONTH_NAMES_RE}\.?\s+(?:1[0-9]{{3}}|20[0-9]{{2}})"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAMES_RE}\.?,?\s+(?:1[0-9]{{3}}|20[0-9]{{2}})"
    rf"|(?:(?:early|mid|late|into\s+the)\s+)?{_ORD_CENTURY_RE}\s+century"
    rf"|(?:1[0-9]{{3}}|20[0-9]{{2}})(?:[–-](?:1[0-9]{{3}}|20[0-9]{{2}}|\d{{2}}))?"
    rf"|{_MONTH_NAMES_RE}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?)\b",
    re.I,
)
_NUM_WORDS_RE = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|half|quarter|third|fourth|fifth|eighth|tenth|dozen)"
)
_QUANTITY_SPAN_RE = re.compile(
    rf"(?:[$€£]\s*\d+(?:[.,]\d+)?(?:\s*(?:million|billion|thousand|trillion|k|m|b))?"
    rf"|\b\d+(?:[.,]\d+)?\s*(?:°[NSEW]?|%|percent|per\s+cent|million|billion|thousand|trillion|km|kilometres?|kilometers?|miles?|meters?|metres?|feet|ft|inches?|cm|mm|kg|lbs?|pounds?|grams?|oz|hours?|hrs?|minutes?|mins?|seconds?|days?|weeks?|months?|years?|decades?|centuries|people|subjects|times|pages|books|items|maps|dollars|euros)\b"
    rf"|\b(?:about\s+|over\s+|nearly\s+|around\s+|approximately\s+)?(?:{_NUM_WORDS_RE}(?:[- ]{_NUM_WORDS_RE}){{0,2}})\s*(?:percent|%|of|million|billion|thousand|years?|months?|weeks?|days?|hours?|times|the\s+number)\b"
    rf"|\b(?:about\s+|over\s+|nearly\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten)-(?:half|thirds?|fourths?|quarters?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?)\b"
    rf"|\b\d+(?:[.,]\d+)?\b)",
    re.I,
)
_PERCENT_FRAC_RE = re.compile(
    rf"(?:\d+(?:[.,]\d+)?\s*(?:%|percent|per\s+cent)|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)-(?:half|thirds?|fourths?|quarters?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?)\b|\b(?:half|quarter|third|majority|minority|proportion|ratio|fraction)\b)",
    re.I,
)
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:yesterday|today|tonight|last\s+(?:night|week|month|year|weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple\s+of|few)\s+"
    r"(?:days?|weeks?|months?|years?)\s+ago|"
    r"earlier\s+(?:today|this\s+week|this\s+month)|"
    r"recently|currently|now|initially|originally|previously|formerly|used\s+to)\b",
    re.I,
)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:;|(?<=,)\s+(?:but|however|while|whereas|although|though|which|because|since|so|and)\s+)",
    re.I,
)


@lru_cache(maxsize=65536)
def morph_stem(word: str) -> str:
    """Deterministic lightweight domain-agnostic morphological stemmer (pure stdlib)."""
    w = word.casefold().strip("'’")
    if len(w) <= 3:
        return w
    if w.endswith("'s") or w.endswith("’s"):
        w = w[:-2]
        if len(w) <= 3:
            return w
    for suf in (
        "ingly", "edly", "iness", "ation", "ition", "ment", "ness",
        "tion", "sion", "able", "ible", "ious", "eous", "ally", "ical",
    ):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    for suf in ("ing", "ies", "ied", "ers", "est", "ful", "ous", "ive", "ize", "ise"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            if suf in ("ies", "ied"):
                return w[:-3] + "y"
            return w[: -len(suf)]
    for suf in ("ed", "er", "es", "ly", "al"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e") and not w.endswith(("ee", "ie", "ye", "oe")):
        w = w[:-1]
    return w


def stemmed_content_tokens(text: str) -> tuple[str, ...]:
    return tuple(morph_stem(t) for t in content_tokens(text))


def char_trigram_sim(a: str, b: str) -> float:
    """Character-trigram Jaccard similarity for typo-tolerant open-world lexical grounding."""
    sa = a.casefold().strip()
    sb = b.casefold().strip()
    if not sa or not sb:
        return 0.0
    if sa == sb:
        return 1.0
    if len(sa) < 3 or len(sb) < 3:
        return 1.0 if sa == sb else 0.0
    if min(len(sa), len(sb)) / max(len(sa), len(sb)) < 0.55:
        return 0.0
    pa = f"_{sa}_"
    pb = f"_{sb}_"
    ta = {pa[i : i + 3] for i in range(len(pa) - 2)}
    tb = {pb[i : i + 3] for i in range(len(pb) - 2)}
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    return inter / len(ta | tb)


def token_grounded_in_set(
    tok: str,
    stem_set: set[str],
    raw_set: set[str],
    *,
    allow_fuzzy: bool = True,
) -> bool:
    """Check whether ``tok`` is grounded in ``raw_set``/``stem_set`` (exact, stemmed, prefix, or typo-fuzzy)."""
    tl = tok.casefold()
    st = morph_stem(tl)
    if tl in raw_set or st in stem_set:
        return True
    if not allow_fuzzy or len(tl) < 5:
        return False
    p4 = tl[:4]
    for cand in raw_set:
        if len(cand) < 5:
            continue
        if cand[:2] == tl[:2]:
            if char_trigram_sim(tl, cand) >= 0.58:
                return True
            if len(tl) >= 6 and len(cand) >= 6 and cand[:5] == tl[:5]:
                return True
        elif cand[:4] == p4 and abs(len(cand) - len(tl)) <= 3:
            return True
    return False


class StructuralPivotTable:
    """Unsupervised corpus-frequency structural pivot discovery (Stage 3, zero domain lists)."""

    def __init__(self) -> None:
        self._doc_freq: Counter[str] = Counter()
        self._num_docs: int = 0
        self._induced_pivots: frozenset[str] = CLAUSE_PIVOTS

    def observe_texts(self, texts: Iterable[str]) -> None:
        for text in texts:
            toks = set(tokenize(text))
            if not toks:
                continue
            self._num_docs += 1
            for t in toks:
                self._doc_freq[t] += 1
        self._recompute_pivots()

    def _recompute_pivots(self) -> None:
        if self._num_docs < 20:
            self._induced_pivots = CLAUSE_PIVOTS
            return
        thresh = max(5, int(0.22 * self._num_docs))
        high_df = {
            tok for tok, c in self._doc_freq.items()
            if c >= thresh and not tok.isdigit()
        }
        self._induced_pivots = CLAUSE_PIVOTS | frozenset(high_df)

    @property
    def pivots(self) -> frozenset[str]:
        return self._induced_pivots

    def idf(self, token: str) -> float:
        st = morph_stem(token)
        c = self._doc_freq.get(token.casefold(), 0) or self._doc_freq.get(st, 0)
        if self._num_docs <= 0:
            return 1.5
        return math.log(1.0 + (self._num_docs - c + 0.5) / (c + 0.5))


@dataclass(frozen=True)
class CandidateEntity:
    """Open-vocabulary candidate entity or answerable span extracted from raw UTF-8 text (Stage 3)."""

    entity_id: str
    surface: str
    canonical_form: str
    entity_shape: str  # PROPER_SPAN | QUOTED_SPAN | DATE_TIME | QUANTITY_NUM | PERCENT_FRAC | NOUN_CHUNK | CLAUSE_SPAN
    char_start: int
    char_end: int
    evidence_id: str
    source_id: str
    coref_confidence: float = 1.0
    context_hv: int = 0
    left_pivot: str = ""
    right_pivot: str = ""


@dataclass(frozen=True)
class CandidateEvent:
    """Open-vocabulary clause/event frame discovered without manual event schemas (Stage 3)."""

    event_id: str
    trigger_span: str
    arg_spans: tuple[str, ...]
    polarity: str  # AFFIRMED | NEGATED | UNCERTAIN
    modality: str  # FACTUAL | HYPOTHETICAL | PREFERENCE | QUERY
    time_ref: str
    evidence_id: str
    source_id: str
    speaker_role: str
    extraction_conf: float
    clause_text: str


def _encode_context_hv(tokens: Sequence[str], center_idx: int, window: int = 4) -> int:
    """Encode local token context around ``center_idx`` into a 10,240-bit BSC hypervector."""
    atoms: list[tuple[int, int]] = []
    n = len(tokens)
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        j = center_idx + offset
        if 0 <= j < n:
            tok = tokens[j]
            if tok in STOPWORDS or len(tok) <= 1:
                continue
            hv = permute10240(hv10240_of(f"s11:ctx:{morph_stem(tok)}"), offset * 7)
            atoms.append((hv, 2 if abs(offset) <= 2 else 1))
    if not atoms:
        return hv10240_of("s11:ctx:empty")
    if len(atoms) == 1:
        return atoms[0][0] & UNIFIED_MASK
    total_w = sum(w for _, w in atoms)
    bd = BitSlicedBundle10240(bits=max(8, (total_w * 2 + 3).bit_length()))
    for hv, w in atoms:
        bd.add(hv, w * 2)
    if total_w % 2 == 0:
        bd.add(UNIFIED_ODD_MASK, 2)
    bd.add(UNIFIED_MASK, 1)
    return bd.binary()


def _clean_span_text(span: str) -> str:
    cleaned = _FOOTNOTE_RE.sub("", span)
    return cleaned.strip(" \t\r\n.,;:!?()[]{}—–-\"'“”")


def _left_right_pivots(text: str, c_start: int, c_end: int) -> tuple[str, str]:
    left_seg = text[:c_start].strip()
    right_seg = text[c_end:].strip()
    l_toks = tokenize(left_seg)
    r_toks = tokenize(right_seg)
    lp = l_toks[-1] if l_toks else "<BOS>"
    if lp in {"the", "a", "an", "my", "your", "his", "her", "our", "their", "this", "that"} and len(l_toks) >= 2:
        lp = l_toks[-2]
    rp = r_toks[0] if r_toks else "<EOS>"
    return lp, rp


def extract_candidate_spans(
    unit: RawEvidenceUnit,
    *,
    pivots: frozenset[str] = CLAUSE_PIVOTS,
    compute_hv: bool = False,
) -> tuple[CandidateEntity, ...]:
    """Discover all schema-free candidate entity and answer spans from one ``RawEvidenceUnit``."""
    raw_text = unit.text
    if not raw_text:
        return ()
    # Replace footnote markers with equal-length spaces to preserve char coordinates cleanly
    text = _FOOTNOTE_RE.sub(lambda m: " " * len(m.group(0)), raw_text)

    raw_matches: list[tuple[int, int, str, str]] = []
    seen_spans: set[tuple[str, str]] = set()

    def _add(cs: int, ce: int, surf: str, shape: str) -> None:
        cleaned = _clean_span_text(surf)
        if not cleaned or len(cleaned) > 120:
            return
        if cleaned.casefold() in {"citation needed", "citation"}:
            return
        key = (cleaned.casefold(), shape)
        if key in seen_spans:
            return
        seen_spans.add(key)
        raw_matches.append((cs, ce, cleaned, shape))

    for m in _QUOTED_SPAN_RE.finditer(text):
        grp = m.group(1) or m.group(2) or m.group(3) or ""
        if grp.strip():
            _add(m.start(), m.end(), grp, "QUOTED_SPAN")

    for m in _DATE_SPAN_RE.finditer(text):
        _add(m.start(), m.end(), m.group(0), "DATE_TIME")

    for m in _PERCENT_FRAC_RE.finditer(text):
        _add(m.start(), m.end(), m.group(0), "PERCENT_FRAC")

    for m in _QUANTITY_SPAN_RE.finditer(text):
        _add(m.start(), m.end(), m.group(0), "QUANTITY_NUM")

    for m in _PROPER_SPAN_RE.finditer(text):
        surf = m.group(0).strip()
        toks = surf.split()
        if m.start() <= 2 and len(toks) == 1 and toks[0].casefold() in _SENT_START_NON_PROPER:
            continue
        if len(toks) >= 2 and toks[0].casefold() in _SENT_START_NON_PROPER:
            trimmed = " ".join(toks[1:])
            if trimmed and trimmed[0].isupper():
                off = surf.find(trimmed)
                _add(m.start() + off, m.end(), trimmed, "PROPER_SPAN")
                continue
        _add(m.start(), m.end(), surf, "PROPER_SPAN")

    tok_matches = list(_TOKEN_SPAN_RE.finditer(text))
    n_toks = len(tok_matches)
    tok_strs = [m.group(0).casefold() for m in tok_matches]

    i = 0
    while i < n_toks:
        if tok_strs[i] in pivots or tok_strs[i] in STOPWORDS:
            i += 1
            continue
        j = i
        while j < n_toks and (j - i) < 7:
            if j > i:
                gap = text[tok_matches[j - 1].end() : tok_matches[j].start()]
                if any(ch in gap for ch in ",.;:!?()[]"):
                    break
            t = tok_strs[j]
            if t in pivots and t not in {"of", "and", "in", "for", "to"}:
                break
            j += 1
        while j > i and tok_strs[j - 1] in {"of", "and", "in", "for", "to", "the", "a", "an"}:
            j -= 1
        if j > i:
            cs = tok_matches[i].start()
            ce = tok_matches[j - 1].end()
            chunk = text[cs:ce]
            _add(cs, ce, chunk, "NOUN_CHUNK")
            if j - i >= 3:
                for sub_len in (2, 3):
                    if sub_len >= (j - i):
                        continue
                    for s_i in range(i, j - sub_len + 1):
                        if tok_strs[s_i] in STOPWORDS or tok_strs[s_i + sub_len - 1] in STOPWORDS:
                            continue
                        sc_s = tok_matches[s_i].start()
                        sc_e = tok_matches[s_i + sub_len - 1].end()
                        _add(sc_s, sc_e, text[sc_s:sc_e], "NOUN_CHUNK")
        i = max(i + 1, j)

    for m_prep in re.finditer(
        r"\b(?:to|by|through|via|because of|due to|called|named|known as|means?|mean|as|into|in|under|on|at|instituted|introducing)\s+(?:a|an|the)?\s*([^.,;:!?()\[\]]{2,75})",
        text,
        re.I,
    ):
        phrase = _clean_span_text(m_prep.group(1))
        if phrase and 1 <= len(phrase.split()) <= 10:
            cs = m_prep.start(1)
            ce = cs + len(phrase)
            shape = "CLAUSE_SPAN" if len(phrase.split()) >= 5 else "NOUN_CHUNK"
            _add(cs, ce, phrase, shape)

    entities: list[CandidateEntity] = []
    for idx, (cs, ce, surf, shape) in enumerate(raw_matches):
        lp, rp = _left_right_pivots(text, cs, ce)
        hv = 0
        if compute_hv and tok_matches:
            center_idx = 0
            for t_i, tm in enumerate(tok_matches):
                if tm.start() >= cs:
                    center_idx = t_i
                    break
            hv = _encode_context_hv(tok_strs, center_idx, window=4)
        canon = " ".join(morph_stem(w) for w in tokenize(surf) if w not in {"a", "an", "the"}) or surf.casefold()
        entities.append(
            CandidateEntity(
                entity_id=f"{unit.evidence_id}:ent{idx}",
                surface=surf,
                canonical_form=canon,
                entity_shape=shape,
                char_start=cs,
                char_end=ce,
                evidence_id=unit.evidence_id,
                source_id=unit.source_id,
                coref_confidence=1.0,
                context_hv=hv,
                left_pivot=lp,
                right_pivot=rp,
            )
        )
    return tuple(entities)


def cluster_coreferent_entities(
    entities: Sequence[CandidateEntity],
) -> tuple[dict[str, str], dict[str, float]]:
    """Unsupervised entity-cluster resolution preserving explicit ambiguity/uncertainty scores."""
    ent_to_cluster: dict[str, str] = {}
    ent_to_conf: dict[str, float] = {}
    canon_to_cluster: dict[str, str] = {}
    cluster_reps: list[tuple[str, str, set[str], int]] = []
    next_cid = 0

    for ent in entities:
        canon = ent.canonical_form
        if canon in canon_to_cluster:
            ent_to_cluster[ent.entity_id] = canon_to_cluster[canon]
            ent_to_conf[ent.entity_id] = 1.0
            continue

        words = set(content_tokens(ent.surface))
        matched_cid: str | None = None
        matched_conf = 1.0

        if ent.entity_shape in {"PROPER_SPAN", "NOUN_CHUNK", "QUOTED_SPAN"} and words:
            acro = "".join(w[0] for w in ent.surface.split() if w and w[0].isalpha()).casefold()
            for cid, rep_canon, rep_words, rep_hv in cluster_reps:
                if not rep_words:
                    continue
                if words < rep_words or rep_words < words:
                    shorter = min(len(words), len(rep_words))
                    longer = max(len(words), len(rep_words))
                    if shorter >= 1 and longer <= 4:
                        matched_cid = cid
                        matched_conf = 0.85 if shorter == 1 else 0.92
                        break
                if len(acro) >= 2 and acro == rep_canon:
                    matched_cid = cid
                    matched_conf = 0.88
                    break
                if ent.context_hv and rep_hv and (words & rep_words):
                    sim = hamming_sim10240(ent.context_hv, rep_hv) / 10240.0
                    if sim >= 0.20:
                        matched_cid = cid
                        matched_conf = round(min(0.82, 0.55 + sim), 4)
                        break

        if matched_cid is None:
            matched_cid = f"cluster_{next_cid}"
            next_cid += 1
            canon_to_cluster[canon] = matched_cid
            cluster_reps.append((matched_cid, canon, words, ent.context_hv))
        ent_to_cluster[ent.entity_id] = matched_cid
        ent_to_conf[ent.entity_id] = matched_conf

    return ent_to_cluster, ent_to_conf


def detect_polarity_and_modality(text: str) -> tuple[str, str]:
    """Detect clause polarity (AFFIRMED/NEGATED/UNCERTAIN) and modality (FACTUAL/HYPOTHETICAL/PREFERENCE/QUERY)."""
    toks = set(tokenize(text))
    tl = text.casefold()
    if (toks & NEGATION_TOKENS) or "no longer" in tl or "not any" in tl or "n't" in tl:
        polarity = "NEGATED"
    elif toks & UNCERTAIN_MODALITY_TOKENS:
        polarity = "UNCERTAIN"
    else:
        polarity = "AFFIRMED"

    if text.strip().endswith("?"):
        modality = "QUERY"
    elif toks & PREFERENCE_TOKENS:
        modality = "PREFERENCE"
    elif toks & HYPOTHETICAL_TOKENS:
        modality = "HYPOTHETICAL"
    else:
        modality = "FACTUAL"
    return polarity, modality


def extract_candidate_events(
    unit: RawEvidenceUnit,
    *,
    pivots: frozenset[str] = CLAUSE_PIVOTS,
) -> tuple[CandidateEvent, ...]:
    """Discover open-vocabulary ``CandidateEvent`` frames from raw UTF-8 clauses (zero manual schemas)."""
    text = unit.text.strip()
    if not text:
        return ()
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
    if not clauses:
        clauses = [text]

    events: list[CandidateEvent] = []
    for c_idx, clause in enumerate(clauses):
        polarity, modality = detect_polarity_and_modality(clause)
        t_match = _RELATIVE_TIME_RE.search(clause)
        if t_match:
            time_ref = t_match.group(0).strip()
        else:
            d_match = _DATE_SPAN_RE.search(clause)
            time_ref = d_match.group(0).strip() if d_match else ""

        words = clause.split()
        content_positions = [
            idx for idx, w in enumerate(words)
            if re.sub(r"[^A-Za-z0-9]", "", w).casefold() not in pivots
        ]
        if not content_positions:
            trigger = clause[:48]
            args: tuple[str, ...] = ()
            conf = 0.45
        elif len(content_positions) == 1:
            trigger = _clean_span_text(words[content_positions[0]])
            args = ()
            conf = 0.65
        else:
            p0 = content_positions[0]
            p_last = content_positions[-1]
            if len(content_positions) == 2:
                arg0 = _clean_span_text(" ".join(words[: p0 + 1]))
                trigger = _clean_span_text(" ".join(words[p0 + 1 :])) or _clean_span_text(words[p_last])
                args = (arg0,)
            else:
                mid = content_positions[1]
                arg0 = _clean_span_text(" ".join(words[:mid]))
                trigger = _clean_span_text(words[mid])
                arg1 = _clean_span_text(" ".join(words[mid + 1 :]))
                args = tuple(a for a in (arg0, arg1) if a)
            conf = 0.80 if polarity == "AFFIRMED" else 0.72

        events.append(
            CandidateEvent(
                event_id=f"{unit.evidence_id}:ev{c_idx}",
                trigger_span=trigger,
                arg_spans=args,
                polarity=polarity,
                modality=modality,
                time_ref=time_ref,
                evidence_id=unit.evidence_id,
                source_id=unit.source_id,
                speaker_role=unit.speaker_role,
                extraction_conf=conf,
                clause_text=clause,
            )
        )
    return tuple(events)