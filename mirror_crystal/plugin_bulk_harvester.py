"""Verifiable Science Plugins & Technical Standards Bulk Concept Harvester for Mirror 7 Phase-2.

Pure Python standard-library harvester (0 neural parameters, 0 GPU, 0 external pip deps) that
distills verified, deduplicated, quality-gated 5-facet ConceptNodes into compact JSONL staging
from official open scientific & technical APIs aligned with Antigravity Science Plugins:

  1. EMBL-EBI Ontology Lookup Service v2 (OLS4 REST API: go, doid, hp, chebi, uberon, cl)
  2. ChEMBL REST API (Phase-4 approved drugs, mechanisms of action, indications, human targets)
  3. PubChem PUG REST API (foundational chemical compounds, biomolecules, physicochemical properties)
  4. UniProtKB REST API (reviewed human Swiss-Prot proteins with curated function/pathway/disease)
  5. Reactome Content Service API (curated human signaling & metabolic biological pathways)
  6. InterPro REST API (curated protein domains, families, superfamilies, and active sites)
  7. openFDA Drug Labeling API (FDA prescription drug pharmacology, indications, safety warnings)
  8. ClinicalTrials.gov APIv2 (Phase 2/3/4 completed clinical therapeutic interventions)
  9. arXiv Export API (18+ STEM, CS, AI, Physics, Math, Astrophysics, Genomics & Economics fields)

Provenance & Licensing Attributions:
  - EMBL-EBI OLS4, ChEMBL, InterPro: https://www.ebi.ac.uk/about/terms-of-use/ (CC BY / CC0 / OBO)
  - UniProtKB: https://www.uniprot.org/help/license (CC BY 4.0)
  - Reactome: https://reactome.org/license (CC0 1.0 Public Domain)
  - PubChem / NCBI / ClinicalTrials.gov: US NLM Public Domain
  - openFDA: https://open.fda.gov/license/ (CC0 1.0 Public Domain)
  - arXiv: https://info.arxiv.org/help/api/tou.html (arXiv / CC metadata)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
import hashlib
import html
import json
from pathlib import Path
import random
import re
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


_USER_AGENT = "Mirror7PluginConceptHarvester/2.0 (https://github.com/realahmad07/Mirror-7; pure-stdlib)"

# Mirror 7's 20 canonical domains (exact match with open_ended_concept_lattice / bulk_concept_harvester)
VALID_MIRROR_DOMAINS: frozenset[str] = frozenset({
    "computer_science",
    "artificial_intelligence",
    "software_engineering",
    "cybersecurity_networking",
    "mathematics",
    "physics",
    "chemistry",
    "biology_genomics",
    "medicine_pharmacology",
    "engineering",
    "economics_finance",
    "business_management",
    "law_governance",
    "psychology_cognitive",
    "philosophy_ethics",
    "history_civilization",
    "earth_climate_science",
    "astronomy_space",
    "linguistics_literature",
    "everyday_life_productivity",
})

# Generic singletons never allowed to be attached as secondary aliases of another concept
_FORBIDDEN_SECONDARY_ALIASES: frozenset[str] = frozenset({
    "cell", "gene", "protein", "drug", "acid", "base", "salt", "water", "cancer",
    "disease", "syndrome", "disorder", "pathway", "domain", "family", "repeat",
    "site", "enzyme", "receptor", "kinase", "channel", "factor", "complex",
    "signal", "signaling", "process", "function", "activity", "system", "model",
    "method", "network", "learning", "analysis", "therapy", "treatment", "vaccine",
    "blood", "brain", "heart", "liver", "kidney", "lung", "skin", "bone", "muscle",
    "nerve", "tissue", "organ", "body", "head", "hand", "foot", "eye", "ear",
    "all", "root", "other", "none", "null", "unknown", "test", "study", "trial",
    "group", "class", "type", "form", "state", "level", "rate", "time", "mass",
    "energy", "force", "power", "light", "heat", "wave", "field", "space", "line",
})

# Raw accession / non-human-readable title rejection patterns
_RAW_ACCESSION_RE = re.compile(
    r"^(?:"
    r"chembl\d+|go:\d+|doid:\d+|hp:\d+|chebi:\d+|uberon:\d+|cl:\d+|ipr\d+|pf\d+|"
    r"nct\d+|cid\s*\d+|r-hsa-\d+|upf\d+|duf\d+|loc\d+|fam\d+[a-z0-9]*|"
    r"tmem\d+[a-z0-9]*|kiaa\d+|c\d+orf\d+|znf\d+|or\d+[a-z]\d+|"
    r"[opq][0-9][a-z0-9]{3}[0-9]|[a-nr-z][0-9](?:[a-z][a-z0-9]{2}[0-9]){1,2}|"
    r"[a-z]{1,4}[-_]?\d{3,}[a-z0-9\-]*"
    r")$",
    re.IGNORECASE,
)

# Obsolete / stub / placeholder rejection cues
_REJECT_TITLE_CUES: tuple[str, ...] = (
    "obsolete",
    "deprecated",
    "uncharacterized",
    "uncharacterised",
    "putative ",
    "probable ",
    "unknown function",
    "hypothetical protein",
    "root of all",
    "disambiguation",
    "list of ",
    "index of ",
    "outline of ",
    "protein of unknown",
    "domain of unknown",
    "predicted protein",
    "open reading frame",
    "pseudogene",
    "fragment",
)

_REJECT_DEF_CUES: tuple[str, ...] = (
    "obsolete.",
    "this term was made obsolete",
    "this term is obsolete",
    "note that this term can be used in place of the obsolete",
    "in place of the obsolete",
    "may refer to",
    "can refer to",
    "root of all terms",
    "the (-)-enantiomer of",
    "the (+)-enantiomer of",
    "the (−)-enantiomer of",
    "an enantiomer of",
    "a stereoisomer of",
    "the conjugate base of",
    "the conjugate acid of",
    "a tautomer of",
    "an organic cation obtained by",
    "an organic anion obtained by",
    "any member of the class",
    "a salt obtained by",
    "a hydrate of",
    "this entry represents uncharacterised",
    "this entry represents uncharacterized",
    "function is unknown",
    "function of this protein is unknown",
)

# Salt suffixes stripped when deduplicating ChEMBL approved drug salts
_DRUG_SALT_SUFFIXES: tuple[str, ...] = (
    " hydrochloride", " dihydrochloride", " sodium", " disodium", " potassium",
    " calcium", " sulfate", " mesylate", " maleate", " tartrate", " phosphate",
    " acetate", " citrate", " fumarate", " bromide", " chloride", " nitrate",
    " besylate", " succinate", " propionate", " benzoate", " lactate", " gluconate",
    " monohydrate", " dihydrate", " trihydrate", " anhydrous", " tromethamine",
    " meglumine", " tosylate", " oxalate", " valerate", " dipropionate",
    " furoate", " pamoate", " palmitate", " decanoate", " enanthate", " cypionate",
)

# OBO relation phrases to humanize inside OLS4 definitions
_OBO_RELATIONS: tuple[tuple[str, str], ...] = (
    ("has_material_basis_in", "has a material basis in"),
    ("derives_from", "derives from"),
    ("located_in", "located in"),
    ("transmitted_by", "transmitted by"),
    ("has_symptom", "is accompanied by the symptom"),
    ("results_in", "results in"),
    ("part_of", "part of"),
    ("develops_from", "develops from"),
    ("capable_of", "capable of"),
    ("is_a", "is a"),
    ("occurs_in", "occurs in"),
    ("regulated_by", "regulated by"),
)


@dataclass(frozen=True)
class PluginConceptRecord:
    concept_id: str
    title: str
    category: str
    aliases: tuple[str, ...]
    definition: str
    mechanism: str
    applications: str
    tradeoffs: str
    provenance: str
    quality_score: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "title": self.title,
            "category": self.category,
            "aliases": list(self.aliases),
            "definition": self.definition,
            "mechanism": self.mechanism,
            "applications": self.applications,
            "tradeoffs": self.tradeoffs,
            "provenance": self.provenance,
            "quality_score": round(self.quality_score, 3),
        }


@dataclass
class PluginHarvestReport:
    total_emitted: int
    raw_candidates_collected: int
    rejected_quality: int
    rejected_duplicate: int
    source_counts: dict[str, int]
    domain_counts: dict[str, int]
    avg_quality_score: float
    min_quality_score: float
    elapsed_seconds: float
    output_path: str
    sample_records: list[dict[str, Any]]


def canonical_concept_key(name: str) -> str:
    """Identical canonical plural/hyphen/punctuation key used by Mirror 7's Concept Lattice."""
    clean = re.sub(r"\([^)]*\)", "", name or "").strip().casefold()
    clean = re.sub(r"[^a-z0-9]+", " ", clean).strip()
    if not clean:
        return ""
    tokens = clean.split()
    norm_tokens: list[str] = []
    for tok in tokens:
        if (
            len(tok) >= 5
            and tok.endswith("s")
            and not tok.endswith(("ss", "us", "is", "ics", "sis", "ness", "ous"))
        ):
            if tok.endswith("ies") and len(tok) >= 6:
                tok = tok[:-3] + "y"
            elif tok.endswith(("ches", "shes", "xes", "zes")):
                tok = tok[:-2]
            else:
                tok = tok[:-1]
        norm_tokens.append(tok)
    return " ".join(norm_tokens)


def content_fingerprint(definition: str) -> str:
    """Identical 64-bit normalized definition hash used by Mirror 7's Concept Lattice."""
    norm = re.sub(r"[^a-z0-9]+", " ", (definition or "").casefold()).strip()
    return hashlib.sha256(norm[:220].encode("utf-8")).hexdigest()[:16]


_ABBREV_PROTECT_RE = re.compile(
    r"\b(et\s+al|e\.g|i\.e|vs|dr|prof|fig|figs|no|vol|sp|spp|approx|ca|cf|ref|refs)\.",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return []
    # Protect common scientific abbreviations before splitting on sentence boundaries
    sentinel = "\x00"
    protected = _ABBREV_PROTECT_RE.sub(lambda m: m.group(1) + sentinel, clean)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", protected)
    return [p.replace(sentinel, ".").strip() for p in parts if len(p.strip()) >= 15]


def _clean_text(raw: str) -> str:
    """Strip HTML tags, UniProt/OBO/InterPro citation tags, LaTeX macros, and normalize whitespace."""
    if not raw:
        return ""
    txt = html.unescape(raw)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\[cite:[^\]]+\]", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\[ec:([0-9.\-]+)\]", r"(EC \1)", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\(PubMed:\d+(?:,\s*PubMed:\d+)*\)", "", txt)
    txt = re.sub(r"\((?:By similarity|Probable|Potential|Microbial infection)\)", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\((?:reviewed in|inferred from)[^)]{0,160}\)", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\{ECO:[^}]+\}", "", txt)
    txt = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", txt)
    txt = re.sub(r"\\(?:emph|textbf|textit|mathrm|text)\{([^}]*)\}", r"\1", txt)
    txt = re.sub(r"\$([^$]{1,40})\$", r"\1", txt)
    for obo_rel, human_rel in _OBO_RELATIONS:
        if obo_rel in txt:
            txt = txt.replace(obo_rel, human_rel)
    # Humanize any remaining lowercase OBO ontology relation predicates (e.g. has_allergic_trigger)
    txt = re.sub(
        r"\b([a-z]{2,}(?:_[a-z]{2,})+)\b",
        lambda m: m.group(1).replace("_", " "),
        txt,
    )
    txt = re.sub(r"\s+([.,;:])", r"\1", txt)
    txt = re.sub(r"\(\s*[;,]?\s*\)", "", txt)
    txt = re.sub(r"\[\s*[;,]?\s*\]", "", txt)
    txt = re.sub(r"\s+([.,;:])", r"\1", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


_TRAILING_DANGLING_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "and", "or", "but", "nor", "that", "which", "who", "whom", "whose", "as",
    "into", "onto", "upon", "via", "per", "than", "between", "among", "during",
    "within",
})


def _truncate_facet(text: str, limit: int = 220) -> str:
    """Ensure every facet stays strictly <= limit chars and ends cleanly without dangling clauses."""
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if not s:
        return ""
    needs_cut = len(s) > limit or (len(s) == limit and s[-1] not in ".!?")
    if needs_cut:
        cut = s[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-([ ")
        if not cut:
            cut = s[: limit - 1].rstrip(",;:-([ ")
        # Trim dangling trailing articles/prepositions left by word-boundary truncation
        words = cut.split(" ")
        while len(words) > 4 and words[-1].rstrip(",;:-").casefold() in _TRAILING_DANGLING_WORDS:
            words.pop()
        s = " ".join(words).rstrip(",;:-([ ")
    # Remove trailing stub lead-in phrases severed during truncation
    s = re.sub(r"\s+(?:Occurs(?:\s+within)?|Localized(?:\s+to)?|Administered(?:\s+via)?|Route)\.?$", "", s, flags=re.IGNORECASE).rstrip(" ,;:-")
    # Remove any dangling unclosed parenthetical fragment left by truncation
    if s.count("(") > s.count(")"):
        last_open = s.rfind("(")
        if last_open >= 35:
            s = s[:last_open].rstrip(" ,;:-")
    if s and s[-1] not in ".!?":
        if len(s) >= limit:
            s = s[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-([ ")
        s += "."
    return s


def _smart_title(raw: str) -> str:
    """Format human-readable scientific title while preserving acronyms & chemical casing."""
    s = re.sub(r"\s+", " ", (raw or "")).strip()
    if not s:
        return ""
    # If all-caps drug/compound name (e.g. "NAPROXEN SODIUM"), title-case cleanly
    letters = [c for c in s if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(letters) >= 5:
        words = s.split(" ")
        out: list[str] = []
        for idx, w in enumerate(words):
            lw = w.lower()
            if idx > 0 and lw in ("of", "in", "on", "for", "and", "or", "to", "by", "with", "via", "from", "the", "a", "an"):
                out.append(lw)
            elif len(w) <= 3 and w.isalpha() and w in ("DNA", "RNA", "ATP", "GTP", "NAD", "HIV", "HCV", "HBV", "HPV", "TNF", "IL", "IgG", "IgA", "IgE", "IgM", "PD", "CAR", "FDA", "MRI", "PET", "CT", "EEG"):
                out.append(w)
            else:
                out.append(w.capitalize())
        return " ".join(out)
    # If all-lowercase ontology label (e.g. "mitochondrion inheritance"), capitalize first char
    if s[0].islower():
        return s[0].upper() + s[1:]
    return s


def _is_ugly_chemical_name(name: str) -> bool:
    """Reject cryptic IUPAC systematic strings with heavy stereochemistry/locant clutter."""
    s = (name or "").strip()
    if len(s) > 52 or len(s) < 3:
        return True
    if "[" in s or "]" in s or "{" in s or "}" in s:
        return True
    if s.count("(") >= 2 or s.count(",") >= 2 or s.count("-") >= 4:
        return True
    digits = sum(c.isdigit() for c in s)
    if digits >= 4:
        return True
    if re.match(r"^(?:\([+\-RSZE0-9,]+\)|[+\-±]|cis-|trans-|alpha-|beta-|gamma-|delta-|\d+[,'\-])", s, re.IGNORECASE):
        return True
    return False


def _http_request_bytes(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 25.0,
    retries: int = 3,
    backoff: float = 1.2,
) -> tuple[bytes, dict[str, str]]:
    """Polite pure-stdlib HTTP GET/POST with gzip decompression and retry backoff on 429/5xx."""
    req_headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json, application/xml, text/xml;q=0.9, */*;q=0.5",
        "Accept-Encoding": "gzip",
    }
    if headers:
        req_headers.update(headers)

    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                # Handle single- or double-gzipped API responses (e.g. UniProt CDN + origin gzip)
                decompress_guard = 0
                while raw[:2] == b"\x1f\x8b" and decompress_guard < 3:
                    raw = gzip.decompress(raw)
                    decompress_guard += 1
                return raw, resp_headers
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                sleep_s = float(retry_after) if (retry_after and retry_after.isdigit()) else backoff * (2 ** attempt)
                time.sleep(min(sleep_s, 12.0))
                continue
            raise
        except Exception as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"HTTP request failed for {url}")


def _http_get_json(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 25.0,
    retries: int = 3,
) -> tuple[Any, dict[str, str]]:
    merged_headers = {"Accept": "application/json"}
    if headers:
        merged_headers.update(headers)
    raw, resp_headers = _http_request_bytes(url, data=data, headers=merged_headers, timeout=timeout, retries=retries)
    return json.loads(raw.decode("utf-8", errors="replace")), resp_headers


def _http_get_text(url: str, *, timeout: float = 25.0, retries: int = 3) -> str:
    raw, _ = _http_request_bytes(url, timeout=timeout, retries=retries)
    return raw.decode("utf-8", errors="replace")


def _extract_ols_str_list(val: Any) -> list[str]:
    """Normalize OLS4 v2 values which may be plain strings, lists of strings, or reification dicts."""
    out: list[str] = []
    if val is None:
        return out
    items = val if isinstance(val, list) else [val]
    for item in items:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            v = item.get("value")
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    return out


def _score_record(
    title: str,
    definition: str,
    mechanism: str,
    applications: str,
    tradeoffs: str,
    aliases: tuple[str, ...],
    provenance: str,
) -> float:
    """Compute a deterministic quality score in [0.0, 1.0] emphasizing factual distinctness & clarity."""
    score = 0.45
    # 1. Definition substance
    if 65 <= len(definition) <= 220:
        score += 0.16
    elif len(definition) >= 50:
        score += 0.10

    # 2. Distinctness across the 4 text facets (penalize repetitive boilerplate)
    def _toks(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]{4,}", s.casefold()) if w not in ("this", "that", "with", "from", "into", "their", "these", "which", "such")}

    d_toks = _toks(definition)
    m_toks = _toks(mechanism)
    a_toks = _toks(applications)
    t_toks = _toks(tradeoffs)

    if len(m_toks) >= 5 and len(m_toks - d_toks) >= 3:
        score += 0.11
    if len(a_toks) >= 5 and len(a_toks - d_toks) >= 3:
        score += 0.08
    if len(t_toks) >= 5 and len(t_toks - d_toks) >= 3:
        score += 0.08

    # 3. Authoritative curated source bonus
    p = provenance.casefold()
    if p.startswith(("openfda:", "uniprot:", "reactome:", "chembl:", "pubchem:")):
        score += 0.07
    elif p.startswith(("ols:", "interpro:", "clinicaltrials:")):
        score += 0.05
    elif p.startswith("arxiv:"):
        score += 0.03

    # 4. Clean title & synonym richness
    if 4 <= len(title) <= 64 and not re.search(r"[\[\]{}|<>\\]", title):
        score += 0.03
    if len(aliases) >= 2:
        score += 0.02

    return round(min(1.0, max(0.0, score)), 3)


def _build_validated_record(
    *,
    concept_id: str,
    title: str,
    category: str,
    aliases: list[str] | tuple[str, ...],
    definition: str,
    mechanism: str,
    applications: str,
    tradeoffs: str,
    provenance: str,
    min_quality: float = 0.65,
) -> PluginConceptRecord | None:
    """Strict quality gate applied to EVERY candidate before admission."""
    cat = (category or "").strip()
    if cat not in VALID_MIRROR_DOMAINS:
        return None

    clean_title = _smart_title(_clean_text(title))
    if not clean_title or len(clean_title) < 3 or len(clean_title) > 84:
        return None

    base_cid = re.sub(r"\s*\([^)]*\)", "", concept_id or clean_title).strip()
    cid = re.sub(r"\s+", " ", base_cid).casefold().strip(" .,:;/-")
    if len(cid) < 3 or len(cid) > 78:
        return None

    # Reject raw accession codes, numeric-only labels, and obsolete/stub titles
    if _RAW_ACCESSION_RE.match(cid) or _RAW_ACCESSION_RE.match(clean_title.strip()):
        return None
    alpha_count = sum(c.isalpha() for c in cid)
    if alpha_count < 3 or alpha_count / max(1, len(cid)) < 0.45:
        return None

    t_low = clean_title.casefold()
    if any(cue in t_low for cue in _REJECT_TITLE_CUES):
        return None

    defn = _truncate_facet(_clean_text(definition), 220)
    mech = _truncate_facet(_clean_text(mechanism), 220)
    apps = _truncate_facet(_clean_text(applications), 220)
    trd = _truncate_facet(_clean_text(tradeoffs), 220)

    if len(defn) < 50 or len(mech) < 32 or len(apps) < 30 or len(trd) < 30:
        return None
    d_low = defn.casefold()
    if any(cue in d_low for cue in _REJECT_DEF_CUES):
        return None
    if defn.casefold() == mech.casefold():
        return None

    # Clean, non-generic alias list
    alias_out: list[str] = [cid]
    t_alias = clean_title.casefold()
    if t_alias != cid and 3 <= len(t_alias) <= 84 and not _RAW_ACCESSION_RE.match(t_alias):
        alias_out.append(t_alias)
    for al in aliases:
        a_clean = re.sub(r"\s+", " ", _clean_text(str(al))).casefold().strip(" .,:;/-")
        if (
            3 <= len(a_clean) <= 68
            and a_clean not in alias_out
            and a_clean not in _FORBIDDEN_SECONDARY_ALIASES
            and not _RAW_ACCESSION_RE.match(a_clean)
            and not _is_ugly_chemical_name(a_clean)
        ):
            alias_out.append(a_clean)
            if len(alias_out) >= 6:
                break

    dedup_aliases = tuple(dict.fromkeys(alias_out))
    qscore = _score_record(clean_title, defn, mech, apps, trd, dedup_aliases, provenance)
    if qscore < min_quality:
        return None

    return PluginConceptRecord(
        concept_id=cid,
        title=clean_title,
        category=cat,
        aliases=dedup_aliases,
        definition=defn,
        mechanism=mech,
        applications=apps,
        tradeoffs=trd,
        provenance=provenance.strip(),
        quality_score=qscore,
    )


# ---------------------------------------------------------------------------
# Worker 1: EMBL-EBI OLS4 v2 Ontologies (GO, DOID, HP, ChEBI, UBERON, CL)
# ---------------------------------------------------------------------------

_OLS4_ONTOLOGY_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "ont": "doid",
        "domain": "medicine_pharmacology",
        "target": 1150,
        "max_pages": 25,
        "page_size": 200,
        "app_tpl": "Used in clinical diagnosis, epidemiological disease classification (Disease Ontology {curie}), and biomedical phenotype-genotype mapping.",
        "trd_tpl": "Clinical presentation varies by patient genetics, comorbidities, and disease stage, requiring differential diagnostic confirmation.",
        "mech_prefix": "Pathological classification in Human Disease Ontology ({curie}):",
    },
    {
        "ont": "cl",
        "domain": "biology_genomics",
        "target": 550,
        "max_pages": 16,
        "page_size": 200,
        "app_tpl": "Used in single-cell RNA-seq cell-type annotation, histology, tissue atlases, and Cell Ontology ({curie}) lineage mapping.",
        "trd_tpl": "Cell identity can exhibit transcriptional plasticity and transitional differentiation states across tissues and microenvironments.",
        "mech_prefix": "Cellular lineage and marker profile in Cell Ontology ({curie}):",
    },
    {
        "ont": "uberon",
        "domain": "biology_genomics",
        "target": 700,
        "max_pages": 20,
        "page_size": 200,
        "app_tpl": "Applied in comparative vertebrate anatomy, developmental biology, spatial transcriptomics, and UBERON ({curie}) cross-species tissue mapping.",
        "trd_tpl": "Anatomical boundaries and developmental homology vary across species, ontogenetic stages, and physiological conditions.",
        "mech_prefix": "Anatomical organization and structural relationships in UBERON ({curie}):",
    },
    {
        "ont": "go",
        "domain": "biology_genomics",
        "target": 1500,
        "max_pages": 32,
        "page_size": 200,
        "app_tpl": "Used in Gene Ontology ({curie}) functional annotation, gene-set enrichment analysis (GSEA), and systems-biology pathway modeling.",
        "trd_tpl": "Functional activity depends on cellular compartment, expression context, post-translational state, and species-specific orthology.",
        "mech_prefix": "Gene Ontology ({curie}) functional mechanism:",
    },
    {
        "ont": "hp",
        "domain": "medicine_pharmacology",
        "target": 750,
        "max_pages": 22,
        "page_size": 200,
        "app_tpl": "Applied in rare-disease genomic diagnostics, clinical phenotyping, electronic health record standardization, and HPO ({curie}) prioritization.",
        "trd_tpl": "Phenotypic expressivity and penetrance vary substantially across individuals and may overlap across distinct genetic syndromes.",
        "mech_prefix": "Clinical phenotype specification in Human Phenotype Ontology ({curie}):",
    },
    {
        "ont": "chebi",
        "domain": "chemistry",
        "target": 750,
        "max_pages": 35,
        "page_size": 200,
        "app_tpl": "Used in cheminformatics, metabolic reaction modeling, biochemical pathway annotation, and ChEBI ({curie}) chemical classification.",
        "trd_tpl": "Chemical reactivity, protonation state, solubility, and biological activity vary with pH, solvent environment, and stereochemistry.",
        "mech_prefix": "Molecular structure and biochemical role in ChEBI ({curie}):",
    },
)


_CHEBI_SEARCH_TOPICS: tuple[str, ...] = (
    "carboxylic acid", "amino acid", "vitamin", "alkaloid", "alcohol",
    "carbohydrate", "monosaccharide", "steroid", "neurotransmitter", "nucleotide",
    "fatty acid", "oxide", "sulfate", "chloride", "carbonate",
    "nitrate", "phosphate", "ester", "ketone", "aldehyde",
    "amine", "phenol", "terpene", "solvent", "catalyst",
    "polymer", "pigment", "hormone", "coenzyme", "flavonoid",
    "antibiotic", "hydrocarbon", "organosulfur", "organophosphorus",
)


def _fetch_ols4_page(ont: str, page: int, size: int) -> list[dict[str, Any]]:
    url = (
        f"https://www.ebi.ac.uk/ols4/api/v2/ontologies/{ont}/classes"
        f"?size={size}&page={page}&isDefiningOntology=true"
    )
    data, _ = _http_get_json(url, timeout=25.0, retries=3)
    return data.get("elements") or []


def _fetch_ols4_search(ont: str, query: str, size: int = 60) -> list[dict[str, Any]]:
    q = urllib.parse.quote(query)
    url = (
        f"https://www.ebi.ac.uk/ols4/api/v2/ontologies/{ont}/classes"
        f"?search={q}&size={size}&isDefiningOntology=true"
    )
    data, _ = _http_get_json(url, timeout=25.0, retries=3)
    return data.get("elements") or []


def harvest_ols4_ontologies(*, max_workers: int = 8) -> dict[str, list[PluginConceptRecord]]:
    """Worker 1: Harvest curated biological, medical, chemical & anatomical terms from EMBL-EBI OLS4 v2."""
    by_ont: dict[str, list[PluginConceptRecord]] = {}

    for cfg in _OLS4_ONTOLOGY_CONFIGS:
        ont: str = cfg["ont"]
        domain: str = cfg["domain"]
        target: int = cfg["target"]
        max_pages: int = cfg["max_pages"]
        page_size: int = cfg["page_size"]

        records: list[PluginConceptRecord] = []
        reg_count = 0
        abn_count = 0
        seen_curies: set[str] = set()

        def _process_elements(els: list[dict[str, Any]]) -> None:
            nonlocal reg_count, abn_count
            for el in els:
                if len(records) >= target:
                    break
                if el.get("isObsolete") is True or el.get("isDefiningOntology") is False:
                    continue

                curie = str(el.get("curie") or el.get("shortForm") or "").strip()
                if not curie or curie in seen_curies:
                    continue

                labels = _extract_ols_str_list(el.get("label"))
                if not labels:
                    continue
                label = labels[0]
                l_low = label.casefold()

                # Ontology-specific high-signal quality filters
                if ont == "chebi" and _is_ugly_chemical_name(label):
                    continue
                if ont == "go":
                    if len(label) > 64:
                        continue
                    if l_low.startswith(("positive regulation of ", "negative regulation of ", "regulation of ")):
                        if reg_count >= 120:
                            continue
                        reg_count += 1
                if ont == "hp":
                    if l_low.startswith("abnormality of "):
                        if abn_count >= 140:
                            continue
                        abn_count += 1

                defs = _extract_ols_str_list(el.get("definition"))
                if not defs:
                    continue
                # Prefer actual definitional sentences over editorial 'Note that...' comments
                chosen_def = next(
                    (d for d in defs if not d.strip().casefold().startswith("note that")),
                    defs[0],
                )
                raw_def = _clean_text(chosen_def)
                if len(raw_def) < 50:
                    continue

                syns = _extract_ols_str_list(el.get("synonym"))
                if not syns:
                    syns = _extract_ols_str_list(
                        el.get("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym")
                    )

                go_ns_list = _extract_ols_str_list(
                    el.get("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace")
                ) if ont == "go" else []
                go_ns = go_ns_list[0].replace("_", " ") if go_ns_list else ""

                sents = _split_sentences(raw_def)
                defn = sents[0] if sents else raw_def
                if len(defn) < 50 and len(sents) >= 2:
                    defn = f"{sents[0]} {sents[1]}"

                if len(sents) >= 2 and sents[1] not in defn:
                    mech = sents[1]
                elif ont == "go" and go_ns:
                    syn_note = f" Also referenced as {', '.join(syns[:2])}." if syns else ""
                    mech = f"Classified under Gene Ontology {go_ns} ({curie}), characterizing the cellular or molecular execution of {label}.{syn_note}"
                else:
                    syn_note = f" Recognized synonyms include {', '.join(syns[:2])}." if syns else ""
                    mech = f"{cfg['mech_prefix'].format(curie=curie)} defines the properties and structured relations of {label}.{syn_note}"

                apps = cfg["app_tpl"].format(curie=curie)
                trd = sents[2] if len(sents) >= 3 else cfg["trd_tpl"]

                rec_domain = domain
                if ont == "uberon" and any(w in raw_def.casefold() for w in ("clinical", "pathology", "surgical", "human organ")):
                    rec_domain = "medicine_pharmacology"

                rec = _build_validated_record(
                    concept_id=label,
                    title=label,
                    category=rec_domain,
                    aliases=syns[:4],
                    definition=defn,
                    mechanism=mech,
                    applications=apps,
                    tradeoffs=trd,
                    provenance=f"ols:{ont}:{curie}",
                )
                if rec is not None:
                    seen_curies.add(curie)
                    records.append(rec)

        if ont == "chebi":
            # Use topic-guided search across ChEBI so we harvest iconic human-readable chemical concepts
            # instead of lexicographically sorted 6-digit natural-product stereoisomers.
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(_fetch_ols4_search, "chebi", q, 70): q for q in _CHEBI_SEARCH_TOPICS}
                for fut in as_completed(futs):
                    try:
                        _process_elements(fut.result())
                    except Exception:
                        continue
            by_ont["ols_chebi"] = records
            continue

        # Fetch pages in parallel batches so we stop immediately once target is reached
        batch_size = max_workers
        for start_page in range(0, max_pages, batch_size):
            if len(records) >= target:
                break
            pages = list(range(start_page, min(max_pages, start_page + batch_size)))
            page_results: dict[int, list[dict[str, Any]]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(_fetch_ols4_page, ont, p, page_size): p for p in pages}
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        page_results[p] = fut.result()
                    except Exception:
                        page_results[p] = []

            empty_pages = 0
            for p in pages:
                els = page_results.get(p, [])
                if not els:
                    empty_pages += 1
                    continue
                _process_elements(els)
            if empty_pages == len(pages):
                break

        by_ont[f"ols_{ont}"] = records
    return by_ont


# ---------------------------------------------------------------------------
# Worker 2: ChEMBL Approved Drugs & Human Pharmacological Targets
# ---------------------------------------------------------------------------

def _strip_drug_salt(name: str) -> str:
    low = name.strip().casefold()
    for suf in _DRUG_SALT_SUFFIXES:
        if low.endswith(suf) and len(low) - len(suf) >= 4:
            return name[: len(name) - len(suf)].strip()
    return name.strip()


def harvest_chembl_drugs_and_targets(*, max_drugs: int = 1350, max_targets: int = 450) -> dict[str, list[PluginConceptRecord]]:
    """Worker 2: Harvest Phase-4 approved drugs and their verified human protein targets from ChEMBL."""
    # Step 1: Parallel fetch of mechanisms, indications, and approved molecules
    mechs_by_mol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mechs_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inds_by_mol: dict[str, list[str]] = defaultdict(list)
    molecules: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    def _fetch_paginated(endpoint: str, key: str, params: str, max_pages: int, limit: int = 1000) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for page in range(max_pages):
            offset = page * limit
            url = f"https://www.ebi.ac.uk/chembl/api/data/{endpoint}.json?{params}&limit={limit}&offset={offset}"
            try:
                data, _ = _http_get_json(url, timeout=30.0, retries=3)
            except Exception:
                break
            chunk = data.get(key) or []
            if not chunk:
                break
            rows.extend(chunk)
            if not (data.get("page_meta") or {}).get("next"):
                break
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_mech = pool.submit(_fetch_paginated, "mechanism", "mechanisms", "max_phase__gte=3", 8, 1000)
        f_ind = pool.submit(_fetch_paginated, "drug_indication", "drug_indications", "max_phase_for_ind=4", 8, 1000)
        f_mol = pool.submit(_fetch_paginated, "molecule", "molecules", "max_phase=4", 5, 1000)
        f_tgt = pool.submit(_fetch_paginated, "target", "targets", "organism=Homo%20sapiens&target_type=SINGLE%20PROTEIN", 5, 1000)

        mech_rows = f_mech.result()
        ind_rows = f_ind.result()
        molecules = f_mol.result()
        targets = f_tgt.result()

    for m in mech_rows:
        mcid = m.get("molecule_chembl_id")
        pmcid = m.get("parent_molecule_chembl_id")
        tcid = m.get("target_chembl_id")
        moa = (m.get("mechanism_of_action") or "").strip()
        if not moa:
            continue
        if mcid:
            mechs_by_mol[mcid].append(m)
        if pmcid and pmcid != mcid:
            mechs_by_mol[pmcid].append(m)
        if tcid:
            mechs_by_target[tcid].append(m)

    for ind in ind_rows:
        term = (ind.get("efo_term") or ind.get("mesh_heading") or "").strip()
        if not term or len(term) < 3:
            continue
        for key_name in ("molecule_chembl_id", "parent_molecule_chembl_id"):
            mcid = ind.get(key_name)
            if mcid and term not in inds_by_mol[mcid]:
                inds_by_mol[mcid].append(term)

    # Prefer non-salt parent molecules first
    molecules.sort(
        key=lambda x: (
            0 if (x.get("molecule_hierarchy") or {}).get("parent_chembl_id") == x.get("molecule_chembl_id") else 1,
            0 if x.get("pref_name") and _strip_drug_salt(x["pref_name"]) == x["pref_name"] else 1,
            x.get("first_approval") or 9999,
        )
    )

    drug_records: list[PluginConceptRecord] = []
    seen_base_drugs: set[str] = set()
    mol_name_by_id: dict[str, str] = {}

    for mol in molecules:
        mcid = str(mol.get("molecule_chembl_id") or "").strip()
        raw_name = str(mol.get("pref_name") or "").strip()
        if not mcid or not raw_name or len(raw_name) < 3 or len(raw_name) > 54:
            continue
        if _RAW_ACCESSION_RE.match(raw_name) or _is_ugly_chemical_name(raw_name):
            continue

        base_name = _strip_drug_salt(raw_name)
        display_name = _smart_title(base_name)
        mol_name_by_id[mcid] = display_name

        base_key = canonical_concept_key(display_name)
        if not base_key or base_key in seen_base_drugs:
            continue

        parent_id = (mol.get("molecule_hierarchy") or {}).get("parent_chembl_id") or mcid
        m_list = mechs_by_mol.get(mcid) or mechs_by_mol.get(parent_id) or []
        i_list = inds_by_mol.get(mcid) or inds_by_mol.get(parent_id) or []

        # Strict quality gate: require verified mechanism OR approved clinical indication
        if not m_list and not i_list:
            continue

        mol_type = str(mol.get("molecule_type") or "therapeutic agent").lower()
        first_appr = mol.get("first_approval")
        props = mol.get("molecule_properties") or {}
        formula = props.get("full_molformula")
        mwt = props.get("full_mwt")
        alogp = props.get("alogp")
        ro5 = props.get("num_ro5_violations")
        bbw = mol.get("black_box_warning")

        moa_texts = list(dict.fromkeys((x.get("mechanism_of_action") or "").strip() for x in m_list if x.get("mechanism_of_action")))
        action_types = list(dict.fromkeys((x.get("action_type") or "").strip().lower() for x in m_list if x.get("action_type")))

        routes: list[str] = []
        if mol.get("oral"):
            routes.append("oral")
        if mol.get("parenteral"):
            routes.append("parenteral")
        if mol.get("topical"):
            routes.append("topical")

        # Build crisp 5-facet drug profile
        appr_str = f"first approved in {first_appr}" if first_appr else "clinically approved (Phase 4)"
        chem_str = f" ({formula}, MW {mwt} Da)" if formula and mwt else (f" ({formula})" if formula else "")
        if moa_texts and i_list:
            defn = f"{display_name} is an approved {mol_type} drug{chem_str}, {appr_str}, acting as a {moa_texts[0]} for {', '.join(i_list[:2])}."
        elif moa_texts:
            defn = f"{display_name} is an approved {mol_type} pharmacological agent{chem_str}, {appr_str}, classified mechanistically as a {moa_texts[0]}."
        else:
            defn = f"{display_name} is an approved {mol_type} therapeutic agent{chem_str}, {appr_str}, clinically indicated for {', '.join(i_list[:3])}."

        if moa_texts:
            act_note = f" ({'/'.join(action_types[:2])} action)" if action_types else ""
            prop_note = f" Physicochemical profile: ALogP {alogp}." if alogp is not None else ""
            mech = f"Exerts therapeutic activity via {'; '.join(moa_texts[:2])}{act_note}.{prop_note}"
        else:
            prop_note = f"Formula {formula}, MW {mwt} Da, ALogP {alogp}." if (formula and mwt and alogp is not None) else f"Molecular class: {mol_type}."
            mech = f"Modulates disease physiology in {', '.join(i_list[:2])} through targeted pharmacological action. {prop_note}"

        route_note = f" Administered via {'/'.join(routes)} routes." if routes else ""
        if i_list:
            apps = f"Clinically approved for treating {', '.join(i_list[:4])}.{route_note}"
        else:
            apps = f"Used in clinical pharmacotherapy targeting {moa_texts[0]}.{route_note}"

        safety_bits: list[str] = []
        if bbw == 1:
            safety_bits.append("Carries an FDA boxed safety warning requiring strict clinical risk monitoring")
        if isinstance(ro5, int) and ro5 > 0:
            safety_bits.append(f"exhibits {ro5} Lipinski Rule-of-5 violation(s) affecting oral bioavailability")
        if not safety_bits:
            safety_bits.append("Requires dose titration, drug-interaction screening, and monitoring for off-target adverse effects")
        trd = "; ".join(safety_bits).capitalize() + "."

        syns: list[str] = []
        for s_obj in (mol.get("molecule_synonyms") or [])[:6]:
            sn = (s_obj.get("molecule_synonym") or "").strip()
            if sn and sn.casefold() != display_name.casefold() and not _is_ugly_chemical_name(sn):
                syns.append(sn)

        rec = _build_validated_record(
            concept_id=display_name,
            title=display_name,
            category="medicine_pharmacology",
            aliases=syns[:4],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"chembl:{mcid}",
        )
        if rec is not None:
            seen_base_drugs.add(base_key)
            drug_records.append(rec)
            if len(drug_records) >= max_drugs:
                break

    # Step 2: Human Pharmacological Protein Targets that have verified clinical drug mechanisms
    target_records: list[PluginConceptRecord] = []
    for tgt in targets:
        tcid = str(tgt.get("target_chembl_id") or "").strip()
        pref_name = str(tgt.get("pref_name") or "").strip()
        if not tcid or not pref_name or tcid not in mechs_by_target:
            continue
        if len(pref_name) < 4 or len(pref_name) > 72:
            continue

        t_mechs = mechs_by_target[tcid]
        moas = list(dict.fromkeys((x.get("mechanism_of_action") or "").strip() for x in t_mechs if x.get("mechanism_of_action")))
        drugs_hitting = list(
            dict.fromkeys(
                mol_name_by_id[x["molecule_chembl_id"]]
                for x in t_mechs
                if x.get("molecule_chembl_id") in mol_name_by_id
            )
        )
        if not moas:
            continue

        comps = tgt.get("target_components") or []
        gene_syns: list[str] = []
        for comp in comps[:2]:
            for syn in (comp.get("target_component_synonyms") or []):
                if syn.get("syn_type") == "GENE_SYMBOL":
                    gs = (syn.get("component_synonym") or "").strip()
                    if 3 <= len(gs) <= 16:
                        gene_syns.append(gs)

        drug_str = f" including {', '.join(drugs_hitting[:3])}" if drugs_hitting else ""
        defn = f"{pref_name} is a validated human pharmacological single-protein target ({tcid}) modulated clinically via {moas[0]}."
        mech = f"Serves as a molecular binding target for therapeutic agents{drug_str} through {'; '.join(moas[:2])}."
        apps = f"Targeted in structure-based drug discovery, medicinal chemistry optimization, and clinical pharmacology ({len(t_mechs)} curated ChEMBL mechanisms)."
        trd = "Drug modulation of human protein targets risks on-target physiological toxicity, isoform cross-reactivity, and acquired resistance mutations."

        rec = _build_validated_record(
            concept_id=pref_name,
            title=pref_name,
            category="medicine_pharmacology",
            aliases=gene_syns[:3],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"chembl:target:{tcid}",
        )
        if rec is not None:
            target_records.append(rec)
            if len(target_records) >= max_targets:
                break

    return {
        "chembl_drugs": drug_records,
        "chembl_targets": target_records,
    }


# ---------------------------------------------------------------------------
# Worker 3: PubChem PUG REST API (Foundational Compounds & Biomolecules)
# ---------------------------------------------------------------------------

# Curated canonical PubChem compound names (amino acids, biomolecules, acids/bases, solvents,
# neurotransmitters, hormones, vitamins, industrial reagents, minerals, organic building blocks)
_PUBCHEM_CANONICAL_NAMES: tuple[str, ...] = (
    # Core organic & inorganic acids/bases/reagents
    "acetic acid", "formic acid", "citric acid", "oxalic acid", "tartaric acid", "lactic acid",
    "malic acid", "succinic acid", "fumaric acid", "pyruvic acid", "benzoic acid", "salicylic acid",
    "uric acid", "phosphoric acid", "sulfuric acid", "nitric acid", "hydrochloric acid", "boric acid",
    "carbonic acid", "hydrofluoric acid", "perchloric acid", "chromate", "ammonia", "hydrazine",
    "sodium hydroxide", "potassium hydroxide", "calcium carbonate", "sodium bicarbonate",
    "sodium chloride", "potassium chloride", "magnesium sulfate", "ammonium nitrate",
    "hydrogen peroxide", "ozone", "carbon monoxide", "carbon dioxide", "sulfur dioxide",
    "nitrogen dioxide", "nitric oxide", "nitrous oxide", "hydrogen sulfide", "phosphine", "silane",
    # Classic hydrocarbons, solvents & organic chemistry
    "methane", "ethane", "propane", "butane", "ethylene", "acetylene", "benzene", "toluene",
    "xylene", "naphthalene", "anthracene", "styrene", "cyclohexane", "methanol", "ethanol",
    "isopropanol", "butanol", "ethylene glycol", "glycerol", "phenol", "catechol", "resorcinol",
    "hydroquinone", "formaldehyde", "acetaldehyde", "benzaldehyde", "acetone", "butanone",
    "diethyl ether", "tetrahydrofuran", "dioxane", "chloroform", "dichloromethane", "carbon tetrachloride",
    "acetonitrile", "dimethylformamide", "dimethyl sulfoxide", "pyridine", "aniline", "urea",
    "thiourea", "guanidine", "indole", "pyrrole", "furan", "thiophene", "imidazole", "purine", "pyrimidine",
    # Amino acids & peptides
    "glycine", "alanine", "valine", "leucine", "isoleucine", "proline", "phenylalanine", "tyrosine",
    "tryptophan", "serine", "threonine", "cysteine", "methionine", "asparagine", "glutamine",
    "aspartic acid", "glutamic acid", "lysine", "arginine", "histidine", "ornithine", "citrulline",
    "taurine", "creatine", "glutathione", "carnosine", "homocysteine", "selenocysteine",
    # Sugars, lipids & energy metabolites
    "glucose", "fructose", "galactose", "mannose", "ribose", "deoxyribose", "sucrose", "lactose",
    "maltose", "trehalose", "sorbitol", "mannitol", "xylitol", "inositol", "palmitic acid",
    "stearic acid", "oleic acid", "linoleic acid", "arachidonic acid", "butyric acid", "cholesterol",
    "ergosterol", "phosphatidylcholine", "sphingosine", "ceramide", "adenosine triphosphate",
    "adenosine diphosphate", "cyclic amp", "nicotinamide adenine dinucleotide", "flavin adenine dinucleotide",
    "coenzyme a", "acetyl-coa", "s-adenosylmethionine", "ubiquinone", "bilirubin", "heme", "chlorophyll a",
    # Nucleobases & nucleosides
    "adenine", "guanine", "cytosine", "thymine", "uracil", "adenosine", "guanosine", "cytidine",
    "uridine", "thymidine", "inosine", "xanthine", "hypoxanthine",
    # Vitamins & cofactors
    "retinol", "thiamine", "riboflavin", "niacin", "pantothenic acid", "pyridoxine", "biotin",
    "folic acid", "cyanocobalamin", "ascorbic acid", "cholecalciferol", "alpha-tocopherol",
    "phylloquinone", "lipoic acid", "choline", "beta-carotene", "lycopene", "lutein",
    # Neurotransmitters, hormones & signaling molecules
    "dopamine", "serotonin", "epinephrine", "norepinephrine", "acetylcholine", "histamine",
    "gamma-aminobutyric acid", "melatonin", "cortisol", "aldosterone", "testosterone", "estradiol",
    "progesterone", "thyroxine", "triiodothyronine", "prostaglandin e2", "leukotriene b4", "anandamide",
    # Natural products, alkaloids, polyphenols & materials
    "caffeine", "theobromine", "theophylline", "nicotine", "quinine", "strychnine", "atropine",
    "scopolamine", "colchicine", "capsaicin", "piperine", "curcumin", "resveratrol", "quercetin",
    "epigallocatechin gallate", "menthol", "camphor", "limonene", "pinene", "eugenol", "vanillin",
    "cinnamaldehyde", "coumarin", "salicin", "berberine", "artemisinin", "taxol",
    "bisphenol a", "phthalic anhydride", "caprolactam", "acrylonitrile", "vinyl chloride",
    "tetrafluoroethylene", "melamine", "epoxypropane", "silicon dioxide", "titanium dioxide",
    "aluminum oxide", "iron oxide", "calcium oxide", "lithium carbonate", "borax", "potassium permanganate",
)


def harvest_pubchem_compounds(*, max_records: int = 240, max_workers: int = 2) -> list[PluginConceptRecord]:
    """Worker 3: Respectfully paced PubChem PUG REST harvester for canonical chemicals & biomolecules."""
    # Step 1: Resolve curated compound names to verified PubChem CIDs in polite threads
    def _resolve_name(name: str) -> int | None:
        time.sleep(0.18)
        enc = urllib.parse.quote(name, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/cids/JSON"
        try:
            data, _ = _http_get_json(url, timeout=15.0, retries=2)
            cids = (data.get("IdentifierList") or {}).get("CID") or []
            return int(cids[0]) if cids else None
        except Exception:
            return None

    valid_cids: list[int] = []
    seen_cids: set[int] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_resolve_name, nm) for nm in _PUBCHEM_CANONICAL_NAMES]
        for fut in as_completed(futs):
            try:
                cid_val = fut.result()
                if cid_val and cid_val not in seen_cids:
                    seen_cids.add(cid_val)
                    valid_cids.append(cid_val)
            except Exception:
                continue

    batches = [valid_cids[i : i + 25] for i in range(0, len(valid_cids), 25)]

    def _fetch_cid_batch(cid_chunk: list[int]) -> list[PluginConceptRecord]:
        time.sleep(0.25)
        cid_csv = ",".join(str(c) for c in cid_chunk)
        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_csv}/property/Title,MolecularFormula,MolecularWeight,XLogP,TPSA/JSON"
        desc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_csv}/description/JSON"
        try:
            prop_data, _ = _http_get_json(prop_url, timeout=20.0, retries=3)
            time.sleep(0.25)
            desc_data, _ = _http_get_json(desc_url, timeout=20.0, retries=3)
        except Exception:
            return []

        props_by_cid: dict[int, dict[str, Any]] = {}
        for row in (prop_data.get("PropertyTable") or {}).get("Properties") or []:
            if "CID" in row:
                props_by_cid[int(row["CID"])] = row

        # Collect best authoritative description per CID (prefer ChEBI / NCI Thesaurus / DrugBank / MeSH)
        descs_by_cid: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        src_rank = {"ChEBI": 0, "NCI Thesaurus (NCIt)": 1, "DrugBank": 2, "MeSH": 3, "CAMEO Chemicals": 4, "HMDB": 5}
        for info in (desc_data.get("InformationList") or {}).get("Information") or []:
            dtxt = (info.get("Description") or "").strip()
            if not dtxt or len(dtxt) < 55:
                continue
            cid_val = int(info.get("CID", 0))
            sname = str(info.get("DescriptionSourceName") or "PubChem")
            descs_by_cid[cid_val].append((src_rank.get(sname, 9), sname, dtxt))

        out: list[PluginConceptRecord] = []
        for cid_val in cid_chunk:
            prop = props_by_cid.get(cid_val)
            dlist = descs_by_cid.get(cid_val)
            if not prop or not dlist:
                continue
            title = str(prop.get("Title") or "").strip()
            if not title or _is_ugly_chemical_name(title):
                continue

            dlist.sort(key=lambda x: (x[0], -len(x[2])))
            _, src_name, raw_desc = dlist[0]
            cleaned = _clean_text(raw_desc)
            sents = _split_sentences(cleaned)
            if not sents:
                continue

            formula = prop.get("MolecularFormula") or ""
            mwt = prop.get("MolecularWeight") or ""
            xlogp = prop.get("XLogP")
            tpsa = prop.get("TPSA")

            defn = sents[0]
            if len(defn) < 55 and len(sents) >= 2:
                defn = f"{sents[0]} {sents[1]}"

            prop_parts = [p for p in (
                f"formula {formula}" if formula else "",
                f"MW {mwt} g/mol" if mwt else "",
                f"XLogP {xlogp}" if xlogp is not None else "",
                f"TPSA {tpsa} A^2" if tpsa is not None else "",
            ) if p]
            prop_summary = ", ".join(prop_parts)

            if len(sents) >= 2 and sents[1] not in defn:
                mech = f"{sents[1]} ({prop_summary})." if prop_summary else sents[1]
            else:
                mech = f"Characterized in PubChem (CID {cid_val}) with {prop_summary}, governing its chemical reactivity and molecular interactions."

            if len(sents) >= 3:
                apps = sents[2]
            else:
                apps = f"Referenced across organic/biochemical synthesis, analytical chemistry, metabolomics, and toxicology ({src_name} curation)."

            trd = "Chemical stability, ionization, solubility, and biological toxicity depend strongly on concentration, pH, solvent, and exposure route."

            cat = "chemistry"
            low_blob = f"{title} {cleaned}".casefold()
            if any(w in low_blob for w in ("drug", "medication", "antibiotic", "analgesic", "therapeutic", "antidepressant", "anesthetic")):
                cat = "medicine_pharmacology"
            elif any(w in low_blob for w in ("amino acid", "nucleotide", "neurotransmitter", "hormone", "coenzyme", "metabolite", "vitamin")):
                cat = "biology_genomics"

            rec = _build_validated_record(
                concept_id=title,
                title=title,
                category=cat,
                aliases=[],
                definition=defn,
                mechanism=mech,
                applications=apps,
                tradeoffs=trd,
                provenance=f"pubchem:CID:{cid_val}",
            )
            if rec is not None:
                out.append(rec)
        return out

    records: list[PluginConceptRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_cid_batch, b) for b in batches]
        for fut in as_completed(futs):
            try:
                records.extend(fut.result())
            except Exception:
                continue
    records.sort(key=lambda r: (-r.quality_score, r.concept_id))
    return records[:max_records]


# ---------------------------------------------------------------------------
# Worker 4: UniProtKB Reviewed Human Swiss-Prot Proteins (Annotation Score 5)
# ---------------------------------------------------------------------------

def harvest_uniprot_human_proteins(*, max_records: int = 1750) -> list[PluginConceptRecord]:
    """Worker 4: Harvest gold-standard reviewed human Swiss-Prot proteins with curated functional summaries."""
    base_params = {
        "query": "reviewed:true AND organism_id:9606 AND cc_function:* AND annotation_score:5",
        "format": "json",
        "size": "500",
        "fields": "accession,id,protein_name,gene_names,cc_function,cc_subcellular_location,cc_disease,cc_pathway,cc_catalytic_activity,cc_caution",
    }
    next_url: str | None = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(base_params)

    records: list[PluginConceptRecord] = []
    znf_count = 0
    pages_fetched = 0

    while next_url and len(records) < max_records and pages_fetched < 8:
        pages_fetched += 1
        try:
            data, hdrs = _http_get_json(next_url, timeout=30.0, retries=3)
        except Exception:
            break

        for entry in data.get("results") or []:
            if len(records) >= max_records:
                break
            acc = str(entry.get("primaryAccession") or "").strip()
            pdesc = entry.get("proteinDescription") or {}
            rec_name_obj = pdesc.get("recommendedName") or {}
            full_name = ((rec_name_obj.get("fullName") or {}).get("value") or "").strip()
            if not acc or not full_name or len(full_name) < 4 or len(full_name) > 74:
                continue

            fn_low = full_name.casefold()
            if any(
                fn_low.startswith(p)
                for p in (
                    "uncharacterized", "putative", "probable", "inactive", "transmembrane protein ",
                    "coiled-coil domain-containing", "leucine-rich repeat-containing",
                    "wd repeat-containing", "ankyrin repeat domain-containing",
                )
            ):
                continue
            if fn_low.startswith("zinc finger protein "):
                if znf_count >= 15:
                    continue
                znf_count += 1

            short_names = [
                (sn.get("value") or "").strip()
                for sn in (rec_name_obj.get("shortNames") or [])
                if (sn.get("value") or "").strip()
            ]
            gene_names: list[str] = []
            for g in (entry.get("genes") or []):
                gn = ((g.get("geneName") or {}).get("value") or "").strip()
                if gn:
                    gene_names.append(gn)
            primary_gene = gene_names[0] if gene_names else ""

            func_texts: list[str] = []
            cat_texts: list[str] = []
            loc_texts: list[str] = []
            path_texts: list[str] = []
            dis_texts: list[str] = []
            caut_texts: list[str] = []

            for cm in (entry.get("comments") or []):
                ctype = cm.get("commentType")
                if ctype == "FUNCTION":
                    for t in (cm.get("texts") or []):
                        if t.get("value"):
                            func_texts.append(t["value"])
                elif ctype == "CATALYTIC ACTIVITY":
                    rxn = (cm.get("reaction") or {}).get("name")
                    if rxn:
                        cat_texts.append(rxn)
                elif ctype == "SUBCELLULAR LOCATION":
                    for sl in (cm.get("subcellularLocations") or []):
                        loc_v = (sl.get("location") or {}).get("value")
                        if loc_v:
                            loc_texts.append(loc_v)
                elif ctype == "PATHWAY":
                    for t in (cm.get("texts") or []):
                        if t.get("value"):
                            path_texts.append(t["value"])
                elif ctype == "DISEASE":
                    dis_obj = cm.get("disease") or {}
                    dname = dis_obj.get("diseaseId") or ""
                    ddesc = dis_obj.get("description") or ""
                    if dname:
                        dis_texts.append(f"{dname}: {ddesc}" if ddesc else dname)
                elif ctype == "CAUTION":
                    for t in (cm.get("texts") or []):
                        if t.get("value"):
                            caut_texts.append(t["value"])

            if not func_texts:
                continue
            func_clean = _clean_text(" ".join(func_texts))
            if len(func_clean) < 60:
                continue
            func_sents = _split_sentences(func_clean)
            if not func_sents:
                continue

            gene_tag = f" ({primary_gene})" if primary_gene else ""
            title = f"{full_name}{gene_tag}" if len(full_name) + len(gene_tag) <= 78 else full_name

            defn = func_sents[0]
            if len(defn) < 55 and len(func_sents) >= 2:
                defn = f"{func_sents[0]} {func_sents[1]}"

            # Mechanism: catalytic reaction or 2nd sentence of function + subcellular localization
            mech_parts: list[str] = []
            if cat_texts:
                mech_parts.append(f"Catalyzes {cat_texts[0]}")
            elif len(func_sents) >= 2 and func_sents[1] not in defn:
                mech_parts.append(func_sents[1].rstrip("."))
            if loc_texts:
                mech_parts.append(f"Localized to {', '.join(list(dict.fromkeys(loc_texts))[:3])}")
            if not mech_parts:
                mech_parts.append(f"Encoded by human gene {primary_gene or acc} (Swiss-Prot reviewed) mediating specialized molecular signaling or enzymatic control")
            mech = ". ".join(mech_parts) + "."

            # Applications: curated pathway or biomedical proteomic context
            if path_texts:
                apps = f"Participates in {_clean_text(path_texts[0])}; studied in human proteomics, functional genomics, and biomarker discovery."
            else:
                apps = f"Investigated in human molecular biology, functional genomics ({primary_gene or acc}), structural proteomics, and therapeutic target profiling."

            # Tradeoffs: curated human genetic disease involvement or caution notes!
            if dis_texts:
                trd = f"Pathogenic mutations or dysregulation are linked to {_clean_text(dis_texts[0])}"
            elif caut_texts:
                trd = _clean_text(caut_texts[0])
            else:
                trd = "Protein activity depends on isoform splicing, post-translational modification, expression tissue, and multiprotein complex stoichiometry."

            cat = "medicine_pharmacology" if dis_texts else "biology_genomics"

            aliases: list[str] = []
            if primary_gene and len(primary_gene) >= 3:
                aliases.append(f"{primary_gene} protein")
                if len(primary_gene) >= 4:
                    aliases.append(primary_gene)
            for sn in short_names[:2]:
                if len(sn) >= 3:
                    aliases.append(sn)

            rec = _build_validated_record(
                concept_id=full_name,
                title=title,
                category=cat,
                aliases=aliases,
                definition=defn,
                mechanism=mech,
                applications=apps,
                tradeoffs=trd,
                provenance=f"uniprot:{acc}",
            )
            if rec is not None:
                records.append(rec)

        # Follow UniProt cursor Link header: <https://...>; rel="next"
        link_hdr = hdrs.get("link", "")
        m_link = re.search(r'<([^>]+)>;\s*rel="next"', link_hdr)
        next_url = m_link.group(1) if m_link else None

    return records


# ---------------------------------------------------------------------------
# Worker 5: Reactome Human Biological Pathways
# ---------------------------------------------------------------------------

def harvest_reactome_pathways(*, max_records: int = 1400, max_workers: int = 8) -> list[PluginConceptRecord]:
    """Worker 5: Harvest curated human signaling & metabolic pathways from Reactome Content Service."""
    try:
        hierarchy, _ = _http_get_json("https://reactome.org/ContentService/data/eventsHierarchy/9606", timeout=25.0)
    except Exception:
        return []

    pathway_meta: list[tuple[str, str]] = []  # (stId, top_level_name)
    seen_stids: set[str] = set()

    def _walk(nodes: list[dict[str, Any]], top_name: str) -> None:
        for n in nodes:
            ntype = n.get("type")
            stid = str(n.get("stId") or "").strip()
            name = str(n.get("name") or "").strip()
            cur_top = name if ntype == "TopLevelPathway" else top_name
            if ntype in ("TopLevelPathway", "Pathway") and stid and stid not in seen_stids:
                seen_stids.add(stid)
                pathway_meta.append((stid, cur_top))
            children = n.get("children") or []
            if children:
                _walk(children, cur_top)

    _walk(hierarchy, "Human Biology")
    top_map = dict(pathway_meta)
    all_stids = [s for s, _ in pathway_meta]

    # Batch POST /data/query/ids (20 stIds per call)
    chunks = [all_stids[i : i + 20] for i in range(0, min(len(all_stids), 2400), 20)]

    def _fetch_chunk(stid_chunk: list[str]) -> list[dict[str, Any]]:
        url = "https://reactome.org/ContentService/data/query/ids"
        body = ",".join(stid_chunk).encode("utf-8")
        data, _ = _http_get_json(url, data=body, headers={"Content-Type": "text/plain"}, timeout=25.0, retries=3)
        return data if isinstance(data, list) else []

    raw_pathways: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_chunk, ch) for ch in chunks]
        for fut in as_completed(futs):
            try:
                raw_pathways.extend(fut.result())
            except Exception:
                continue

    records: list[PluginConceptRecord] = []
    for pw in raw_pathways:
        if len(records) >= max_records:
            break
        stid = str(pw.get("stId") or "").strip()
        name = str(pw.get("displayName") or "").strip()
        if not stid or not name or len(name) < 4 or len(name) > 76:
            continue
        # Skip mutation-allele micro-subpathways with code noise like "Signaling by FGFR2 IIIa TM"
        if re.search(r"\b(?:mutants?|variants?)\b.*\b[A-Z]\d{2,4}[A-Z]\b", name):
            continue

        sums = [str(s.get("text") or "").strip() for s in (pw.get("summation") or []) if s.get("text")]
        if not sums:
            continue
        sum_clean = _clean_text(" ".join(sums))
        if len(sum_clean) < 60:
            continue
        sents = _split_sentences(sum_clean)
        if not sents:
            continue

        top_domain = top_map.get(stid, "Human Cellular Physiology")
        in_disease = bool(pw.get("isInDisease"))
        compartments = list(
            dict.fromkeys(
                str(c.get("displayName") or "").strip()
                for c in (pw.get("compartment") or [])
                if isinstance(c, dict) and c.get("displayName")
            )
        )

        defn = sents[0]
        if len(defn) < 55 and len(sents) >= 2:
            defn = f"{sents[0]} {sents[1]}"

        comp_note = f" Occurs within {', '.join(compartments[:3])}." if compartments else ""
        if len(sents) >= 2 and sents[1] not in defn:
            mech = f"{sents[1]}{comp_note}"
        else:
            mech = f"Curated human biochemical pathway under {top_domain} ({stid}) coordinating sequential molecular reactions.{comp_note}"

        apps = f"Mapped in Reactome ({stid}) under {top_domain} for pathway enrichment analysis, systems pharmacology, and multi-omics interpretation."
        if len(sents) >= 3 and sents[2] not in (defn, mech):
            trd = sents[2]
        elif in_disease:
            trd = "Pathological pathway perturbation alters downstream signaling homeostasis and contributes to human disease progression or drug resistance."
        else:
            trd = "Pathway flux is modulated by tissue-specific expression, feedback loops, cross-talk with parallel cascades, and metabolic state."

        cat = "medicine_pharmacology" if in_disease else "biology_genomics"
        aliases = [f"{name} pathway"] if "pathway" not in name.casefold() and len(name) <= 50 else []

        rec = _build_validated_record(
            concept_id=name,
            title=name,
            category=cat,
            aliases=aliases,
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"reactome:{stid}",
        )
        if rec is not None:
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Worker 6: InterPro Protein Domains, Families & Conserved Sites
# ---------------------------------------------------------------------------

def harvest_interpro_entries(*, max_records: int = 1350) -> list[PluginConceptRecord]:
    """Worker 6: Harvest curated protein domains, families & active sites from InterPro REST API."""
    records: list[PluginConceptRecord] = []
    next_url: str | None = (
        "https://www.ebi.ac.uk/interpro/api/entry/interpro/"
        "?page_size=200&extra_fields=description,short_name"
    )
    pages = 0

    while next_url and len(records) < max_records and pages < 14:
        pages += 1
        try:
            data, _ = _http_get_json(next_url, timeout=25.0, retries=3)
        except Exception:
            break

        for item in data.get("results") or []:
            if len(records) >= max_records:
                break
            meta = item.get("metadata") or {}
            ef = item.get("extra_fields") or {}
            acc = str(meta.get("accession") or "").strip()
            name = str(meta.get("name") or "").strip()
            etype = str(meta.get("type") or "domain").replace("_", " ")
            if not acc or not name or len(name) < 4 or len(name) > 74:
                continue
            n_low = name.casefold()
            if any(bad in n_low for bad in ("unknown function", "uncharacterised", "uncharacterized", "duf", "upf0")):
                continue

            desc_blocks = ef.get("description") or []
            raw_parts: list[str] = []
            for blk in desc_blocks:
                if isinstance(blk, dict) and blk.get("text"):
                    raw_parts.append(str(blk["text"]))
                elif isinstance(blk, str):
                    raw_parts.append(blk)
            if not raw_parts:
                continue

            clean_desc = _clean_text(" ".join(raw_parts))
            if len(clean_desc) < 65:
                continue
            sents = _split_sentences(clean_desc)
            if not sents:
                continue

            # Extract member signature databases (Pfam, SMART, CDD, PROSITE, etc.)
            mem_dbs = sorted(list((meta.get("member_databases") or {}).keys()))[:4]
            mem_str = ", ".join(db.upper() for db in mem_dbs) if mem_dbs else "InterPro"

            defn = sents[0]
            if len(defn) < 55 and len(sents) >= 2:
                defn = f"{sents[0]} {sents[1]}"

            if len(sents) >= 2 and sents[1] not in defn:
                mech = sents[1]
            else:
                mech = f"Conserved protein {etype} ({acc}) identified via sequence profile HMMs and structural signatures integrated across {mem_str}."

            apps = f"Used in automated proteome annotation, evolutionary domain architecture analysis, and functional classification ({acc}; {mem_str})."
            if len(sents) >= 3 and sents[2] not in (defn, mech):
                trd = sents[2]
            else:
                trd = "Sequence divergence across distant homologs can reduce profile sensitivity, and domain presence alone does not guarantee identical substrate specificity."

            # Add domain/family qualifier in title if helpful for disambiguation
            title = name
            if etype in ("domain", "active site", "binding site", "conserved site") and etype not in n_low:
                title = f"{name} ({etype.title()})" if len(name) <= 54 else name

            short_name = str(ef.get("short_name") or "").replace("_", " ").strip()
            aliases = [short_name] if (short_name and 4 <= len(short_name) <= 40 and short_name.casefold() != name.casefold()) else []

            rec = _build_validated_record(
                concept_id=name,
                title=title,
                category="biology_genomics",
                aliases=aliases,
                definition=defn,
                mechanism=mech,
                applications=apps,
                tradeoffs=trd,
                provenance=f"interpro:{acc}",
            )
            if rec is not None:
                records.append(rec)

        next_url = data.get("next")

    return records


# ---------------------------------------------------------------------------
# Worker 7: openFDA Prescription Drug Labeling & Clinical Safety
# ---------------------------------------------------------------------------

def _strip_fda_section_header(text: str) -> str:
    s = _clean_text(text)
    s = re.sub(
        r"^(?:\d+(?:\.\d+)?\s*)?(?:INDICATIONS AND USAGE|MECHANISM OF ACTION|CLINICAL PHARMACOLOGY|"
        r"BOXED WARNING|WARNINGS AND PRECAUTIONS|WARNINGS|ADVERSE REACTIONS|CONTRAINDICATIONS|"
        r"WARNING:\s*[A-Z\s,/-]{3,60})\s*[:\-]?\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def harvest_openfda_drugs(*, max_records: int = 550) -> list[PluginConceptRecord]:
    """Worker 7: Harvest FDA-approved prescription drug pharmacology & safety profiles from openFDA."""
    records: list[PluginConceptRecord] = []
    seen_generics: set[str] = set()
    search_q = urllib.parse.quote(
        'openfda.product_type:"HUMAN PRESCRIPTION DRUG" AND _exists_:mechanism_of_action AND _exists_:indications_and_usage'
    )

    for skip in range(0, 2600, 100):
        if len(records) >= max_records:
            break
        url = f"https://api.fda.gov/drug/label.json?search={search_q}&limit=100&skip={skip}"
        try:
            data, _ = _http_get_json(url, timeout=25.0, retries=3)
        except Exception:
            break

        results = data.get("results") or []
        if not results:
            break

        for row in results:
            if len(records) >= max_records:
                break
            ofda = row.get("openfda") or {}
            gen_list = ofda.get("generic_name") or []
            if not gen_list:
                continue
            raw_gen = str(gen_list[0]).strip()
            # Reject messy multi-drug cocktails (> 2 ingredients) or long strings
            if not raw_gen or len(raw_gen) > 46 or raw_gen.count(",") >= 2 or raw_gen.count(" AND ") >= 2:
                continue
            base_gen = _strip_drug_salt(raw_gen)
            display_name = _smart_title(base_gen)
            gkey = canonical_concept_key(display_name)
            if not gkey or gkey in seen_generics or _is_ugly_chemical_name(display_name):
                continue

            moa_raw = (row.get("mechanism_of_action") or [""])[0]
            ind_raw = (row.get("indications_and_usage") or [""])[0]
            bw_raw = (row.get("boxed_warning") or [""])[0]
            warn_raw = (row.get("warnings_and_cautions") or row.get("warnings") or row.get("adverse_reactions") or [""])[0]

            moa_clean = _strip_fda_section_header(moa_raw)
            ind_clean = _strip_fda_section_header(ind_raw)
            safety_clean = _strip_fda_section_header(bw_raw or warn_raw)

            if len(moa_clean) < 50 or len(ind_clean) < 45:
                continue

            epc_list = [re.sub(r"\s*\[EPC\]\s*$", "", e).strip() for e in (ofda.get("pharm_class_epc") or []) if e]
            routes = [r.lower() for r in (ofda.get("route") or [])[:2] if r]
            brands = [
                _smart_title(b.strip())
                for b in (ofda.get("brand_name") or [])[:2]
                if b and len(b.strip()) <= 32 and b.strip().casefold() != display_name.casefold()
            ]

            ind_sents = _split_sentences(ind_clean)
            moa_sents = _split_sentences(moa_clean)
            saf_sents = _split_sentences(safety_clean)

            epc_str = f" ({epc_list[0]})" if epc_list else ""
            ind_first = ind_sents[0] if ind_sents else ind_clean
            if display_name.casefold() not in ind_first.casefold():
                defn = f"{display_name}{epc_str} is an FDA-approved prescription drug: {ind_first}"
            else:
                defn = ind_first

            mech = moa_sents[0] if moa_sents else moa_clean
            route_note = f" Administered via {'/'.join(routes)} route." if routes else ""
            if len(ind_sents) >= 2 and ind_sents[1] != ind_first:
                apps = f"{ind_sents[1]}{route_note}"
            elif epc_list:
                apps = f"Prescribed clinically as a {', '.join(epc_list[:2])} under FDA drug labeling guidelines.{route_note}"
            else:
                apps = f"Prescribed clinically under FDA-approved therapeutic labeling for targeted patient management.{route_note}"
            if bw_raw and saf_sents:
                trd = f"FDA Boxed Warning: {saf_sents[0]}"
            elif saf_sents:
                trd = saf_sents[0]
            else:
                trd = "Subject to FDA prescription contraindications, adverse reaction monitoring, and dose adjustment in renal or hepatic impairment."

            set_id = str(row.get("set_id") or row.get("id") or gkey).split("-")[0][:12]
            rec = _build_validated_record(
                concept_id=display_name,
                title=display_name,
                category="medicine_pharmacology",
                aliases=brands[:2],
                definition=defn,
                mechanism=mech,
                applications=apps,
                tradeoffs=trd,
                provenance=f"openfda:{set_id}",
            )
            if rec is not None:
                seen_generics.add(gkey)
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Worker 8: ClinicalTrials.gov APIv2 Verified Clinical Therapeutic Interventions
# ---------------------------------------------------------------------------

_CTGOV_QUERIES: tuple[str, ...] = (
    "immunotherapy",
    "gene therapy",
    "monoclonal antibody",
    "CAR-T",
    "targeted therapy",
    "mRNA vaccine",
    "deep brain stimulation",
    "transcatheter",
    "CRISPR",
    "stem cell transplantation",
    "checkpoint inhibitor",
    "radiotherapy",
)

_CTGOV_SKIP_INTERVENTIONS: frozenset[str] = frozenset({
    "placebo", "saline", "standard of care", "usual care", "control", "sham",
    "observation", "no intervention", "questionnaire", "blood draw", "biopsy",
    "best supportive care", "active comparator", "normal saline", "dextrose",
})


def harvest_clinicaltrials_interventions(*, max_records: int = 420, max_workers: int = 4) -> list[PluginConceptRecord]:
    """Worker 8: Harvest completed Phase 2/3/4 clinical intervention concepts from ClinicalTrials.gov APIv2."""

    def _fetch_area(area: str) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({
            "filter.overallStatus": "COMPLETED",
            "query.term": area,
            "pageSize": "100",
            "fields": "NCTId,BriefTitle,BriefSummary,Condition,InterventionName,InterventionType,Phase",
        })
        url = f"https://clinicaltrials.gov/api/v2/studies?{params}"
        try:
            data, _ = _http_get_json(url, timeout=25.0, retries=3)
            return data.get("studies") or []
        except Exception:
            return []

    all_studies: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_area, a) for a in _CTGOV_QUERIES]
        for fut in as_completed(futs):
            try:
                all_studies.extend(fut.result())
            except Exception:
                continue

    records: list[PluginConceptRecord] = []
    seen_interventions: set[str] = set()

    for study in all_studies:
        if len(records) >= max_records:
            break
        ps = study.get("protocolSection") or {}
        phases = (ps.get("designModule") or {}).get("phases") or []
        if not any(p in ("PHASE2", "PHASE3", "PHASE4") for p in phases):
            continue

        nct_id = str((ps.get("identificationModule") or {}).get("nctId") or "").strip()
        summary = _clean_text((ps.get("descriptionModule") or {}).get("briefSummary") or "")
        conditions = [c.strip() for c in ((ps.get("conditionsModule") or {}).get("conditions") or []) if c and len(c.strip()) <= 50]
        interventions = (ps.get("armsInterventionsModule") or {}).get("interventions") or []

        if not nct_id or len(summary) < 80 or not conditions:
            continue

        # Pick the highest-signal biomedical intervention (drug, biological, genetic, device, procedure)
        picked_name = ""
        picked_type = ""
        for iv in interventions:
            itype = str(iv.get("type") or "").upper()
            iname = str(iv.get("name") or "").strip()
            if itype not in ("DRUG", "BIOLOGICAL", "GENETIC", "DEVICE", "PROCEDURE", "RADIATION"):
                continue
            in_low = iname.casefold()
            if (
                not iname
                or len(iname) < 5
                or len(iname) > 54
                or in_low in _CTGOV_SKIP_INTERVENTIONS
                or any(w in in_low for w in ("placebo", "sham", "saline", "standard of care", "mg ", "mg/", "dose ", "arm ", "cohort", "group "))
                or _RAW_ACCESSION_RE.match(iname)
                or re.search(r"\b[A-Z]{2,5}-\d{3,6}\b", iname)
            ):
                continue
            picked_name = iname
            picked_type = itype.lower()
            break

        if not picked_name:
            continue

        cond_primary = conditions[0]
        concept_title = f"{_smart_title(picked_name)} Clinical Intervention for {cond_primary}"
        if len(concept_title) > 78:
            concept_title = f"{_smart_title(picked_name)} ({picked_type.title()} Therapy)"

        ikey = canonical_concept_key(picked_name)
        ckey = canonical_concept_key(concept_title)
        if not ikey or ikey in seen_interventions or ckey in seen_interventions:
            continue

        sents = _split_sentences(summary)
        # Filter out meta-sentences like "This trial is..." when better biomedical sentences exist
        bio_sents = [s for s in sents if not re.match(r"^(?:this\s+(?:study|trial|protocol)|the\s+purpose\s+of\s+this\s+(?:study|trial))", s, re.IGNORECASE)]
        ordered_sents = bio_sents + [s for s in sents if s not in bio_sents]
        if not ordered_sents:
            continue

        phase_str = "/".join(p.replace("PHASE", "Phase ") for p in phases[:2])
        defn = f"{_smart_title(picked_name)} is a {phase_str}-evaluated {picked_type} intervention investigated for {', '.join(conditions[:2])}: {ordered_sents[0]}"
        mech = ordered_sents[1] if len(ordered_sents) >= 2 else f"Evaluated in completed {phase_str} clinical investigation ({nct_id}) assessing biological efficacy and safety mechanisms."
        apps = f"Clinically studied for managing {', '.join(conditions[:3])} under verified trial protocol {nct_id}."
        trd = "Clinical deployment requires randomized safety validation, patient-eligibility stratification, adverse-event surveillance, and regulatory approval."

        rec = _build_validated_record(
            concept_id=concept_title,
            title=concept_title,
            category="medicine_pharmacology",
            aliases=[f"{picked_name} therapy"] if len(picked_name) <= 36 else [],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"clinicaltrials:{nct_id}",
        )
        if rec is not None:
            seen_interventions.add(ikey)
            seen_interventions.add(ckey)
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Worker 9: arXiv Export API (18 Scientific, Computing, Math & Econ Categories)
# ---------------------------------------------------------------------------

_ARXIV_CATEGORIES: tuple[tuple[str, str, int], ...] = (
    ("cat:cs.AI", "artificial_intelligence", 65),
    ("cat:cs.LG", "artificial_intelligence", 65),
    ("cat:cs.CL", "artificial_intelligence", 60),
    ("cat:cs.CV", "artificial_intelligence", 60),
    ("cat:cs.CR", "cybersecurity_networking", 65),
    ("cat:cs.DS", "computer_science", 65),
    ("cat:cs.DB", "software_engineering", 60),
    ("cat:cs.DC", "computer_science", 60),
    ("cat:cs.SE", "software_engineering", 60),
    ("cat:cs.PL", "computer_science", 55),
    ("cat:quant-ph", "physics", 65),
    ("cat:cond-mat.mes-hall", "physics", 55),
    ("cat:astro-ph.CO", "astronomy_space", 60),
    ("cat:math.OC", "mathematics", 65),
    ("cat:math.PR", "mathematics", 55),
    ("cat:stat.ML", "mathematics", 60),
    ("cat:q-bio.GN", "biology_genomics", 55),
    ("cat:econ.TH", "economics_finance", 55),
    ("cat:physics.geo-ph", "earth_climate_science", 50),
    ("cat:eess.SP", "engineering", 55),
)


def _fetch_arxiv_category(query: str, domain: str, max_results: int) -> list[PluginConceptRecord]:
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    raw_xml = _http_get_text(url, timeout=25.0, retries=3)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw_xml)
    records: list[PluginConceptRecord] = []

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        if title_el is None or summary_el is None:
            continue

        raw_title = _clean_text(title_el.text or "")
        raw_summary = _clean_text(summary_el.text or "")
        arxiv_id = (id_el.text or "").strip().rsplit("/", 1)[-1] if id_el is not None else ""
        if not raw_title or not arxiv_id or len(raw_summary) < 120:
            continue
        if "\\" in raw_title or "?" in raw_title or "$" in raw_title:
            continue

        # Extract clean concise method/concept phrase (prefer pre-colon concept name if informative)
        if ":" in raw_title:
            prefix = raw_title.split(":", 1)[0].strip()
            short_title = prefix if (10 <= len(prefix) <= 68 and len(prefix.split()) >= 2) else raw_title
        else:
            short_title = raw_title
        if len(short_title) > 72:
            short_title = short_title[:72].rsplit(" ", 1)[0].strip(" ,;:-")
        if len(short_title.split()) < 2 or len(short_title) < 8:
            continue
        # Reject conversational or hyper-generic titles
        if re.match(r"^(?:on\s+the\s|a\s+note\s+on|towards?\s|can\s|why\s|how\s|when\s|what\s|is\s|are\s|do\s)", short_title, re.IGNORECASE):
            continue

        sents = _split_sentences(raw_summary)
        if len(sents) < 2:
            continue

        defn = sents[0]
        mech = ""
        apps = ""
        trd = ""
        for s in sents[1:8]:
            low = s.casefold()
            if not trd and any(w in low for w in ("limit", "challenge", "however", "cost", "overhead", "trade-off", "bottleneck", "difficult", "bound", "constraint", "error", "noise", "risk")):
                trd = s
            elif not mech and any(w in low for w in ("propose", "method", "algorithm", "model", "framework", "operat", "comput", "derive", "formulat", "construct", "train", "optimiz", "protocol", "proof")):
                mech = s
            elif not apps and any(w in low for w in ("demonstrat", "experiment", "outperform", "benchmark", "achiev", "appli", "enable", "evaluat", "result", "improv", "task", "system")):
                apps = s

        unused = [s for s in sents[1:8] if s not in (defn, mech, apps, trd)]
        if not mech:
            mech = unused.pop(0) if unused else f"Formalized via mathematical and algorithmic constructs in {domain.replace('_', ' ')} (arXiv:{arxiv_id})."
        if not apps:
            apps = unused.pop(0) if unused else f"Applied to research benchmarks, computational analysis, and technical systems in {domain.replace('_', ' ')}."
        if not trd:
            trd = unused.pop(0) if unused else f"Subject to theoretical assumptions, computational scaling bounds, and empirical generalization limits in {domain.replace('_', ' ')}."

        rec = _build_validated_record(
            concept_id=short_title,
            title=short_title,
            category=domain,
            aliases=[],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"arxiv:{arxiv_id}",
        )
        if rec is not None:
            records.append(rec)

    return records


def harvest_arxiv_concepts(*, max_records: int = 1050) -> list[PluginConceptRecord]:
    """Worker 9: Respectfully paced multi-category arXiv Export API harvester (20 scientific fields)."""
    records: list[PluginConceptRecord] = []

    def _paced_fetch(args: tuple[str, str, int]) -> list[PluginConceptRecord]:
        q, dom, lim = args
        time.sleep(0.35)
        try:
            return _fetch_arxiv_category(q, dom, lim)
        except Exception:
            return []

    # Polite concurrency of 2 workers to respect arXiv API fair-use limits
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_paced_fetch, cfg) for cfg in _ARXIV_CATEGORIES]
        for fut in as_completed(futs):
            try:
                records.extend(fut.result())
            except Exception:
                continue

    return records[:max_records]


# ---------------------------------------------------------------------------
# Unified Orchestrator with 4-Layer Canonical Deduplication & JSONL Output
# ---------------------------------------------------------------------------

def deduplicate_plugin_records(
    ordered_source_batches: list[tuple[str, list[PluginConceptRecord]]],
    *,
    skip_canonical_keys: set[str] | None = None,
    target_max: int = 12000,
) -> tuple[list[PluginConceptRecord], dict[str, int], int]:
    """Enforce Mirror 7's exact 4-layer deduplication (exact ID, canonical key, alias key, 64-bit content hash)."""
    seen_cids: set[str] = set()
    seen_ckeys: set[str] = set(skip_canonical_keys or ())
    seen_alias_keys: set[str] = set()
    seen_fps: set[str] = set()

    final_records: list[PluginConceptRecord] = []
    source_counts: Counter[str] = Counter()
    dup_dropped = 0

    for src_name, batch in ordered_source_batches:
        for rec in batch:
            if len(final_records) >= target_max:
                break
            cid = rec.concept_id.casefold()
            ck = canonical_concept_key(cid)
            fp = content_fingerprint(rec.definition)

            if not ck or cid in seen_cids or ck in seen_ckeys or ck in seen_alias_keys or fp in seen_fps:
                dup_dropped += 1
                continue

            # Filter secondary aliases so they never collide with any previously claimed canonical/alias key
            clean_aliases: list[str] = [cid]
            for al in rec.aliases:
                if al == cid:
                    continue
                ak = canonical_concept_key(al)
                if ak and ak != ck and ak not in seen_ckeys and ak not in seen_alias_keys:
                    clean_aliases.append(al)

            final_rec = PluginConceptRecord(
                concept_id=cid,
                title=rec.title,
                category=rec.category,
                aliases=tuple(dict.fromkeys(clean_aliases)),
                definition=rec.definition,
                mechanism=rec.mechanism,
                applications=rec.applications,
                tradeoffs=rec.tradeoffs,
                provenance=rec.provenance,
                quality_score=rec.quality_score,
            )

            seen_cids.add(cid)
            seen_ckeys.add(ck)
            for al in final_rec.aliases:
                ak = canonical_concept_key(al)
                if ak:
                    seen_alias_keys.add(ak)
            seen_fps.add(fp)

            final_records.append(final_rec)
            source_counts[src_name] += 1

    return final_records, dict(source_counts), dup_dropped


def harvest_all_plugin_concepts(
    output_path: str | Path,
    *,
    skip_canonical_keys: set[str] | None = None,
    target_max: int = 12000,
    seed: int = 42,
) -> PluginHarvestReport:
    """Run all official scientific plugin harvesters in parallel, deduplicate cleanly, and write JSONL."""
    t0 = time.perf_counter()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    worker_outputs: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {
            pool.submit(harvest_openfda_drugs, max_records=550): "openfda",
            pool.submit(harvest_chembl_drugs_and_targets, max_drugs=1250, max_targets=400): "chembl",
            pool.submit(harvest_pubchem_compounds, max_records=240): "pubchem",
            pool.submit(harvest_uniprot_human_proteins, max_records=1500): "uniprot",
            pool.submit(harvest_reactome_pathways, max_records=1250): "reactome",
            pool.submit(harvest_interpro_entries, max_records=1150): "interpro",
            pool.submit(harvest_ols4_ontologies, max_workers=6): "ols4",
            pool.submit(harvest_clinicaltrials_interventions, max_records=320): "clinicaltrials",
            pool.submit(harvest_arxiv_concepts, max_records=950): "arxiv",
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                worker_outputs[name] = fut.result()
            except Exception:
                worker_outputs[name] = {} if name in ("chembl", "ols4") else []

    chembl_map = worker_outputs.get("chembl") or {}
    ols_map = worker_outputs.get("ols4") or {}

    # Priority order (richest curated clinical/molecular entries first so best record wins overlaps!)
    ordered_batches: list[tuple[str, list[PluginConceptRecord]]] = [
        ("openfda", worker_outputs.get("openfda") or []),
        ("chembl_drugs", chembl_map.get("chembl_drugs") or []),
        ("pubchem", worker_outputs.get("pubchem") or []),
        ("uniprot", worker_outputs.get("uniprot") or []),
        ("reactome", worker_outputs.get("reactome") or []),
        ("interpro", worker_outputs.get("interpro") or []),
        ("ols_doid", ols_map.get("ols_doid") or []),
        ("ols_cl", ols_map.get("ols_cl") or []),
        ("ols_uberon", ols_map.get("ols_uberon") or []),
        ("ols_go", ols_map.get("ols_go") or []),
        ("ols_hp", ols_map.get("ols_hp") or []),
        ("ols_chebi", ols_map.get("ols_chebi") or []),
        ("chembl_targets", chembl_map.get("chembl_targets") or []),
        ("clinicaltrials", worker_outputs.get("clinicaltrials") or []),
        ("arxiv", worker_outputs.get("arxiv") or []),
    ]

    raw_total = sum(len(b) for _, b in ordered_batches)
    final_records, source_counts, dup_dropped = deduplicate_plugin_records(
        ordered_batches,
        skip_canonical_keys=skip_canonical_keys,
        target_max=target_max,
    )

    # Atomic write via temp file to ensure clean JSONL output
    tmp_file = out_file.with_suffix(".jsonl.tmp")
    with tmp_file.open("w", encoding="utf-8") as fh:
        for rec in final_records:
            fh.write(json.dumps(rec.to_json_dict(), ensure_ascii=False) + "\n")
    tmp_file.replace(out_file)

    domain_counts = dict(Counter(r.category for r in final_records).most_common())
    scores = [r.quality_score for r in final_records]
    avg_q = round(sum(scores) / len(scores), 3) if scores else 0.0
    min_q = round(min(scores), 3) if scores else 0.0

    rng = random.Random(seed)
    sample_recs = [
        r.to_json_dict()
        for r in rng.sample(final_records, min(5, len(final_records)))
    ]

    return PluginHarvestReport(
        total_emitted=len(final_records),
        raw_candidates_collected=raw_total,
        rejected_quality=0,
        rejected_duplicate=dup_dropped,
        source_counts=source_counts,
        domain_counts=domain_counts,
        avg_quality_score=avg_q,
        min_quality_score=min_q,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
        output_path=str(out_file),
        sample_records=sample_recs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror 7 Phase-2 Verifiable Science Plugins & Technical Standards Bulk Concept Harvester"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for deduplicated, quality-gated plugin ConceptNodes.",
    )
    parser.add_argument(
        "--target-max",
        type=int,
        default=12000,
        help="Maximum unique verified ConceptNodes to emit (default: 12000).",
    )
    args = parser.parse_args()

    report = harvest_all_plugin_concepts(
        output_path=args.output,
        target_max=args.target_max,
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
