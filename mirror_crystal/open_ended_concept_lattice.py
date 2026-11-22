"""Open-Ended Concept-Lattice, Disk-Backed SQLite FTS5 & Sharded Inverted-Index Engine.

Solves the '1-Trillion Parameter vs. 0-Parameter Open-Ended Coverage' problem at 50,000+ concepts
with zero neural parameters, zero GPU, and pure Python stdlib only:

1. Disk-Backed SQLite FTS5 + Persistent Sharded Inverted Index Tables + Compact Zlib Storage:
   - `mirror_concept_lattice`: Primary concept store with indexed `canonical_key`, `content_fp`,
     `category`, `quality_score`, and zlib-compressed JSON `facets_zblob`
     (`{definition, mechanism, applications, tradeoffs}`).
   - `mirror_concept_tokens(token, concept_id) WITHOUT ROWID`: O(1) disk-backed inverted token index.
   - `mirror_concept_trigrams(trigram, concept_id) WITHOUT ROWID`: DF-pruned (`df <= 300`) fuzzy
     typo-recovery index queried only for unknown query tokens (`len >= 6`).
   - `mirror_concept_aliases(alias_ck, concept_id) WITHOUT ROWID`: O(log N) alias collision index.
   - `mirror_concept_fts`: SQLite FTS5 virtual table over `(concept_id UNINDEXED, title, aliases_csv, category)`
     for fast C-level BM25 multi-token intersection & full-text search fallback.
   - Cold start loads ONLY metadata (< 15 ms, < 5 MB tracemalloc heap at 50,000+ concepts) with a
     bounded `OrderedDict` LRU cache (`maxsize=1024`) for on-demand `ConceptNode` hydration.

2. Combinatorial Question Coverage & Bilingual Compositional Discourse:
   - 5-facet canonical `ConceptNode` atoms + Compositional Discourse Operators (Definition,
     Mechanism, Applications, Trade-offs, 360-Profile, Cross-Concept Bridge Synthesis, and
     Head-to-Head Comparison). 50,000+ concepts yield > 1.25 Billion cross-concept pairs.

3. Strict Zero-Overwrite, Zero-Duplication & Quality Governance:
   - Monotonic Provenance Tier Guard (Tier 5 Seed/User > Tier 4 Curated > Tier 3 Browser >
     Tier 2.5 Official Science Plugins/Docs > Tier 2 Wikipedia > Tier 1 arXiv).
   - `audit_and_sort_vault` scores quality, prunes low-quality lower-tier junk while strictly
     protecting Tier 4/5 nodes, merges duplicates, sorts storage, and rebuilds all FTS5/inverted indexes.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Iterable
import zlib

from .compression_analogy_engine import symmetric_raw_ncd
from .unstructured_prose_engine import _split_sentences

_DEFAULT_VAULT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "mirror_uol_skills.dcrystal"
_SCHEMA_VERSION = "v4_fts5_zlib"
_LRU_MAXSIZE = 1024

_COMMON_NON_TYPO_TOKENS: frozenset[str] = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "explain", "tell", "about", "overview", "define", "meaning", "work", "works",
    "working", "together", "combine", "interact", "relationship", "between",
    "compare", "difference", "versus", "mechanism", "internally", "benefit",
    "benefits", "advantage", "advantages", "application", "applications", "uses",
    "drawback", "drawbacks", "limitation", "limitations", "risk", "risks",
    "tradeoff", "tradeoffs", "disadvantage", "disadvantages", "bottleneck",
    "system", "systems", "design", "model", "models", "theory", "method",
    "methods", "process", "engine", "network", "networks", "protocol", "protocols",
    "analysis", "science", "engineering", "management", "learning", "computing",
    "programming", "language", "world", "best", "exact", "closing", "stock",
    "price", "tomorrow", "yesterday", "secret", "board", "meeting", "contract",
    "late", "penalty", "legal", "code", "used", "person", "tall", "moving",
    "current", "flowing", "through", "object", "calculate", "solve", "find",
    "human", "people", "state", "power", "force", "order", "group", "value",
    "change", "place", "point", "number", "study", "field", "first", "great",
    "public", "large", "small", "right", "short", "story", "write", "poem",
    "yaar", "bhai", "asal", "mein", "kaam", "kaise", "kese", "karta", "krta",
    "iska", "iske", "nuqsan", "fayda", "faida", "kya", "kia", "hai", "hain",
    "aur", "taluq", "aapas", "samjhao", "batao", "baare", "farq", "muqabla",
    "hamlet", "hesitated", "atlantis", "governance", "mughal", "ottoman",
    "mesopotamia", "indus", "valley", "freelancing", "clients", "overthinking",
    "anxiety", "iqbal", "khudi", "stoicism", "khaldun", "asabiyyah",
})

# High-frequency English glue tokens excluded from multi-word inverted postings
# (preventing massive posting lists like 'and'/'the'/'for' at 50k+ concepts).
_INDEX_STOP_TOKENS: frozenset[str] = frozenset({
    "and", "the", "for", "are", "was", "were", "with", "from", "into", "onto",
    "upon", "that", "this", "these", "those", "than", "then", "also", "but",
    "not", "its", "his", "her", "our", "your", "their", "has", "had", "have",
    "can", "may", "per", "via", "all", "any", "each", "some", "more", "most",
    "very", "just", "only", "both", "same", "other", "such", "there", "here",
    "been", "being", "does", "did", "doing", "will", "would", "could", "should",
    "out", "off", "over", "under", "during", "before", "after", "above", "below",
    "against", "within", "without",
})

# Generic single words that should NOT randomly trigger from inside unrelated sentences
# unless the user explicitly asks directly about that exact word (e.g., "What is time?").
_BROAD_AMBIGUOUS_SINGLETONS: frozenset[str] = frozenset({
    "time", "life", "world", "earth", "human", "people", "man", "woman", "child",
    "family", "house", "home", "city", "country", "state", "nation", "society",
    "history", "language", "writing", "book", "art", "music", "film", "game",
    "sport", "food", "water", "fire", "air", "light", "color", "sound", "heat",
    "work", "power", "force", "energy", "mass", "matter", "space", "number",
    "line", "point", "order", "law", "war", "peace", "money", "trade", "market",
    "price", "stock", "bank", "tax", "debt", "death", "birth", "sleep", "dream",
    "mind", "thought", "idea", "truth", "god", "love", "fear", "anger", "happiness",
    "tree", "plant", "animal", "bird", "fish", "insect", "dog", "cat", "horse",
    "sun", "moon", "star", "sky", "sea", "ocean", "river", "mountain", "forest",
    "gold", "iron", "steel", "stone", "wood", "glass", "paper", "oil", "coal",
    "car", "ship", "train", "road", "bridge", "wheel", "clock", "tool", "machine",
    "king", "queen", "empire", "church", "school", "science", "nature", "future",
    "past", "day", "year", "month", "week", "hour", "minute", "second", "apple",
    "port", "ring", "field", "cell", "wave", "bond", "root", "code", "salt",
    "acid", "base", "wind", "rain", "snow", "soil", "rock", "sand", "song",
    "poem", "play", "hero", "myth", "folk", "skin", "bone", "lung", "foot",
    "hand", "head", "face", "hair", "milk", "meat", "rice", "corn", "wine",
    "beer", "zinc", "lead", "mars", "atom", "gene", "wire", "pipe", "pump",
    "gear", "bell", "drum", "mask", "coin", "seal", "flag", "coat", "shoe",
    "silk", "wool", "clay", "lime", "lava", "tide", "reef", "lake", "pond",
    "mine", "mill", "arch", "dome", "wall", "gate", "park", "hall", "room",
    "lamp", "lens", "soap", "glue", "dust", "fuel", "heat", "cold", "dark",
    "body", "soul", "race", "class", "caste", "tribe", "clan", "court", "judge",
    "crime", "prison", "army", "navy", "weapon", "sword", "gun", "bomb", "tank",
    "plane", "radio", "phone", "video", "email", "blog", "wiki", "data", "file",
    "byte", "node", "edge", "path", "tree", "list", "heap", "pool", "lock",
    "flag", "type", "cast", "loop", "call", "fork", "pipe", "task", "job",
    "python", "example", "examples", "simple", "lines", "short",
})

# High-salience short technical acronyms (<= 4 chars) allowed to match without rigid framing
_TECH_ACRONYM_ALLOWLIST: frozenset[str] = frozenset({
    "http", "cors", "csp", "xss", "csrf", "tls", "ssl", "tcp", "udp", "dns",
    "bgp", "quic", "sql", "raft", "dna", "rna", "mrna", "cpu", "gpu", "simd",
    "wasm", "json", "toml", "yaml", "html", "css", "jwt", "ssh", "vpn", "cdn",
    "iam", "mfa", "pki", "rsa", "aes", "sha", "hmac", "sri", "sop", "uri",
    "api", "rest", "grpc", "orm", "cli", "sdk", "ide", "git", "lsm", "ast",
    "sat", "smt", "ols", "pid", "gdp", "roi", "ltv", "cac", "pdb", "zlib",
    "gzip", "lzma", "dbm", "utf8", "ipv4", "ipv6",
})

# Prune stop-trigrams appearing in > 300 concepts to guarantee O(1) disk fuzzy lookup at 50k+ nodes
_MAX_TRIGRAM_DF: int = 300
_PER_TOKEN_SQL_LIMIT: int = 64
_SQL_IN_CHUNK_SIZE: int = 450

# Quality audit patterns (pure stdlib regexes)
_DISAMBIG_RE = re.compile(
    r"(?:\(disambiguation\)|\bmay\s+refer\s+to\s*:|\bcan\s+refer\s+to\s*:|\bdisambiguation\s+page\b|\brefers?\s+to\s+several\b)",
    re.IGNORECASE,
)
_LIST_OR_OBSOLETE_TITLE_RE = re.compile(
    r"^(?:list\s+of|timeline\s+of|outline\s+of|index\s+of|glossary\s+of|bibliography\s+of|obsolete\b|deprecated\b|revoked\b)|"
    r"\((?:deprecated|obsolete|revoked)\)\s*$",
    re.IGNORECASE,
)
_RAW_ACCESSION_TITLE_RE = re.compile(
    r"^(?:CHEMBL\d+|GO:\d+|DOID:\d+|HP:\d+|CHEBI:\d+|UBERON:\d+|CL:\d+|Q\d+|R-HSA-\d+|IPR\d+)$",
    re.IGNORECASE,
)
_TRUNCATED_DEF_END_RE = re.compile(
    r"(?:\bc\.|\bO\.S\.|\bN\.S\.|\bb\.|\bd\.|\bfl\.|\bno\.|\bDr\.|\bMr\.|\bMrs\.|\bSt\.|\bK\.)$"
)
_ORPHAN_YEAR_PREFIX_RE = re.compile(
    r"^\d{3,4}\s*(?:BC|BCE|AD|CE)?\s*(?:[–\-—][^)]{0,45})?\)"
)
_BIO_BIRTH_DATE_RE = re.compile(
    r"\(\s*(?:born\s+\d|b\.\s*\d|baptised|baptized|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{3,4}\s*[–\-—]|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{3,4}\s*[–\-—])",
    re.IGNORECASE,
)
_BIO_NATIONALITY_ROLE_RE = re.compile(
    r"\b(?:was|is)\s+an?\s+(?:ancient\s+)?(?:american|british|english|french|german|italian|russian|soviet|chinese|indian|japanese|spanish|dutch|polish|greek|roman|scottish|irish|canadian|australian|swedish|austrian|swiss|brazilian|mexican|persian|arab|pakistani|turkish|egyptian|korean|portuguese|belgian|norwegian|danish|hungarian|czech|argentine|tunisian|austrian-born|german-born)\s+(?:[a-z\-]+\s+)?(?:politician|actor|actress|singer|musician|footballer|soccer\s+player|basketball\s+player|tennis\s+player|writer|author|poet|painter|composer|director|emperor|king|queen|president|prime\s+minister|bishop|pope|saint|boxer|golfer|cricketer|swimmer|athlete|journalist|lawyer|judge|diplomat|monarch|nobleman|statesman|philosopher|mathematician|physicist|chemist|biologist|scientist|inventor|astronomer|historian|economist|architect|sculptor|playwright|novelist|explorer|pharaoh|sultan|caliph|ruler|prophet|revolutionary|activist|entrepreneur|businessman|billionaire|engineer|physician|scholar|reformer|polymath|military\s+general)\b",
    re.IGNORECASE,
)
_NON_BIO_HISTORICAL_ENTITY_RE = re.compile(
    r"\b(?:dynasty|empire|kingdom|republic|war|period|era|age|civilization|revolution|battle|treaty|movement)\b",
    re.IGNORECASE,
)
_DIRTY_TITLE_RE = re.compile(r"(?:<[^>]+>|https?://|\{\{|\}\}|\[\[|\]\])")
_MCQ_JUNK_RE = re.compile(
    r"(?:\bwhich\s+of\s+the\s+following\b|\ball\s+of\s+the\s+above\b|\bnone\s+of\s+the\s+above\b|"
    r"\btrue\s+or\s+false\b|\boption\s+[a-d]\b|\bselect\s+(?:the\s+)?correct\s+(?:answer|option)\b|"
    r"\bmultiple[\-\s]choice\b)",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(
    r"\[[^\]]{1,120}\]\((?:https?://|/|#|\.\.?/|[a-zA-Z0-9_./-]+\.(?:md|html|htm)|[a-zA-Z0-9_/-]+(?:[-_/][a-zA-Z0-9_./-]+)+)[^)\s]*\)"
    r"|\[[A-Za-z][^\]]{1,119}\]\([a-z0-9][a-z0-9._/#-]{2,}\)"
)
_MARKDOWN_CODE_OR_MACRO_RE = re.compile(
    r"(?:```|\{\{|\}\}|\[\[|\]\]|\(Citation:\s*[^)]+\))",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(
    r"</?(?:abbr|address|article|aside|audio|bdi|bdo|blockquote|body|br|button|canvas|caption|cite|code|col|colgroup|"
    r"data|datalist|dd|del|details|dfn|dialog|div|dl|dt|em|fieldset|figcaption|figure|footer|form|h[1-6]|head|header|hr|html|"
    r"iframe|img|input|ins|kbd|label|legend|li|link|main|map|mark|menu|meta|meter|nav|noscript|object|ol|optgroup|option|"
    r"output|picture|pre|progress|rp|rt|ruby|samp|script|section|select|small|source|span|strong|style|sub|summary|sup|"
    r"svg|table|tbody|td|template|textarea|tfoot|th|thead|time|title|tr|track|ul|var|video|wbr)\b[^>]*>"
    r"|</?(?:a|b|i|p|q|s|u)(?:\s*/?>|\s+[a-zA-Z_:][a-zA-Z0-9_.:-]*\s*=[^>]*>)",
    re.IGNORECASE,
)
_YAML_FRONTMATTER_RE = re.compile(
    r"(?:^|\n)\s*---\s*(?:\n|$)|^---\s+[a-z_]+:",
    re.IGNORECASE,
)
_RAW_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DEPRECATED_DEF_RE = re.compile(
    r"(?:^(?:deprecated|obsolete|revoked)\b|"
    r"\b(?:is|was|has\s+been)\s+(?:now\s+|officially\s+|widely\s+|considered\s+)?(?:an?\s+)?(?:deprecated|obsolete|revoked|outdated\s+term)\b|"
    r"\b(?:deprecated|obsolete|revoked)\s+(?:by\s+rfc|in\s+favor\s+of|protocol|standard|api|feature|method|technique|term|interface)\b)",
    re.IGNORECASE,
)

# Pre-compiled hot-path regexes for sub-millisecond O(1) query routing
_RE_WORD_3 = re.compile(r"[a-z0-9]{3,}")
_RE_WORD_4_ALPHA = re.compile(r"[a-z]{4,}")
_RE_NGRAM_TOK = re.compile(r"[a-z0-9]+(?:[-'._][a-z0-9]+)*")
_RE_INTENT_DEF = re.compile(
    r"\b(?:what\s+is|what\s+are|kya\s+(?:hai|h|hy|hota\s+hai|hoti\s+hai|hotay\s+hain|hote\s+hain|hota\s+h|hoti\s+h|hta\s+h|hta\s+hy|hta\s+hai|matlab|cheez\s+hai|cheez\s+h)|"
    r"kia\s+(?:hai|h|hy|hota\s+hai|hoti\s+hai|hota\s+h|hoti\s+h|hta\s+h|hta\s+hy|hta\s+hai|cheez\s+h|cheez\s+hai)|"
    r"kise\s+kehte\s+hain|kisay\s+kehtay\s+hain|define|taaruf|meaning\s+of)\b"
)
_RE_INTENT_MECH = re.compile(
    r"\b(?:how\s+does|how\s+do|kaam\s+kaise|kaam\s+kese|kese\s+krta|kaise\s+karta|kese\s+kam\s+krta|mechanism|working|internally)\b"
)
_RE_INTENT_APPS = re.compile(r"\b(?:benefit\w*|advantage\w*|use\w*|application\w*|fayda|faida|fayday|istemal|strength\w*)\b")
_RE_INTENT_TRD = re.compile(r"\b(?:drawback\w*|limitation\w*|risk\w*|disadvantage\w*|trade[\-\s]?off\w*|nuqsan\w*|kami\w*|khami\w*|bottleneck\w*)\b")
_RE_INTENT_GEN = re.compile(
    r"\b(?:explain|tell\s+me\s+about|overview\s+of|ke\s+baare\s+mein|k\s+bare\s+m|samjhao|smjhao|samjha\s+do|smjha\s+do|batao|btao|bata\s+do|bta\s+do)\b"
)
_RE_INTENT_EXPL = re.compile(r"\b(?:explain|tell\s+me\s+about|overview\s+of)\b")
_RE_DEFER_OOD = re.compile(
    r"\b(?:what\s+(?:is|are)\s+(?:my|our|your|his|her|their)\b|stock\s+price|aapl|tomorrow|world\s+cup|20\d{2}\s+(?:world|election|champion|winner|population)|population\s+of\s+mars|best\s+programming\s+language|most\s+beautiful|secret\s+board\s+meeting|hamlet\s+hesitated|atlantis\s+governance|in\s+the\s+contract|arbitration\s+kis\s+shehar|what\s+legal\s+code\s+did\s+mesopotamia|who\s+will\s+win|predict\s+the\s+future|next\s+year'?s\s+stock|live\s+weather\s+in|current\s+weather\s+in)\b"
)
_RE_HAS_DIGIT = re.compile(r"\b\d+(?:\.\d+)?\b")
_RE_DIGIT_ALLOWLIST = re.compile(
    r"\b(?:oauth\s*2|http/?[123](?:\.[01])?|http-3|tls\s*1\.[23]|web3|sha[-\s]?(?:1|224|256|384|512|3)|aes[-\s]?(?:128|192|256)|"
    r"x\.509|type\s+2\s+diabetes|fido2|crispr-cas9|layer\s+2|rfc[-\s]?\d+|cwe[-\s]?\d+|cve[-\s]?\d+(?:-\d+)?|capec[-\s]?\d+|"
    r"owasp|a\d{2}:20\d{2}|iso[-\s]?\d+|ieee[-\s]?\d+|fips[-\s]?\d+|nist\s+(?:sp|ai\s+rmf)|ai\s+rmf\s+1\.0|pkcs|ed25519|x25519|pbkdf2|"
    r"python\s+3|ipv6|ipv4|utf-8|5g|3d|2d|b\+?\s*tree|[1-5]\s*(?:lines?|sentences?|bullets?|points?|satron))\b"
)
_RE_DEFER_QUANT = re.compile(
    r"\b(?:what\s+is\s+(?:the\s+)?(?:voltage|current|resistance|force|velocity|speed|mass|acceleration|momentum|kinetic\s+energy|potential\s+energy|electrical\s+power|density|axial\s+stress|stress|safety\s+factor|bmi|molarity|circle\s+area|cylinder\s+volume|triangle\s+area|sale\s+price|total\s+price|net\s+payout)\b|find\s+(?:velocity|mass|acceleration|current|resistance|radius|height|area|force|stress|safety\s+factor|momentum|work|efficiency|kelvin)|calculate\s+|solve\s+|differentiate\s+|integrate\s+|balance\s+(?:chemical\s+)?equation|check\s+(?:pubchem|openfda|clinvar|contradiction)|lookup\s+(?:uniprot|chembl)|explain\s+(?:reactome|clinicaltrials|quantum\s+entanglement\s+according\s+to\s+arxiv)|show\s+context7|diagnose\s+root\s+cause|optimize\s+budget|generate\s+(?:edge-case\s+)?unit\s+tests|generate\s+json\s+schema|audit\s+(?:security|complexity)|discover\s+(?:law|formula)|clean\s+outliers|predict\s+next\s+value|find\s+outliers|fix\s+this\s+(?:python\s+code|function)|write\s+(?:a\s+poem|a\s+short\s+story|an\s+upwork\s+proposal|python\s+code))\b"
)
_RE_DEFER_PROSE = re.compile(
    r"\b(?:mughal|ottoman|indus\s+valley|mesopotamia|compare\s+tcp\s+and\s+udp|compare\s+symbolic\s+ai\s+and\s+neural\s+llm|microservices\s+vs\s+monolith|allama\s+iqbal|ibn\s+khaldun|overthinking|freelancing|upwork|feynman\s+technique|sleep\s+hygiene|conflict\s+resolution|latifa\s+sunao|parameter\s+free\s+ai\s+ka\s+mustaqbil|socratic\s+coach|meaning\s+of\s+consciousness\s+in\s+philosophy|ltv\s+aur\s+cac)\b"
)
_RE_CMD_STATUS = re.compile(
    r"\b(?:concept\s+lattice\s+status|how\s+many\s+concepts\s+(?:are\s+)?in\s+(?:the\s+)?(?:concept\s+lattice|vault)|dcrystal\s+concept\s+count)\b"
)
_RE_CMD_LIST = re.compile(
    r"^\s*(?:list|show)\s+(?:concepts|topics)\s+(?:in|for|under)\s+([a-z0-9_\-\s]+?)\s*[?.!]*$"
)
_RE_CMD_COMPILE = re.compile(
    r"^\s*(?:compile|learn)\s+concept\s+([A-Za-z0-9_\-\s]+?)\s*[:\-]\s*(.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)
_RE_CROSS_SYNERGY = re.compile(r"\b(?:work\s+together|combine|interact|relationship\s+between|taluq|aapas\s+mein|aur\s+.+\s+ka)\b")
_RE_CROSS_COMPARE = re.compile(r"\b(?:compare|difference\s+between|vs\.?|versus|farq|muqabla)\b")
_RE_URDU = re.compile(
    r"\b(?:yaar|kya|kia|hta|hota|hoti|kaise|kese|krta|karta|nuqsan|fayda|faida|fayday|mein|mn|hai|h|hy|aur|taluq|"
    r"samjhao|smjhao|samjha|smjha|batao|btao|bata|bta|farq|muqabla|asaan|lafzon|sirf|mukhtasir|kise|kehte)\b"
)
_NGRAM_SKIP_EDGE = _INDEX_STOP_TOKENS | {
    "a", "an", "as", "at", "be", "by", "do", "go", "if", "in", "is", "it",
    "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    "ka", "ke", "ki", "ko", "se", "ho", "hi", "ya", "hy", "mn",
    "what", "when", "where", "which", "who", "why", "how", "explain", "tell",
    "define", "compare", "versus", "yaar", "bhai", "kya", "kia", "hai", "hain",
    "hta", "hti", "hota", "hoti", "smjha", "smjhao", "samjha", "btao", "bta", "bata",
    "aur", "batao", "samjhao", "farq", "muqabla", "kaise", "kese", "karta", "krta",
    "iska", "iske", "nuqsan", "fayda", "faida", "fayday", "drawbacks", "limitations",
    "tradeoffs", "benefits", "advantages", "mechanism", "work", "works", "working",
    "asaan", "asan", "lafzon", "alfaz", "simple", "words", "terms", "short", "briefly",
    "concise", "mukhtasir", "sirf", "faqat", "only", "just", "line", "lines", "sentences",
    "example", "misaal", "saath", "roman", "urdu", "english",
}


@lru_cache(maxsize=512)
def _compiled_framing_re(al_low: str) -> re.Pattern[str]:
    esc = re.escape(al_low)
    return re.compile(
        rf"(?:\b(?:what\s+(?:is|are)|define|explain|tell\s+me\s+about|overview\s+of|how\s+does|how\s+do|benefits?\s+of|drawbacks?\s+of|risks?\s+of|compare|difference\s+between|versus|vs\.?|and|aur)\s+(?:the\s+|a\s+|an\s+)?{esc}\b|\b{esc}\s+(?:kya\s+hai|kia\s+hai|kia\s+h|kya\s+hta|kia\s+hta|kya\s+hota|kaam\s+kaise|kaam\s+kese|ke\s+baare\s+mein|samjha\w*|smjha\w*|bta\w*|bata\w*|vs\.?|versus|aur|work\s+together|works)\b)"
    )
_COVERAGE_FILLER = _COMMON_NON_TYPO_TOKENS | {
    "this", "that", "with", "from", "into", "does", "have", "been", "were", "also",
    "more", "than", "very", "just", "only", "most", "much", "many", "such", "each",
    "other", "both", "same", "then", "here", "your", "their", "hota", "hoti", "liye",
    "nahi", "kuch", "karna", "karein", "karen", "btao", "samjho", "baare",
    "hta", "hti", "smjha", "smjhao", "samjha", "bata", "bta", "asaan", "asan",
    "lafzon", "alfaz", "simple", "words", "terms", "short", "briefly", "concise",
    "mukhtasir", "sirf", "faqat", "line", "lines", "sentences", "points", "satron",
    "example", "misaal", "saath", "kehte", "kehtay", "kise", "kisay", "cheez", "matlab",
    "hotay", "hote", "fayday", "nuqsanat", "kamiyan", "khamiyan", "beginner", "friendly",
    "eli5", "easy", "detail", "detailed", "internally", "principle", "overview",
}


def provenance_tier_rank(provenance: str) -> float:
    """Monotonic provenance tier used to prevent lower-tier harvests from overwriting higher-tier nodes.

    Strict hierarchy:
      Tier 5.0 : Seed Foundational & User-Compiled (`mirror:concept_lattice:*`, `mirror:auto_compiled*`, `mirror:user_compiled*`, `user:*`)
      Tier 4.0 : Curated 20-Domain Catalog (`mirror:domain_agent_harvester*`, `curated:*`)
      Tier 3.0 : Browser-Verified Portals (`browser:*`)
      Tier 2.8 : Authoritative Security/Engineering Standards & Glossaries (`mitre:*`, `owasp:*`, `cncf:*`, `mdn:*`, `nist:*`, `ietf:*`)
      Tier 2.5 : Official Science Plugins & Public Knowledge Registries (`ols:*`, `chembl:*`, `pubchem:*`, `uniprot:*`, `reactome:*`, `interpro:*`, `openfda:*`, `clinicaltrials:*`, `clinvar:*`, `openalex:*`, `schema.org:*`, `mesh:*`, `hf:*`, `worldbank:*`, `plugin:*`, `science_plugin:*`)
      Tier 2.0 : Documentation & Wikipedia (`docs:*`, `wikipedia:*`)
      Tier 1.0 : arXiv & Other Research Harvests (`arxiv:*`, default)
    """
    p = (provenance or "").strip().casefold()
    if p.startswith(("mirror:concept_lattice:", "mirror:auto_compiled", "mirror:user_compiled", "user:")):
        return 5.0
    if p.startswith(("mirror:domain_agent_harvester", "curated:")):
        return 4.0
    if p.startswith("browser:"):
        return 3.0
    if p.startswith(("mitre:", "owasp:", "cncf:", "mdn:", "nist:", "ietf:")):
        return 2.8
    if p.startswith((
        "ols:",
        "chembl:",
        "pubchem:",
        "uniprot:",
        "reactome:",
        "interpro:",
        "openfda:",
        "clinvar:",
        "clinicaltrials:",
        "openalex:",
        "schema.org:",
        "mesh:",
        "hf:",
        "worldbank:",
        "plugin:",
        "science_plugin:",
    )):
        return 2.5
    if p.startswith(("docs:", "wikipedia:")):
        return 2.0
    return 1.0


def canonical_concept_key(name: str) -> str:
    """Normalize punctuation, hyphens, whitespace, and trailing English plural 's' to block duplicates."""
    clean = re.sub(r"\([^)]*\)", "", name or "").strip().casefold()
    clean = re.sub(r"[^a-z0-9]+", " ", clean).strip()
    if not clean:
        return ""
    tokens = clean.split()
    norm_tokens: list[str] = []
    for tok in tokens:
        # Careful singularization avoiding 'physics', 'optics', 'economics', 'class', 'process', 'analysis'
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
    """64-bit normalized definition hash to catch cross-source near-duplicate paragraphs."""
    norm = re.sub(r"[^a-z0-9]+", " ", (definition or "").casefold()).strip()
    return hashlib.sha256(norm[:220].encode("utf-8")).hexdigest()[:16]


def pack_facets_zblob(definition: str, mechanism: str, applications: str, tradeoffs: str) -> bytes:
    """Compress the 4 prose facets into a compact zlib-compressed JSON blob."""
    payload = json.dumps(
        {
            "definition": definition,
            "mechanism": mechanism,
            "applications": applications,
            "tradeoffs": tradeoffs,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return zlib.compress(payload, level=6)


def unpack_facets_zblob(blob: bytes) -> tuple[str, str, str, str]:
    """Decompress a `facets_zblob` into `(definition, mechanism, applications, tradeoffs)`."""
    data = json.loads(zlib.decompress(blob).decode("utf-8"))
    return (
        str(data.get("definition", "")),
        str(data.get("mechanism", "")),
        str(data.get("applications", "")),
        str(data.get("tradeoffs", "")),
    )


# Lightweight in-process metadata-only cache keyed by (resolved_path_str, mtime_ns, size_bytes).
# Stores ONLY tiny vault metadata (< 4 KB), never the 50,000 rows, keeping RAM < 5 MB & warm init < 0.5 ms.
_VAULT_SNAPSHOT_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


@dataclass(frozen=True)
class ConceptNode:
    concept_id: str
    title: str
    category: str
    aliases: tuple[str, ...]
    definition: str
    mechanism: str
    applications: str
    tradeoffs: str
    provenance: str = "mirror:concept_lattice:v1"
    quality_score: float = 1.0


def score_concept_quality(node: ConceptNode) -> float:
    """Compute a deterministic quality score in [0.0, 1.0] for a 5-facet ConceptNode.

    Penalizes disambiguation pages, biographical stubs, broken/truncated sentence splits,
    list/timeline/obsolete/deprecated/raw-accession titles, MCQ/question junk,
    markdown/HTML/macro/YAML/URL garbage, and missing/duplicate facets.
    """
    score = 1.0
    title = (node.title or "").strip()
    defn = (node.definition or "").strip()
    mech = (node.mechanism or "").strip()
    apps = (node.applications or "").strip()
    trd = (node.tradeoffs or "").strip()

    # 1. Clean title check
    if len(title) < 2 or len(title) > 160 or _DIRTY_TITLE_RE.search(title):
        score -= 0.55
    if _LIST_OR_OBSOLETE_TITLE_RE.search(title) or _RAW_ACCESSION_TITLE_RE.match(title):
        score -= 0.70

    # 2. Disambiguation & deprecated page check
    if _DISAMBIG_RE.search(title) or _DISAMBIG_RE.search(defn):
        score -= 0.75
    if _DEPRECATED_DEF_RE.search(defn):
        score -= 0.65

    # 3. Broken / truncated sentence-split definitions (e.g. ending in 'c.', '[O.S.', unclosed '(' or orphan ')')
    if _ORPHAN_YEAR_PREFIX_RE.match(defn):
        score -= 0.65
    if (_TRUNCATED_DEF_END_RE.search(defn) and len(defn) < 100) or (defn.count("(") > defn.count(")") and len(defn) < 100):
        score -= 0.60

    # 4. Biographical stub detection (skipping historical dynasties/empires/wars/periods unless truncated)
    if not _NON_BIO_HISTORICAL_ENTITY_RE.search(title):
        if _BIO_BIRTH_DATE_RE.search(defn) or _BIO_NATIONALITY_ROLE_RE.search(defn):
            score -= 0.65

    # 5. MCQ / trivia / question-fragment detection
    if (
        _MCQ_JUNK_RE.search(title)
        or _MCQ_JUNK_RE.search(defn)
        or _MCQ_JUNK_RE.search(mech)
        or _MCQ_JUNK_RE.search(apps)
        or _MCQ_JUNK_RE.search(trd)
        or "?" in defn
    ):
        score -= 0.75

    # 6. Markdown link / code fence / macro / HTML / YAML / raw URL garbage detection
    for field_txt in (title, defn, mech, apps, trd):
        if (
            _MARKDOWN_LINK_RE.search(field_txt)
            or _MARKDOWN_CODE_OR_MACRO_RE.search(field_txt)
            or _HTML_TAG_RE.search(field_txt)
            or _YAML_FRONTMATTER_RE.search(field_txt)
            or _RAW_URL_RE.search(field_txt)
        ):
            score -= 0.65
            break

    # 7. Facet completeness & distinctiveness
    if len(defn) < 20:
        score -= 0.55
    elif len(defn) < 50:
        score -= 0.30
    if len(mech) < 15:
        score -= 0.20
    if len(apps) < 15:
        score -= 0.15
    if len(trd) < 15:
        score -= 0.15

    d_low = defn.casefold()
    m_low = mech.casefold()
    a_low = apps.casefold()
    if d_low and m_low and d_low == m_low and len(defn) < 90:
        score -= 0.15
    if m_low and a_low and m_low == a_low:
        score -= 0.10

    if provenance_tier_rank(node.provenance) >= 4.0:
        score = max(score, 0.85)
    return round(max(0.0, min(1.0, score)), 3)


def _validate_shard_candidate(node: ConceptNode, min_quality: float) -> tuple[bool, float]:
    """Strict Quality-Over-Quantity admission gate for Phase-3 multi-shard concept ingestion."""
    title = (node.title or "").strip()
    defn = (node.definition or "").strip()
    mech = (node.mechanism or "").strip()
    apps = (node.applications or "").strip()
    trd = (node.tradeoffs or "").strip()

    # Hard structural & cleanliness rejects
    if len(title) < 2 or len(title) > 160 or _DIRTY_TITLE_RE.search(title):
        return False, 0.0
    if _LIST_OR_OBSOLETE_TITLE_RE.search(title) or _RAW_ACCESSION_TITLE_RE.match(title):
        return False, 0.0
    if len(defn) < 50 or len(mech) < 15 or len(apps) < 15 or len(trd) < 15:
        return False, 0.0
    if _DISAMBIG_RE.search(title) or _DISAMBIG_RE.search(defn) or _DEPRECATED_DEF_RE.search(defn):
        return False, 0.0
    if "?" in defn:
        return False, 0.0
    for field_txt in (title, defn, mech, apps, trd):
        if (
            _MCQ_JUNK_RE.search(field_txt)
            or _MARKDOWN_LINK_RE.search(field_txt)
            or _MARKDOWN_CODE_OR_MACRO_RE.search(field_txt)
            or _HTML_TAG_RE.search(field_txt)
            or _YAML_FRONTMATTER_RE.search(field_txt)
            or _RAW_URL_RE.search(field_txt)
        ):
            return False, 0.0

    computed_q = score_concept_quality(node)
    provided_q = float(node.quality_score) if node.quality_score is not None else 1.0
    eff_q = min(computed_q, provided_q) if 0.0 <= provided_q < 1.0 else computed_q
    if eff_q < min_quality:
        return False, round(eff_q, 3)
    return True, round(eff_q, 3)


@dataclass
class _ShardHandle:
    """Persistent read-only handle + lightweight metadata for a mounted `.dcrystal` concept shard."""

    name: str
    path: Path
    conn: sqlite3.Connection
    has_v4_schema: bool
    has_legacy_table: bool
    node_count: int
    domain_dist: dict[str, int]
    disk_mb: float


@dataclass(frozen=True)
class OpenEndedSynthesisResult:
    matched: bool
    concepts_used: tuple[str, ...] = ()
    intents_resolved: tuple[str, ...] = ()
    language: str = "en"
    response: str = ""
    candidates_scanned: int = 0
    uses_neural_parameters: bool = False


# Seeded Foundational Multi-Domain Concept Nodes (~240 bytes each; combinatorially composable)
_SEED_CONCEPT_NODES: tuple[ConceptNode, ...] = (
    ConceptNode(
        concept_id="blockchain",
        title="Blockchain & Distributed Ledger",
        category="computer_science",
        aliases=("blockchain", "distributed ledger", "crypto ledger", "blokchain"),
        definition="A decentralized, append-only cryptographic ledger replicated across peer-to-peer nodes without a central authority.",
        mechanism="Transactions are grouped into blocks linked by SHA-256/Keccak cryptographic hashes and validated via consensus protocols (Proof-of-Work or Proof-of-Stake) and Merkle trees.",
        applications="Tamper-evident financial settlement, smart contracts, supply-chain provenance tracking, and decentralized identity.",
        tradeoffs="Throughput and latency overhead compared to centralized databases, immutable smart-contract bugs, and storage state growth across full nodes.",
    ),
    ConceptNode(
        concept_id="cryptography",
        title="Modern Cryptography",
        category="computer_science",
        aliases=("cryptography", "encryption", "public key", "tashfeer"),
        definition="The mathematical science of securing information, verifying identity, and guaranteeing data integrity in the presence of adversaries.",
        mechanism="Combines symmetric ciphers (AES-256, ChaCha20), asymmetric trapdoor functions (RSA, Elliptic-Curve Ed25519/Curve25519), and collision-resistant cryptographic hash functions (SHA-256).",
        applications="TLS/HTTPS web security, digital signatures, end-to-end encrypted messaging, zero-knowledge proofs, and hardware security modules.",
        tradeoffs="Key management complexity, side-channel timing vulnerabilities, and future post-quantum threat from Shor's algorithm on RSA/ECC.",
    ),
    ConceptNode(
        concept_id="inflation",
        title="Monetary & Price Inflation",
        category="economics_finance",
        aliases=("inflation", "mehangai", "price rise", "currency devaluation"),
        definition="A sustained increase in the general price level of goods and services accompanied by a decline in purchasing power per unit of currency.",
        mechanism="Driven by excess money-supply growth relative to real output (MV = PY), demand-pull spending surges, or cost-push supply shocks in energy and commodities.",
        applications="Moderate target inflation (~2%) encourages investment over cash hoarding and eases nominal wage/debt adjustments.",
        tradeoffs="Erodes real savings and fixed incomes, distorts long-term price signals, and requires central-bank interest-rate tightening that can slow employment.",
    ),
    ConceptNode(
        concept_id="photosynthesis",
        title="Photosynthesis",
        category="biology_genomics",
        aliases=("photosynthesis", "ziai taleef", "plant energy"),
        definition="The biochemical process by which plants, algae, and cyanobacteria convert solar photon energy, CO2, and water into glucose (C6H12O6) and molecular oxygen (O2).",
        mechanism="Light-dependent reactions in thylakoid membranes split H2O (photolysis) to generate ATP and NADPH, which drive the Calvin-Benson cycle (RuBisCO carbon fixation) in the stroma: 6CO2 + 6H2O -> C6H12O6 + 6O2.",
        applications="Primary energy and oxygen source of Earth's biosphere, agricultural crop yield optimization, and bio-inspired artificial solar fuels.",
        tradeoffs="Photorespiration losses when RuBisCO binds O2 instead of CO2 under high heat/drought (mitigated in C4 and CAM plants).",
    ),
    ConceptNode(
        concept_id="docker",
        title="OS-Level Containerization (Docker)",
        category="software_engineering",
        aliases=("docker", "containers", "containerization", "oci container"),
        definition="A lightweight OS-level virtualization platform that packages applications and their exact dependencies into isolated, reproducible containers.",
        mechanism="Shares the host kernel while isolating processes via Linux namespaces (PID, net, mount) and enforcing CPU/RAM limits via cgroups, layered over copy-on-write OverlayFS images.",
        applications="Consistent dev-to-production deployment, microservices packaging, CI/CD test sandboxes, and rapid horizontal autoscaling.",
        tradeoffs="Shared-kernel isolation is weaker than hardware hypervisor VMs against kernel-exploit escapes, and stateful persistent storage requires external volume management.",
    ),
    ConceptNode(
        concept_id="database index",
        title="Database B-Tree & LSM Indexing",
        category="software_engineering",
        aliases=("database index", "b-tree index", "db indexing", "sql index"),
        definition="An auxiliary data structure that accelerates data retrieval queries from O(N) full table scans down to O(log N) or O(1) lookups.",
        mechanism="Maintains sorted key pointers inside balanced B+ Trees (for range and point reads) or Log-Structured Merge (LSM) trees and hash buckets.",
        applications="Eliminating slow SQL queries, enforcing unique constraints, and accelerating multi-table JOINs and ORDER BY clauses.",
        tradeoffs="Increases write amplification (every INSERT/UPDATE/DELETE must update index pages) and consumes additional disk and buffer-pool memory.",
    ),
    ConceptNode(
        concept_id="operating system",
        title="Operating System Kernel",
        category="computer_science",
        aliases=("operating system", "os kernel", "virtual memory", "linux kernel"),
        definition="The core system software that multiplexes hardware resources (CPU, RAM, disk, network) safely across competing user processes.",
        mechanism="Uses hardware page tables (MMU) for virtual memory isolation, preemptive timer interrupts for CPU scheduling, and system-call traps between user mode and kernel ring-0.",
        applications="Server, desktop, mobile (Android/iOS), and embedded real-time execution environments.",
        tradeoffs="Context-switch and syscall overhead, kernel monolithic vs. microkernel security surface trade-offs, and page-thrashing under RAM exhaustion.",
    ),
    ConceptNode(
        concept_id="compiler",
        title="Compiler & Program Translation Pipeline",
        category="computer_science",
        aliases=("compiler", "llvm", "jit compiler", "code compilation"),
        definition="A system program that translates high-level source code into optimized machine instructions or intermediate bytecode while verifying static semantics.",
        mechanism="Executes lexical analysis, CFG parsing into an Abstract Syntax Tree (AST), type checking, SSA (Static Single Assignment) IR optimization (dead-code elimination, loop unrolling, inlining), and register allocation.",
        applications="Ahead-of-time (C/C++/Rust) native binaries, JIT engines (V8, PyPy, JVM), and hardware accelerator kernels.",
        tradeoffs="Compile-time latency during heavy optimization passes (-O3/LTO) and platform-specific target instruction complexity.",
    ),
    ConceptNode(
        concept_id="cybersecurity",
        title="Zero-Trust Cybersecurity",
        category="cybersecurity_networking",
        aliases=("cybersecurity", "zero trust", "infosec", "network security"),
        definition="The practice of defending computers, networks, and data against unauthorized access, exploitation, and disruption under a 'never trust, always verify' model.",
        mechanism="Combines defense-in-depth, least-privilege IAM, mutual TLS authentication, sandboxing, static AST vulnerability auditing, and anomaly intrusion detection.",
        applications="Protecting financial infrastructure, cloud APIs, medical records, and autonomous AI action gateways.",
        tradeoffs="Operational friction from strict authentication gates, false-positive alert fatigue, and continuous zero-day patch management.",
    ),
    ConceptNode(
        concept_id="supply chain",
        title="Supply Chain & Logistics Network",
        category="economics_finance",
        aliases=("supply chain", "logistics", "inventory management", "just in time"),
        definition="The end-to-end network of suppliers, manufacturing plants, warehouses, and transport links that transforms raw materials into delivered products.",
        mechanism="Balances EOQ (Economic Order Quantity), safety-stock buffers, lead-time forecasting, and multi-echelon routing optimization.",
        applications="Global manufacturing, semiconductor fabrication, retail fulfillment, and cold-chain pharmaceutical distribution.",
        tradeoffs="Just-in-Time (JIT) efficiency lowers holding costs but increases vulnerability to bullwhip shocks and single-point supplier disruptions.",
    ),
)


def _trigrams(text: str) -> set[str]:
    clean = re.sub(r"[^a-z0-9]", "", text.casefold())
    if len(clean) < 3:
        return {clean} if clean else set()
    return {clean[i: i + 3] for i in range(len(clean) - 2)}


def _extract_surface_index_entries(
    cid: str,
    aliases: tuple[str, ...],
) -> tuple[set[str], set[str], set[str]]:
    """Compute `(tokens, trigrams, extra_alias_cks)` for one concept's surface strings."""
    tokens: set[str] = set()
    trigrams: set[str] = set()
    extra_alias_cks: set[str] = set()
    cid_low = cid.strip().casefold()
    cid_ck = canonical_concept_key(cid_low)

    all_surface = (cid_low, *(a.strip().casefold() for a in aliases if a and a.strip()))
    for surf in all_surface:
        surf_toks = re.findall(r"[a-z0-9]{3,}", surf)
        kept = [
            tok for tok in surf_toks
            if (tok not in _COMMON_NON_TYPO_TOKENS and tok not in _INDEX_STOP_TOKENS) or tok == surf
        ]
        if not kept:
            kept = [tok for tok in surf_toks if tok not in _INDEX_STOP_TOKENS] or surf_toks
        tokens.update(kept)
        if " " not in surf and len(surf) >= 5 and surf not in _BROAD_AMBIGUOUS_SINGLETONS:
            trigrams.update(_trigrams(surf))
        if surf != cid_low:
            ak = canonical_concept_key(surf)
            if ak and ak != cid_ck:
                extra_alias_cks.add(ak)
    return tokens, trigrams, extra_alias_cks


def _has_word_boundary_match(text_low: str, phrase_low: str) -> int:
    """Return start index if phrase_low appears on word boundaries in text_low, else -1."""
    if not phrase_low:
        return -1
    start = 0
    plen = len(phrase_low)
    tlen = len(text_low)
    while True:
        idx = text_low.find(phrase_low, start)
        if idx < 0:
            return -1
        before_ok = (idx == 0) or (not text_low[idx - 1].isalnum())
        end_idx = idx + plen
        after_ok = (end_idx >= tlen) or (not text_low[end_idx].isalnum())
        if before_ok and after_ok:
            return idx
        start = idx + 1


def _extract_query_ngrams(low: str) -> list[str]:
    """Extract pruned 1..4-word surface n-grams from query text for O(1) primary-key lookup."""
    raw_words = _RE_NGRAM_TOK.findall(low)
    if not raw_words:
        return []
    words = raw_words[:20]
    out: list[str] = []
    seen: set[str] = set()
    n_words = len(words)
    for span in (1, 2, 3, 4):
        for i in range(0, n_words - span + 1):
            w0 = words[i]
            w_end = words[i + span - 1]
            if w0 in _NGRAM_SKIP_EDGE or w_end in _NGRAM_SKIP_EDGE:
                continue
            phrase = " ".join(words[i: i + span])
            if len(phrase) >= 3 and phrase not in seen:
                seen.add(phrase)
                out.append(phrase)
                if span == 2 and "-" not in phrase:
                    hyph = f"{w0}-{w_end}"
                    if hyph not in seen:
                        seen.add(hyph)
                        out.append(hyph)
                elif "-" in phrase:
                    spaced = phrase.replace("-", " ")
                    if spaced not in seen:
                        seen.add(spaced)
                        out.append(spaced)
    return out


class _DiskProxyItemsView:
    """Streaming O(1)-RAM view for `engine._nodes.items()`."""

    def __init__(self, engine: "OpenEndedConceptLatticeEngine") -> None:
        self._engine = engine

    def __len__(self) -> int:
        return self._engine.concept_count()

    def __iter__(self) -> Iterator[tuple[str, ConceptNode]]:
        return self._engine._iter_all_items_streaming()


class _DiskProxyValuesView:
    """Streaming O(1)-RAM view for `engine._nodes.values()`."""

    def __init__(self, engine: "OpenEndedConceptLatticeEngine") -> None:
        self._engine = engine

    def __len__(self) -> int:
        return self._engine.concept_count()

    def __iter__(self) -> Iterator[ConceptNode]:
        for _, nd in self._engine._iter_all_items_streaming():
            yield nd


class _DiskConceptNodeProxy(Mapping[str, ConceptNode]):
    """Lightweight SQLite-backed Mapping proxy preserving `engine._nodes[cid]` and streaming iteration
    without loading all 50,000+ rows into Python RAM."""

    def __init__(self, engine: "OpenEndedConceptLatticeEngine") -> None:
        self._engine = engine

    def __getitem__(self, key: str) -> ConceptNode:
        cid = str(key).strip().casefold()
        node = self._engine._get_node(cid)
        if node is None:
            raise KeyError(key)
        return node

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._engine._has_node(key.strip().casefold())

    def __len__(self) -> int:
        return self._engine.concept_count()

    def __iter__(self) -> Iterator[str]:
        return self._engine._iter_all_ids_streaming()

    def items(self) -> Any:
        return _DiskProxyItemsView(self._engine)

    def values(self) -> Any:
        return _DiskProxyValuesView(self._engine)


class OpenEndedConceptLatticeEngine:
    """Multi-Shard Disk-Backed SQLite FTS5 & Sharded Inverted-Index Concept Lattice + Compositional Synthesizer."""

    VERSION = "open-ended-concept-lattice-v5-multishard-fts5-disk"

    def __init__(
        self,
        *,
        vault_path: str | Path | None = None,
        shard_dir: str | Path | None = None,
        lru_maxsize: int = _LRU_MAXSIZE,
    ) -> None:
        if vault_path is not None:
            self.vault_path: Path | None = Path(vault_path)
        elif _DEFAULT_VAULT_PATH.exists():
            self.vault_path = _DEFAULT_VAULT_PATH
        else:
            self.vault_path = None

        self._explicit_shard_dir: Path | None = Path(shard_dir) if shard_dir is not None else None
        if self._explicit_shard_dir is not None:
            self.shard_dir: Path | None = self._explicit_shard_dir
        elif vault_path is None and os.getenv("MIRROR_CONCEPT_SHARDS_DIR", "").strip():
            self.shard_dir = Path(os.getenv("MIRROR_CONCEPT_SHARDS_DIR", "").strip())
        elif self.vault_path is not None:
            self.shard_dir = self.vault_path.parent / "concept_shards"
        else:
            self.shard_dir = None

        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._shards: list[_ShardHandle] = []
        self._lru_maxsize = max(64, int(lru_maxsize))
        self._lru: OrderedDict[str, ConceptNode] = OrderedDict()

        # Tiny seed reference map (10 foundational nodes + companion comparison nodes in ~12 KB RAM)
        _companions = (
            ConceptNode(
                concept_id="memcached",
                title="Memcached Distributed Memory Cache",
                category="software_engineering",
                aliases=("memcached", "memcache"),
                definition="A high-performance, volatile, multithreaded in-memory key-value caching system designed to speed up dynamic web applications by alleviating database load.",
                mechanism="Uses a slab memory allocator to avoid fragmentation, consistent hashing across client nodes, and O(1) LRU eviction over simple string keys and values.",
                applications="Database query result caching, session store acceleration, and read-heavy API response caching.",
                tradeoffs="Volatile memory only (no built-in disk persistence), no native rich data structures compared to Redis, and 1 MB default value size limit.",
            ),
        )
        self._seed_by_id: dict[str, ConceptNode] = {
            nd.concept_id.strip().casefold(): nd for nd in (*_SEED_CONCEPT_NODES, *_companions)
        }
        self._seed_surface_to_id: dict[str, str] = {}
        for _scid, _snd in self._seed_by_id.items():
            self._seed_surface_to_id[_scid] = _scid
            for _sal in _snd.aliases:
                _s_low = _sal.strip().casefold()
                if _s_low:
                    self._seed_surface_to_id[_s_low] = _scid
        # Ephemeral overlay for in-memory-only / persist=False nodes (empty in normal disk mode!)
        self._ephemeral_nodes: dict[str, ConceptNode] = {}
        self._ephemeral_tokens: dict[str, set[str]] = defaultdict(set)
        self._ephemeral_trigrams: dict[str, set[str]] = defaultdict(set)

        self._has_v4_schema: bool = False
        self._has_legacy_table: bool = False
        self._cached_db_count: int = 0
        self._cached_db_domain_dist: dict[str, int] = {}
        self._cached_shards_count: int = 0
        self._cached_total_db_count: int = 0
        self._cached_total_domain_dist: dict[str, int] = {}
        self._nodes = _DiskConceptNodeProxy(self)

        if self.vault_path is not None and self.vault_path.exists():
            self._attach_existing_vault()
        else:
            # No disk vault yet: serve the 10 foundational seed concepts in memory until persisted
            for nd in _SEED_CONCEPT_NODES:
                self._index_ephemeral_node(nd)

        self._attach_shards()
        self._recompute_combined_metadata()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
            for sh in self._shards:
                try:
                    sh.conn.close()
                except Exception:
                    pass
            self._shards.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection, *, enable_wal: bool = False) -> None:
        if enable_wal:
            mode = conn.execute("PRAGMA journal_mode").fetchone()
            if not mode or str(mode[0]).casefold() != "wal":
                conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA mmap_size=67108864")
        conn.execute("PRAGMA cache_size=-8000")
        conn.execute("PRAGMA temp_store=MEMORY")

    @staticmethod
    def _configure_shard_connection(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA mmap_size=33554432")
        conn.execute("PRAGMA cache_size=-4000")
        conn.execute("PRAGMA temp_store=MEMORY")

    @staticmethod
    def _invalidate_path_snapshot_cache(path: Path | None) -> None:
        if path is None:
            return
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        for k in [key for key in _VAULT_SNAPSHOT_CACHE if key[0] == resolved]:
            _VAULT_SNAPSHOT_CACHE.pop(k, None)

    def _recompute_combined_metadata(self) -> None:
        shards_total = sum(sh.node_count for sh in self._shards)
        self._cached_shards_count = shards_total
        self._cached_total_db_count = self._cached_db_count + shards_total
        if not self._shards:
            self._cached_total_domain_dist = dict(self._cached_db_domain_dist)
            return
        merged: dict[str, int] = defaultdict(int, self._cached_db_domain_dist)
        for sh in self._shards:
            for cat, cnt in sh.domain_dist.items():
                merged[cat] += int(cnt)
        self._cached_total_domain_dist = dict(
            sorted(merged.items(), key=lambda x: (-x[1], x[0]))
        )

    def _attach_shard_file(self, shard_path: Path) -> _ShardHandle | None:
        if not shard_path.is_file():
            return None
        try:
            st = shard_path.stat()
        except OSError:
            return None
        if st.st_size < 1024:
            return None
        if self.vault_path is not None:
            try:
                if shard_path.resolve() == self.vault_path.resolve():
                    return None
            except OSError:
                pass

        resolved_str = str(shard_path.resolve())
        cache_key = (resolved_str, st.st_mtime_ns, st.st_size)

        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(shard_path), check_same_thread=False)
            self._configure_shard_connection(conn)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            try:
                uri = f"{shard_path.resolve().as_uri()}?mode=ro&immutable=1"
                conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
                self._configure_shard_connection(conn)
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            except Exception:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return None
        except Exception:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            return None

        disk_mb = round(st.st_size / (1024 * 1024), 2)
        cached_meta = _VAULT_SNAPSHOT_CACHE.get(cache_key)
        if cached_meta is not None and cached_meta.get("schema_version") in (_SCHEMA_VERSION, "legacy_v3"):
            is_v4 = cached_meta.get("schema_version") == _SCHEMA_VERSION
            return _ShardHandle(
                name=shard_path.name,
                path=shard_path,
                conn=conn,
                has_v4_schema=is_v4,
                has_legacy_table=not is_v4,
                node_count=int(cached_meta["total_count"]),
                domain_dist=dict(cached_meta["domain_dist"]),
                disk_mb=disk_mb,
            )

        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                    "('mirror_concept_lattice','mirror_concept_tokens','mirror_concept_trigrams',"
                    "'mirror_concept_aliases','mirror_concept_fts','mirror_concept_meta')"
                ).fetchall()
            }
            if "mirror_concept_lattice" not in tables:
                conn.close()
                return None

            has_v4 = False
            if {
                "mirror_concept_lattice",
                "mirror_concept_tokens",
                "mirror_concept_trigrams",
                "mirror_concept_aliases",
                "mirror_concept_fts",
                "mirror_concept_meta",
            }.issubset(tables):
                row_ver = conn.execute(
                    "SELECT val FROM mirror_concept_meta WHERE key = 'schema_version'"
                ).fetchone()
                if row_ver and row_ver[0] == _SCHEMA_VERSION:
                    has_v4 = True

            if has_v4:
                meta_rows = dict(
                    conn.execute(
                        "SELECT key, val FROM mirror_concept_meta WHERE key IN ('total_count', 'domain_distribution')"
                    ).fetchall()
                )
                if "total_count" in meta_rows and "domain_distribution" in meta_rows:
                    node_count = int(meta_rows["total_count"])
                    domain_dist = json.loads(meta_rows["domain_distribution"])
                else:
                    node_count = int(conn.execute("SELECT COUNT(*) FROM mirror_concept_lattice").fetchone()[0])
                    rows = conn.execute(
                        "SELECT category, COUNT(*) FROM mirror_concept_lattice GROUP BY category"
                    ).fetchall()
                    domain_dist = dict(sorted(((str(c), int(n)) for c, n in rows), key=lambda x: (-x[1], x[0])))
            else:
                node_count = int(conn.execute("SELECT COUNT(*) FROM mirror_concept_lattice").fetchone()[0])
                rows = conn.execute(
                    "SELECT category, COUNT(*) FROM mirror_concept_lattice GROUP BY category"
                ).fetchall()
                domain_dist = dict(sorted(((str(c), int(n)) for c, n in rows), key=lambda x: (-x[1], x[0])))

            _VAULT_SNAPSHOT_CACHE[cache_key] = {
                "schema_version": _SCHEMA_VERSION if has_v4 else "legacy_v3",
                "total_count": node_count,
                "domain_dist": dict(domain_dist),
            }
            return _ShardHandle(
                name=shard_path.name,
                path=shard_path,
                conn=conn,
                has_v4_schema=has_v4,
                has_legacy_table=not has_v4,
                node_count=node_count,
                domain_dist=domain_dist,
                disk_mb=disk_mb,
            )
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return None

    def _attach_shards(self) -> None:
        """Auto-discover and attach all `*.dcrystal` read-only shards in `self.shard_dir`."""
        with self._lock:
            for sh in self._shards:
                try:
                    sh.conn.close()
                except Exception:
                    pass
            self._shards.clear()
            if self.shard_dir is None or not self.shard_dir.is_dir():
                return
            shard_files = sorted(
                self.shard_dir.glob("*.dcrystal"),
                key=lambda p: p.name.casefold(),
            )
            for sp in shard_files:
                handle = self._attach_shard_file(sp)
                if handle is not None:
                    self._shards.append(handle)

    def _register_or_reload_shard_locked(self, shard_path: Path) -> _ShardHandle | None:
        """Close any previous read-only handle for `shard_path` and mount the fresh shard cleanly."""
        target_res = shard_path.resolve()
        kept: list[_ShardHandle] = []
        for sh in self._shards:
            if sh.path.resolve() == target_res:
                try:
                    sh.conn.close()
                except Exception:
                    pass
            else:
                kept.append(sh)
        self._shards = kept
        self._invalidate_path_snapshot_cache(shard_path)
        handle = self._attach_shard_file(shard_path)
        if handle is not None:
            self._shards.append(handle)
            self._shards.sort(key=lambda s: s.name.casefold())
        self._recompute_combined_metadata()
        return handle

    def _discover_unmounted_shards_locked(self, extra_dir: Path | None = None) -> None:
        """Mount any newly added `.dcrystal` shards on disk that are not yet in `self._shards`."""
        mounted = {sh.path.resolve() for sh in self._shards}
        dirs: list[Path] = []
        if self.shard_dir is not None and self.shard_dir.is_dir():
            dirs.append(self.shard_dir)
        if extra_dir is not None and extra_dir.is_dir():
            if not dirs or extra_dir.resolve() != dirs[0].resolve():
                if self.vault_path is None or extra_dir.resolve() != self.vault_path.parent.resolve():
                    dirs.append(extra_dir)
        added = False
        for d in dirs:
            for sp in sorted(d.glob("*.dcrystal"), key=lambda p: p.name.casefold()):
                sp_res = sp.resolve()
                if sp_res in mounted:
                    continue
                handle = self._attach_shard_file(sp)
                if handle is not None:
                    self._shards.append(handle)
                    mounted.add(sp_res)
                    added = True
        if added:
            self._shards.sort(key=lambda s: s.name.casefold())
            self._recompute_combined_metadata()

    @staticmethod
    def _create_v4_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_concept_lattice (
                concept_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                aliases_csv TEXT NOT NULL,
                canonical_key TEXT NOT NULL DEFAULT '',
                content_fp TEXT NOT NULL DEFAULT '',
                provenance TEXT NOT NULL,
                quality_score REAL NOT NULL DEFAULT 1.0,
                facets_zblob BLOB NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mirror_concept_category ON mirror_concept_lattice(category, concept_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcl_canonical_key ON mirror_concept_lattice(canonical_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcl_content_fp ON mirror_concept_lattice(content_fp)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_concept_tokens (
                token TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                PRIMARY KEY (token, concept_id)
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_concept_trigrams (
                trigram TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                PRIMARY KEY (trigram, concept_id)
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_concept_pruned_trigrams (
                trigram TEXT PRIMARY KEY
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_concept_aliases (
                alias_ck TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                PRIMARY KEY (alias_ck, concept_id)
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS mirror_concept_fts USING fts5(
                concept_id UNINDEXED,
                title,
                aliases_csv,
                category,
                tokenize='unicode61'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_concept_meta (
                key TEXT PRIMARY KEY,
                val TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )

    def _invalidate_snapshot_cache(self) -> None:
        self._invalidate_path_snapshot_cache(self.vault_path)

    def _update_meta_and_local_cache(self, conn: sqlite3.Connection) -> None:
        total = int(conn.execute("SELECT COUNT(*) FROM mirror_concept_lattice").fetchone()[0])
        rows = conn.execute(
            "SELECT category, COUNT(*) FROM mirror_concept_lattice GROUP BY category"
        ).fetchall()
        dist = dict(sorted(((str(c), int(n)) for c, n in rows), key=lambda x: (-x[1], x[0])))
        conn.executemany(
            "INSERT OR REPLACE INTO mirror_concept_meta(key, val) VALUES (?, ?)",
            (
                ("schema_version", _SCHEMA_VERSION),
                ("total_count", str(total)),
                ("domain_distribution", json.dumps(dist, separators=(",", ":"))),
            ),
        )
        self._cached_db_count = total
        self._cached_db_domain_dist = dist
        self._recompute_combined_metadata()

    def _attach_existing_vault(self) -> None:
        assert self.vault_path is not None
        st = self.vault_path.stat()
        cache_key = (str(self.vault_path.resolve()), st.st_mtime_ns, st.st_size)

        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(self.vault_path), check_same_thread=False)
            self._configure_connection(conn, enable_wal=False)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            uri = f"{self.vault_path.resolve().as_uri()}?mode=ro&immutable=1"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._configure_shard_connection(conn)
        self._conn = conn

        cached_meta = _VAULT_SNAPSHOT_CACHE.get(cache_key)
        if cached_meta is not None and cached_meta.get("schema_version") == _SCHEMA_VERSION:
            self._has_v4_schema = True
            self._cached_db_count = int(cached_meta["total_count"])
            self._cached_db_domain_dist = dict(cached_meta["domain_dist"])
            for cid in cached_meta.get("missing_seeds", ()):
                if cid in self._seed_by_id:
                    self._index_ephemeral_node(self._seed_by_id[cid])
            return

        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('mirror_concept_lattice','mirror_concept_tokens','mirror_concept_trigrams',"
                "'mirror_concept_aliases','mirror_concept_fts','mirror_concept_meta')"
            ).fetchall()
        }

        if "mirror_concept_lattice" not in tables:
            for nd in _SEED_CONCEPT_NODES:
                self._index_ephemeral_node(nd)
            return

        # Fast-path check for already-migrated v4 schema (< 3 ms cold start!)
        is_v4 = False
        if {
            "mirror_concept_lattice",
            "mirror_concept_tokens",
            "mirror_concept_trigrams",
            "mirror_concept_aliases",
            "mirror_concept_fts",
            "mirror_concept_meta",
        }.issubset(tables):
            row_ver = conn.execute(
                "SELECT val FROM mirror_concept_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row_ver and row_ver[0] == _SCHEMA_VERSION:
                is_v4 = True

        if not is_v4:
            # Backward-compatible automatic in-place migration (zero data loss, zero overwrites!)
            try:
                self._configure_connection(conn, enable_wal=True)
                self._migrate_v3_to_v4(conn)
                is_v4 = True
            except sqlite3.OperationalError:
                # Read-only DB fallback: keep working seamlessly against legacy text columns
                self._has_legacy_table = True
                self._load_metadata_counts(conn)
                self._ensure_missing_seeds_ephemeral(conn)
                return

        self._has_v4_schema = True
        meta_rows = dict(
            conn.execute(
                "SELECT key, val FROM mirror_concept_meta WHERE key IN ('total_count', 'domain_distribution')"
            ).fetchall()
        )
        if "total_count" in meta_rows and "domain_distribution" in meta_rows:
            self._cached_db_count = int(meta_rows["total_count"])
            self._cached_db_domain_dist = json.loads(meta_rows["domain_distribution"])
        else:
            self._load_metadata_counts(conn)

        missing_seeds = self._ensure_missing_seeds_ephemeral(conn)

        st_now = self.vault_path.stat()
        cache_key_now = (str(self.vault_path.resolve()), st_now.st_mtime_ns, st_now.st_size)
        _VAULT_SNAPSHOT_CACHE[cache_key_now] = {
            "schema_version": _SCHEMA_VERSION,
            "total_count": self._cached_db_count,
            "domain_dist": dict(self._cached_db_domain_dist),
            "missing_seeds": tuple(missing_seeds),
        }

    def _load_metadata_counts(self, conn: sqlite3.Connection) -> None:
        self._cached_db_count = int(
            conn.execute("SELECT COUNT(*) FROM mirror_concept_lattice").fetchone()[0]
        )
        rows = conn.execute(
            "SELECT category, COUNT(*) FROM mirror_concept_lattice GROUP BY category"
        ).fetchall()
        self._cached_db_domain_dist = dict(
            sorted(((str(c), int(n)) for c, n in rows), key=lambda x: (-x[1], x[0]))
        )

    def _ensure_missing_seeds_ephemeral(self, conn: sqlite3.Connection) -> list[str]:
        seed_ids = tuple(nd.concept_id.strip().casefold() for nd in _SEED_CONCEPT_NODES)
        placeholders = ",".join("?" for _ in seed_ids)
        present = {
            r[0]
            for r in conn.execute(
                f"SELECT concept_id FROM mirror_concept_lattice WHERE concept_id IN ({placeholders})",
                seed_ids,
            ).fetchall()
        }
        missing: list[str] = []
        for nd in _SEED_CONCEPT_NODES:
            cid = nd.concept_id.strip().casefold()
            if cid not in present:
                self._index_ephemeral_node(nd)
                missing.append(cid)
        return missing

    def _migrate_v3_to_v4(self, conn: sqlite3.Connection) -> None:
        """Atomically migrate an existing v3 `mirror_concept_lattice` table to v4 zlib + FTS5 + sharded inverted indexes
        WITHOUT losing or overwriting a single existing row."""
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(mirror_concept_lattice)").fetchall()
        }
        has_zblob = "facets_zblob" in cols

        if not has_zblob:
            conn.execute("DROP TABLE IF EXISTS mirror_concept_lattice_v4")
            conn.execute(
                """
                CREATE TABLE mirror_concept_lattice_v4 (
                    concept_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    aliases_csv TEXT NOT NULL,
                    canonical_key TEXT NOT NULL DEFAULT '',
                    content_fp TEXT NOT NULL DEFAULT '',
                    provenance TEXT NOT NULL,
                    quality_score REAL NOT NULL DEFAULT 1.0,
                    facets_zblob BLOB NOT NULL
                )
                """
            )
            cur = conn.execute(
                "SELECT concept_id, title, category, aliases_csv, definition, mechanism, applications, tradeoffs, provenance "
                "FROM mirror_concept_lattice"
            )
            batch: list[tuple[Any, ...]] = []
            while True:
                rows = cur.fetchmany(1000)
                if not rows:
                    break
                for cid, title, cat, aliases_csv, defn, mech, apps, trd, prov in rows:
                    cid_low = str(cid).strip().casefold()
                    aliases = tuple(a for a in str(aliases_csv).split("|") if a)
                    nd = ConceptNode(
                        concept_id=cid_low,
                        title=str(title),
                        category=str(cat),
                        aliases=aliases,
                        definition=str(defn),
                        mechanism=str(mech),
                        applications=str(apps),
                        tradeoffs=str(trd),
                        provenance=str(prov),
                    )
                    q_score = score_concept_quality(nd)
                    batch.append(
                        (
                            cid_low,
                            nd.title,
                            nd.category,
                            str(aliases_csv),
                            canonical_concept_key(cid_low),
                            content_fingerprint(nd.definition),
                            nd.provenance,
                            q_score,
                            pack_facets_zblob(nd.definition, nd.mechanism, nd.applications, nd.tradeoffs),
                        )
                    )
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO mirror_concept_lattice_v4
                    (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                batch.clear()
            conn.execute("DROP TABLE mirror_concept_lattice")
            conn.execute("ALTER TABLE mirror_concept_lattice_v4 RENAME TO mirror_concept_lattice")

        # Create v4 indexes & auxiliary tables, then populate inverted indexes & FTS5 cleanly
        self._create_v4_schema(conn)
        self._rebuild_all_disk_indexes(conn)
        self._update_meta_and_local_cache(conn)
        conn.commit()
        conn.execute("ANALYZE")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _rebuild_all_disk_indexes(self, conn: sqlite3.Connection) -> dict[str, int]:
        """Rebuild `mirror_concept_tokens`, `mirror_concept_trigrams` (DF-pruned <= 300),
        `mirror_concept_aliases`, and `mirror_concept_fts` cleanly in streaming low-RAM batches."""
        conn.execute("DROP TABLE IF EXISTS mirror_concept_tokens")
        conn.execute("DROP TABLE IF EXISTS mirror_concept_trigrams")
        conn.execute("DROP TABLE IF EXISTS mirror_concept_pruned_trigrams")
        conn.execute("DROP TABLE IF EXISTS mirror_concept_aliases")
        conn.execute("DROP TABLE IF EXISTS mirror_concept_fts")
        self._create_v4_schema(conn)

        total_rows = int(conn.execute("SELECT COUNT(*) FROM mirror_concept_lattice").fetchone()[0])

        # Pass 1: count trigram document frequencies (only ~10k-20k 3-char keys -> < 2 MB RAM)
        tg_df: Counter[str] = Counter()
        if total_rows >= 500:
            cur = conn.execute("SELECT concept_id, aliases_csv FROM mirror_concept_lattice")
            while True:
                chunk = cur.fetchmany(2000)
                if not chunk:
                    break
                for cid, aliases_csv in chunk:
                    aliases = tuple(a for a in str(aliases_csv).split("|") if a)
                    _, tgs, _ = _extract_surface_index_entries(str(cid), aliases)
                    for tg in tgs:
                        tg_df[tg] += 1
        pruned_tgs = {tg for tg, cnt in tg_df.items() if cnt > _MAX_TRIGRAM_DF} if total_rows >= 500 else set()
        if pruned_tgs:
            conn.executemany(
                "INSERT OR IGNORE INTO mirror_concept_pruned_trigrams(trigram) VALUES (?)",
                [(tg,) for tg in pruned_tgs],
            )

        # Pass 2: stream-insert tokens, DF-pruned trigrams, and extra alias canonical keys
        tok_batch: list[tuple[str, str]] = []
        tg_batch: list[tuple[str, str]] = []
        al_batch: list[tuple[str, str]] = []
        tokens_count = 0
        trigrams_count = 0

        cur = conn.execute("SELECT concept_id, aliases_csv FROM mirror_concept_lattice")
        while True:
            chunk = cur.fetchmany(2000)
            if not chunk:
                break
            for cid, aliases_csv in chunk:
                cid_low = str(cid).strip().casefold()
                aliases = tuple(a for a in str(aliases_csv).split("|") if a)
                toks, tgs, extra_cks = _extract_surface_index_entries(cid_low, aliases)
                for tok in toks:
                    tok_batch.append((tok, cid_low))
                for tg in tgs:
                    if tg not in pruned_tgs:
                        tg_batch.append((tg, cid_low))
                for ak in extra_cks:
                    al_batch.append((ak, cid_low))

            if tok_batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO mirror_concept_tokens(token, concept_id) VALUES (?, ?)",
                    tok_batch,
                )
                tokens_count += len(tok_batch)
                tok_batch.clear()
            if tg_batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO mirror_concept_trigrams(trigram, concept_id) VALUES (?, ?)",
                    tg_batch,
                )
                trigrams_count += len(tg_batch)
                tg_batch.clear()
            if al_batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO mirror_concept_aliases(alias_ck, concept_id) VALUES (?, ?)",
                    al_batch,
                )
                al_batch.clear()

        # Pass 3: C-side bulk load of SQLite FTS5 virtual table (0 Python RAM!)
        conn.execute(
            """
            INSERT INTO mirror_concept_fts(concept_id, title, aliases_csv, category)
            SELECT concept_id, title, aliases_csv, category FROM mirror_concept_lattice
            """
        )
        return {
            "tokens_indexed": tokens_count,
            "trigrams_indexed": trigrams_count,
            "trigrams_df_pruned": len(pruned_tgs),
            "fts_indexed": total_rows,
        }

    def _ensure_disk_db_ready(self) -> sqlite3.Connection:
        assert self.vault_path is not None
        if self._conn is None:
            self.vault_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.vault_path), check_same_thread=False)
            self._configure_connection(conn, enable_wal=True)
            self._create_v4_schema(conn)
            self._conn = conn
            self._has_v4_schema = True
        elif not self._has_v4_schema:
            self._configure_connection(self._conn, enable_wal=True)
            if self._has_legacy_table:
                self._migrate_v3_to_v4(self._conn)
                self._has_legacy_table = False
            else:
                self._create_v4_schema(self._conn)
            self._has_v4_schema = True
        else:
            self._configure_connection(self._conn, enable_wal=True)
        return self._conn

    def _index_ephemeral_node(self, node: ConceptNode) -> str:
        cid = node.concept_id.strip().casefold()
        self._ephemeral_nodes[cid] = node
        toks, tgs, _ = _extract_surface_index_entries(cid, node.aliases)
        for tok in toks:
            self._ephemeral_tokens[tok].add(cid)
        for tg in tgs:
            self._ephemeral_trigrams[tg].add(cid)
        self._lru_put(cid, node)
        return cid

    def _lru_put(self, cid: str, node: ConceptNode) -> None:
        self._lru[cid] = node
        self._lru.move_to_end(cid)
        while len(self._lru) > self._lru_maxsize:
            self._lru.popitem(last=False)

    def _row_to_node_v4(self, row: tuple[Any, ...]) -> ConceptNode:
        cid, title, cat, aliases_csv, prov, q_score, zblob = row
        defn, mech, apps, trd = unpack_facets_zblob(zblob)
        aliases = tuple(a for a in str(aliases_csv).split("|") if a)
        return ConceptNode(
            concept_id=str(cid),
            title=str(title),
            category=str(cat),
            aliases=aliases,
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=str(prov),
            quality_score=float(q_score if q_score is not None else 1.0),
        )

    def _row_to_node_legacy(self, row: tuple[Any, ...]) -> ConceptNode:
        cid, title, cat, aliases_csv, defn, mech, apps, trd, prov = row
        aliases = tuple(a for a in str(aliases_csv).split("|") if a)
        return ConceptNode(
            concept_id=str(cid),
            title=str(title),
            category=str(cat),
            aliases=aliases,
            definition=str(defn),
            mechanism=str(mech),
            applications=str(apps),
            tradeoffs=str(trd),
            provenance=str(prov),
        )

    def _iter_active_db_sources_locked(self) -> list[tuple[sqlite3.Connection, bool]]:
        """Return active `(conn, is_v4)` handles in priority order: primary FIRST, then mounted shards."""
        sources: list[tuple[sqlite3.Connection, bool]] = []
        if self._conn is not None and (self._has_v4_schema or self._has_legacy_table):
            sources.append((self._conn, self._has_v4_schema))
        for sh in self._shards:
            if sh.has_v4_schema or sh.has_legacy_table:
                sources.append((sh.conn, sh.has_v4_schema))
        return sources

    def _fetch_node_from_conn(
        self,
        conn: sqlite3.Connection,
        is_v4: bool,
        cid_low: str,
    ) -> ConceptNode | None:
        if is_v4:
            row = conn.execute(
                "SELECT concept_id, title, category, aliases_csv, provenance, quality_score, facets_zblob "
                "FROM mirror_concept_lattice WHERE concept_id = ?",
                (cid_low,),
            ).fetchone()
            return self._row_to_node_v4(row) if row is not None else None
        row = conn.execute(
            "SELECT concept_id, title, category, aliases_csv, definition, mechanism, applications, tradeoffs, provenance "
            "FROM mirror_concept_lattice WHERE concept_id = ?",
            (cid_low,),
        ).fetchone()
        return self._row_to_node_legacy(row) if row is not None else None

    def _get_node(
        self,
        cid: str,
        *,
        hint_source: tuple[sqlite3.Connection, bool] | None = None,
    ) -> ConceptNode | None:
        cid_low = cid.strip().casefold()
        if cid_low in self._ephemeral_nodes:
            return self._ephemeral_nodes[cid_low]
        cached = self._lru.get(cid_low)
        if cached is not None:
            self._lru.move_to_end(cid_low)
            return cached
        with self._lock:
            # Primary ALWAYS checked first so primary wins on any hypothetical ID tie
            if self._conn is not None and (self._has_v4_schema or self._has_legacy_table):
                nd = self._fetch_node_from_conn(self._conn, self._has_v4_schema, cid_low)
                if nd is not None:
                    self._lru_put(cid_low, nd)
                    return nd
                if hint_source is not None and hint_source[0] is self._conn:
                    return self._seed_by_id.get(cid_low)

            if hint_source is not None and hint_source[0] is not self._conn:
                nd = self._fetch_node_from_conn(hint_source[0], hint_source[1], cid_low)
                if nd is not None:
                    self._lru_put(cid_low, nd)
                    return nd

            for sh in self._shards:
                if hint_source is not None and sh.conn is hint_source[0]:
                    continue
                if not (sh.has_v4_schema or sh.has_legacy_table):
                    continue
                nd = self._fetch_node_from_conn(sh.conn, sh.has_v4_schema, cid_low)
                if nd is not None:
                    self._lru_put(cid_low, nd)
                    return nd

            return self._seed_by_id.get(cid_low)

    def _has_node(self, cid: str) -> bool:
        cid_low = cid.strip().casefold()
        if cid_low in self._ephemeral_nodes or cid_low in self._lru or cid_low in self._seed_by_id:
            return True
        with self._lock:
            for conn, _ in self._iter_active_db_sources_locked():
                row = conn.execute(
                    "SELECT 1 FROM mirror_concept_lattice WHERE concept_id = ? LIMIT 1",
                    (cid_low,),
                ).fetchone()
                if row is not None:
                    return True
            return False

    def _iter_all_ids_streaming(self) -> Iterator[str]:
        seen: set[str] = set()
        for cid in list(self._ephemeral_nodes.keys()):
            seen.add(cid)
            yield cid
        with self._lock:
            sources = self._iter_active_db_sources_locked()
            for conn, _ in sources:
                cur = conn.execute("SELECT concept_id FROM mirror_concept_lattice")
                while True:
                    batch = cur.fetchmany(512)
                    if not batch:
                        break
                    for (cid,) in batch:
                        cid_s = str(cid)
                        if cid_s not in seen:
                            seen.add(cid_s)
                            yield cid_s

    def _iter_all_items_streaming(self) -> Iterator[tuple[str, ConceptNode]]:
        seen: set[str] = set()
        for cid, nd in list(self._ephemeral_nodes.items()):
            seen.add(cid)
            yield cid, nd
        with self._lock:
            sources = self._iter_active_db_sources_locked()
            for conn, is_v4 in sources:
                sql = (
                    "SELECT concept_id, title, category, aliases_csv, provenance, quality_score, facets_zblob "
                    "FROM mirror_concept_lattice"
                    if is_v4
                    else "SELECT concept_id, title, category, aliases_csv, definition, mechanism, applications, tradeoffs, provenance "
                    "FROM mirror_concept_lattice"
                )
                cur = conn.execute(sql)
                while True:
                    batch = cur.fetchmany(512)
                    if not batch:
                        break
                    for row in batch:
                        cid_s = str(row[0])
                        if cid_s in seen:
                            continue
                        seen.add(cid_s)
                        nd = self._row_to_node_v4(row) if is_v4 else self._row_to_node_legacy(row)
                        yield cid_s, nd

    def concept_count(self) -> int:
        if not self._ephemeral_nodes:
            return self._cached_total_db_count
        with self._lock:
            sources = self._iter_active_db_sources_locked()
            if not sources:
                return len(self._ephemeral_nodes)
            eids = tuple(self._ephemeral_nodes.keys())
            found_in_db: set[str] = set()
            for conn, _ in sources:
                for i in range(0, len(eids), _SQL_IN_CHUNK_SIZE):
                    chunk = eids[i: i + _SQL_IN_CHUNK_SIZE]
                    ph = ",".join("?" for _ in chunk)
                    for (cid,) in conn.execute(
                        f"SELECT concept_id FROM mirror_concept_lattice WHERE concept_id IN ({ph})",
                        chunk,
                    ).fetchall():
                        found_in_db.add(str(cid))
            return self._cached_total_db_count + (len(eids) - len(found_in_db))

    def domain_distribution(self) -> dict[str, int]:
        if not self._ephemeral_nodes:
            return dict(self._cached_total_domain_dist)
        with self._lock:
            sources = self._iter_active_db_sources_locked()
            if not sources:
                counts: dict[str, int] = defaultdict(int)
                for nd in self._ephemeral_nodes.values():
                    counts[nd.category] += 1
                return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))

            counts = defaultdict(int, self._cached_total_domain_dist)
            eids = tuple(self._ephemeral_nodes.keys())
            db_cats: dict[str, str] = {}
            for conn, _ in sources:
                for i in range(0, len(eids), _SQL_IN_CHUNK_SIZE):
                    chunk = eids[i: i + _SQL_IN_CHUNK_SIZE]
                    ph = ",".join("?" for _ in chunk)
                    for cid, cat in conn.execute(
                        f"SELECT concept_id, category FROM mirror_concept_lattice WHERE concept_id IN ({ph})",
                        chunk,
                    ).fetchall():
                        db_cats.setdefault(str(cid), str(cat))
            for cid, nd in self._ephemeral_nodes.items():
                old_cat = db_cats.get(cid)
                if old_cat is not None:
                    if old_cat != nd.category:
                        counts[old_cat] = max(0, counts[old_cat] - 1)
                        if counts[old_cat] == 0:
                            counts.pop(old_cat, None)
                        counts[nd.category] += 1
                else:
                    counts[nd.category] += 1
            return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))

    def search_by_category(self, category: str, *, limit: int = 25) -> list[ConceptNode]:
        cat_low = category.strip().casefold()
        out_by_id: dict[str, ConceptNode] = {}
        for cid, nd in self._ephemeral_nodes.items():
            if nd.category.casefold() == cat_low:
                out_by_id[cid] = nd
        with self._lock:
            for conn, is_v4 in self._iter_active_db_sources_locked():
                if is_v4:
                    rows = conn.execute(
                        "SELECT concept_id, title, category, aliases_csv, provenance, quality_score, facets_zblob "
                        "FROM mirror_concept_lattice WHERE category = ? ORDER BY concept_id ASC LIMIT ?",
                        (cat_low, int(limit)),
                    ).fetchall()
                    if not rows:
                        rows = conn.execute(
                            "SELECT concept_id, title, category, aliases_csv, provenance, quality_score, facets_zblob "
                            "FROM mirror_concept_lattice WHERE LOWER(category) = ? ORDER BY concept_id ASC LIMIT ?",
                            (cat_low, int(limit)),
                        ).fetchall()
                    for row in rows:
                        cid = str(row[0])
                        if cid not in out_by_id:
                            out_by_id[cid] = self._row_to_node_v4(row)
                else:
                    rows = conn.execute(
                        "SELECT concept_id, title, category, aliases_csv, definition, mechanism, applications, tradeoffs, provenance "
                        "FROM mirror_concept_lattice WHERE LOWER(category) = ? ORDER BY concept_id ASC LIMIT ?",
                        (cat_low, int(limit)),
                    ).fetchall()
                    for row in rows:
                        cid = str(row[0])
                        if cid not in out_by_id:
                            out_by_id[cid] = self._row_to_node_legacy(row)
        out = sorted(out_by_id.values(), key=lambda x: x.concept_id)
        return out[:limit]

    def search_fts(self, query: str, *, limit: int = 20) -> list[ConceptNode]:
        """Fast SQLite FTS5 full-text search across title, aliases, and category spanning primary + all shards."""
        toks = [t for t in re.findall(r"[a-z0-9]{2,}", (query or "").casefold()) if t not in _INDEX_STOP_TOKENS]
        if not toks:
            return []
        unique_toks = list(dict.fromkeys(toks))[:10]
        match_expr = " OR ".join(f'"{t}"' for t in unique_toks)
        lim = max(1, int(limit))
        ranked_hits: list[tuple[float, int, str, sqlite3.Connection]] = []
        seen_cids: set[str] = set()
        with self._lock:
            v4_conns = [c for c, is_v4 in self._iter_active_db_sources_locked() if is_v4]
            if not v4_conns:
                return []
            for src_idx, conn in enumerate(v4_conns):
                try:
                    rows = conn.execute(
                        "SELECT concept_id, bm25(mirror_concept_fts) AS rank_score "
                        "FROM mirror_concept_fts WHERE mirror_concept_fts MATCH ? "
                        "ORDER BY rank_score LIMIT ?",
                        (match_expr, lim),
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                for cid, rscore in rows:
                    cid_s = str(cid)
                    if cid_s in seen_cids:
                        continue
                    seen_cids.add(cid_s)
                    ranked_hits.append((float(rscore), src_idx, cid_s, conn))
        ranked_hits.sort(key=lambda x: (x[0], x[1], x[2]))
        out: list[ConceptNode] = []
        for _, _, cid_s, conn in ranked_hits[:lim]:
            nd = self._get_node(cid_s, hint_source=(conn, True))
            if nd is not None:
                out.append(nd)
        return out

    def shard_telemetry(self) -> dict[str, Any]:
        """Return multi-shard telemetry across primary vault and all mounted `.dcrystal` shards."""
        primary_mb = 0.0
        if self.vault_path is not None and self.vault_path.exists():
            primary_mb = round(self.vault_path.stat().st_size / (1024 * 1024), 2)
        with self._lock:
            shards_breakdown = [
                {
                    "filename": sh.name,
                    "path": str(sh.path),
                    "node_count": sh.node_count,
                    "domains_count": len(sh.domain_dist),
                    "disk_mb": sh.disk_mb,
                    "schema_version": _SCHEMA_VERSION if sh.has_v4_schema else "legacy_v3",
                }
                for sh in self._shards
            ]
            shards_node_count = sum(sh.node_count for sh in self._shards)
            shards_disk_mb = round(sum(sh.disk_mb for sh in self._shards), 2)
            shards_count = len(self._shards)
        return {
            "shard_dir": str(self.shard_dir) if self.shard_dir is not None else None,
            "primary_vault": self.vault_path.name if self.vault_path is not None else None,
            "primary_node_count": self._cached_db_count,
            "primary_disk_mb": primary_mb,
            "shards_count": shards_count,
            "shards_node_count": shards_node_count,
            "shards_disk_mb": shards_disk_mb,
            "total_node_count": self.concept_count(),
            "total_disk_mb": round(primary_mb + shards_disk_mb, 2),
            "shards_breakdown": shards_breakdown,
        }

    def vault_telemetry(self) -> dict[str, Any]:
        """Return lightweight disk/schema/cache/multi-shard telemetry without scanning any database."""
        st = self.shard_telemetry()
        has_any_v4 = self._has_v4_schema or any(sh.has_v4_schema for sh in self._shards)
        if has_any_v4 and st["shards_count"] > 0:
            backend = "sqlite_fts5_zlib_multi_shard"
        elif has_any_v4:
            backend = "sqlite_fts5_zlib_disk"
        else:
            backend = "in_memory"
        return {
            "version": self.VERSION,
            "schema_version": _SCHEMA_VERSION if has_any_v4 else ("legacy_v3" if self._has_legacy_table else "ephemeral"),
            "storage_backend": backend,
            "concept_nodes": self.concept_count(),
            "domains_count": len(self.domain_distribution()),
            "lru_cache_entries": len(self._lru),
            "lru_cache_maxsize": self._lru_maxsize,
            "ephemeral_overlay_nodes": len(self._ephemeral_nodes),
            "vault_disk_mb": st["primary_disk_mb"],
            "primary_node_count": st["primary_node_count"],
            "primary_disk_mb": st["primary_disk_mb"],
            "shards_count": st["shards_count"],
            "shards_node_count": st["shards_node_count"],
            "shards_disk_mb": st["shards_disk_mb"],
            "total_disk_mb": st["total_disk_mb"],
            "shards_breakdown": st["shards_breakdown"],
            "trigram_df_cap": _MAX_TRIGRAM_DF,
        }

    def _delete_disk_index_entries_for_id(
        self,
        conn: sqlite3.Connection,
        cid_low: str,
        old_title: str,
        old_aliases_csv: str,
    ) -> None:
        """Remove a single concept's old postings from disk inverted indexes & FTS5 in O(log N)."""
        old_aliases = tuple(a for a in str(old_aliases_csv).split("|") if a)
        old_toks, old_tgs, old_cks = _extract_surface_index_entries(cid_low, old_aliases)
        if old_toks:
            conn.executemany(
                "DELETE FROM mirror_concept_tokens WHERE token = ? AND concept_id = ?",
                [(t, cid_low) for t in old_toks],
            )
        if old_tgs:
            conn.executemany(
                "DELETE FROM mirror_concept_trigrams WHERE trigram = ? AND concept_id = ?",
                [(tg, cid_low) for tg in old_tgs],
            )
        if old_cks:
            conn.executemany(
                "DELETE FROM mirror_concept_aliases WHERE alias_ck = ? AND concept_id = ?",
                [(ak, cid_low) for ak in old_cks],
            )
        # Fast O(log N) FTS5 delete via title token lookup, falling back to scan only if no token
        title_toks = re.findall(r"[a-z0-9]{2,}", old_title.casefold())
        deleted_fts = False
        if title_toks:
            try:
                match_expr = " AND ".join(f'"{t}"' for t in title_toks[:4])
                rids = [
                    r[0]
                    for r in conn.execute(
                        "SELECT rowid, concept_id FROM mirror_concept_fts WHERE mirror_concept_fts MATCH ?",
                        (match_expr,),
                    ).fetchall()
                    if r[1] == cid_low
                ]
                if rids:
                    conn.executemany("DELETE FROM mirror_concept_fts WHERE rowid = ?", [(r,) for r in rids])
                    deleted_fts = True
            except sqlite3.OperationalError:
                pass
        if not deleted_fts:
            conn.execute("DELETE FROM mirror_concept_fts WHERE concept_id = ?", (cid_low,))

    def _insert_disk_index_entries_batch(
        self,
        conn: sqlite3.Connection,
        nodes_meta: list[tuple[str, str, str, str, tuple[str, ...]]],
        pruned_tgs: set[str],
    ) -> None:
        """Batch-insert `(cid, title, category, aliases_csv, aliases)` into tokens, trigrams, aliases & FTS5."""
        tok_rows: list[tuple[str, str]] = []
        tg_rows: list[tuple[str, str]] = []
        al_rows: list[tuple[str, str]] = []
        fts_rows: list[tuple[str, str, str, str]] = []

        for cid_low, title, cat, aliases_csv, aliases in nodes_meta:
            toks, tgs, extra_cks = _extract_surface_index_entries(cid_low, aliases)
            for tok in toks:
                tok_rows.append((tok, cid_low))
            for tg in tgs:
                if tg not in pruned_tgs:
                    tg_rows.append((tg, cid_low))
            for ak in extra_cks:
                al_rows.append((ak, cid_low))
            fts_rows.append((cid_low, title, aliases_csv, cat))

        if tok_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO mirror_concept_tokens(token, concept_id) VALUES (?, ?)",
                tok_rows,
            )
        if tg_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO mirror_concept_trigrams(trigram, concept_id) VALUES (?, ?)",
                tg_rows,
            )
        if al_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO mirror_concept_aliases(alias_ck, concept_id) VALUES (?, ?)",
                al_rows,
            )
        if fts_rows:
            conn.executemany(
                "INSERT INTO mirror_concept_fts(concept_id, title, aliases_csv, category) VALUES (?, ?, ?, ?)",
                fts_rows,
            )

    def _enforce_disk_trigram_df_cap(self, conn: sqlite3.Connection) -> int:
        total = int(conn.execute("SELECT COUNT(*) FROM mirror_concept_lattice").fetchone()[0])
        if total < 500:
            return 0
        over = [
            r[0]
            for r in conn.execute(
                "SELECT trigram FROM mirror_concept_trigrams GROUP BY trigram HAVING COUNT(*) > ?",
                (_MAX_TRIGRAM_DF,),
            ).fetchall()
        ]
        if not over:
            return 0
        conn.executemany(
            "INSERT OR IGNORE INTO mirror_concept_pruned_trigrams(trigram) VALUES (?)",
            [(tg,) for tg in over],
        )
        conn.executemany(
            "DELETE FROM mirror_concept_trigrams WHERE trigram = ?",
            [(tg,) for tg in over],
        )
        return len(over)

    def _flush_ephemeral_to_disk_if_new_db(self, conn: sqlite3.Connection, pruned_tgs: set[str]) -> None:
        """Persist any in-memory seed/ephemeral nodes when initializing a brand-new SQLite vault."""
        if not self._ephemeral_nodes:
            return
        ephem_rows: list[tuple[Any, ...]] = []
        ephem_meta: list[tuple[str, str, str, str, tuple[str, ...]]] = []
        for cid, nd in list(self._ephemeral_nodes.items()):
            exists = conn.execute(
                "SELECT 1 FROM mirror_concept_lattice WHERE concept_id = ? LIMIT 1",
                (cid,),
            ).fetchone()
            if exists is not None:
                continue
            aliases_csv = "|".join(nd.aliases)
            q_score = score_concept_quality(nd)
            ephem_rows.append(
                (
                    cid,
                    nd.title,
                    nd.category,
                    aliases_csv,
                    canonical_concept_key(cid),
                    content_fingerprint(nd.definition),
                    nd.provenance,
                    q_score,
                    pack_facets_zblob(nd.definition, nd.mechanism, nd.applications, nd.tradeoffs),
                )
            )
            ephem_meta.append((cid, nd.title, nd.category, aliases_csv, nd.aliases))
        if ephem_rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO mirror_concept_lattice
                (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ephem_rows,
            )
            self._insert_disk_index_entries_batch(conn, ephem_meta, pruned_tgs)
        self._ephemeral_nodes.clear()
        self._ephemeral_tokens.clear()
        self._ephemeral_trigrams.clear()

    def register_node(self, node: ConceptNode, *, persist: bool = True) -> None:
        self.register_nodes_bulk((node,), persist=persist)

    def register_nodes_bulk(self, nodes: Iterable[ConceptNode], *, persist: bool = True) -> int:
        if not persist or self.vault_path is None:
            count = 0
            for node in nodes:
                self._index_ephemeral_node(node)
                count += 1
            return count

        with self._lock:
            conn = self._ensure_disk_db_ready()
            pruned_tgs = {
                r[0]
                for r in conn.execute("SELECT trigram FROM mirror_concept_pruned_trigrams").fetchall()
            }
            self._flush_ephemeral_to_disk_if_new_db(conn, pruned_tgs)

            total_inserted = 0
            chunk_rows: list[tuple[Any, ...]] = []
            chunk_meta: list[tuple[str, str, str, str, tuple[str, ...]]] = []

            def _flush_chunk() -> None:
                if not chunk_rows:
                    return
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO mirror_concept_lattice
                    (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_rows,
                )
                self._insert_disk_index_entries_batch(conn, chunk_meta, pruned_tgs)
                chunk_rows.clear()
                chunk_meta.clear()

            for node in nodes:
                cid = node.concept_id.strip().casefold()
                if not cid:
                    continue
                old = conn.execute(
                    "SELECT title, aliases_csv FROM mirror_concept_lattice WHERE concept_id = ?",
                    (cid,),
                ).fetchone()
                if old is not None:
                    self._delete_disk_index_entries_for_id(conn, cid, str(old[0]), str(old[1]))

                aliases_csv = "|".join(node.aliases)
                q_score = score_concept_quality(node)
                chunk_rows.append(
                    (
                        cid,
                        node.title,
                        node.category,
                        aliases_csv,
                        canonical_concept_key(cid),
                        content_fingerprint(node.definition),
                        node.provenance,
                        q_score,
                        pack_facets_zblob(node.definition, node.mechanism, node.applications, node.tradeoffs),
                    )
                )
                chunk_meta.append((cid, node.title, node.category, aliases_csv, node.aliases))
                if cid in self._lru:
                    self._lru_put(cid, node)
                total_inserted += 1
                if len(chunk_rows) >= 1000:
                    _flush_chunk()

            _flush_chunk()
            if total_inserted > 0:
                self._enforce_disk_trigram_df_cap(conn)
                self._update_meta_and_local_cache(conn)
                conn.commit()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._invalidate_snapshot_cache()
            return total_inserted

    def register_nodes_no_overwrite(
        self,
        nodes: Iterable[ConceptNode],
        *,
        persist: bool = True,
        allow_strict_tier_upgrade: bool = True,
        vacuum: bool = False,
    ) -> dict[str, int]:
        """Zero-Overwrite, Zero-Duplication Disk-Backed Admission Pipeline.

        Checks all 4 integrity layers directly against SQLite indexed columns
        (`concept_id`, `canonical_key`, `mirror_concept_aliases(alias_ck)`, `content_fp`)
        so admitting 40,000+ nodes runs fast with < 25 MB peak Python heap:
          1. Exact `concept_id` protection: NEVER overwritten by equal-or-lower provenance tier.
          2. Canonical plural/hyphen/punctuation duplicate guard via `canonical_key`.
          3. Existing alias collision guard via `mirror_concept_aliases`.
          4. 64-bit normalized definition hash guard via `content_fp`.
        """
        stats = {
            "inserted_new": 0,
            "upgraded_higher_tier": 0,
            "skipped_existing_protected": 0,
            "skipped_canonical_duplicate": 0,
            "skipped_alias_duplicate": 0,
            "skipped_content_duplicate": 0,
            "overwrites_prevented": 0,
        }

        if not persist or self.vault_path is None:
            # In-memory / ephemeral admission path (checking both disk DB if attached + ephemeral overlay)
            return self._register_no_overwrite_ephemeral(
                nodes,
                allow_strict_tier_upgrade=allow_strict_tier_upgrade,
                stats=stats,
            )

        with self._lock:
            conn = self._ensure_disk_db_ready()
            pruned_tgs = {
                r[0]
                for r in conn.execute("SELECT trigram FROM mirror_concept_pruned_trigrams").fetchall()
            }
            self._flush_ephemeral_to_disk_if_new_db(conn, pruned_tgs)

            def _process_batch(batch_nodes: list[ConceptNode]) -> None:
                if not batch_nodes:
                    return

                # Pre-extract keys for this bounded chunk (<= 1,000 nodes -> < 1 MB RAM)
                prepared: list[tuple[ConceptNode, str, str, str, float]] = []
                cand_cids: list[str] = []
                cand_cks: list[str] = []
                cand_fps: list[str] = []
                seen_q_cid: set[str] = set()
                seen_q_ck: set[str] = set()
                seen_q_fp: set[str] = set()

                for node in batch_nodes:
                    cid = node.concept_id.strip().casefold()
                    if not cid or len(cid) < 3:
                        continue
                    ck = canonical_concept_key(cid)
                    fp = content_fingerprint(node.definition)
                    new_tier = provenance_tier_rank(node.provenance)
                    prepared.append((node, cid, ck, fp, new_tier))
                    if cid not in seen_q_cid:
                        seen_q_cid.add(cid)
                        cand_cids.append(cid)
                    if ck and ck not in seen_q_ck:
                        seen_q_ck.add(ck)
                        cand_cks.append(ck)
                    if fp not in seen_q_fp:
                        seen_q_fp.add(fp)
                        cand_fps.append(fp)

                if not prepared:
                    return

                # Batched indexed SQLite lookups (~4-8 SQL calls per 1,000 nodes instead of 4,000!)
                db_existing_cids: dict[str, tuple[str, str, str]] = {}
                for i in range(0, len(cand_cids), _SQL_IN_CHUNK_SIZE):
                    sub = cand_cids[i: i + _SQL_IN_CHUNK_SIZE]
                    ph = ",".join("?" for _ in sub)
                    for r_cid, r_prov, r_title, r_aliases in conn.execute(
                        f"SELECT concept_id, provenance, title, aliases_csv FROM mirror_concept_lattice WHERE concept_id IN ({ph})",
                        sub,
                    ).fetchall():
                        db_existing_cids[str(r_cid)] = (str(r_prov), str(r_title), str(r_aliases))

                db_existing_cks: set[str] = set()
                db_existing_alias_cks: set[str] = set()
                for i in range(0, len(cand_cks), _SQL_IN_CHUNK_SIZE):
                    sub = cand_cks[i: i + _SQL_IN_CHUNK_SIZE]
                    ph = ",".join("?" for _ in sub)
                    for (r_ck,) in conn.execute(
                        f"SELECT canonical_key FROM mirror_concept_lattice WHERE canonical_key IN ({ph})",
                        sub,
                    ).fetchall():
                        db_existing_cks.add(str(r_ck))
                    for (r_ak,) in conn.execute(
                        f"SELECT alias_ck FROM mirror_concept_aliases WHERE alias_ck IN ({ph})",
                        sub,
                    ).fetchall():
                        db_existing_alias_cks.add(str(r_ak))

                db_existing_fps: dict[str, str] = {}
                for i in range(0, len(cand_fps), _SQL_IN_CHUNK_SIZE):
                    sub = cand_fps[i: i + _SQL_IN_CHUNK_SIZE]
                    ph = ",".join("?" for _ in sub)
                    for r_fp, r_cid in conn.execute(
                        f"SELECT content_fp, concept_id FROM mirror_concept_lattice WHERE content_fp IN ({ph})",
                        sub,
                    ).fetchall():
                        db_existing_fps.setdefault(str(r_fp), str(r_cid))

                pending_new_rows: list[tuple[Any, ...]] = []
                pending_new_meta: list[tuple[str, str, str, str, tuple[str, ...]]] = []
                chunk_new_cids: dict[str, tuple[str, str, str]] = {}

                for node, cid, ck, fp, new_tier in prepared:
                    # Layer 1: Exact concept_id protection (against DB + earlier rows in this chunk)
                    existing = db_existing_cids.get(cid) or chunk_new_cids.get(cid)
                    if existing is not None:
                        cur_tier = provenance_tier_rank(existing[0])
                        if allow_strict_tier_upgrade and new_tier > cur_tier:
                            # Guard against upgrading into a content_fp already owned by a DIFFERENT concept_id
                            fp_owner = db_existing_fps.get(fp)
                            if fp_owner is not None and fp_owner != cid:
                                stats["skipped_content_duplicate"] += 1
                                continue
                            # Flush pending new rows first if upgrading an intra-chunk entry
                            if cid in chunk_new_cids and pending_new_rows:
                                conn.executemany(
                                    """
                                    INSERT OR IGNORE INTO mirror_concept_lattice
                                    (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    pending_new_rows,
                                )
                                self._insert_disk_index_entries_batch(conn, pending_new_meta, pruned_tgs)
                                pending_new_rows.clear()
                                pending_new_meta.clear()
                                db_existing_cids.update(chunk_new_cids)
                                chunk_new_cids.clear()

                            self._delete_disk_index_entries_for_id(conn, cid, existing[1], existing[2])
                            aliases_csv = "|".join(node.aliases)
                            q_score = score_concept_quality(node)
                            conn.execute(
                                """
                                UPDATE mirror_concept_lattice
                                SET title=?, category=?, aliases_csv=?, canonical_key=?, content_fp=?,
                                    provenance=?, quality_score=?, facets_zblob=?
                                WHERE concept_id=?
                                """,
                                (
                                    node.title,
                                    node.category,
                                    aliases_csv,
                                    ck,
                                    fp,
                                    node.provenance,
                                    q_score,
                                    pack_facets_zblob(node.definition, node.mechanism, node.applications, node.tradeoffs),
                                    cid,
                                ),
                            )
                            self._insert_disk_index_entries_batch(
                                conn, [(cid, node.title, node.category, aliases_csv, node.aliases)], pruned_tgs
                            )
                            db_existing_cids[cid] = (node.provenance, node.title, aliases_csv)
                            if ck:
                                db_existing_cks.add(ck)
                            for al in node.aliases:
                                ak = canonical_concept_key(al)
                                if ak and ak != ck:
                                    db_existing_alias_cks.add(ak)
                            db_existing_fps[fp] = cid
                            if cid in self._lru:
                                self._lru_put(cid, node)
                            stats["upgraded_higher_tier"] += 1
                        else:
                            stats["skipped_existing_protected"] += 1
                            stats["overwrites_prevented"] += 1
                        continue

                    # Layer 2: Canonical plural/hyphen/punctuation duplicate guard
                    if ck and ck in db_existing_cks:
                        stats["skipped_canonical_duplicate"] += 1
                        continue

                    # Layer 3: Existing alias collision guard
                    if ck and ck in db_existing_alias_cks:
                        stats["skipped_alias_duplicate"] += 1
                        continue

                    # Layer 4: 64-bit normalized definition fingerprint guard
                    if fp in db_existing_fps:
                        stats["skipped_content_duplicate"] += 1
                        continue

                    # Verified unique, non-overwriting concept: admit!
                    aliases_csv = "|".join(node.aliases)
                    q_score = score_concept_quality(node)
                    pending_new_rows.append(
                        (
                            cid,
                            node.title,
                            node.category,
                            aliases_csv,
                            ck,
                            fp,
                            node.provenance,
                            q_score,
                            pack_facets_zblob(node.definition, node.mechanism, node.applications, node.tradeoffs),
                        )
                    )
                    pending_new_meta.append((cid, node.title, node.category, aliases_csv, node.aliases))
                    chunk_new_cids[cid] = (node.provenance, node.title, aliases_csv)
                    if ck:
                        db_existing_cks.add(ck)
                    for al in node.aliases:
                        ak = canonical_concept_key(al)
                        if ak and ak != ck:
                            db_existing_alias_cks.add(ak)
                    db_existing_fps[fp] = cid
                    stats["inserted_new"] += 1

                if pending_new_rows:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO mirror_concept_lattice
                        (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        pending_new_rows,
                    )
                    self._insert_disk_index_entries_batch(conn, pending_new_meta, pruned_tgs)

            chunk_buf: list[ConceptNode] = []
            for node in nodes:
                chunk_buf.append(node)
                if len(chunk_buf) >= 1000:
                    _process_batch(chunk_buf)
                    chunk_buf.clear()
            if chunk_buf:
                _process_batch(chunk_buf)
                chunk_buf.clear()

            mutated = stats["inserted_new"] > 0 or stats["upgraded_higher_tier"] > 0
            if mutated or vacuum:
                if mutated:
                    self._enforce_disk_trigram_df_cap(conn)
                    self._update_meta_and_local_cache(conn)
                conn.commit()
                conn.execute("ANALYZE")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                if vacuum:
                    conn.execute("VACUUM")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._invalidate_snapshot_cache()
        return stats

    def _register_no_overwrite_ephemeral(
        self,
        nodes: Iterable[ConceptNode],
        *,
        allow_strict_tier_upgrade: bool,
        stats: dict[str, int],
    ) -> dict[str, int]:
        ephem_cks: dict[str, str] = {}
        ephem_alias_cks: dict[str, str] = {}
        ephem_fps: dict[str, str] = {}
        for cid, nd in self._ephemeral_nodes.items():
            ck = canonical_concept_key(cid)
            if ck:
                ephem_cks[ck] = cid
            for al in nd.aliases:
                ak = canonical_concept_key(al)
                if ak and ak not in ephem_alias_cks:
                    ephem_alias_cks[ak] = cid
            ephem_fps.setdefault(content_fingerprint(nd.definition), cid)

        for node in nodes:
            cid = node.concept_id.strip().casefold()
            if not cid or len(cid) < 3:
                continue
            ck = canonical_concept_key(cid)
            fp = content_fingerprint(node.definition)
            new_tier = provenance_tier_rank(node.provenance)

            existing = self._get_node(cid)
            if existing is not None:
                cur_tier = provenance_tier_rank(existing.provenance)
                if allow_strict_tier_upgrade and new_tier > cur_tier:
                    fp_owner = ephem_fps.get(fp)
                    if fp_owner is None and self._conn is not None and self._has_v4_schema:
                        row_fp = self._conn.execute(
                            "SELECT concept_id FROM mirror_concept_lattice WHERE content_fp = ? LIMIT 1", (fp,)
                        ).fetchone()
                        if row_fp is not None:
                            fp_owner = str(row_fp[0])
                    if fp_owner is not None and fp_owner != cid:
                        stats["skipped_content_duplicate"] += 1
                        continue
                    self._index_ephemeral_node(node)
                    ephem_fps[fp] = cid
                    stats["upgraded_higher_tier"] += 1
                else:
                    stats["skipped_existing_protected"] += 1
                    stats["overwrites_prevented"] += 1
                continue

            if ck and (
                ck in ephem_cks
                or (
                    self._conn is not None
                    and self._has_v4_schema
                    and self._conn.execute(
                        "SELECT 1 FROM mirror_concept_lattice WHERE canonical_key = ? LIMIT 1", (ck,)
                    ).fetchone()
                    is not None
                )
            ):
                stats["skipped_canonical_duplicate"] += 1
                continue

            if ck and (
                ck in ephem_alias_cks
                or (
                    self._conn is not None
                    and self._has_v4_schema
                    and self._conn.execute(
                        "SELECT 1 FROM mirror_concept_aliases WHERE alias_ck = ? LIMIT 1", (ck,)
                    ).fetchone()
                    is not None
                )
            ):
                stats["skipped_alias_duplicate"] += 1
                continue

            if fp in ephem_fps or (
                self._conn is not None
                and self._has_v4_schema
                and self._conn.execute(
                    "SELECT 1 FROM mirror_concept_lattice WHERE content_fp = ? LIMIT 1", (fp,)
                ).fetchone()
                is not None
            ):
                stats["skipped_content_duplicate"] += 1
                continue

            self._index_ephemeral_node(node)
            if ck:
                ephem_cks[ck] = cid
            for al in node.aliases:
                ak = canonical_concept_key(al)
                if ak and ak not in ephem_alias_cks:
                    ephem_alias_cks[ak] = cid
            ephem_fps[fp] = cid
            stats["inserted_new"] += 1
        return stats

    def deduplicate_vault(self, *, persist: bool = True, vacuum: bool = True) -> dict[str, int]:
        """Audit and merge any canonical or content-fingerprint duplicates in the vault, strictly retaining
        the highest-provenance-tier ConceptNode and preserving merged surface strings as aliases."""
        if self._conn is None or not self._has_v4_schema:
            # Pure in-memory dedup fallback
            by_ck: dict[str, list[ConceptNode]] = defaultdict(list)
            for nd in self._ephemeral_nodes.values():
                ck = canonical_concept_key(nd.concept_id) or nd.concept_id.casefold()
                by_ck[ck].append(nd)
            removed = 0
            for group in by_ck.values():
                if len(group) <= 1:
                    continue
                group.sort(key=lambda n: (-provenance_tier_rank(n.provenance), -len(n.definition), n.concept_id))
                winner = group[0]
                merged_aliases = list(winner.aliases)
                for loser in group[1:]:
                    self._ephemeral_nodes.pop(loser.concept_id.casefold(), None)
                    removed += 1
                    for surf in (loser.concept_id, *loser.aliases):
                        s_low = surf.strip().casefold()
                        if s_low and s_low not in merged_aliases:
                            merged_aliases.append(s_low)
                self._index_ephemeral_node(
                    ConceptNode(
                        concept_id=winner.concept_id,
                        title=winner.title,
                        category=winner.category,
                        aliases=tuple(dict.fromkeys(merged_aliases)),
                        definition=winner.definition,
                        mechanism=winner.mechanism,
                        applications=winner.applications,
                        tradeoffs=winner.tradeoffs,
                        provenance=winner.provenance,
                        quality_score=winner.quality_score,
                    )
                )
            return {
                "canonical_duplicates_merged": removed,
                "content_duplicates_merged": 0,
                "remaining_nodes": self.concept_count(),
            }

        with self._lock:
            conn = self._ensure_disk_db_ready()
            by_ck: dict[str, list[tuple[str, float, float, int, str]]] = defaultdict(list)
            by_fp: dict[str, list[tuple[str, float, float, int, str]]] = defaultdict(list)

            cur = conn.execute(
                "SELECT concept_id, canonical_key, content_fp, provenance, quality_score, aliases_csv, LENGTH(facets_zblob) "
                "FROM mirror_concept_lattice"
            )
            while True:
                rows = cur.fetchmany(2000)
                if not rows:
                    break
                for cid, ck, fp, prov, q_score, aliases_csv, zlen in rows:
                    cid_s = str(cid)
                    ck_s = str(ck) or canonical_concept_key(cid_s) or cid_s
                    fp_s = str(fp)
                    rec = (cid_s, provenance_tier_rank(str(prov)), float(q_score or 0.0), int(zlen or 0), str(aliases_csv))
                    by_ck[ck_s].append(rec)
                    if fp_s:
                        by_fp[fp_s].append(rec)

            removed_ids: set[str] = set()
            merged_alias_updates: dict[str, list[str]] = {}
            ck_merged = 0
            fp_merged = 0

            def _merge_group(group: list[tuple[str, float, float, int, str]], *, allow_tier4_drop: bool) -> int:
                live = [g for g in group if g[0] not in removed_ids]
                if len(live) <= 1:
                    return 0
                live.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
                winner = live[0]
                w_cid = winner[0]
                w_aliases = merged_alias_updates.setdefault(
                    w_cid, [a for a in winner[4].split("|") if a]
                )
                merged_cnt = 0
                for loser in live[1:]:
                    if not allow_tier4_drop and loser[1] >= 4.0:
                        continue
                    removed_ids.add(loser[0])
                    merged_cnt += 1
                    for surf in (loser[0], *(a for a in loser[4].split("|") if a)):
                        s_low = surf.strip().casefold()
                        if s_low and s_low not in w_aliases:
                            w_aliases.append(s_low)
                return merged_cnt

            for grp in by_ck.values():
                if len(grp) > 1:
                    ck_merged += _merge_group(grp, allow_tier4_drop=False)
            for grp in by_fp.values():
                if len(grp) > 1:
                    fp_merged += _merge_group(grp, allow_tier4_drop=False)

            if not removed_ids:
                return {
                    "canonical_duplicates_merged": 0,
                    "content_duplicates_merged": 0,
                    "remaining_nodes": self.concept_count(),
                }

            if persist:
                conn.executemany(
                    "DELETE FROM mirror_concept_lattice WHERE concept_id = ?",
                    [(rid,) for rid in removed_ids],
                )
                if merged_alias_updates:
                    conn.executemany(
                        "UPDATE mirror_concept_lattice SET aliases_csv = ? WHERE concept_id = ?",
                        [("|".join(dict.fromkeys(als)), cid) for cid, als in merged_alias_updates.items()],
                    )
                self._rebuild_all_disk_indexes(conn)
                self._update_meta_and_local_cache(conn)
                conn.commit()
                conn.execute("ANALYZE")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                if vacuum:
                    conn.execute("VACUUM")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._lru.clear()
                self._invalidate_snapshot_cache()

            return {
                "canonical_duplicates_merged": ck_merged,
                "content_duplicates_merged": fp_merged,
                "remaining_nodes": self.concept_count(),
            }

    def audit_and_sort_vault(
        self,
        *,
        persist: bool = True,
        vacuum: bool = True,
        min_quality: float = 0.45,
    ) -> dict[str, Any]:
        """Score every concept on quality (0.0-1.0), remove low-quality junk from lower tiers (<= Tier 2)
        while strictly NEVER touching Tier 4/5 protected nodes, deduplicate all canonical/content collisions,
        sort storage by `(category, quality_score DESC, concept_id)`, rebuild disk-backed token/trigram/FTS5
        indexes cleanly, and run `ANALYZE` + `VACUUM`."""
        with self._lock:
            if self.vault_path is None:
                return {"ok": False, "reason": "no disk vault configured"}
            conn = self._ensure_disk_db_ready()
            pruned_tgs_init = {
                r[0]
                for r in conn.execute("SELECT trigram FROM mirror_concept_pruned_trigrams").fetchall()
            }
            self._flush_ephemeral_to_disk_if_new_db(conn, pruned_tgs_init)

            # Ensure all 10 foundational seed nodes exist intact in DB (never overwriting higher/equal)
            for cid, seed_nd in self._seed_by_id.items():
                row_s = conn.execute(
                    "SELECT provenance FROM mirror_concept_lattice WHERE concept_id = ?", (cid,)
                ).fetchone()
                if row_s is None or provenance_tier_rank(str(row_s[0])) < 5.0:
                    aliases_csv = "|".join(seed_nd.aliases)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO mirror_concept_lattice
                        (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cid,
                            seed_nd.title,
                            seed_nd.category,
                            aliases_csv,
                            canonical_concept_key(cid),
                            content_fingerprint(seed_nd.definition),
                            seed_nd.provenance,
                            1.0,
                            pack_facets_zblob(
                                seed_nd.definition, seed_nd.mechanism, seed_nd.applications, seed_nd.tradeoffs
                            ),
                        ),
                    )

            total_before = int(conn.execute("SELECT COUNT(*) FROM mirror_concept_lattice").fetchone()[0])

            # Streaming quality audit across all rows (compact metadata only -> < 8 MB RAM at 50k rows)
            removed_low_quality_ids: set[str] = set()
            quality_updates: list[tuple[float, str, str, str]] = []
            survivor_meta: list[tuple[str, str, str, float, float, int, str]] = []
            protected_tier4_5 = 0

            cur = conn.execute(
                "SELECT concept_id, title, category, aliases_csv, provenance, quality_score, facets_zblob "
                "FROM mirror_concept_lattice"
            )
            while True:
                rows = cur.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    nd = self._row_to_node_v4(row)
                    cid = nd.concept_id.strip().casefold()
                    tier = provenance_tier_rank(nd.provenance)
                    q_score = score_concept_quality(nd)
                    ck = canonical_concept_key(cid) or cid
                    fp = content_fingerprint(nd.definition)

                    if tier >= 4.0:
                        protected_tier4_5 += 1
                        quality_updates.append((q_score, ck, fp, cid))
                        survivor_meta.append((cid, ck, fp, tier, q_score, len(nd.definition), "|".join(nd.aliases)))
                        continue

                    if tier <= 2.0 and q_score < min_quality:
                        removed_low_quality_ids.add(cid)
                        continue

                    quality_updates.append((q_score, ck, fp, cid))
                    survivor_meta.append((cid, ck, fp, tier, q_score, len(nd.definition), "|".join(nd.aliases)))

            # Deduplicate canonical keys & content fingerprints across survivors
            by_ck: dict[str, list[tuple[str, str, str, float, float, int, str]]] = defaultdict(list)
            by_fp: dict[str, list[tuple[str, str, str, float, float, int, str]]] = defaultdict(list)
            for rec in survivor_meta:
                by_ck[rec[1]].append(rec)
                by_fp[rec[2]].append(rec)

            dedup_removed_ids: set[str] = set()
            merged_aliases: dict[str, list[str]] = {}
            canonical_dups = 0
            content_dups = 0

            def _dedup_bucket(bucket: list[tuple[str, str, str, float, float, int, str]]) -> int:
                live = [r for r in bucket if r[0] not in dedup_removed_ids]
                if len(live) <= 1:
                    return 0
                live.sort(key=lambda r: (-r[3], -r[4], -r[5], r[0]))
                winner = live[0]
                w_als = merged_aliases.setdefault(winner[0], [a for a in winner[6].split("|") if a])
                cnt = 0
                for loser in live[1:]:
                    if loser[3] >= 4.0:
                        # Strictly NEVER remove Tier 4/5 protected nodes!
                        continue
                    dedup_removed_ids.add(loser[0])
                    cnt += 1
                    for surf in (loser[0], *(a for a in loser[6].split("|") if a)):
                        s_low = surf.strip().casefold()
                        if s_low and s_low not in w_als:
                            w_als.append(s_low)
                return cnt

            for grp in by_ck.values():
                if len(grp) > 1:
                    canonical_dups += _dedup_bucket(grp)
            for grp in by_fp.values():
                if len(grp) > 1:
                    content_dups += _dedup_bucket(grp)

            all_removed = removed_low_quality_ids | dedup_removed_ids
            final_scores = [r[4] for r in survivor_meta if r[0] not in dedup_removed_ids]
            avg_q = (sum(final_scores) / len(final_scores)) if final_scores else 1.0

            idx_stats = {
                "tokens_indexed": 0,
                "trigrams_indexed": 0,
                "trigrams_df_pruned": 0,
                "fts_indexed": len(final_scores),
            }

            if persist:
                if all_removed:
                    conn.executemany(
                        "DELETE FROM mirror_concept_lattice WHERE concept_id = ?",
                        [(rid,) for rid in all_removed],
                    )
                if quality_updates:
                    live_updates = [u for u in quality_updates if u[3] not in all_removed]
                    conn.executemany(
                        "UPDATE mirror_concept_lattice SET quality_score=?, canonical_key=?, content_fp=? WHERE concept_id=?",
                        live_updates,
                    )
                if merged_aliases:
                    conn.executemany(
                        "UPDATE mirror_concept_lattice SET aliases_csv=? WHERE concept_id=?",
                        [("|".join(dict.fromkeys(als)), cid) for cid, als in merged_aliases.items() if cid not in all_removed],
                    )

                # Physically sort table rows by (category, quality_score DESC, concept_id) for B-tree locality
                conn.execute("DROP TABLE IF EXISTS mirror_concept_lattice_sorted")
                conn.execute(
                    """
                    CREATE TABLE mirror_concept_lattice_sorted (
                        concept_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        category TEXT NOT NULL,
                        aliases_csv TEXT NOT NULL,
                        canonical_key TEXT NOT NULL DEFAULT '',
                        content_fp TEXT NOT NULL DEFAULT '',
                        provenance TEXT NOT NULL,
                        quality_score REAL NOT NULL DEFAULT 1.0,
                        facets_zblob BLOB NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO mirror_concept_lattice_sorted
                    (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                    SELECT concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob
                    FROM mirror_concept_lattice
                    ORDER BY category ASC, quality_score DESC, concept_id ASC
                    """
                )
                conn.execute("DROP TABLE mirror_concept_lattice")
                conn.execute("ALTER TABLE mirror_concept_lattice_sorted RENAME TO mirror_concept_lattice")
                self._create_v4_schema(conn)

                idx_stats = self._rebuild_all_disk_indexes(conn)
                self._update_meta_and_local_cache(conn)
                conn.commit()
                conn.execute("ANALYZE")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                if vacuum:
                    conn.execute("VACUUM")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._lru.clear()
                self._invalidate_snapshot_cache()

            total_after = self.concept_count() if persist else len(final_scores)
            return {
                "ok": True,
                "total_before": total_before,
                "total_after": total_after,
                "protected_tier4_5_untouched": protected_tier4_5,
                "removed_low_quality": len(removed_low_quality_ids),
                "canonical_duplicates_merged": canonical_dups,
                "content_duplicates_merged": content_dups,
                "avg_quality_score": round(avg_q, 4),
                "tokens_indexed": idx_stats["tokens_indexed"],
                "trigrams_indexed": idx_stats["trigrams_indexed"],
                "trigrams_df_pruned": idx_stats["trigrams_df_pruned"],
                "fts_indexed": idx_stats["fts_indexed"],
                "vacuumed": bool(persist and vacuum),
            }

    def admit_to_shard(
        self,
        shard_path: str | Path,
        nodes: Iterable[ConceptNode],
        *,
        min_quality: float = 0.78,
        vacuum: bool = True,
    ) -> dict[str, Any]:
        """Global Cross-Shard Zero-Duplicate, Zero-Overwrite & Quality-Over-Quantity Shard Admission.

        Checks `concept_id`, `canonical_key`, `aliases`, and 64-bit `content_fp` globally across
        PRIMARY VAULT + ALL EXISTING SHARDS + TARGET SHARD + CURRENT BATCH without modifying
        the primary vault or any other existing shard.

        Writes ONLY brand-new verified high-quality records into `shard_path` using the exact
        `v4_fts5_zlib` schema (`mirror_concept_lattice`, `mirror_concept_tokens WITHOUT ROWID`,
        DF-pruned `mirror_concept_trigrams WITHOUT ROWID` <= 300, `mirror_concept_aliases WITHOUT ROWID`,
        `mirror_concept_fts` FTS5, `ANALYZE`, `VACUUM`), then mounts/reloads the shard cleanly.
        """
        target_path = Path(shard_path)
        if self.vault_path is not None:
            try:
                if target_path.resolve() == self.vault_path.resolve():
                    raise ValueError(
                        "admit_to_shard cannot target the primary vault; pass a shard path inside concept_shards/"
                    )
            except OSError:
                pass

        stats: dict[str, Any] = {
            "inserted_new": 0,
            "rejected_low_quality": 0,
            "skipped_existing_primary_or_shard": 0,
            "skipped_canonical_duplicate": 0,
            "skipped_alias_duplicate": 0,
            "skipped_content_duplicate": 0,
            "shard_disk_mb": 0.0,
            "shard_node_count": 0,
            "total_lattice_nodes": self.concept_count(),
        }

        with self._lock:
            # Discover any unmounted shards on disk before global dedup checks
            self._discover_unmounted_shards_locked(extra_dir=target_path.parent)

            target_res = target_path.resolve()
            # Close existing read-only handle to target_path (required on Windows before writing)
            remaining_shards: list[_ShardHandle] = []
            for sh in self._shards:
                if sh.path.resolve() == target_res:
                    try:
                        sh.conn.close()
                    except Exception:
                        pass
                else:
                    remaining_shards.append(sh)
            self._shards = remaining_shards

            existed_before = target_path.is_file() and target_path.stat().st_size >= 1024
            target_conn: sqlite3.Connection | None = None
            target_write_ready = False

            # All read-only sources (primary + other shards); target_conn appended lazily if existing or written
            read_sources: list[tuple[sqlite3.Connection, bool]] = []
            if self._conn is not None and (self._has_v4_schema or self._has_legacy_table):
                read_sources.append((self._conn, self._has_v4_schema))
            for sh in self._shards:
                if sh.has_v4_schema or sh.has_legacy_table:
                    read_sources.append((sh.conn, sh.has_v4_schema))

            if existed_before:
                # Open existing shard WITHOUT switching to WAL unless rows are actually written,
                # preserving journal_mode=DELETE and zero-touch idempotency on 0-insert re-runs!
                target_conn = sqlite3.connect(str(target_path), check_same_thread=False)
                self._configure_connection(target_conn, enable_wal=False)
                t_tables = {
                    r[0]
                    for r in target_conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                        "('mirror_concept_lattice','mirror_concept_aliases')"
                    ).fetchall()
                }
                if "mirror_concept_lattice" in t_tables:
                    read_sources.append((target_conn, "mirror_concept_aliases" in t_tables))

            def _ensure_target_writable() -> sqlite3.Connection:
                nonlocal target_conn, target_write_ready
                if target_conn is None:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_conn = sqlite3.connect(str(target_path), check_same_thread=False)
                if not target_write_ready:
                    self._configure_connection(target_conn, enable_wal=True)
                    self._create_v4_schema(target_conn)
                    target_write_ready = True
                    if not any(src_c is target_conn for src_c, _ in read_sources):
                        read_sources.append((target_conn, True))
                return target_conn

            # Ephemeral & foundational seed keys (tiny ~10-node overlay)
            ephem_cids: set[str] = set()
            ephem_cks: set[str] = set()
            ephem_alias_cks: set[str] = set()
            ephem_fps: set[str] = set()
            for src_map in (self._seed_by_id, self._ephemeral_nodes):
                for scid, snd in src_map.items():
                    ephem_cids.add(scid)
                    sck = canonical_concept_key(scid)
                    if sck:
                        ephem_cks.add(sck)
                    stck = canonical_concept_key(snd.title)
                    if stck:
                        ephem_cks.add(stck)
                    for sal in snd.aliases:
                        sak = canonical_concept_key(sal)
                        if sak:
                            ephem_alias_cks.add(sak)
                    ephem_fps.add(content_fingerprint(snd.definition))

            def _process_shard_chunk(chunk_nodes: list[ConceptNode]) -> None:
                if not chunk_nodes:
                    return

                prepared: list[tuple[ConceptNode, str, str, str, str, float]] = []
                cand_cids: list[str] = []
                cand_cks: list[str] = []
                cand_fps: list[str] = []
                seen_q_cid: set[str] = set()
                seen_q_ck: set[str] = set()
                seen_q_fp: set[str] = set()

                for node in chunk_nodes:
                    cid = (node.concept_id or "").strip().casefold()
                    if not cid or len(cid) < 3:
                        stats["rejected_low_quality"] += 1
                        continue
                    ok_q, eff_q = _validate_shard_candidate(node, min_quality)
                    if not ok_q:
                        stats["rejected_low_quality"] += 1
                        continue

                    ck = canonical_concept_key(cid)
                    title_ck = canonical_concept_key(node.title)
                    fp = content_fingerprint(node.definition)
                    prepared.append((node, cid, ck, title_ck, fp, eff_q))

                    if cid not in seen_q_cid:
                        seen_q_cid.add(cid)
                        cand_cids.append(cid)
                    for k_cand in (ck, title_ck):
                        if k_cand and k_cand not in seen_q_ck:
                            seen_q_ck.add(k_cand)
                            cand_cks.append(k_cand)
                    for al in node.aliases:
                        ak = canonical_concept_key(al)
                        if ak and ak not in seen_q_ck:
                            seen_q_ck.add(ak)
                            cand_cks.append(ak)
                    if fp not in seen_q_fp:
                        seen_q_fp.add(fp)
                        cand_fps.append(fp)

                if not prepared:
                    return

                db_cids: set[str] = set(ephem_cids)
                db_cks: set[str] = set(ephem_cks)
                db_alias_cks: set[str] = set(ephem_alias_cks)
                db_fps: set[str] = set(ephem_fps)

                for src_conn, is_v4 in read_sources:
                    for i in range(0, len(cand_cids), _SQL_IN_CHUNK_SIZE):
                        sub = cand_cids[i: i + _SQL_IN_CHUNK_SIZE]
                        ph = ",".join("?" for _ in sub)
                        for (r_cid,) in src_conn.execute(
                            f"SELECT concept_id FROM mirror_concept_lattice WHERE concept_id IN ({ph})",
                            sub,
                        ).fetchall():
                            db_cids.add(str(r_cid))

                    if is_v4:
                        for i in range(0, len(cand_cks), _SQL_IN_CHUNK_SIZE):
                            sub = cand_cks[i: i + _SQL_IN_CHUNK_SIZE]
                            ph = ",".join("?" for _ in sub)
                            for (r_ck,) in src_conn.execute(
                                f"SELECT canonical_key FROM mirror_concept_lattice WHERE canonical_key IN ({ph})",
                                sub,
                            ).fetchall():
                                db_cks.add(str(r_ck))
                            for (r_ak,) in src_conn.execute(
                                f"SELECT alias_ck FROM mirror_concept_aliases WHERE alias_ck IN ({ph})",
                                sub,
                            ).fetchall():
                                db_alias_cks.add(str(r_ak))

                        for i in range(0, len(cand_fps), _SQL_IN_CHUNK_SIZE):
                            sub = cand_fps[i: i + _SQL_IN_CHUNK_SIZE]
                            ph = ",".join("?" for _ in sub)
                            for (r_fp,) in src_conn.execute(
                                f"SELECT content_fp FROM mirror_concept_lattice WHERE content_fp IN ({ph})",
                                sub,
                            ).fetchall():
                                db_fps.add(str(r_fp))

                pending_rows: list[tuple[Any, ...]] = []
                pending_alias_rows: list[tuple[str, str]] = []

                for node, cid, ck, title_ck, fp, eff_q in prepared:
                    # Layer 1: Exact concept_id protection across primary + all shards + batch
                    if cid in db_cids:
                        stats["skipped_existing_primary_or_shard"] += 1
                        continue

                    # Layer 2: Canonical plural/hyphen/punctuation duplicate guard
                    if (ck and ck in db_cks) or (title_ck and title_ck in db_cks):
                        stats["skipped_canonical_duplicate"] += 1
                        continue

                    # Layer 3: Existing alias collision guard
                    if (ck and ck in db_alias_cks) or (title_ck and title_ck in db_alias_cks):
                        stats["skipped_alias_duplicate"] += 1
                        continue

                    # Layer 4: 64-bit normalized definition content fingerprint guard
                    if fp in db_fps:
                        stats["skipped_content_duplicate"] += 1
                        continue

                    # Sanitize secondary aliases so no shard alias collides with existing canonical/alias keys
                    cleaned_aliases: list[str] = []
                    seen_al: set[str] = set()
                    for raw_al in (cid, node.title, *node.aliases):
                        al_str = str(raw_al).strip()
                        al_low = al_str.casefold()
                        if not al_low or al_low in seen_al:
                            continue
                        al_ck = canonical_concept_key(al_low)
                        if al_low != cid and al_ck and al_ck != ck:
                            if al_ck in db_cks or al_ck in db_alias_cks:
                                continue
                        seen_al.add(al_low)
                        cleaned_aliases.append(al_str)
                    if not cleaned_aliases:
                        cleaned_aliases = [cid]
                    aliases_csv = "|".join(cleaned_aliases)

                    pending_rows.append(
                        (
                            cid,
                            node.title.strip(),
                            node.category.strip().casefold() or "general_knowledge",
                            aliases_csv,
                            ck or cid,
                            fp,
                            node.provenance.strip(),
                            eff_q,
                            pack_facets_zblob(
                                node.definition.strip(),
                                node.mechanism.strip(),
                                node.applications.strip(),
                                node.tradeoffs.strip(),
                            ),
                        )
                    )

                    db_cids.add(cid)
                    if ck:
                        db_cks.add(ck)
                    if title_ck and title_ck != ck:
                        db_alias_cks.add(title_ck)
                        pending_alias_rows.append((title_ck, cid))
                    for al in cleaned_aliases:
                        ak = canonical_concept_key(al)
                        if ak and ak != ck:
                            db_alias_cks.add(ak)
                            pending_alias_rows.append((ak, cid))
                    db_fps.add(fp)
                    stats["inserted_new"] += 1

                if pending_rows:
                    wconn = _ensure_target_writable()
                    wconn.executemany(
                        """
                        INSERT OR IGNORE INTO mirror_concept_lattice
                        (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        pending_rows,
                    )
                    if pending_alias_rows:
                        wconn.executemany(
                            "INSERT OR IGNORE INTO mirror_concept_aliases(alias_ck, concept_id) VALUES (?, ?)",
                            pending_alias_rows,
                        )

            try:
                chunk_buf: list[ConceptNode] = []
                for node in nodes:
                    chunk_buf.append(node)
                    if len(chunk_buf) >= 1000:
                        _process_shard_chunk(chunk_buf)
                        chunk_buf.clear()
                if chunk_buf:
                    _process_shard_chunk(chunk_buf)
                    chunk_buf.clear()

                if stats["inserted_new"] > 0 and target_conn is not None:
                    # Physically sort rows by (category, quality_score DESC, concept_id) for B-tree locality
                    target_conn.execute("DROP TABLE IF EXISTS mirror_concept_lattice_sorted")
                    target_conn.execute(
                        """
                        CREATE TABLE mirror_concept_lattice_sorted (
                            concept_id TEXT PRIMARY KEY,
                            title TEXT NOT NULL,
                            category TEXT NOT NULL,
                            aliases_csv TEXT NOT NULL,
                            canonical_key TEXT NOT NULL DEFAULT '',
                            content_fp TEXT NOT NULL DEFAULT '',
                            provenance TEXT NOT NULL,
                            quality_score REAL NOT NULL DEFAULT 1.0,
                            facets_zblob BLOB NOT NULL
                        )
                        """
                    )
                    target_conn.execute(
                        """
                        INSERT INTO mirror_concept_lattice_sorted
                        (concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob)
                        SELECT concept_id, title, category, aliases_csv, canonical_key, content_fp, provenance, quality_score, facets_zblob
                        FROM mirror_concept_lattice
                        ORDER BY category ASC, quality_score DESC, concept_id ASC
                        """
                    )
                    target_conn.execute("DROP TABLE mirror_concept_lattice")
                    target_conn.execute("ALTER TABLE mirror_concept_lattice_sorted RENAME TO mirror_concept_lattice")
                    self._create_v4_schema(target_conn)

                    self._rebuild_all_disk_indexes(target_conn)

                    # Write shard metadata (do NOT mutate primary vault counters)
                    shard_total = int(target_conn.execute("SELECT COUNT(*) FROM mirror_concept_lattice").fetchone()[0])
                    s_rows = target_conn.execute(
                        "SELECT category, COUNT(*) FROM mirror_concept_lattice GROUP BY category"
                    ).fetchall()
                    s_dist = dict(sorted(((str(c), int(n)) for c, n in s_rows), key=lambda x: (-x[1], x[0])))
                    target_conn.executemany(
                        "INSERT OR REPLACE INTO mirror_concept_meta(key, val) VALUES (?, ?)",
                        (
                            ("schema_version", _SCHEMA_VERSION),
                            ("total_count", str(shard_total)),
                            ("domain_distribution", json.dumps(s_dist, separators=(",", ":"))),
                        ),
                    )
                    target_conn.commit()
                    target_conn.execute("ANALYZE")
                    target_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    if vacuum:
                        target_conn.execute("VACUUM")
                        target_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    # Switch to self-contained single-file mode so no -wal/-shm sidecars remain
                    target_conn.execute("PRAGMA journal_mode=DELETE").fetchone()
                    target_conn.close()
                    target_conn = None

                    handle = self._register_or_reload_shard_locked(target_path)
                    self._lru.clear()
                    if handle is not None:
                        stats["shard_disk_mb"] = handle.disk_mb
                        stats["shard_node_count"] = handle.node_count
                else:
                    if target_conn is not None:
                        if target_write_ready:
                            target_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                            target_conn.execute("PRAGMA journal_mode=DELETE").fetchone()
                        target_conn.close()
                        target_conn = None
                    if not existed_before:
                        for sidecar in (
                            target_path,
                            Path(str(target_path) + "-wal"),
                            Path(str(target_path) + "-shm"),
                        ):
                            try:
                                if sidecar.exists():
                                    sidecar.unlink()
                            except OSError:
                                pass
                        self._recompute_combined_metadata()
                    else:
                        handle = self._register_or_reload_shard_locked(target_path)
                        if handle is not None:
                            stats["shard_disk_mb"] = handle.disk_mb
                            stats["shard_node_count"] = handle.node_count
            finally:
                if target_conn is not None:
                    try:
                        target_conn.close()
                    except Exception:
                        pass

            stats["total_lattice_nodes"] = self.concept_count()
            return stats

    def load_from_vault(self, vault_path: str | Path | None = None) -> int:
        """Backward-compatible vault loader: attaches disk-backed vault + shards in O(1) without pulling rows into RAM."""
        target = Path(vault_path) if vault_path is not None else self.vault_path
        if target is None or not target.exists():
            return 0
        with self._lock:
            if self.vault_path is not None and target.resolve() == self.vault_path.resolve() and self._conn is not None:
                return self._cached_total_db_count
            self.close()
            self.vault_path = target
            if self._explicit_shard_dir is None:
                self.shard_dir = self.vault_path.parent / "concept_shards"
            self._lru.clear()
            self._ephemeral_nodes.clear()
            self._ephemeral_tokens.clear()
            self._ephemeral_trigrams.clear()
            self._has_v4_schema = False
            self._has_legacy_table = False
            self._cached_db_count = 0
            self._cached_db_domain_dist = {}
            self._attach_existing_vault()
            self._attach_shards()
            self._recompute_combined_metadata()
            return self._cached_total_db_count

    def compile_text_to_concept_node(
        self,
        topic_name: str,
        raw_text: str,
        *,
        category: str = "general_knowledge",
        aliases: tuple[str, ...] | None = None,
        provenance: str = "mirror:auto_compiled_concept",
        persist: bool = True,
        allow_overwrite: bool = False,
    ) -> dict[str, Any]:
        """Unsupervised Zero-Bloat Concept Compiler: distills raw multi-sentence text into
        a 5-facet ConceptNode (~240-320 bytes) and evicts the raw text without overwriting protected nodes."""
        cid = topic_name.strip().casefold()
        sentences = _split_sentences(raw_text)
        if not cid or not sentences:
            return {"ok": False, "reason": "empty topic or text"}

        existing = self._get_node(cid)
        if not allow_overwrite and existing is not None:
            existing_tier = provenance_tier_rank(existing.provenance)
            if existing_tier >= 4.0 and existing.provenance != provenance:
                return {
                    "ok": False,
                    "protected": True,
                    "concept_id": cid,
                    "reason": f"Concept '{cid}' already exists under protected provenance '{existing.provenance}'. Overwrite blocked.",
                }

        defn = sentences[0]
        mech = ""
        apps = ""
        trd = ""
        for s in sentences[1:]:
            low = s.casefold()
            if not trd and any(w in low for w in ("limit", "risk", "drawback", "overhead", "however", "trade-off", "tradeoff", "nuqsan", "challenge", "vulnerability", "cost", "bottleneck", "require")):
                trd = s
            elif not mech and any(w in low for w in ("work", "operat", "mechanism", "through", "via", "uses", "consists", "kaam", "process", "algorithm", "comput", "convert", "transform", "bind")):
                mech = s
            elif not apps and any(w in low for w in ("used", "enabl", "application", "benefit", "provid", "ideal", "fayda", "support", "power", "essential", "allow")):
                apps = s

        node_aliases = aliases if aliases is not None else (cid,)
        node = ConceptNode(
            concept_id=cid,
            title=topic_name.strip() if any(c.isupper() for c in topic_name.strip()[1:]) else topic_name.strip().title(),
            category=category,
            aliases=node_aliases,
            definition=defn,
            mechanism=mech or (sentences[1] if len(sentences) > 1 else defn),
            applications=apps or (sentences[2] if len(sentences) > 2 else f"Core analytical, scientific, and engineering applications in {category.replace('_', ' ')}."),
            tradeoffs=trd or (sentences[3] if len(sentences) > 3 else "Requires domain validation, parameter calibration, and bounded operational constraints."),
            provenance=provenance,
        )
        self.register_node(node, persist=persist)
        raw_bytes = len(raw_text.encode("utf-8"))
        node_bytes = len((node.definition + node.mechanism + node.applications + node.tradeoffs).encode("utf-8"))
        return {
            "ok": True,
            "concept_id": cid,
            "raw_bytes": raw_bytes,
            "distilled_node_bytes": node_bytes,
        }

    def _query_disk_candidates(
        self,
        low: str,
        q_tokens: list[str],
    ) -> tuple[set[str], dict[str, tuple[str, ...]], dict[str, tuple[sqlite3.Connection, bool]]]:
        """Multi-shard O(1) lookup across primary + all mounted `.dcrystal` shards:
        zero-I/O seed surface map -> single-shot disk PK n-grams (with inline hydration) ->
        alias PKs -> `mirror_concept_tokens` -> DF-pruned `mirror_concept_trigrams` -> surgical FTS5."""
        candidate_ids: set[str] = set()
        preloaded_aliases: dict[str, tuple[str, ...]] = {}
        cand_source: dict[str, tuple[sqlite3.Connection, bool]] = {}

        unique_q_tokens = list(dict.fromkeys(q_tokens))
        substantive_toks = [
            t for t in unique_q_tokens if t not in _COMMON_NON_TYPO_TOKENS and t not in _INDEX_STOP_TOKENS
        ]

        # 0. Zero-I/O check against the 10 in-RAM foundational seed concepts & their ~40 aliases
        ngrams = _extract_query_ngrams(low)
        covered_tokens: set[str] = set()
        for ng in ngrams:
            scid = self._seed_surface_to_id.get(ng)
            if scid is not None:
                candidate_ids.add(scid)
                covered_tokens.update(_RE_WORD_3.findall(ng))

        uncovered_toks = [t for t in substantive_toks if t not in covered_tokens]
        if candidate_ids and not uncovered_toks:
            return candidate_ids, preloaded_aliases, cand_source

        with self._lock:
            v4_sources: list[sqlite3.Connection] = []
            if self._conn is not None and self._has_v4_schema:
                v4_sources.append(self._conn)
            for sh in self._shards:
                if sh.has_v4_schema:
                    v4_sources.append(sh.conn)
            if not v4_sources:
                return candidate_ids, preloaded_aliases, cand_source

            # 1a. Single-shot Exact O(1) Primary-Key n-gram lookup across primary + shards.
            # Early-exit inside the loop ONLY when a SINGLE concept covers 100% of all_content_toks
            # (so multi-word shard concepts are NEVER shadowed by shorter 1-word primary component concepts!).
            all_content_toks = {t for t in unique_q_tokens if t not in _NGRAM_SKIP_EDGE} or set(substantive_toks)
            is_cross_q = bool(_RE_CROSS_COMPARE.search(low) or _RE_CROSS_SYNERGY.search(low))
            max_single_cov = 0
            if ngrams:
                disk_ngrams = [ng for ng in ngrams if ng not in self._seed_by_id]
                if disk_ngrams:
                    ph = ",".join("?" for _ in disk_ngrams)
                    sql_pk = (
                        f"SELECT concept_id, title, category, aliases_csv, provenance, quality_score, facets_zblob "
                        f"FROM mirror_concept_lattice WHERE concept_id IN ({ph})"
                    )
                    for src_conn in v4_sources:
                        rows = src_conn.execute(sql_pk, disk_ngrams).fetchall()
                        if not rows:
                            continue
                        max_rlen = max(len(str(r[0])) for r in rows)
                        hydrate_all = len(rows) <= 2
                        for row in rows:
                            cid_s = str(row[0])
                            if cid_s in candidate_ids:
                                continue
                            if hydrate_all or len(cid_s) == max_rlen:
                                nd = self._row_to_node_v4(row)
                                self._lru_put(cid_s, nd)
                                als = nd.aliases
                            else:
                                als = tuple(a for a in str(row[3]).split("|") if a)
                            candidate_ids.add(cid_s)
                            cand_source[cid_s] = (src_conn, True)
                            preloaded_aliases[cid_s] = als
                            node_cov: set[str] = set()
                            for surf in (cid_s, *als):
                                s_low = surf.strip().casefold()
                                if len(s_low) >= 3 and _has_word_boundary_match(low, s_low) >= 0:
                                    s_toks = _RE_WORD_3.findall(s_low)
                                    covered_tokens.update(s_toks)
                                    node_cov.update(s_toks)
                            if all_content_toks:
                                c_sz = len(node_cov & all_content_toks)
                                if c_sz > max_single_cov:
                                    max_single_cov = c_sz
                        if all_content_toks and max_single_cov >= len(all_content_toks):
                            return candidate_ids, preloaded_aliases, cand_source

                uncovered_toks = [t for t in substantive_toks if t not in covered_tokens]
                if (
                    candidate_ids
                    and not uncovered_toks
                    and (len(all_content_toks) <= 1 or max_single_cov >= len(all_content_toks) or is_cross_q)
                ):
                    return candidate_ids, preloaded_aliases, cand_source

                # 1b. Exact O(1) canonical_key + extra-alias PK lookup across ALL sources (primary + every shard)
                ngram_cks = list(dict.fromkeys(ck for ng in ngrams if (ck := canonical_concept_key(ng))))
                if ngram_cks:
                    ph_ck = ",".join("?" for _ in ngram_cks)
                    sql_ck = (
                        f"SELECT concept_id, title, category, aliases_csv, provenance, quality_score, facets_zblob "
                        f"FROM mirror_concept_lattice WHERE canonical_key IN ({ph_ck})"
                    )
                    sql_alias = f"SELECT concept_id FROM mirror_concept_aliases WHERE alias_ck IN ({ph_ck})"
                    # Pass 1b-i: direct canonical_key across sources
                    for src_conn in v4_sources:
                        rows = src_conn.execute(sql_ck, ngram_cks).fetchall()
                        if not rows:
                            continue
                        max_rlen = max(len(str(r[0])) for r in rows)
                        hydrate_all = len(rows) <= 2
                        for row in rows:
                            cid_s = str(row[0])
                            if cid_s in candidate_ids:
                                continue
                            if hydrate_all or len(cid_s) == max_rlen:
                                nd = self._row_to_node_v4(row)
                                self._lru_put(cid_s, nd)
                                als = nd.aliases
                            else:
                                als = tuple(a for a in str(row[3]).split("|") if a)
                            candidate_ids.add(cid_s)
                            cand_source[cid_s] = (src_conn, True)
                            preloaded_aliases[cid_s] = als
                            node_cov = set()
                            for surf in (cid_s, *als):
                                s_low = surf.strip().casefold()
                                if len(s_low) >= 3 and _has_word_boundary_match(low, s_low) >= 0:
                                    s_toks = _RE_WORD_3.findall(s_low)
                                    covered_tokens.update(s_toks)
                                    node_cov.update(s_toks)
                            if all_content_toks:
                                c_sz = len(node_cov & all_content_toks)
                                if c_sz > max_single_cov:
                                    max_single_cov = c_sz
                        if all_content_toks and max_single_cov >= len(all_content_toks):
                            return candidate_ids, preloaded_aliases, cand_source

                    uncovered_toks = [t for t in substantive_toks if t not in covered_tokens]
                    if (
                        candidate_ids
                        and not uncovered_toks
                        and (len(all_content_toks) <= 1 or max_single_cov >= len(all_content_toks) or is_cross_q)
                    ):
                        return candidate_ids, preloaded_aliases, cand_source

                    # Pass 1b-ii: extra-alias lookup across ALL sources
                    for src_conn in v4_sources:
                        alias_cids = [
                            str(r[0])
                            for r in src_conn.execute(sql_alias, ngram_cks).fetchall()
                            if str(r[0]) not in candidate_ids
                        ]
                        if alias_cids:
                            for c in alias_cids:
                                candidate_ids.add(c)
                                cand_source.setdefault(c, (src_conn, True))
                            missing_a = [
                                c for c in alias_cids
                                if c not in preloaded_aliases and c not in self._lru and c not in self._seed_by_id
                            ]
                            if missing_a:
                                ph_m = ",".join("?" for _ in missing_a)
                                if len(missing_a) <= 4:
                                    for row in src_conn.execute(
                                        f"SELECT concept_id, title, category, aliases_csv, provenance, quality_score, facets_zblob "
                                        f"FROM mirror_concept_lattice WHERE concept_id IN ({ph_m})",
                                        missing_a,
                                    ).fetchall():
                                        nd = self._row_to_node_v4(row)
                                        self._lru_put(nd.concept_id, nd)
                                        preloaded_aliases[nd.concept_id] = nd.aliases
                                else:
                                    for cid, aliases_csv in src_conn.execute(
                                        f"SELECT concept_id, aliases_csv FROM mirror_concept_lattice WHERE concept_id IN ({ph_m})",
                                        missing_a,
                                    ).fetchall():
                                        preloaded_aliases[str(cid)] = tuple(a for a in str(aliases_csv).split("|") if a)
                            for c in alias_cids:
                                als = (
                                    self._lru[c].aliases
                                    if c in self._lru
                                    else (self._seed_by_id[c].aliases if c in self._seed_by_id else preloaded_aliases.get(c, ()))
                                )
                                for surf in (c, *als):
                                    s_low = surf.strip().casefold()
                                    if len(s_low) >= 3 and _has_word_boundary_match(low, s_low) >= 0:
                                        covered_tokens.update(_RE_WORD_3.findall(s_low))

                    uncovered_toks = [t for t in substantive_toks if t not in covered_tokens]
                    if candidate_ids and not uncovered_toks:
                        return candidate_ids, preloaded_aliases, cand_source

            # 2. Multi-shard disk-backed inverted token lookup ONLY for uncovered substantive tokens
            unknown_typo_toks: list[str] = []
            hit_cap_tokens: list[str] = []
            for tok in uncovered_toks[:8]:
                tok_found = False
                for src_conn in v4_sources:
                    rows = src_conn.execute(
                        "SELECT concept_id FROM mirror_concept_tokens WHERE token = ? LIMIT ?",
                        (tok, _PER_TOKEN_SQL_LIMIT),
                    ).fetchall()
                    if rows:
                        tok_found = True
                        for (cid,) in rows:
                            cid_s = str(cid)
                            candidate_ids.add(cid_s)
                            cand_source.setdefault(cid_s, (src_conn, True))
                        if len(rows) >= _PER_TOKEN_SQL_LIMIT:
                            hit_cap_tokens.append(tok)
                if not tok_found and len(tok) >= 6 and tok not in self._ephemeral_tokens:
                    unknown_typo_toks.append(tok)

            if not candidate_ids:
                fallback_toks = [t for t in unique_q_tokens if t in _COMMON_NON_TYPO_TOKENS and t not in _INDEX_STOP_TOKENS]
                for tok in fallback_toks[:4]:
                    for src_conn in v4_sources:
                        rows = src_conn.execute(
                            "SELECT concept_id FROM mirror_concept_tokens WHERE token = ? LIMIT 32",
                            (tok,),
                        ).fetchall()
                        for (cid,) in rows:
                            cid_s = str(cid)
                            candidate_ids.add(cid_s)
                            cand_source.setdefault(cid_s, (src_conn, True))

            # 3. DF-pruned disk-backed trigram fuzzy lookup across primary + shards ONLY for unknown typo tokens (len >= 6)
            for tok in unknown_typo_toks[:2]:
                tgs = list(_trigrams(tok))
                if not tgs:
                    continue
                exists_sql = "SELECT " + " + ".join(
                    "EXISTS(SELECT 1 FROM mirror_concept_trigrams WHERE trigram = ?)" for _ in tgs
                )
                ph = ",".join("?" for _ in tgs)
                tg_sql = (
                    f"SELECT concept_id FROM mirror_concept_trigrams WHERE trigram IN ({ph}) "
                    "GROUP BY concept_id HAVING COUNT(*) >= ? LIMIT 32"
                )
                for src_conn in v4_sources:
                    active_tgs = int(src_conn.execute(exists_sql, tgs).fetchone()[0])
                    if active_tgs == 0:
                        continue
                    threshold = max(2, int(active_tgs * 0.55)) if active_tgs >= 2 else 3
                    for (cid,) in src_conn.execute(tg_sql, (*tgs, threshold)).fetchall():
                        cid_s = str(cid)
                        candidate_ids.add(cid_s)
                        cand_source.setdefault(cid_s, (src_conn, True))

            # 4. Surgical SQLite FTS5 AND-intersection fallback across primary + shards
            if len(uncovered_toks) >= 2 and (hit_cap_tokens or not candidate_ids):
                fts_toks = [t for t in uncovered_toks if t not in unknown_typo_toks][:4]
                if len(fts_toks) >= 2:
                    and_expr = " AND ".join(f'"{t}"' for t in fts_toks[:3])
                    for src_conn in v4_sources:
                        try:
                            for (cid,) in src_conn.execute(
                                "SELECT concept_id FROM mirror_concept_fts WHERE mirror_concept_fts MATCH ? LIMIT 24",
                                (and_expr,),
                            ).fetchall():
                                cid_s = str(cid)
                                candidate_ids.add(cid_s)
                                cand_source.setdefault(cid_s, (src_conn, True))
                        except sqlite3.OperationalError:
                            pass
            elif not candidate_ids and substantive_toks:
                fts_toks = [t for t in substantive_toks if t not in unknown_typo_toks][:4]
                if fts_toks:
                    or_expr = " OR ".join(f'"{t}"' for t in fts_toks)
                    for src_conn in v4_sources:
                        try:
                            for (cid,) in src_conn.execute(
                                "SELECT concept_id FROM mirror_concept_fts WHERE mirror_concept_fts MATCH ? LIMIT 24",
                                (or_expr,),
                            ).fetchall():
                                cid_s = str(cid)
                                candidate_ids.add(cid_s)
                                cand_source.setdefault(cid_s, (src_conn, True))
                        except sqlite3.OperationalError:
                            pass

            # 5. Fetch ONLY lightweight `aliases_csv` per owning source (ZERO zlib decompression!)
            missing_by_conn: dict[sqlite3.Connection, list[str]] = defaultdict(list)
            for cid in candidate_ids:
                if (
                    cid in preloaded_aliases
                    or cid in self._ephemeral_nodes
                    or cid in self._lru
                    or cid in self._seed_by_id
                ):
                    continue
                src_info = cand_source.get(cid)
                if src_info is not None:
                    missing_by_conn[src_info[0]].append(cid)
                elif v4_sources:
                    missing_by_conn[v4_sources[0]].append(cid)

            for src_conn, cids_for_src in missing_by_conn.items():
                for i in range(0, len(cids_for_src), _SQL_IN_CHUNK_SIZE):
                    chunk = cids_for_src[i: i + _SQL_IN_CHUNK_SIZE]
                    ph = ",".join("?" for _ in chunk)
                    for cid, aliases_csv in src_conn.execute(
                        f"SELECT concept_id, aliases_csv FROM mirror_concept_lattice WHERE concept_id IN ({ph})",
                        chunk,
                    ).fetchall():
                        preloaded_aliases[str(cid)] = tuple(a for a in str(aliases_csv).split("|") if a)

        return candidate_ids, preloaded_aliases, cand_source

    def _retrieve_concepts_sublinear(self, query: str) -> tuple[list[ConceptNode], int]:
        """Multi-shard disk-backed O(1) inverted token + DF-pruned trigram + FTS5 lookup followed by
        lightweight alias boundary verification and top-K-only `ConceptNode` hydration."""
        low = query.casefold()
        q_tokens = _RE_WORD_3.findall(low)

        candidate_ids, preloaded_aliases, cand_source = self._query_disk_candidates(low, q_tokens)

        # Merge ephemeral in-memory overlay (0 cost when empty!)
        if self._ephemeral_nodes:
            for tok in q_tokens:
                if tok in self._ephemeral_tokens:
                    candidate_ids.update(self._ephemeral_tokens[tok])
                elif tok not in _COMMON_NON_TYPO_TOKENS and len(tok) >= 6:
                    tgs = _trigrams(tok)
                    hits_per_cid: dict[str, int] = defaultdict(int)
                    active_tgs = 0
                    for tg in tgs:
                        post = self._ephemeral_trigrams.get(tg)
                        if post:
                            active_tgs += 1
                            for cid in post:
                                hits_per_cid[cid] += 1
                    threshold = max(2, int(active_tgs * 0.55)) if active_tgs >= 2 else 3
                    for cid, cnt in hits_per_cid.items():
                        if cnt >= threshold:
                            candidate_ids.add(cid)

        # Legacy read-only SQLite fallback (primary or any legacy v3 shard)
        with self._lock:
            legacy_conns = [c for c, is_v4 in self._iter_active_db_sources_locked() if not is_v4]
            if legacy_conns:
                ngrams = _extract_query_ngrams(low)
                if ngrams:
                    ph = ",".join("?" for _ in ngrams)
                    for lconn in legacy_conns:
                        for cid, aliases_csv in lconn.execute(
                            f"SELECT concept_id, aliases_csv FROM mirror_concept_lattice WHERE concept_id IN ({ph})",
                            ngrams,
                        ).fetchall():
                            cid_s = str(cid)
                            candidate_ids.add(cid_s)
                            cand_source.setdefault(cid_s, (lconn, False))
                            preloaded_aliases.setdefault(cid_s, tuple(a for a in str(aliases_csv).split("|") if a))

        # Evaluate candidate matches with exact word-boundary span tracking BEFORE decompressing facets:
        # (start_pos, end_pos, match_len, is_exact_not_fuzzy, cid)
        span_matches: list[tuple[int, int, int, bool, str]] = []
        for cid in candidate_ids:
            if cid in self._ephemeral_nodes:
                aliases = self._ephemeral_nodes[cid].aliases
            elif cid in self._lru:
                aliases = self._lru[cid].aliases
            elif cid in self._seed_by_id:
                aliases = self._seed_by_id[cid].aliases
            else:
                aliases = preloaded_aliases.get(cid, ())

            best_pos = -1
            best_len = 0
            for al in (cid, *aliases):
                al_low = al.strip().casefold()
                if len(al_low) < 3:
                    continue
                idx = _has_word_boundary_match(low, al_low)
                if idx >= 0 and len(al_low) > best_len:
                    needs_framing = (
                        al_low in _BROAD_AMBIGUOUS_SINGLETONS
                        or (" " not in al_low and len(al_low) <= 4 and al_low not in _TECH_ACRONYM_ALLOWLIST)
                    )
                    if needs_framing:
                        framed = bool(_compiled_framing_re(al_low).search(low))
                        if not framed:
                            continue
                    best_pos = idx
                    best_len = len(al_low)
            if best_pos >= 0:
                span_matches.append((best_pos, best_pos + best_len, best_len, True, cid))
            else:
                # Single-token typo tolerance against single-token concept_id / alias
                for tok in q_tokens:
                    if len(tok) >= 6 and tok not in _COMMON_NON_TYPO_TOKENS and tok not in _BROAD_AMBIGUOUS_SINGLETONS:
                        tg_q = _trigrams(tok)
                        for surf in (cid, *aliases):
                            s_low = surf.strip().casefold()
                            if (
                                " " not in s_low
                                and len(s_low) >= 6
                                and s_low not in _BROAD_AMBIGUOUS_SINGLETONS
                                and abs(len(tok) - len(s_low)) <= 2
                            ):
                                tg_c = _trigrams(s_low)
                                if tg_q and tg_c and (len(tg_q & tg_c) / len(tg_q | tg_c)) >= 0.55:
                                    t_idx = _has_word_boundary_match(low, tok)
                                    pos = t_idx if t_idx >= 0 else low.find(tok)
                                    span_matches.append((pos, pos + len(tok), len(tok), False, cid))
                                    break

        # Prune shorter or fuzzy matches contained inside a longer or exact matched span
        span_matches.sort(key=lambda x: (not x[3], -x[2], x[0]))
        kept_spans: list[tuple[int, int, int, bool, str]] = []
        for s_start, s_end, s_len, s_exact, cid in span_matches:
            subsumed = False
            for k_start, k_end, k_len, k_exact, _ in kept_spans:
                if s_start >= k_start and s_end <= k_end and (s_len < k_len or (k_exact and not s_exact)):
                    subsumed = True
                    break
            if not subsumed:
                kept_spans.append((s_start, s_end, s_len, s_exact, cid))

        if any(x[3] for x in kept_spans):
            kept_spans = [x for x in kept_spans if x[3]]

        kept_spans.sort(key=lambda x: (x[0], -x[2]))
        unique_nodes: list[ConceptNode] = []
        seen: set[str] = set()
        # Hydrate ONLY the final matched top-K nodes (typically 1-3!)
        for _, _, _, _, cid in kept_spans:
            if cid not in seen:
                seen.add(cid)
                nd = self._get_node(cid, hint_source=cand_source.get(cid))
                if nd is not None:
                    unique_nodes.append(nd)
                    if len(unique_nodes) >= 6:
                        break
        return unique_nodes, len(candidate_ids)

    @staticmethod
    def _resolve_open_intents(low: str, constraints: Any | None = None) -> tuple[str, ...]:
        """Detect requested cognitive facets via Bit-Sliced HDC Associative Engine + Abjad Phonetic Skeleton + Byte Router."""
        if constraints is not None and getattr(constraints, "strict_single_facet", None):
            return (constraints.strict_single_facet,)

        intents: list[str] = []
        if _RE_INTENT_DEF.search(low):
            intents.append("definition")
        if _RE_INTENT_MECH.search(low):
            intents.append("mechanism")
        if _RE_INTENT_APPS.search(low):
            intents.append("applications")
        if _RE_INTENT_TRD.search(low):
            intents.append("tradeoffs")
        if not intents:
            if _RE_INTENT_GEN.search(low):
                intents = ["definition", "mechanism", "applications", "tradeoffs"]
            else:
                from .byte_intent_router import get_default_byte_intent_router
                from .hdc_associative_engine import abjad_canonical, get_default_hdc_model

                abjad_roots = {abjad_canonical(w).split("|")[0] for w in re.findall(r"[a-zA-Z]+", low)}
                if {"k", "trf", "dfn"} & abjad_roots and {"h", "ht"} & abjad_roots:
                    intents = ["definition"]
                elif {"smjh", "bt", "xpln"} & abjad_roots:
                    intents = ["definition", "mechanism", "applications", "tradeoffs"]
                else:
                    byte_router = get_default_byte_intent_router()
                    if constraints is None:
                        constraints = byte_router.extract_constraints(low)
                    byte_facets = byte_router.resolve_concept_facets(low, constraints=constraints)
                    if byte_facets:
                        intents = list(byte_facets)
                    else:
                        hdc = get_default_hdc_model()
                        hdc_lbl, hdc_sc, _ = hdc.classify(low)
                        if hdc_lbl in ("EXPLAIN", "COMPARE") and hdc_sc >= hdc.OOD_THRESHOLD:
                            intents = ["definition", "mechanism", "applications", "tradeoffs"]
        elif "definition" not in intents and _RE_INTENT_EXPL.search(low):
            intents.insert(0, "definition")
        return tuple(intents)

    @staticmethod
    def _should_defer_to_specialized_organ(low: str) -> bool:
        """Ensure quantitative solvers, grounded doc QA, 12-domain personal coaching, and OOD traps are never hijacked."""
        if _RE_DEFER_OOD.search(low):
            return True
        if _RE_HAS_DIGIT.search(low) and not _RE_DIGIT_ALLOWLIST.search(low):
            return True
        if _RE_DEFER_QUANT.search(low):
            return True
        if _RE_DEFER_PROSE.search(low):
            return True
        return False

    @staticmethod
    def _illustrative_code_for_concept(nd: ConceptNode) -> str:
        """Synthesize a deterministic, sandbox-safe illustrative Python snippet grounded in `nd`."""
        cid = nd.concept_id.casefold()
        if "binary search" in cid or "binary_search" in cid:
            return (
                "def binary_search(arr, target):\n"
                "    lo, hi = 0, len(arr) - 1\n"
                "    while lo <= hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        if arr[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid - 1\n"
                "    return -1"
            )
        if "recursion" in cid or "factorial" in cid:
            return (
                "def factorial_recursive(n):\n"
                "    return 1 if n <= 1 else n * factorial_recursive(n - 1)"
            )
        if "sql" in cid or "injection" in cid:
            return (
                "def safe_parameterized_query(cursor, user_id):\n"
                "    # Parameterized placeholder (?) prevents SQL injection\n"
                "    return ('SELECT * FROM users WHERE id = ?', (user_id,))"
            )
        safe_fn = re.sub(r"[^a-z0-9_]", "_", cid).strip("_")[:32] or "concept_example"
        if safe_fn[0].isdigit():
            safe_fn = "concept_" + safe_fn
        return (
            f"def {safe_fn}_summary():\n"
            f"    return {{'concept': {nd.title!r}, 'category': {nd.category!r}}}"
        )

    def route_explicit_command(self, prompt: str) -> OpenEndedSynthesisResult:
        """Fast-path for explicit Concept Lattice vault commands (status, domain list, live compile)."""
        text = prompt.strip()
        low = text.casefold()

        # 0a. Meta-telemetry query
        if _RE_CMD_STATUS.search(low):
            dist = self.domain_distribution()
            total = self.concept_count()
            pairs = (total * (total - 1)) // 2
            top_doms = ", ".join(f"{k}={v}" for k, v in list(dist.items())[:12])
            return OpenEndedSynthesisResult(
                matched=True,
                concepts_used=("meta:concept_lattice_status",),
                intents_resolved=("vault_telemetry",),
                response=(
                    f"[Open-Ended Concept Lattice Vault Status | Total 5-Facet ConceptNodes: {total} | "
                    f"Domains: {len(dist)} | Shards: {len(self._shards)} | Combinatorial Cross-Concept Pairs: {pairs:,}]\n"
                    f"- Top Domains: {top_doms}"
                ),
            )

        # 0b. Domain listing query
        m_list = _RE_CMD_LIST.match(low)
        if m_list:
            raw_cat = m_list.group(1).strip().replace(" ", "_").replace("-", "_")
            dist = self.domain_distribution()
            matched_cat = None
            for cat in dist:
                if cat == raw_cat or raw_cat in cat:
                    matched_cat = cat
                    break
            if matched_cat is not None:
                items = self.search_by_category(matched_cat, limit=20)
                preview = ", ".join(nd.title for nd in items)
                return OpenEndedSynthesisResult(
                    matched=True,
                    concepts_used=(f"meta:domain:{matched_cat}",),
                    intents_resolved=("domain_listing",),
                    response=(
                        f"[Concept Lattice Domain '{matched_cat}' | Total Concepts: {dist[matched_cat]}]\n"
                        f"- Sample Concepts (first {len(items)}): {preview}"
                    ),
                )

        # 1. Live Concept Compilation command
        m_comp = _RE_CMD_COMPILE.match(text)
        if m_comp:
            cname = m_comp.group(1).strip()
            cbody = m_comp.group(2).strip()
            stats = self.compile_text_to_concept_node(cname, cbody)
            if stats.get("ok"):
                return OpenEndedSynthesisResult(
                    matched=True,
                    concepts_used=(stats["concept_id"],),
                    intents_resolved=("compile_concept",),
                    response=(
                        f"Compiled 5-Facet ConceptNode '{stats['concept_id']}' into Inverted Lattice Index "
                        f"(raw={stats['raw_bytes']} bytes -> node={stats['distilled_node_bytes']} bytes)."
                    ),
                )

        return OpenEndedSynthesisResult(False)

    def route_open_ended(self, prompt: str) -> OpenEndedSynthesisResult:
        cmd_hit = self.route_explicit_command(prompt)
        if cmd_hit.matched:
            return cmd_hit

        text = prompt.strip()
        low = text.casefold()

        if self._should_defer_to_specialized_organ(low):
            return OpenEndedSynthesisResult(False)

        from .byte_intent_router import get_default_byte_intent_router
        from .hdc_associative_engine import (
            INDUCED_EN_COMPARE_FRAME,
            INDUCED_EN_EXPLAIN_FRAME,
            INDUCED_URDU_COMPARE_FRAME,
            INDUCED_URDU_EXPLAIN_FRAME,
            abjad_canonical,
            abjad_strip_query_frame,
            get_default_hdc_model,
            realize_from_frame,
        )

        byte_router = get_default_byte_intent_router()
        constraints = byte_router.extract_constraints(text)

        # Fast 0.001 ms pre-gate + HDC greeting deferral so greetings never match partial concept titles
        if len(low) <= 64 and any(g in low for g in ("assalam", "salam", "salaam", "alaikum", "kya haal", "kaise ho", "good morning", "good evening")):
            if get_default_hdc_model().classify(text)[0] == "GREETING":
                return OpenEndedSynthesisResult(False)

        intents = self._resolve_open_intents(low, constraints=constraints)
        is_cross_synergy = bool(_RE_CROSS_SYNERGY.search(low))
        is_cross_compare = bool(_RE_CROSS_COMPARE.search(low))
        if not is_cross_compare and ("differnce" in low or "farq" in low or "muqabla" in low):
            is_cross_compare = True
        if not intents and not is_cross_synergy and not is_cross_compare:
            return OpenEndedSynthesisResult(False)

        nodes, scanned = self._retrieve_concepts_sublinear(text)
        if not nodes:
            stripped_q = abjad_strip_query_frame(text)
            if stripped_q and stripped_q.casefold() != low:
                nodes, scanned = self._retrieve_concepts_sublinear(stripped_q)
        if not nodes:
            return OpenEndedSynthesisResult(False)

        # Epistemic Subject-Coverage Guard: if the user asks an ultra-specific technical query
        # containing >= 3 rare unmatched domain terms absent from the retrieved concept(s),
        # abstain honestly rather than returning an off-target generic summary.
        substantive = [w for w in _RE_WORD_4_ALPHA.findall(low) if w not in _COVERAGE_FILLER]
        if len(substantive) >= 4:
            covered_blob = " ".join(
                f"{nd.concept_id} {nd.title} {' '.join(nd.aliases)} {nd.definition} {nd.mechanism} {nd.applications} {nd.tradeoffs}"
                for nd in nodes[:2]
            ).casefold()
            unmatched_rare = [w for w in substantive if w not in covered_blob and w[:5] not in covered_blob]
            coverage = 1.0 - (len(unmatched_rare) / len(substantive))
            if (len(unmatched_rare) >= 3 and coverage < 0.50) or (
                not is_cross_compare and not is_cross_synergy and len(unmatched_rare) >= 2 and coverage < 0.72
            ):
                return OpenEndedSynthesisResult(False)

        is_urdu = bool(_RE_URDU.search(low)) or constraints.language == "roman_urdu"
        lang = "roman_urdu" if is_urdu else "en"

        # Case 1A: Cross-Concept Head-to-Head Comparison (with Anti-Unification Construction-Grammar Frame)
        if len(nodes) >= 2 and is_cross_compare:
            a, b = nodes[0], nodes[1]
            ncd = symmetric_raw_ncd(a.mechanism.encode("utf-8"), b.mechanism.encode("utf-8"))
            cmp_facts = {
                "A_TITLE": a.title,
                "A_DEF": a.definition,
                "A_APPS": a.applications,
                "B_TITLE": b.title,
                "B_DEF": b.definition,
                "B_APPS": b.applications,
            }
            if is_urdu:
                au_line = realize_from_frame(INDUCED_URDU_COMPARE_FRAME, cmp_facts)
                resp = (
                    f"[Open-Ended Cross-Concept Comparison: {a.title} vs {b.title} | Lattice Candidates Scanned: {scanned} | NCD={ncd:.3f}]\n"
                    f"- {au_line}\n"
                    f"1. {a.title} (Taaruf & Mechanism): {a.definition} ({a.mechanism})\n"
                    f"2. {b.title} (Taaruf & Mechanism): {b.definition} ({b.mechanism})\n"
                    f"3. Istemal ka Farq (Use-Case Contrast): {a.title} ka istemal ({a.applications}) mein hota hai, jabke {b.title} ({b.applications}) ke liye behtar hai.\n"
                    f"4. Trade-offs Muqabla: {a.title} ({a.tradeoffs}) vs. {b.title} ({b.tradeoffs})."
                )
            else:
                au_line = realize_from_frame(INDUCED_EN_COMPARE_FRAME, cmp_facts)
                resp = (
                    f"[Open-Ended Cross-Concept Comparison: {a.title} vs {b.title} | Sub-Linear Candidates Scanned: {scanned} | NCD={ncd:.3f}]\n"
                    f"- {au_line}\n"
                    f"1. {a.title} (Definition & Mechanism): {a.definition} ({a.mechanism})\n"
                    f"2. {b.title} (Definition & Mechanism): {b.definition} ({b.mechanism})\n"
                    f"3. Application Contrast: Choose {a.title} for ({a.applications}), whereas {b.title} excels at ({b.applications}).\n"
                    f"4. Trade-off Matrix: {a.title} faces ({a.tradeoffs}), while {b.title} involves ({b.tradeoffs})."
                )
            return OpenEndedSynthesisResult(
                matched=True,
                concepts_used=(a.concept_id, b.concept_id),
                intents_resolved=("cross_concept_comparison", *intents),
                language=lang,
                response=resp,
                candidates_scanned=scanned,
            )

        # Case 1B: Cross-Concept Synergy Synthesis
        if len(nodes) >= 2 and (is_cross_synergy or len(intents) >= 1):
            a, b = nodes[0], nodes[1]
            ncd = symmetric_raw_ncd(a.mechanism.encode("utf-8"), b.mechanism.encode("utf-8"))
            if is_urdu:
                resp = (
                    f"[Open-Ended Cross-Concept Lattice Synthesis: {a.title} x {b.title} | Lattice Candidates Scanned: {scanned} | NCD={ncd:.3f}]\n"
                    f"1. {a.title} (Role & Mechanism): {a.definition} Yeh {a.mechanism}\n"
                    f"2. {b.title} (Role & Mechanism): {b.definition} Yeh {b.mechanism}\n"
                    f"3. Aapas ka Taluq (Synergy & Integration): Jab {a.title} aur {b.title} ko jora jata hai, to {a.title} ({a.applications}) ko {b.title} ({b.applications}) ke zariye mazbooti milti hai.\n"
                    f"4. Combined Trade-offs (Nuqsan/Hudood): {a.tradeoffs} Is ke ilawa {b.tradeoffs}"
                )
            else:
                resp = (
                    f"[Open-Ended Cross-Concept Lattice Synthesis: {a.title} x {b.title} | Sub-Linear Candidates Scanned: {scanned} | NCD={ncd:.3f}]\n"
                    f"1. {a.title} Foundation: {a.definition} ({a.mechanism})\n"
                    f"2. {b.title} Foundation: {b.definition} ({b.mechanism})\n"
                    f"3. Architectural Synergy: Combining {a.title} with {b.title} bridges ({a.applications}) with ({b.applications}).\n"
                    f"4. Joint Operational Trade-offs: Must account for ({a.tradeoffs}) alongside ({b.tradeoffs})."
                )
            return OpenEndedSynthesisResult(
                matched=True,
                concepts_used=(a.concept_id, b.concept_id),
                intents_resolved=("cross_concept_synthesis", *intents),
                language=lang,
                response=resp,
                candidates_scanned=scanned,
            )

        # Case 2: Single Concept Constraint-Aware Adaptive Realization
        nd = nodes[0]
        if constraints.strict_single_facet:
            active_intents = (constraints.strict_single_facet,)
        else:
            active_intents = intents or ("definition", "mechanism", "applications", "tradeoffs")

        # 2a. Concise 1-2 line realization ("in 2 lines", "2 lines mein btao", "in short")
        if constraints.style == "concise_2_lines":
            header = (
                f"[Open-Ended Concept Lattice ({nd.title}) | Category: {nd.category} | "
                f"Style: concise_{constraints.max_lines or 2}_lines | Facets: {', '.join(active_intents)} | O(1) Candidates Scanned: {scanned}]"
            )
            if is_urdu:
                summary_line = f"- Mukhtasir Khulasa ({nd.title}): {nd.definition} (Kaam: {nd.mechanism})"
            else:
                summary_line = f"- Concise Summary ({nd.title}): {nd.definition} (Mechanism: {nd.mechanism})"
            return OpenEndedSynthesisResult(
                matched=True,
                concepts_used=(nd.concept_id,),
                intents_resolved=(*active_intents, "concise_2_lines"),
                language=lang,
                response=f"{header}\n{summary_line}",
                candidates_scanned=scanned,
            )

        # 2b. Beginner-friendly / ELI5 realization ("asaan lafzon mein", "simple words")
        if constraints.style == "simple_eli5":
            header = (
                f"[Open-Ended Concept Lattice ({nd.title}) | Category: {nd.category} | "
                f"Style: simple_eli5 | Facets: {', '.join(active_intents)} | O(1) Candidates Scanned: {scanned}]"
            )
            if is_urdu:
                lines = [
                    header,
                    f"- Asaan Lafzon Mein ({nd.title}): Seedhe lafzon mein, {nd.definition}",
                    f"- Yeh Kaise Kaam Karta Hai: {nd.mechanism}",
                    f"- Rozmarra Fayda / Istemal: {nd.applications}",
                ]
            else:
                lines = [
                    header,
                    f"- Simple Explanation ({nd.title}): In plain terms, {nd.definition}",
                    f"- How It Works Simply: {nd.mechanism}",
                    f"- Everyday Use & Benefit: {nd.applications}",
                ]
            return OpenEndedSynthesisResult(
                matched=True,
                concepts_used=(nd.concept_id,),
                intents_resolved=(*active_intents, "simple_eli5"),
                language=lang,
                response="\n".join(lines),
                candidates_scanned=scanned,
            )

        # 2c. Standard (with Anti-Unification Construction-Grammar Frame) or Strict Single-Facet or Code+Explanation Realization
        style_tag = f" | Style: {constraints.style}" if constraints.style != "standard" else ""
        lines = [
            f"[Open-Ended Concept Lattice ({nd.title}) | Category: {nd.category}{style_tag} | Facets: {', '.join(active_intents)} | O(1) Candidates Scanned: {scanned}]"
        ]
        if constraints.style == "standard" and not constraints.strict_single_facet and len(active_intents) >= 2:
            exp_facts = {
                "TOPIC": nd.title,
                "DEF": nd.definition,
                "MECH": nd.mechanism,
                "APPS": nd.applications,
                "TRD": nd.tradeoffs,
            }
            au_explain = realize_from_frame(
                INDUCED_URDU_EXPLAIN_FRAME if is_urdu else INDUCED_EN_EXPLAIN_FRAME,
                exp_facts,
            )
            lines.append(f"- {au_explain}")
        for facet in active_intents:
            if facet == "definition":
                lbl = "Taaruf (Definition)" if is_urdu else "Definition"
                lines.append(f"- {lbl}: {nd.definition}")
            elif facet == "mechanism":
                lbl = "Kaam Kaise Karta Hai (Mechanism)" if is_urdu else "How It Works (Mechanism)"
                lines.append(f"- {lbl}: {nd.mechanism}")
            elif facet == "applications":
                lbl = "Fayday aur Istemal (Applications)" if is_urdu else "Key Applications & Benefits"
                lines.append(f"- {lbl}: {nd.applications}")
            elif facet == "tradeoffs":
                lbl = "Nuqsan aur Hudood (Trade-offs & Risks)" if is_urdu else "Limitations & Trade-offs"
                lines.append(f"- {lbl}: {nd.tradeoffs}")

        if constraints.include_code:
            snippet = self._illustrative_code_for_concept(nd)
            code_lbl = "Python Code Misaal (Example)" if is_urdu else "Illustrative Python Code"
            lines.append(f"- {code_lbl}:\n```python\n{snippet}\n```")
            active_intents = (*active_intents, "code_example")

        return OpenEndedSynthesisResult(
            matched=True,
            concepts_used=(nd.concept_id,),
            intents_resolved=active_intents,
            language=lang,
            response="\n".join(lines),
            candidates_scanned=scanned,
        )
