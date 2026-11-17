"""Shared, dependency-free text helpers for the native intelligence organs."""
from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset("""
a an the and or but if then else of to in on at by for with from into onto about as is are was were be
been being am do does did doing have has had having i me my mine you your yours he him his she her hers it
its we us our ours they them their theirs this that these those what which who whom whose when where why
how can could would should will shall may might must not no yes so than too very just also please tell
give show explain describe define simple simply words word short brief briefly some any all much many
more most less least there here up down out over under again once only own same such both each few other
let lets let's want need know like i'm it's what's who's where's whats whos wheres im
""".split())

# Words that describe the *form* of a request rather than its topic.
INSTRUCTION_WORDS = frozenset("""
write explain describe define tell give show list summarize summarise compose create make generate
simple simply short brief briefly detail detailed words word sentence sentences paragraph example examples
meaning mean means kind type please quick quickly
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text).casefold())


def stem(word: str) -> str:
    """Tiny suffix stripper; deliberately conservative to avoid false merges."""
    w = word.casefold()
    if w.endswith("'s"):
        w = w[:-2]
    for suffix, min_len in (("ies", 5), ("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if w.endswith(suffix) and len(w) >= min_len:
            if suffix == "ies":
                return w[:-3] + "y"
            if suffix == "es" and not w.endswith(("ses", "xes", "zes", "ches", "shes")):
                return w[:-1]
            if suffix == "s" and w.endswith("ss"):
                return w
            return w[: -len(suffix)]
    return w


def content_words(text: str, *, drop_instructions: bool = True) -> set[str]:
    words = set()
    for tok in tokens(text):
        if tok in STOPWORDS or len(tok) < 3:
            continue
        if drop_instructions and tok in INSTRUCTION_WORDS:
            continue
        words.add(stem(tok))
    return words


def strip_question(text: str) -> str:
    return normalize(text).rstrip(" ?!.").strip()


def clean_entity(value: str) -> str:
    value = normalize(value).strip(" \"'.,!?;:")
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    return value.strip()
