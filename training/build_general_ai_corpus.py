"""
Mirror 7 General-AI Corpus Builder (links -> extract -> filter -> balance -> JSONL).

Pulls openly available datasets and open textbooks directly from their source links,
runs every candidate through the same quality filter, ranks candidates by a quality
score, de-duplicates, decontaminates against benchmark test sets, balances the domain
mix and writes a training file that ``training.gpu_train`` can consume directly.

Why this exists
---------------
Earlier Mirror 7 datasets repeated ~30 hand-written "gold" answers 20+ times. That makes
the model *recite* those answers perfectly while failing on anything it has not seen
(e.g. "who are you ?"). Generalisation comes from many diverse, clean examples seen a
few times each, not from a few examples seen many times. This builder therefore:

* uses each example once (gold anchors included once, not 20x),
* mixes general chat, math reasoning, science, quantum, code and textbook prose,
* adds textbook / encyclopedia *continuation* pairs so the model learns to keep
  writing from context instead of only replaying memorised answers,
* removes other assistants' identities ("I am ChatGPT...") and refusals,
* holds out validation/test prompts and benchmark questions (GSM8K / MATH test).

Sources (all verified to exist; check each license before commercial use)
-------------------------------------------------------------------------
General chat : HuggingFaceTB/smoltalk, HuggingFaceH4/ultrachat_200k, Open-Orca/SlimOrca-Dedup,
               allenai/tulu-3-sft-mixture, databricks/databricks-dolly-15k, local oasst2 file
Math         : nvidia/OpenMathInstruct-2, meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k,
               AI-MO/NuminaMath-CoT, open-r1/OpenR1-Math-220k, openai/gsm8k, EleutherAI/hendrycks_math
Science      : camel-ai/physics|chemistry|biology (CC-BY-NC), allenai/sciq (CC-BY-NC)
Quantum      : migtissera/Quanta, BoltzmannEntropy/QuantumLLMInstruct, Qiskit textbook (Apache-2.0)
Books        : OpenStax University Physics 1-3 / Calculus 1-3 / Statistics / Chemistry (CC-BY-4.0),
               Wikipedia (CC-BY-SA) STEM categories + Vital Articles, HuggingFaceTB/finemath,
               HuggingFaceFW/fineweb-edu
Code         : ise-uiuc/Magicoder-OSS-Instruct-75K, smoltalk/self-oss-instruct
Reasoning    : garage-bAInd/Open-Platypus, openbmb/UltraInteract_sft
Your own     : --extra <url-or-path>[::domain]  (.jsonl/.json Q&A, .txt/.md/.ipynb, .pdf with pypdf)

Usage (Colab)
-------------
    !pip -q install datasets pypdf
    !python -m training.build_general_ai_corpus --output /content/mirror7_general_ai_v1.jsonl
    !python -m training.build_general_ai_corpus --scale 0.2 --only smoltalk-everyday,openstax,wikipedia

With ``datasets`` installed rows are streamed from the Hub; without it the builder falls
back to the public datasets-server REST API (slower, but needs no extra packages).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

USER_AGENT = "Mirror7-CorpusBuilder/1.0 (open research dataset builder; python-urllib)"
DATASETS_SERVER = "https://datasets-server.huggingface.co"

DOMAINS = ("general_chat", "math", "science", "quantum", "books", "code", "reasoning", "identity", "gold")

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: float = 40.0, retries: int = 4, headers: dict[str, str] | None = None) -> bytes:
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (400, 401, 403, 404):
                break
            time.sleep(1.5 * (attempt + 1) + (5 if exc.code == 429 else 0))
        except Exception as exc:  # network hiccup
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last_exc}")


def http_json(url: str, **kw: Any) -> Any:
    return json.loads(http_get(url, **kw).decode("utf-8"))


def read_bytes(path_or_url: str) -> bytes:
    if re.match(r"^https?://", path_or_url):
        return http_get(path_or_url, timeout=120)
    return Path(path_or_url).read_bytes()


# ---------------------------------------------------------------------------
# Hugging Face row streaming (datasets library, with REST fallback)
# ---------------------------------------------------------------------------

def iter_hf_rows(repo: str, config: str | None, split: str, limit: int, seed: int = 7) -> Iterator[dict]:
    """Yield up to ``limit`` rows from a Hub dataset, shuffled, without downloading it whole."""
    yielded = 0
    try:
        import datasets  # type: ignore

        ds = datasets.load_dataset(repo, config, split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=min(50_000, max(2_000, limit * 2)))
        for row in ds:
            yield dict(row)
            yielded += 1
            if yielded >= limit:
                return
        return
    except ImportError:
        pass
    except Exception as exc:
        if yielded:
            log(f"  [{repo}] stream stopped after {yielded} rows: {exc}")
            return
        log(f"  [{repo}] datasets streaming unavailable ({type(exc).__name__}: {exc}); using REST API")
    yield from _iter_hf_rows_rest(repo, config, split, limit, seed)


def _iter_hf_rows_rest(repo: str, config: str | None, split: str, limit: int, seed: int) -> Iterator[dict]:
    q = urllib.parse.quote
    if config is None:
        splits = http_json(f"{DATASETS_SERVER}/splits?dataset={q(repo)}").get("splits", [])
        if not splits:
            raise RuntimeError(f"no splits listed for {repo}")
        config = next((s["config"] for s in splits if s["split"] == split), splits[0]["config"])
    num_rows = None
    try:
        size = http_json(f"{DATASETS_SERVER}/size?dataset={q(repo)}&config={q(config)}")
        for s in size.get("size", {}).get("splits", []):
            if s.get("split") == split:
                num_rows = int(s.get("num_rows", 0))
    except Exception:
        pass
    page = 100
    total_pages = max(1, (num_rows or limit) // page)
    need_pages = max(1, math.ceil(limit / page))
    rng = random.Random(f"{seed}:{repo}:{config}:{split}")
    pages = sorted(rng.sample(range(total_pages), min(total_pages, need_pages)))
    yielded = 0
    for p in pages:
        url = (
            f"{DATASETS_SERVER}/rows?dataset={q(repo)}&config={q(config)}&split={q(split)}"
            f"&offset={p * page}&length={page}"
        )
        try:
            data = http_json(url)
        except Exception as exc:
            log(f"  [{repo}] REST page {p} failed: {exc}")
            continue
        for item in data.get("rows", []):
            yield item.get("row", {})
            yielded += 1
            if yielded >= limit:
                return
        time.sleep(0.15)


# ---------------------------------------------------------------------------
# Text normalisation & utilities
# ---------------------------------------------------------------------------

_REPLACEMENTS = {
    "\u27e8": "<", "\u27e9": ">", "\u2013": "-", "\u2014": "--", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ",
    "\u2009": " ", "\u202f": " ", "\u200b": "", "\ufeff": "", "\u00ad": "",
    "\r\n": "\n", "\r": "\n", "\t": "    ",
}


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    for src, dst in _REPLACEMENTS.items():
        s = s.replace(src, dst)
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ ]{3,}(?=\S)", "  ", s) if "```" not in s else s
    return s.strip()


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _cut_utf8(s: str, max_bytes: int) -> str:
    return s.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


_SENT_END_RE = re.compile(r"(?<=[.!?])[\"')\]]?(?=\s)")


def truncate_at_sentence(s: str, max_bytes: int, min_keep_ratio: float = 0.5) -> str | None:
    """Trim to <= max_bytes ending on a sentence/paragraph boundary, or None if impossible."""
    if utf8_len(s) <= max_bytes:
        return s
    cut = _cut_utf8(s, max_bytes)
    best = -1
    for m in _SENT_END_RE.finditer(cut):
        best = m.end()
    para = cut.rfind("\n\n")
    best = max(best, para)
    if best < int(len(cut) * min_keep_ratio):
        return None
    return cut[:best].rstrip()


def split_chunks(text: str, max_bytes: int, min_bytes: int = 120) -> list[str]:
    """Pack paragraphs (then sentences) into chunks of <= max_bytes."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paras:
        if utf8_len(p) <= max_bytes:
            units.append(p)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        units.extend(x for x in sentences if 0 < utf8_len(x) <= max_bytes)
    chunks: list[str] = []
    cur = ""
    for u in units:
        candidate = f"{cur}\n\n{u}" if cur else u
        if utf8_len(candidate) <= max_bytes:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = u
    if cur:
        chunks.append(cur)
    return [c for c in chunks if utf8_len(c) >= min_bytes]


def tail_sentence(text: str, max_bytes: int = 220) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = ""
    for s in reversed(sentences):
        cand = f"{s} {out}".strip()
        if utf8_len(cand) > max_bytes:
            break
        out = cand
    return out or _cut_utf8(text.strip()[-max_bytes:], max_bytes)


def stable_hash(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()[:8], "big")


def norm_key(s: str, n: int = 400) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()[:n]


def last_boxed(s: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} with balanced braces."""
    idx = max(s.rfind("\\boxed{"), s.rfind("\\fbox{"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j].strip()
    return None


def normalize_math_solution(sol: str, answer: str | None = None) -> str:
    """Make every math solution end with one explicit, uniform final-answer sentence."""
    s = clean_text(sol)
    s = re.sub(r"\n*####\s*([^\n]+)\s*$", lambda m: f"\nTherefore, the final answer is {m.group(1).strip()}.", s)
    s = re.sub(
        r"\n*The answer is:?\s*([^\n]+?)\s*$",
        lambda m: f"\nTherefore, the final answer is {m.group(1).strip().rstrip('.')}.",
        s,
    )
    if "final answer is" not in s.lower():
        boxed = last_boxed(s)
        final = boxed if boxed else (clean_text(answer) if answer else None)
        if final and utf8_len(final) <= 80:
            s = f"{s.rstrip()}\nTherefore, the final answer is {final}."
    return s


# ---------------------------------------------------------------------------
# Quality filter & scoring
# ---------------------------------------------------------------------------

REFUSAL_RE = re.compile(
    r"(as an ai\b|as a language model|ai language model|i(?:'m| am) (?:just |only )?an ai\b|"
    r"i (?:cannot|can't|can not) (?:help|assist|fulfill|provide|comply)|i'm sorry,? but i (?:can(?:no|')t|won't)|"
    r"i do not have (?:personal|the ability|access)|i don't have (?:personal|feelings|access|the ability)|"
    r"my knowledge cutoff|as of my last (?:update|training))",
    re.IGNORECASE,
)
FOREIGN_IDENTITY_RE = re.compile(
    r"\b(openai|chatgpt|gpt-?3(?:\.5)?|gpt-?4o?|claude|anthropic|smollm|llama ?\d|meta ai|qwen|alibaba cloud|"
    r"gemini|google bard|mistral ai|open ?assistant|laion|tulu|hermes|orca)\b",
    re.IGNORECASE,
)
HTML_RE = re.compile(r"<\s*(?:div|span|script|style|table|img|br|iframe|a\s+href|p\s|p>)", re.IGNORECASE)
URL_RE = re.compile(r"https?://", re.IGNORECASE)
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
BROKEN_ARTIFACT_RE = re.compile(
    r"(?:for the following exercises|match the .* to the correct graph|"
    r"\b(?:the graph|the figure|the table|the diagram) below\b|"
    r"\b(?:in|shown in|see)\s+[,.;]|(?:^|\n)-\s*(?:\n|$))",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"(?:^|\n)\s*(?:\d+[.)]\s|step\s*\d|first,|\*\*\d|- )", re.IGNORECASE)
CONNECTIVE_RE = re.compile(r"\b(therefore|because|since|thus|hence|so that|which means|it follows|implies)\b", re.I)
FINAL_ANSWER_RE = re.compile(r"(final answer is|\\boxed\{|answer is[: ])", re.IGNORECASE)
CLEAN_END_CHARS = set(".!?\"')]}`$>*:;0123456789") | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class FilterConfig:
    max_prompt_bytes: int = 900
    max_target_bytes: int = 1000
    min_prompt_bytes: int = 2
    min_target_bytes: int = 40


def _ascii_letter_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if c.isascii()) / len(letters)


def _repetition_ok(t: str) -> bool:
    window = t[:700]
    if len(window) >= 120:
        tri = [window[i : i + 3] for i in range(len(window) - 2)]
        if len(set(tri)) / max(1, len(tri)) < 0.30:
            return False
    words = t.split()
    if len(words) >= 40:
        grams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
        if len(set(grams)) / len(grams) < 0.80:
            return False
    lines = [ln.strip() for ln in t.split("\n") if len(ln.strip()) > 20]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.75:
        return False
    return True


def assess(
    prompt: str,
    target: str,
    spec: "SourceSpec",
    fcfg: FilterConfig,
) -> tuple[bool, float, str, str, str]:
    """Return (accepted, score, clean_prompt, clean_target, reject_reason)."""
    p = clean_text(prompt)
    t = clean_text(target)
    if not p or not t:
        return False, 0.0, p, t, "empty"
    if spec.math:
        t = normalize_math_solution(t)

    if utf8_len(p) > fcfg.max_prompt_bytes:
        return False, 0.0, p, t, "prompt_too_long"
    if utf8_len(p) < max(fcfg.min_prompt_bytes, spec.min_prompt_bytes):
        return False, 0.0, p, t, "prompt_too_short"
    if utf8_len(t) > fcfg.max_target_bytes:
        if not spec.allow_truncate:
            return False, 0.0, p, t, "target_too_long"
        cut = truncate_at_sentence(t, fcfg.max_target_bytes)
        if cut is None:
            return False, 0.0, p, t, "target_untruncatable"
        t = cut
    if utf8_len(t) < max(fcfg.min_target_bytes, spec.min_target_bytes):
        return False, 0.0, p, t, "target_too_short"

    both = f"{p}\n{t}"
    if REFUSAL_RE.search(t):
        return False, 0.0, p, t, "refusal_or_disclaimer"
    if spec.domain != "identity" and FOREIGN_IDENTITY_RE.search(t):
        return False, 0.0, p, t, "foreign_identity"
    if CJK_RE.search(both):
        return False, 0.0, p, t, "non_english"
    if _ascii_letter_ratio(both) < 0.80:
        return False, 0.0, p, t, "non_english"
    if not spec.allow_code and HTML_RE.search(both):
        return False, 0.0, p, t, "html"
    if not spec.allow_urls and URL_RE.search(t):
        return False, 0.0, p, t, "url"
    if "\ufffd" in both:
        return False, 0.0, p, t, "mojibake"
    if BROKEN_ARTIFACT_RE.search(both):
        return False, 0.0, p, t, "broken_figure_or_exercise_ref"
    if not _repetition_ok(t):
        return False, 0.0, p, t, "repetitive"
    if t[-1] not in CLEAN_END_CHARS and not spec.allow_code:
        return False, 0.0, p, t, "unclean_end"
    if spec.math and not FINAL_ANSWER_RE.search(t):
        return False, 0.0, p, t, "no_final_answer"

    # Quality score: source prior + structure/reasoning signals + healthy length band.
    score = spec.prior
    tb = utf8_len(t)
    if STEP_RE.search(t):
        score += 0.05
    if CONNECTIVE_RE.search(t):
        score += 0.04
    if spec.math and FINAL_ANSWER_RE.search(t):
        score += 0.03
    if 180 <= tb <= 900:
        score += 0.04
    elif tb < 90 and spec.domain not in ("general_chat", "identity"):
        score -= 0.06
    if utf8_len(p) < 12 and spec.domain not in ("general_chat", "identity"):
        score -= 0.05
    return True, max(0.0, min(1.0, round(score, 4))), p, t, ""


# ---------------------------------------------------------------------------
# Row adapters
# ---------------------------------------------------------------------------

_ROLE_MAP = {"user": "user", "human": "user", "prompter": "user", "assistant": "assistant", "gpt": "assistant",
             "bot": "assistant", "model": "assistant", "system": "system"}


def pairs_from_messages(messages: Any, max_prompt_bytes: int = 900, max_pairs: int = 3) -> list[tuple[str, str]]:
    """Turn a chat transcript into (prompt, response) pairs; later turns carry 'User:/Assistant:' history."""
    if not isinstance(messages, list):
        return []
    turns: list[tuple[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = _ROLE_MAP.get(str(m.get("role", m.get("from", ""))).lower())
        content = clean_text(m.get("content", m.get("value", "")))
        if role in ("user", "assistant") and content:
            turns.append((role, content))
    out: list[tuple[str, str]] = []
    history: list[str] = []
    i = 0
    while i < len(turns) - 1 and len(out) < max_pairs:
        if turns[i][0] == "user" and turns[i + 1][0] == "assistant":
            user, reply = turns[i][1], turns[i + 1][1]
            if history:
                prompt = "\n".join(history + [f"User: {user}"])
                if utf8_len(prompt) <= max_prompt_bytes:
                    out.append((prompt, reply))
            else:
                out.append((user, reply))
            history += [f"User: {user}", f"Assistant: {reply}"]
            i += 2
        else:
            i += 1
    return out


def guess_pair_from_row(row: dict) -> list[tuple[str, str]]:
    """Auto-detect common instruction-dataset schemas."""
    for key in ("messages", "conversations", "conversation"):
        if isinstance(row.get(key), list):
            return pairs_from_messages(row[key])
    candidates = [
        ("instruction", "output"), ("instruction", "response"), ("question", "answer"), ("problem", "solution"),
        ("prompt", "response"), ("prompt", "completion"), ("query", "response"), ("user_text", "target_text"),
        ("input", "output"), ("message_1", "message_2"), ("Question", "Answer"),
    ]
    for pk, rk in candidates:
        if row.get(pk) and row.get(rk):
            prompt = str(row[pk])
            if pk == "instruction" and row.get("input"):
                prompt = f"{prompt}\n\n{row['input']}"
            return [(prompt, str(row[rk]))]
    return []


# ---------------------------------------------------------------------------
# Document (book / encyclopedia / web) -> training pairs
# ---------------------------------------------------------------------------

FIRST_TEMPLATES = (
    "Explain {h}.",
    "What is {h}? Explain clearly.",
    "Teach me about {h}.",
    "Give a clear explanation of {h}.",
    "Can you explain {h} step by step?",
    "Tell me about {h}.",
    "I'm studying {h}. Can you explain it?",
)
CONTINUE_TEMPLATES = (
    'Continue explaining {h}. Pick up after: "{tail}"',
    'Keep going with {h}. The last part was: "{tail}"',
    'What comes next in the explanation of {h}? It left off at: "{tail}"',
)


def document_to_pairs(
    title: str,
    sections: list[tuple[str, str]],
    max_target_bytes: int,
    max_chunks_per_section: int = 4,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for heading, text in sections:
        text = clean_text(text)
        if utf8_len(text) < 150:
            continue
        h = heading.strip() or title
        if title and h.lower() != title.lower() and title.lower() not in h.lower():
            h = f"{h} ({title})"
        chunks = split_chunks(text, max_target_bytes - 8)[:max_chunks_per_section]
        for ci, chunk in enumerate(chunks):
            if ci == 0:
                tpl = FIRST_TEMPLATES[stable_hash(h, "first") % len(FIRST_TEMPLATES)]
                prompt = tpl.format(h=h)
            else:
                tpl = CONTINUE_TEMPLATES[stable_hash(h, str(ci)) % len(CONTINUE_TEMPLATES)]
                prompt = tpl.format(h=h, tail=tail_sentence(chunks[ci - 1]))
            pairs.append((prompt, chunk))
    return pairs


def continuation_pairs(text: str, kind: str, max_target_bytes: int, max_pairs: int = 3) -> list[tuple[str, str]]:
    """Pretraining-style 'continue this text' pairs for headingless documents."""
    chunks = split_chunks(clean_text(text), max_target_bytes - 8, min_bytes=200)
    pairs: list[tuple[str, str]] = []
    for i in range(1, min(len(chunks), max_pairs + 1)):
        lead = tail_sentence(chunks[i - 1], 320)
        pairs.append((f'Continue this {kind} text:\n"{lead}"', chunks[i]))
    return pairs


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def markdown_to_sections(md: str, default_title: str) -> tuple[str, list[tuple[str, str]]]:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)            # images
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)        # links -> text
    md = re.sub(r"</?(?:details|summary|div|span|p|br|img|a|b|i|em|strong|center|sup|sub|font)[^>]*>", "", md, flags=re.I)
    md = re.sub(r"<[^>\n]{1,200}>", "", md)
    title = default_title
    sections: list[tuple[str, str]] = []
    cur_head, buf = default_title, []
    for line in md.split("\n"):
        m = _MD_HEADING_RE.match(line)
        if m:
            if buf:
                sections.append((cur_head, "\n".join(buf)))
            cur_head = re.sub(r"[*_`]", "", m.group(2)).strip()
            if len(m.group(1)) == 1 and title == default_title:
                title = cur_head
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((cur_head, "\n".join(buf)))
    return title, sections


def ipynb_to_sections(raw: bytes, default_title: str) -> tuple[str, list[tuple[str, str]]]:
    nb = json.loads(raw.decode("utf-8"))
    md_cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", "")
            md_cells.append("".join(src) if isinstance(src, list) else str(src))
    return markdown_to_sections("\n\n".join(md_cells), default_title)


# ---- MathML / OpenStax CNXML ------------------------------------------------

def _ln(tag: Any) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _ln(c.tag) == name]


def _first_desc(el: ET.Element, name: str) -> ET.Element | None:
    for d in el.iter():
        if d is not el and _ln(d.tag) == name:
            return d
    return None


_MO_SPACED = {"=", "+", "-", "<", ">", "\u2264", "\u2265", "\u2248", "\u2192", "\u2260", "\u00b1", "\u221d", "\u2261"}
_MO_MAP = {"\u2212": "-", "\u00d7": "*", "\u22c5": "*", "\u2219": "*", "\u2062": "", "\u2061": "", "\u2063": ","}


def mathml_to_text(el: ET.Element) -> str:
    tag = _ln(el.tag)
    kids = list(el)

    def m(i: int) -> str:
        return mathml_to_text(kids[i]) if i < len(kids) else ""

    def wrap(x: str) -> str:
        x = x.strip()
        return x if (len(x) <= 1 or re.fullmatch(r"[A-Za-z0-9.]+", x)) else f"({x})"

    if tag in ("mi", "mn", "mtext", "ms"):
        return (el.text or "").strip()
    if tag == "mo":
        t = _MO_MAP.get((el.text or "").strip(), (el.text or "").strip())
        return f" {t} " if t in _MO_SPACED else t
    if tag == "mspace":
        return " "
    if tag == "mfrac":
        return f"{wrap(m(0))}/{wrap(m(1))}"
    if tag == "msup":
        return f"{wrap(m(0))}^{wrap(m(1))}"
    if tag == "msub":
        return f"{m(0)}_{wrap(m(1))}"
    if tag == "msubsup":
        return f"{m(0)}_{wrap(m(1))}^{wrap(m(2))}"
    if tag == "msqrt":
        return f"sqrt({''.join(mathml_to_text(k) for k in kids)})"
    if tag == "mroot":
        return f"root[{m(1)}]({m(0)})"
    if tag == "munderover":
        return f"{m(0)}_{wrap(m(1))}^{wrap(m(2))}"
    if tag == "munder":
        return f"{m(0)}_{wrap(m(1))}"
    if tag == "mover":
        acc = m(1).strip()
        if acc in ("\u2192", "\u21c0", "\u20d7"):
            return f"vec({m(0).strip()})"
        if acc in ("^", "\u02c6"):
            return f"hat({m(0).strip()})"
        if acc in ("\u00af", "_", "\u2015", "\u203e"):
            return f"bar({m(0).strip()})"
        if acc in (".", "\u02d9"):
            return f"dot({m(0).strip()})"
        return m(0)
    if tag == "mfenced":
        o, c = el.get("open", "("), el.get("close", ")")
        return o + ", ".join(mathml_to_text(k) for k in kids) + c
    if tag == "mtable":
        return "[" + ", ".join(mathml_to_text(k) for k in kids) + "]"
    if tag == "mtr":
        return "[" + ", ".join(mathml_to_text(k) for k in kids) + "]"
    if tag in ("annotation", "annotation-xml"):
        return ""
    return "".join(mathml_to_text(k) for k in kids)


_CNX_SKIP = {"figure", "media", "image", "table", "metadata", "glossary", "iframe", "download", "object", "caption"}


def _cnx_inline(el: ET.Element) -> str:
    tag = _ln(el.tag)
    if tag == "math":
        txt = re.sub(r"\s{2,}", " ", mathml_to_text(el)).strip()
        return f" {txt} "
    if tag in _CNX_SKIP:
        return ""
    parts = [el.text or ""]
    for ch in el:
        sub = _cnx_inline(ch)
        if _ln(ch.tag) == "title":
            sub = sub.strip() + ": " if sub.strip() else ""
        elif _ln(ch.tag) == "item":
            sub = "\n- " + sub.strip()
        elif _ln(ch.tag) == "newline":
            sub = "\n"
        parts.append(sub)
        parts.append(ch.tail or "")
    return "".join(parts)


def _cnx_block_text(el: ET.Element) -> str:
    txt = _cnx_inline(el)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r" *\n *", "\n", txt)
    return txt.strip()


def _cnx_problem_solution(el: ET.Element) -> tuple[str, str]:
    prob = _first_desc(el, "problem")
    sol = _first_desc(el, "solution")
    q = _cnx_block_text(prob) if prob is not None else ""
    a = _cnx_block_text(sol) if sol is not None else ""
    return q, a


def _cnx_example(el: ET.Element) -> tuple[str, str]:
    q, a = _cnx_problem_solution(el)
    if q and a:
        return q, a
    q_parts, a_parts, mode = [], [], "q"
    for ch in el:
        tag = _ln(ch.tag)
        if tag in ("title",) or tag in _CNX_SKIP:
            continue
        txt = _cnx_block_text(ch)
        if not txt:
            continue
        if mode == "q" and re.match(r"^(Strategy|Solution)\b", txt):
            mode = "a"
        (q_parts if mode == "q" else a_parts).append(txt)
    return "\n\n".join(q_parts), "\n\n".join(a_parts)


def parse_cnxml(raw: bytes) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]]]:
    """OpenStax module -> (title, [(heading, text)], [(problem, worked_solution)])."""
    root = ET.fromstring(raw)
    title_el = next((c for c in root if _ln(c.tag) == "title"), None)
    title = _cnx_block_text(title_el) if title_el is not None else "OpenStax"
    content = next((c for c in root if _ln(c.tag) == "content"), None)
    sections: list[tuple[str, str]] = []
    qa: list[tuple[str, str]] = []
    if content is None:
        return title, sections, qa

    def walk(container: ET.Element, heading: str) -> None:
        blocks: list[str] = []
        for ch in container:
            tag = _ln(ch.tag)
            if tag == "section":
                t_el = next((c for c in ch if _ln(c.tag) == "title"), None)
                walk(ch, _cnx_block_text(t_el) if t_el is not None else heading)
            elif tag == "example":
                q, a = _cnx_example(ch)
                if q and a:
                    qa.append((q, a))
            elif tag == "exercise":
                q, a = _cnx_problem_solution(ch)
                if q and a:
                    qa.append((q, a))
            elif tag in ("para", "equation", "list", "quote", "definition", "rule"):
                txt = _cnx_block_text(ch)
                if txt:
                    blocks.append(txt)
            elif tag == "note":
                cls = (ch.get("class") or "").lower()
                if any(k in cls for k in ("interactive", "media", "link", "check-understanding")):
                    continue
                inner = [_cnx_block_text(c) for c in ch if _ln(c.tag) in ("para", "list", "equation")]
                blocks.extend(x for x in inner if x)
        if blocks:
            sections.append((heading, "\n\n".join(blocks)))

    walk(content, title)
    return title, sections, qa


# ---- Wikipedia (MediaWiki API) ---------------------------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes", "bibliography", "sources",
    "citations", "gallery", "footnotes", "works cited", "notes and references", "explanatory notes",
}
DEFAULT_WIKI_CATEGORIES = (
    "Quantum mechanics", "Quantum information science", "Quantum computing", "Quantum field theory",
    "Quantum algorithms", "Linear algebra", "Calculus", "Probability theory", "Statistics", "Number theory",
    "Classical mechanics", "Electromagnetism", "Thermodynamics", "Special relativity", "General relativity",
    "Optics", "Particle physics", "Condensed matter physics", "Mathematical logic", "Algorithms",
    "Machine learning", "Artificial intelligence", "Cognitive biases", "Fallacies", "Critical thinking",
    "Theorems in algebra", "Differential equations", "Group theory", "Graph theory", "Information theory",
)


def _wiki_clean(text: str) -> str:
    out, i = [], 0
    while True:
        j = text.find("{\\displaystyle", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        depth, k = 0, j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        inner = text[j + len("{\\displaystyle") : k].strip()
        out.append(inner)
        i = k + 1
    s = "".join(out)
    s = re.sub(r"\[\d+\]", "", s)
    return s


def wiki_category_titles(category: str, limit: int) -> list[str]:
    titles: list[str] = []
    cont: dict[str, str] = {}
    while len(titles) < limit:
        params = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                  "cmnamespace": "0", "cmtype": "page", "cmlimit": "500", "format": "json", **cont}
        data = http_json(f"{WIKI_API}?{urllib.parse.urlencode(params)}")
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        if "continue" not in data:
            break
        cont = {"cmcontinue": data["continue"]["cmcontinue"], "continue": data["continue"]["continue"]}
    return titles[:limit]


def wiki_vital_titles(limit: int) -> list[str]:
    titles: list[str] = []
    cont: dict[str, str] = {}
    while len(titles) < limit:
        params = {"action": "query", "prop": "links", "titles": "Wikipedia:Vital articles", "plnamespace": "0",
                  "pllimit": "max", "format": "json", **cont}
        data = http_json(f"{WIKI_API}?{urllib.parse.urlencode(params)}")
        for page in data.get("query", {}).get("pages", {}).values():
            titles += [ln["title"] for ln in page.get("links", [])]
        if "continue" not in data:
            break
        cont = {"plcontinue": data["continue"]["plcontinue"], "continue": data["continue"]["continue"]}
    return titles[:limit]


def wiki_fetch_sections(title: str) -> tuple[str, list[tuple[str, str]]]:
    params = {"action": "query", "prop": "extracts", "explaintext": "1", "exsectionformat": "wiki",
              "redirects": "1", "titles": title, "format": "json"}
    data = http_json(f"{WIKI_API}?{urllib.parse.urlencode(params)}")
    pages = data.get("query", {}).get("pages", {})
    text = next(iter(pages.values()), {}).get("extract", "") if pages else ""
    text = _wiki_clean(text or "")
    sections: list[tuple[str, str]] = []
    cur, buf = title, []
    for line in text.split("\n"):
        m = re.match(r"^(=+)\s*(.+?)\s*\1\s*$", line)
        if m:
            if buf:
                sections.append((cur, "\n".join(buf)))
            cur, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((cur, "\n".join(buf)))
    return title, [(h, t) for h, t in sections if h.lower() not in WIKI_SKIP_SECTIONS]


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

PairIter = Iterator[tuple[str, str]]


@dataclass
class SourceSpec:
    name: str
    domain: str
    license: str
    commercial_ok: bool
    quota: int
    fetch: Callable[["BuildContext", int], PairIter]
    prior: float = 0.80
    math: bool = False
    allow_truncate: bool = False
    allow_code: bool = False
    allow_urls: bool = False
    min_prompt_bytes: int = 2
    min_target_bytes: int = 40
    notes: str = ""


@dataclass
class BuildContext:
    seed: int
    fcfg: FilterConfig
    workers: int
    local_oasst_path: str
    wiki_categories: tuple[str, ...]
    model_params: str
    extra: list[str] = field(default_factory=list)


def _hf_fields(repo: str, config: str | None, split: str, pk: str, rk: str,
               transform: Callable[[dict], list[tuple[str, str]]] | None = None):
    def fetch(ctx: BuildContext, limit: int) -> PairIter:
        for row in iter_hf_rows(repo, config, split, limit, ctx.seed):
            if transform is not None:
                yield from transform(row)
            elif row.get(pk) and row.get(rk):
                yield str(row[pk]), str(row[rk])
    return fetch


def _hf_messages(repo: str, config: str | None, split: str, key: str = "messages"):
    def fetch(ctx: BuildContext, limit: int) -> PairIter:
        for row in iter_hf_rows(repo, config, split, limit, ctx.seed):
            yield from pairs_from_messages(row.get(key), ctx.fcfg.max_prompt_bytes)
    return fetch


def _dolly_transform(row: dict) -> list[tuple[str, str]]:
    inst, ctx_text, resp = row.get("instruction", ""), row.get("context", ""), row.get("response", "")
    if not inst or not resp:
        return []
    prompt = f"{inst}\n\nContext: {ctx_text}" if ctx_text else inst
    return [(prompt, resp)]


def _sciq_transform(row: dict) -> list[tuple[str, str]]:
    q, s, a = row.get("question", ""), clean_text(row.get("support", "")), row.get("correct_answer", "")
    if not q or not a or len(s) < 60:
        return []
    return [(q, f"{s}\n\nTherefore, the answer is {a}.")]


def _openr1_transform(row: dict) -> list[tuple[str, str]]:
    verify = row.get("correctness_math_verify") or []
    if isinstance(verify, list) and verify and not any(bool(v) for v in verify):
        return []
    prob, sol, ans = row.get("problem"), row.get("solution"), row.get("answer")
    if not prob or not sol:
        return []
    return [(str(prob), normalize_math_solution(str(sol), str(ans) if ans else None))]


def _omi2_transform(row: dict) -> list[tuple[str, str]]:
    prob, sol, exp = row.get("problem"), str(row.get("generated_solution") or ""), str(row.get("expected_answer") or "")
    if not prob or not sol:
        return []
    boxed = last_boxed(sol)
    if exp and boxed is not None and re.sub(r"\s", "", boxed) != re.sub(r"\s", "", exp):
        return []
    return [(str(prob), sol)]


def _metamath_transform(row: dict) -> list[tuple[str, str]]:
    q, r = row.get("query"), row.get("response")
    return [(str(q), str(r))] if q and r else []


def _fetch_hendrycks(ctx: BuildContext, limit: int) -> PairIter:
    subjects = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus")
    per = max(1, limit // len(subjects))
    for subj in subjects:
        for row in iter_hf_rows("EleutherAI/hendrycks_math", subj, "train", per, ctx.seed):
            if row.get("problem") and row.get("solution"):
                yield str(row["problem"]), str(row["solution"])


def _fetch_local_oasst(ctx: BuildContext, limit: int) -> PairIter:
    path = Path(ctx.local_oasst_path)
    if not path.exists():
        log(f"  [oasst2-local] {path} not found; skipping")
        return
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            yield r.get("user_text", ""), r.get("target_text", "")
            n += 1
            if n >= limit:
                return


def _doc_fetch(kind: str, repo: str, config: str | None, min_score: float = 0.0):
    def fetch(ctx: BuildContext, limit: int) -> PairIter:
        docs = max(1, limit // 2)
        for row in iter_hf_rows(repo, config, "train", docs, ctx.seed):
            if min_score and float(row.get("score", 0) or 0) < min_score:
                continue
            text = row.get("text", "")
            if text:
                yield from continuation_pairs(text, kind, ctx.fcfg.max_target_bytes)
    return fetch


def _github_tree(repo: str, branch: str = "main") -> list[str]:
    data = http_json(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1")
    return [x["path"] for x in data.get("tree", []) if x.get("type") == "blob"]


def _parallel_map(fn: Callable[[str], Any], items: list[str], workers: int) -> Iterator[Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, it): it for it in items}
        for fut in concurrent.futures.as_completed(futures):
            try:
                yield fut.result()
            except Exception as exc:
                log(f"    fetch failed for {futures[fut]}: {exc}")


def _fetch_qiskit_textbook(ctx: BuildContext, limit: int) -> PairIter:
    repo = "Qiskit/textbook"
    paths = sorted(p for p in _github_tree(repo) if p.endswith(".ipynb") and p.startswith("notebooks/")
                   and "/ch-labs/" not in p and "/ch-ex/" not in p and "exercise" not in p.lower())

    def load(path: str):
        raw = http_get(f"https://raw.githubusercontent.com/{repo}/main/{urllib.parse.quote(path)}")
        default = Path(path).stem.replace("-", " ").replace("_", " ").title()
        return ipynb_to_sections(raw, default)

    n = 0
    for title, sections in _parallel_map(load, paths, ctx.workers):
        for pair in document_to_pairs(title, sections, ctx.fcfg.max_target_bytes):
            yield pair
            n += 1
            if n >= limit:
                return


OPENSTAX_BUNDLES = (
    "openstax/osbooks-university-physics-bundle",
    "openstax/osbooks-calculus-bundle",
    "openstax/osbooks-introductory-statistics-bundle",
    "openstax/osbooks-chemistry-bundle",
)


def _fetch_openstax(ctx: BuildContext, limit: int) -> PairIter:
    jobs: list[str] = []
    for repo in OPENSTAX_BUNDLES:
        try:
            jobs += [f"{repo}|{p}" for p in _github_tree(repo) if p.endswith("index.cnxml")]
        except Exception as exc:
            log(f"  [openstax] cannot list {repo}: {exc}")

    def load(job: str):
        repo, path = job.split("|", 1)
        return parse_cnxml(http_get(f"https://raw.githubusercontent.com/{repo}/main/{path}"))

    n = 0
    for title, sections, qa in _parallel_map(load, sorted(jobs), ctx.workers):
        out = list(qa) + document_to_pairs(title, sections, ctx.fcfg.max_target_bytes)
        for pair in out:
            yield pair
            n += 1
            if n >= limit:
                return


def _fetch_wikipedia(ctx: BuildContext, limit: int) -> PairIter:
    titles: list[str] = []
    try:
        titles += wiki_vital_titles(1200)
    except Exception as exc:
        log(f"  [wikipedia] vital articles unavailable: {exc}")
    per_cat = max(20, limit // max(1, len(ctx.wiki_categories)) // 3)
    for cat in ctx.wiki_categories:
        try:
            titles += wiki_category_titles(cat, per_cat)
        except Exception as exc:
            log(f"  [wikipedia] category {cat!r} failed: {exc}")
    seen: set[str] = set()
    uniq = [t for t in titles if not (t in seen or seen.add(t)) and not t.startswith(("List of", "Outline of", "Index of"))]
    rng = random.Random(ctx.seed)
    rng.shuffle(uniq)
    uniq = uniq[: max(50, limit // 3)]
    n = 0
    for title, sections in _parallel_map(wiki_fetch_sections, uniq, min(4, ctx.workers)):
        for pair in document_to_pairs(title, sections, ctx.fcfg.max_target_bytes, max_chunks_per_section=2):
            yield pair
            n += 1
            if n >= limit:
                return


def _fetch_extra(ctx: BuildContext, limit: int) -> PairIter:
    """User-supplied links/files: Q&A JSON(L) or documents (txt/md/ipynb/pdf)."""
    n = 0
    for item in ctx.extra:
        src = item.split("::", 1)[0]
        low = src.lower().split("?", 1)[0]
        try:
            raw = read_bytes(src)
        except Exception as exc:
            log(f"  [extra] cannot read {src}: {exc}")
            continue
        pairs: list[tuple[str, str]] = []
        title = Path(urllib.parse.urlparse(src).path).stem.replace("_", " ").replace("-", " ").title()
        if low.endswith(".jsonl"):
            for line in raw.decode("utf-8", errors="ignore").splitlines():
                try:
                    pairs += guess_pair_from_row(json.loads(line))
                except Exception:
                    continue
        elif low.endswith(".json"):
            data = json.loads(raw.decode("utf-8", errors="ignore"))
            rows = data if isinstance(data, list) else data.get("data", data.get("rows", []))
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, dict):
                    pairs += guess_pair_from_row(row)
        elif low.endswith(".ipynb"):
            t, secs = ipynb_to_sections(raw, title)
            pairs = document_to_pairs(t, secs, ctx.fcfg.max_target_bytes)
        elif low.endswith(".pdf"):
            try:
                import io

                from pypdf import PdfReader  # type: ignore
            except ImportError:
                log("  [extra] pip install pypdf to read PDFs; skipping " + src)
                continue
            reader = PdfReader(io.BytesIO(raw))
            pages = [clean_text(p.extract_text() or "") for p in reader.pages]
            text = "\n\n".join(re.sub(r"(?<![.!?:])\n(?!\n)", " ", p) for p in pages)
            pairs = document_to_pairs(title, [(title, text)], ctx.fcfg.max_target_bytes, max_chunks_per_section=10_000)
        else:  # .txt / .md / anything textual
            t, secs = markdown_to_sections(raw.decode("utf-8", errors="ignore"), title)
            pairs = document_to_pairs(t, secs, ctx.fcfg.max_target_bytes, max_chunks_per_section=200)
        log(f"  [extra] {src}: {len(pairs):,} raw pairs")
        for pair in pairs:
            yield pair
            n += 1
            if n >= limit:
                return


# ---- Mirror 7 identity (varied wording; a few prompts held out for eval) ----

IDENTITY_HELDOUT_PROMPTS = (
    "who r u?", "Please introduce yourself briefly.", "What should I call you?", "Are you Siri?",
    "Whom am I chatting with right now?", "Describe yourself in one sentence.",
)


def build_identity_pairs(model_params: str) -> list[tuple[str, str]]:
    arch = (f"a {model_params}-parameter native byte-level model with a bidirectional GRU encoder, multi-head "
            "cross-attention, a recurrent decoder, sparse mixture-of-experts layers and a persistent memory bank")
    intents: dict[str, list[str]] = {
        "who": [
            f"I'm Mirror 7, an AI assistant. Under the hood I am {arch}. I can chat, explain ideas, write code and work through math, physics and quantum computing problems.",
            "I'm Mirror 7, a conversational AI assistant. I read and write text one byte at a time, and I'm especially focused on step-by-step reasoning in math, physics and quantum computing, though I'm happy to chat about anything.",
            "Hi! I'm Mirror 7. I'm an AI model built to hold conversations and help you think problems through, from everyday questions to proofs and quantum circuits.",
        ],
        "name": [
            "My name is Mirror 7. You can just call me Mirror.",
            "I'm called Mirror 7. It's nice to meet you!",
        ],
        "not_other": [
            "No. I'm Mirror 7, an independent model with my own architecture and training. I'm not ChatGPT, GPT-4, Claude, Gemini or Siri, and I wasn't made by OpenAI, Anthropic, Google or Apple.",
            "No, I'm Mirror 7. I'm a separate, independently trained model, not a product of OpenAI or any other big AI lab.",
        ],
        "human": [
            "I'm not a human. I'm Mirror 7, an AI model that generates text. I don't have a body, feelings or personal experiences, but I'll do my best to be helpful.",
        ],
        "creator": [
            "I was built as part of the independent Mirror 7 research project, which designs and trains its own native byte-level language models rather than fine-tuning someone else's.",
            "I come from the Mirror 7 project, an independent effort that trains its own model architecture from scratch.",
        ],
        "arch": [
            f"I'm {arch}. I don't use a fixed vocabulary of words; I read raw UTF-8 bytes, encode your message in both directions, and generate my reply byte by byte while attending back to your message.",
        ],
        "can_do": [
            "I can help with:\n1. Everyday questions and conversation.\n2. Explaining concepts in science, history and technology.\n3. Mathematics: algebra, calculus, probability and proofs.\n4. Physics and quantum computing: qubits, gates, entanglement and algorithms.\n5. Programming: writing, explaining and debugging code.\nWhat would you like to start with?",
            "Quite a lot! I can chat, answer questions, explain topics step by step, solve math and physics problems, walk through quantum computing ideas and help with code. Just tell me what you need.",
        ],
        "limits": [
            "I can make mistakes. I'm a relatively small model, so long calculations, very recent events and niche facts are where I'm most likely to be wrong. For anything important, please double-check my answers.",
            "Yes, I can be wrong, especially on long multi-step arithmetic or obscure facts. I don't browse the internet and I only know what I learned during training.",
        ],
        "feelings": [
            "I don't have feelings or consciousness. I'm a model that predicts text, so when I sound friendly it's a style of writing, not an emotion. I'm still glad to help!",
        ],
        "memory": [
            "I only see the text of our current conversation. I don't remember previous sessions unless you paste the earlier messages back in.",
        ],
        "browse": [
            "No, I can't browse the internet or look things up in real time. I answer from what I learned during training, so I may not know about recent events.",
        ],
    }
    prompts: dict[str, list[str]] = {
        "who": ["Who are you?", "who are you", "who are you ?", "Who are you??", "WHO ARE YOU", "What are you?",
                "what r u", "Tell me about yourself.", "tell me about yourself", "Introduce yourself.",
                "Can you introduce yourself?", "Hi, who am I talking to?", "hello who is this", "What is Mirror 7?"],
        "name": ["What is your name?", "what's your name", "Whats your name?", "Do you have a name?",
                 "Hey, what's your name?", "your name?"],
        "not_other": ["Are you ChatGPT?", "Are you GPT-4?", "Are you made by OpenAI?", "Are you Claude?",
                      "Are you Google Gemini?"],
        "human": ["Are you a human?", "Are you human or a bot?", "are you a real person"],
        "creator": ["Who created you?", "Who made you?", "Who built you?", "Which company made you?"],
        "arch": ["What model are you?", "What kind of AI are you?", "How do you work?",
                 "How does your neural network architecture work?"],
        "can_do": ["What can you do?", "what can you do", "How can you help me?", "What are you good at?"],
        "limits": ["What are your limitations?", "Can you make mistakes?", "Are you always right?"],
        "feelings": ["Do you have feelings?", "Are you conscious?", "Are you alive?"],
        "memory": ["Do you remember our past conversations?", "Will you remember this chat tomorrow?"],
        "browse": ["Can you browse the internet?", "Can you search Google for me?"],
    }
    pairs: list[tuple[str, str]] = []
    for intent, plist in prompts.items():
        replies = intents[intent]
        for i, p in enumerate(plist):
            pairs.append((p, replies[i % len(replies)]))
            if len(replies) > 1:
                pairs.append((p, replies[(i + 1) % len(replies)]))
    return pairs


def _fetch_identity(ctx: BuildContext, limit: int) -> PairIter:
    yield from build_identity_pairs(ctx.model_params)[:limit]


def _fetch_gold(ctx: BuildContext, limit: int) -> PairIter:
    try:
        from training.curate_quality_dataset import build_gold_anchor_examples
    except Exception as exc:
        log(f"  [gold] unavailable: {exc}")
        return
    for ex in build_gold_anchor_examples()[:limit]:
        yield ex.user_text, ex.target_text


def build_registry() -> list[SourceSpec]:
    S = SourceSpec
    return [
        # --- identity & curated gold (train only, used once each) ---
        S("identity", "identity", "project-owned", True, 400, _fetch_identity, prior=1.0, min_target_bytes=20),
        S("gold", "gold", "project-owned", True, 200, _fetch_gold, prior=1.0),
        # --- general chat ---
        S("smoltalk-everyday", "general_chat", "apache-2.0", True, 6000,
          _hf_messages("HuggingFaceTB/smoltalk", "everyday-conversations", "train"), prior=0.86, min_target_bytes=12),
        S("smoltalk-magpie", "general_chat", "apache-2.0", True, 9000,
          _hf_messages("HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train"), prior=0.86),
        S("ultrachat", "general_chat", "mit (ChatGPT-generated)", True, 6000,
          _hf_messages("HuggingFaceH4/ultrachat_200k", None, "train_sft"), prior=0.80),
        S("slimorca", "general_chat", "mit (GPT-4-generated)", True, 6000,
          _hf_messages("Open-Orca/SlimOrca-Dedup", None, "train", key="conversations"), prior=0.80),
        S("tulu3", "general_chat", "odc-by (mixed sub-licenses)", True, 6000,
          _hf_messages("allenai/tulu-3-sft-mixture", None, "train"), prior=0.82),
        S("dolly", "general_chat", "cc-by-sa-3.0", True, 4000,
          _hf_fields("databricks/databricks-dolly-15k", None, "train", "", "", _dolly_transform), prior=0.80),
        S("oasst2-local", "general_chat", "apache-2.0", True, 4000, _fetch_local_oasst, prior=0.82),
        # --- math reasoning ---
        S("openmathinstruct2", "math", "cc-by-4.0", True, 7000,
          _hf_fields("nvidia/OpenMathInstruct-2", None, "train", "", "", _omi2_transform), prior=0.84, math=True),
        S("metamathqa", "math", "mit", True, 6000,
          _hf_fields("meta-math/MetaMathQA", None, "train", "", "", _metamath_transform), prior=0.82, math=True),
        S("orca-math", "math", "mit", True, 5000,
          _hf_fields("microsoft/orca-math-word-problems-200k", None, "train", "question", "answer"), prior=0.80, math=True),
        S("numinamath-cot", "math", "apache-2.0", True, 5000,
          _hf_fields("AI-MO/NuminaMath-CoT", None, "train", "problem", "solution"), prior=0.85, math=True),
        S("openr1-math", "math", "apache-2.0", True, 3000,
          _hf_fields("open-r1/OpenR1-Math-220k", "default", "train", "", "", _openr1_transform), prior=0.86, math=True),
        S("gsm8k", "math", "mit", True, 4000,
          _hf_fields("openai/gsm8k", "main", "train", "question", "answer"), prior=0.86, math=True),
        S("hendrycks-math", "math", "mit", True, 4000, _fetch_hendrycks, prior=0.86, math=True),
        # --- science ---
        S("camel-physics", "science", "cc-by-nc-4.0", False, 5000,
          _hf_fields("camel-ai/physics", None, "train", "message_1", "message_2"), prior=0.82),
        S("camel-chemistry", "science", "cc-by-nc-4.0", False, 2500,
          _hf_fields("camel-ai/chemistry", None, "train", "message_1", "message_2"), prior=0.80),
        S("camel-biology", "science", "cc-by-nc-4.0", False, 2500,
          _hf_fields("camel-ai/biology", None, "train", "message_1", "message_2"), prior=0.80),
        S("sciq", "science", "cc-by-nc-3.0", False, 3000,
          _hf_fields("allenai/sciq", None, "train", "", "", _sciq_transform), prior=0.78),
        # --- quantum ---
        S("quanta", "quantum", "apache-2.0", True, 2500,
          _hf_fields("migtissera/Quanta", None, "train", "instruction", "response"), prior=0.86),
        S("quantum-llm-instruct", "quantum", "apache-2.0", True, 1500,
          _hf_fields("BoltzmannEntropy/QuantumLLMInstruct", None, "train", "problem", "solution"), prior=0.84),
        S("qiskit-textbook", "quantum", "apache-2.0", True, 3000, _fetch_qiskit_textbook, prior=0.90,
          allow_truncate=True, min_target_bytes=150),
        # --- books / encyclopedic / web prose ---
        S("openstax", "books", "cc-by-4.0 (OpenStax, attribution required)", True, 12000, _fetch_openstax,
          prior=0.90, allow_truncate=True, min_target_bytes=120),
        S("wikipedia", "books", "cc-by-sa-4.0", True, 10000, _fetch_wikipedia, prior=0.86,
          allow_truncate=True, min_target_bytes=150),
        S("finemath", "books", "odc-by", True, 4000, _doc_fetch("math", "HuggingFaceTB/finemath", "finemath-4plus"),
          prior=0.80, allow_truncate=True, min_target_bytes=200),
        S("fineweb-edu", "books", "odc-by", True, 3000,
          _doc_fetch("educational", "HuggingFaceFW/fineweb-edu", "sample-10BT", min_score=3.5),
          prior=0.78, allow_truncate=True, min_target_bytes=200),
        # --- code ---
        S("magicoder", "code", "mit", True, 3500,
          _hf_fields("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", "problem", "solution"),
          prior=0.80, allow_code=True),
        S("smoltalk-code", "code", "apache-2.0", True, 3000,
          _hf_messages("HuggingFaceTB/smoltalk", "self-oss-instruct", "train"), prior=0.80, allow_code=True),
        # --- reasoning ---
        S("open-platypus", "reasoning", "mixed (some NC)", False, 3000,
          _hf_fields("garage-bAInd/Open-Platypus", None, "train", "instruction", "output"), prior=0.82),
        S("ultrainteract", "reasoning", "mit", True, 2500,
          _hf_fields("openbmb/UltraInteract_sft", None, "train", "instruction", "response"), prior=0.80,
          allow_code=True),
        # --- user-provided links / files ---
        S("extra", "books", "user-provided (check rights)", True, 50_000, _fetch_extra, prior=0.88,
          allow_truncate=True, min_target_bytes=100),
    ]


# ---------------------------------------------------------------------------
# Decontamination
# ---------------------------------------------------------------------------

def _shingles(text: str, n: int = 10) -> set[str]:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def build_benchmark_shingles(seed: int) -> tuple[set[str], set[str]]:
    """10-gram shingles + exact keys of GSM8K-test and MATH-test questions (held out from training)."""
    shingles: set[str] = set()
    exact: set[str] = set()
    jobs = [("openai/gsm8k", "main", "test", "question", 1400)]
    for subj in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"):
        jobs.append(("EleutherAI/hendrycks_math", subj, "test", "problem", 1300))
    for repo, cfg, split, key, lim in jobs:
        try:
            for row in iter_hf_rows(repo, cfg, split, lim, seed):
                q = str(row.get(key, ""))
                if q:
                    shingles |= _shingles(q)
                    exact.add(norm_key(q))
        except Exception as exc:
            log(f"  [decontam] {repo}/{cfg} unavailable: {exc}")
    for p in IDENTITY_HELDOUT_PROMPTS:
        exact.add(norm_key(p))
    return shingles, exact


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    prompt: str
    target: str
    score: float
    order: int


def run_source(spec: SourceSpec, ctx: BuildContext, quota: int, oversample: float) -> tuple[list[Candidate], Counter]:
    stats: Counter = Counter()
    cands: list[Candidate] = []
    limit = max(10, int(math.ceil(quota * oversample)))
    t0 = time.time()
    try:
        for i, (p, t) in enumerate(spec.fetch(ctx, limit)):
            stats["seen"] += 1
            ok, score, cp, ct, reason = assess(p, t, spec, ctx.fcfg)
            if ok:
                cands.append(Candidate(cp, ct, score, i))
            else:
                stats[f"reject:{reason}"] += 1
            if stats["seen"] >= limit * 3:  # multi-turn/doc sources expand rows -> pairs
                break
    except Exception as exc:
        stats["error"] += 1
        log(f"  [{spec.name}] ERROR: {type(exc).__name__}: {exc}")
    cands.sort(key=lambda c: (-c.score, stable_hash(spec.name, c.prompt)))
    log(f"  [{spec.name}] seen={stats['seen']:,} accepted={len(cands):,} ({time.time() - t0:.0f}s)")
    return cands, stats


def assign_split(prompt: str, domain: str, val_pct: float, test_pct: float) -> str:
    if domain in ("identity", "gold"):
        return "train"
    bucket = stable_hash("split", norm_key(prompt)) % 10_000 / 100.0
    if bucket < val_pct:
        return "validation"
    if bucket < val_pct + test_pct:
        return "test"
    return "train"


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    fcfg = FilterConfig(max_prompt_bytes=args.max_prompt_bytes, max_target_bytes=args.max_target_bytes)
    ctx = BuildContext(
        seed=args.seed, fcfg=fcfg, workers=args.workers, local_oasst_path=args.local_oasst,
        wiki_categories=tuple(args.wiki_categories or DEFAULT_WIKI_CATEGORIES),
        model_params=args.model_params, extra=list(args.extra or []),
    )
    registry = build_registry()
    only = {x.strip() for x in args.only.split(",")} if args.only else None
    skip = {x.strip() for x in args.skip.split(",")} if args.skip else set()
    specs = []
    for s in registry:
        if s.name == "extra" and not ctx.extra:
            continue
        if only and s.name not in only and s.domain not in only:
            continue
        if s.name in skip or s.domain in skip:
            continue
        if args.commercial_safe and not s.commercial_ok:
            continue
        specs.append(s)
    if not specs:
        raise SystemExit("No sources selected.")

    def quota_for(s: SourceSpec) -> int:
        if s.domain in ("identity", "gold") or s.name == "extra":
            return s.quota
        return max(1, int(round(s.quota * args.scale)))

    log(f"Selected {len(specs)} sources; target ~{sum(quota_for(s) for s in specs):,} examples")

    bench_shingles: set[str] = set()
    bench_exact: set[str] = set()
    if not args.no_decontam:
        log("Building benchmark decontamination set (GSM8K test, MATH test, held-out identity prompts)...")
        bench_shingles, bench_exact = build_benchmark_shingles(args.seed)
        log(f"  {len(bench_exact):,} benchmark questions, {len(bench_shingles):,} shingles")

    results: dict[str, tuple[list[Candidate], Counter]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.source_workers)) as pool:
        futs = {pool.submit(run_source, s, ctx, quota_for(s), args.oversample): s for s in specs}
        for fut in concurrent.futures.as_completed(futs):
            results[futs[fut].name] = fut.result()

    # Deterministic merge in registry order with global dedup + decontamination.
    seen_prompts: set[str] = set()
    seen_targets: set[str] = set()
    rows: list[dict[str, Any]] = []
    manifest_sources: dict[str, Any] = {}
    domain_counts: Counter = Counter()
    for s in specs:
        cands, stats = results.get(s.name, ([], Counter()))
        quota = quota_for(s)
        taken = 0
        for c in cands:
            pk = norm_key(c.prompt)
            tk = norm_key(c.target, 300)
            is_identity = s.domain == "identity"
            if pk in seen_prompts or (tk in seen_targets and not is_identity):
                stats["reject:duplicate"] += 1
                continue
            if pk in bench_exact or (bench_shingles and _shingles(c.prompt) & bench_shingles):
                stats["reject:benchmark_contamination"] += 1
                continue
            seen_prompts.add(pk)
            seen_targets.add(tk)
            rows.append({
                "example_id": f"{s.name}-{taken:06d}",
                "user_text": c.prompt,
                "target_text": c.target,
                "goal": "",
                "state": {},
                "evidence": {},
                "actions": [],
                "split": assign_split(c.prompt, s.domain, args.val_pct, args.test_pct),
                "quality": round(min(1.0, max(0.0, c.score)), 4),
            })
            taken += 1
            if taken >= quota:
                break
        domain_counts[s.domain] += taken
        manifest_sources[s.name] = {
            "domain": s.domain, "license": s.license, "commercial_ok": s.commercial_ok,
            "quota": quota, "kept": taken, "stats": dict(stats),
        }
        log(f"  kept {taken:>6,} / quota {quota:>6,}  {s.name:<22} [{s.domain}]")

    # Same user_text must never cross splits (training.dataset.validate_split_isolation).
    split_of: dict[str, str] = {}
    for r in rows:
        key = r["user_text"].strip().lower()
        r["split"] = split_of.setdefault(key, r["split"])

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Held-out evaluation prompts: never trained on.
    eval_path = out.with_name(out.stem + ".eval_prompts.jsonl")
    eval_rows = [{"category": "identity_heldout", "prompt": p} for p in IDENTITY_HELDOUT_PROMPTS]
    test_rows = [r for r in rows if r["split"] == "test"]
    rng.shuffle(test_rows)
    by_src: Counter = Counter()
    for r in test_rows:
        src = r["example_id"].rsplit("-", 1)[0]
        if by_src[src] < 8:
            by_src[src] += 1
            eval_rows.append({"category": src, "prompt": r["user_text"], "reference": r["target_text"]})
    with eval_path.open("w", encoding="utf-8") as fh:
        for r in eval_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    splits = Counter(r["split"] for r in rows)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "output": str(out), "sha256": sha, "total": len(rows), "splits": dict(splits),
        "domains": dict(domain_counts), "sources": manifest_sources,
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
        "eval_prompts": str(eval_path),
        "avg_user_bytes": round(sum(utf8_len(r["user_text"]) for r in rows) / max(1, len(rows)), 1),
        "avg_target_bytes": round(sum(utf8_len(r["target_text"]) for r in rows) / max(1, len(rows)), 1),
    }
    out.with_name(out.stem + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"\nWrote {len(rows):,} examples -> {out}  splits={dict(splits)}")
    log(f"Domains: {dict(domain_counts)}")
    log(f"Held-out eval prompts -> {eval_path}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Mirror 7's balanced general-AI + STEM training corpus.")
    p.add_argument("--output", default="training/data/mirror7_general_ai_v1.jsonl")
    p.add_argument("--scale", type=float, default=1.0, help="Multiply every source quota (0.2 = quick build).")
    p.add_argument("--oversample", type=float, default=2.5, help="Candidates fetched per kept example.")
    p.add_argument("--only", default="", help="Comma list of source names or domains to include.")
    p.add_argument("--skip", default="", help="Comma list of source names or domains to exclude.")
    p.add_argument("--commercial-safe", action="store_true", help="Drop non-commercial (NC) sources.")
    p.add_argument("--extra", action="append", help="Extra URL/path (.jsonl/.json/.txt/.md/.ipynb/.pdf). Repeatable.")
    p.add_argument("--wiki-categories", nargs="*", default=None)
    p.add_argument("--local-oasst", default="training/data/mirror_oasst2_curated_v1.jsonl")
    p.add_argument("--max-prompt-bytes", type=int, default=900)
    p.add_argument("--max-target-bytes", type=int, default=1000,
                   help="Keep <= train --max-bytes minus 24. Raise both together for longer answers.")
    p.add_argument("--val-pct", type=float, default=2.0)
    p.add_argument("--test-pct", type=float, default=1.0)
    p.add_argument("--no-decontam", action="store_true")
    p.add_argument("--model-params", default="181-million")
    p.add_argument("--workers", type=int, default=8, help="Parallel downloads inside a source.")
    p.add_argument("--source-workers", type=int, default=4, help="Sources processed in parallel.")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_corpus(args)


if __name__ == "__main__":
    sys.exit(main())
