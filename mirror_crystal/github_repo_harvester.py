"""Public GitHub Repositories & Open Engineering Standards Harvester for Mirror 7 Phase-3.

Pure Python standard-library harvester (0 neural parameters, 0 GPU, 0 external pip deps) that
harvests, sanitizes, strictly quality-filters, and deduplicates authoritative engineering,
security, cloud-native, web, database, and computer-science concepts into compact 5-facet JSONL
staging for Mirror 7's Multi-Shard Disk-Backed SQLite FTS5 Lattice (`shard_github_engineering.dcrystal`).

Authoritative Public Sources (Priority Order):
  1. MITRE ATT&CK STIX 2.0 (Enterprise + ICS + Mobile: `github.com/mitre/cti`) — Apache 2.0
  2. MITRE CAPEC STIX 2.1 (`github.com/mitre/cti` `capec/2.1/stix-capec.json`) — Apache 2.0
  3. CNCF Cloud Native Glossary (`github.com/cncf/glossary` `content/en/*.md`) — CC BY 4.0
  4. CNCF Kubernetes Official Glossary (`github.com/kubernetes/website`) — CC BY 4.0
  5. OWASP Cheat Sheet Series (`github.com/OWASP/CheatSheetSeries`) — CC BY-SA 4.0
  6. MDN Web Docs Glossary (`github.com/mdn/content` `files/en-us/glossary/*/index.md`) — CC BY-SA 2.5
  7. NIST CSRC Official Glossary (`csrc.nist.gov/csrc/media/glossary/glossary-export.zip`) — US Gov Public Domain
  8. IETF RFC 4949 Internet Security Glossary (`rfc-editor.org/rfc/rfc4949.txt`) — IETF Trust
  9. PostgreSQL Official Database Glossary (`github.com/postgres/postgres`) — PostgreSQL License
 10. Python Official Documentation Glossary (`github.com/python/cpython`) — PSF License
 11. Rust Language Reference Glossary (`github.com/rust-lang/reference`) — MIT / Apache 2.0
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import zipfile


_USER_AGENT = "Mirror7RepoHarvester/3.0 (https://github.com/realahmad07/Mirror-7; pure-stdlib)"

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

# Generic singletons never allowed as secondary aliases (prevents cross-domain hijack)
_FORBIDDEN_SECONDARY_ALIASES: frozenset[str] = frozenset({
    "access", "account", "action", "agent", "alert", "alias", "all", "analysis",
    "api", "app", "application", "array", "asset", "attack", "audit", "auth",
    "base", "batch", "block", "body", "boot", "branch", "bridge", "buffer",
    "build", "byte", "cache", "call", "card", "case", "cast", "cell", "chain",
    "channel", "check", "class", "client", "clock", "cloud", "cluster", "code",
    "command", "commit", "component", "config", "container", "content", "context",
    "control", "cookie", "core", "cost", "count", "crash", "data", "database",
    "date", "debug", "default", "deploy", "design", "device", "disk", "domain",
    "driver", "edge", "element", "email", "engine", "entity", "entry", "error",
    "event", "exec", "export", "factor", "field", "file", "filter", "flag",
    "flow", "folder", "font", "force", "fork", "form", "frame", "function",
    "gate", "global", "graph", "group", "guard", "handle", "hash", "head",
    "header", "heap", "help", "hook", "host", "icon", "id", "identity", "image",
    "import", "index", "info", "input", "instance", "interface", "item", "job",
    "join", "kernel", "key", "label", "layer", "layout", "level", "library",
    "limit", "line", "link", "list", "load", "local", "lock", "log", "login",
    "loop", "machine", "mail", "main", "map", "mark", "mask", "match", "matrix",
    "media", "memory", "merge", "mesh", "message", "meta", "method", "metric",
    "mirror", "mode", "model", "module", "monitor", "mount", "name", "namespace",
    "native", "net", "network", "node", "none", "null", "number", "object",
    "offset", "open", "operator", "option", "order", "origin", "other", "output",
    "owner", "package", "packet", "page", "pair", "panel", "param", "parent",
    "parse", "parser", "patch", "path", "pattern", "payload", "peer", "phase",
    "pipe", "pixel", "plan", "plane", "platform", "plugin", "pod", "point",
    "pointer", "policy", "pool", "port", "post", "power", "prefix", "probe",
    "process", "profile", "program", "project", "prompt", "proof", "property",
    "protocol", "proxy", "public", "pull", "push", "query", "queue", "range",
    "rate", "raw", "read", "record", "ref", "region", "registry", "release",
    "remote", "render", "repo", "report", "request", "reset", "resource",
    "response", "result", "retry", "return", "review", "ring", "risk", "role",
    "root", "route", "router", "row", "rule", "run", "runtime", "salt", "sample",
    "scale", "scan", "schema", "scheme", "scope", "score", "script", "search",
    "secret", "section", "sector", "security", "seed", "segment", "select",
    "selector", "send", "sensor", "server", "service", "session", "set", "shadow",
    "shard", "share", "shell", "shield", "shift", "shim", "signal", "signature",
    "sink", "site", "size", "slice", "slot", "slug", "snapshot", "socket", "sort",
    "source", "space", "span", "spec", "split", "stack", "stage", "standard",
    "start", "state", "static", "status", "step", "stop", "store", "stream",
    "string", "struct", "stub", "style", "subject", "subnet", "suite", "switch",
    "symbol", "sync", "syntax", "system", "table", "tag", "taint", "target",
    "task", "template", "tenant", "term", "test", "text", "thread", "threshold",
    "ticket", "tier", "time", "timeout", "timer", "title", "token", "tool",
    "topic", "trace", "track", "traffic", "trait", "transfer", "trap", "tree",
    "trigger", "trust", "tunnel", "tuple", "type", "unit", "unknown", "update",
    "upgrade", "upload", "uri", "url", "usage", "user", "value", "var", "vault",
    "vector", "vendor", "version", "view", "virtual", "volume", "wait", "wall",
    "watch", "wave", "web", "weight", "window", "wire", "word", "work", "worker",
    "workflow", "workload", "workspace", "world", "write", "zone",
})

# Generic single-word Kubernetes / PostgreSQL / Rust / Python terms contextualized cleanly
_GENERIC_K8S_SINGLETONS: frozenset[str] = frozenset({
    "affinity", "annotation", "annotations", "api", "app", "application",
    "cluster", "config", "container", "controller", "daemon", "deployment",
    "device", "drain", "endpoint", "eviction", "extension", "finalizer",
    "image", "ingress", "job", "label", "labels", "limit", "manifest",
    "master", "mirror", "name", "namespace", "node", "object", "operator",
    "patch", "pod", "probe", "proxy", "quota", "replica", "resource",
    "rollout", "runtime", "scheduler", "secret", "selector", "service",
    "shim", "sidecar", "spec", "state", "status", "storage", "taint",
    "target", "token", "toleration", "uid", "volume", "watch", "workload",
})

_GENERIC_PG_SINGLETONS: frozenset[str] = frozenset({
    "aggregate", "analyze", "attribute", "backend", "bloat", "cast", "catalog",
    "checkpoint", "client", "cluster", "column", "commit", "concurrency",
    "connection", "constraint", "database", "datum", "deadlock", "delete",
    "extension", "file", "fork", "function", "grant", "heap", "host", "index",
    "insert", "instance", "isolation", "join", "key", "lock", "log", "member",
    "null", "object", "operator", "page", "partition", "planner", "postmaster",
    "primary", "procedure", "process", "query", "record", "relation", "replica",
    "revoke", "role", "rollback", "row", "rule", "savepoint", "schema",
    "segment", "select", "sequence", "server", "session", "snapshot", "standby",
    "statistic", "table", "tablespace", "target", "temporary", "toast", "token",
    "transaction", "trigger", "tuple", "type", "unlogged", "update", "user",
    "vacuum", "view", "wal", "window", "worker", "wrapper",
})

_GENERIC_PY_SINGLETONS: frozenset[str] = frozenset({
    "annotation", "argument", "asynchronous", "attribute", "awaitable",
    "bytecode", "callback", "class", "coercion", "complex", "comprehension",
    "context", "coroutine", "decorator", "descriptor", "dictionary", "docstring",
    "expression", "extension", "file", "finder", "floor", "function", "future",
    "generator", "generic", "hashable", "immutable", "importer", "interactive",
    "interpreted", "iterable", "iterator", "key", "keyword", "lambda", "list",
    "loader", "locale", "mapping", "metaclass", "method", "module", "mutable",
    "namespace", "object", "package", "parameter", "path", "portion", "property",
    "qualified", "reference", "sequence", "slice", "statement", "string",
    "text", "tuple", "type", "universal", "variable", "view",
})

_GENERIC_RUST_SINGLETONS: frozenset[str] = frozenset({
    "alignment", "arity", "attribute", "binding", "block", "borrow", "closure",
    "coercion", "combinator", "crate", "destructor", "dispatch", "dynamically",
    "expression", "field", "function", "identifier", "item", "keyword", "lifetime",
    "literal", "macro", "method", "module", "move", "namespace", "nominal",
    "object", "operand", "operator", "owner", "ownership", "panic", "path",
    "pattern", "place", "pointer", "prelude", "reference", "scope", "slice",
    "statement", "string", "struct", "trait", "tuple", "type", "union",
    "unsized", "value", "variable", "variance",
})

_REJECT_TITLE_PREFIX_RE = re.compile(
    r"^(?:list\s+of|index\s+of|glossary\s+of|outline\s+of|overview\s+of|"
    r"table\s+of|bibliography\s+of|timeline\s+of|appendix\b|template\b|"
    r"definition\s+template\b|preface\b|obsolete\b|deprecated\b|revoked\b|"
    r"see\s+also\b|untitled\b)",
    re.IGNORECASE,
)

_RAW_ID_TITLE_RE = re.compile(
    r"^(?:"
    r"0x[0-9a-f]+|\d+(?:\.\d+)?[a-z]?|"
    r"[a-z]{1,4}[-_]?\d{3,}[a-z0-9.\-]*|"
    r"[^a-zA-Z]*|"
    r"\([a-z0-9,\s+\-]+\)\s*-\s*[a-z]+|"
    r"[A-Z0-9]{1,3}$"
    r")$",
    re.IGNORECASE,
)

_RESIDUAL_SYNTAX_RE = re.compile(
    r"(?:\{\{|\}\}|\{\{<|>\}\}|\]\(|\]\[|https?://|www\.[a-z0-9\-]+\.|"
    r"</?[a-z][a-z0-9:_-]*(?:\s[^>]*)?>|<!--|-->|```|\[\[|\]\]|\[![A-Z]+\]|"
    r"\(Citation:|:(?:term|ref|func|class|mod|pep|rfc|meth|attr|exc|data):`|"
    r"(?:^|\s)---(?:\s|$))",
    re.IGNORECASE,
)

_DEPRECATED_PROSE_RE = re.compile(
    r"\b(?:is|was|are|were|has\s+been|have\s+been)\s+(?:an?\s+|now\s+|considered\s+)?(?:deprecated|obsolete|revoked|outdated\s+term)\b",
    re.IGNORECASE,
)

_REJECT_PROSE_CUES: tuple[str, ...] = (
    "which of the following",
    "all of the above",
    "none of the above",
    "true or false",
    "option a",
    "may refer to",
    "can refer to",
    "disambiguation page",
    "this term is obsolete",
    "this term was made obsolete",
    "this entry is deprecated",
    "is deprecated",
    "was deprecated",
    "is obsolete",
    "was obsolete",
    "was revoked",
    "have been revoked",
    "has been deprecated",
    "has been revoked",
    "quick summary of the concept",
    "define the problem it addresses",
    "add related glossary terms",
    "no definition available",
    "not yet defined",
    "placeholder",
    "todo:",
    "tbd.",
)

_ABBREV_PROTECT_RE = re.compile(
    r"\b(e\.g|i\.e|vs|etc|dr|prof|fig|no|vol|approx|cf|ref|refs|rev|sec|st|inc|corp|ltd|al|u\.s)\.",
    re.IGNORECASE,
)

_TRAILING_DANGLING_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "and", "or", "but", "nor", "that", "which", "who", "whom", "whose", "as",
    "into", "onto", "upon", "via", "per", "than", "between", "among", "during",
    "within", "without", "through", "such", "including", "using", "where", "when",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "can", "could", "may", "might", "will", "would", "shall", "should", "must",
    "do", "does", "did", "supports", "support", "assume", "assumes", "allows",
    "allow", "enables", "enable", "provides", "provide", "requires", "require",
    "contains", "contain", "includes", "include", "uses", "use", "makes", "make",
})


@dataclass(frozen=True)
class RepoConceptRecord:
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
class RepoHarvestReport:
    total_emitted: int
    raw_candidates_collected: int
    raw_bytes_fetched: int
    rejected_breakdown: dict[str, int]
    source_counts: dict[str, int]
    source_status: dict[str, str]
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


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").casefold()).strip("_")
    return s[:64] or "entry"


# ---------------------------------------------------------------------------
# Pure-Stdlib HTTP Helper (gzip, retry backoff, optional GITHUB_TOKEN)
# ---------------------------------------------------------------------------

def _http_get_bytes(
    url: str,
    *,
    timeout: float = 25.0,
    retries: int = 3,
    backoff: float = 1.2,
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json, text/plain, text/markdown, */*;q=0.5",
        "Accept-Encoding": "gzip",
    }
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if gh_token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {gh_token}"
    if extra_headers:
        headers.update(extra_headers)

    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                guard = 0
                while raw[:2] == b"\x1f\x8b" and guard < 3:
                    raw = gzip.decompress(raw)
                    guard += 1
                return raw
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                sleep_s = float(retry_after) if (retry_after and retry_after.isdigit()) else backoff * (2 ** attempt)
                time.sleep(min(sleep_s, 10.0))
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
    raise RuntimeError(f"HTTP fetch failed for {url}")


def _http_get_json(url: str, *, timeout: float = 30.0) -> tuple[Any, int]:
    raw = _http_get_bytes(url, timeout=timeout, extra_headers={"Accept": "application/json"})
    return json.loads(raw.decode("utf-8-sig", errors="replace")), len(raw)


def _http_get_text(url: str, *, timeout: float = 20.0) -> tuple[str, int]:
    raw = _http_get_bytes(url, timeout=timeout)
    return raw.decode("utf-8-sig", errors="replace"), len(raw)


# ---------------------------------------------------------------------------
# Strict Markdown / MDN / Hugo / reST / SGML / STIX Sanitizers
# ---------------------------------------------------------------------------

def _normalize_unicode(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\ufeff", "").replace("\u00ad", "")
    s = re.sub(r"[\u200b-\u200f\u2060]", "", s)
    s = s.replace("\u00a0", " ")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", " - ")
    s = s.replace("\u2026", "...")
    return s


def _parse_yaml_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Extract simple YAML frontmatter key/values and return (meta, markdown_body)."""
    text = _normalize_unicode(raw).lstrip()
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", text, flags=re.DOTALL)
    if not m:
        return {}, text
    fm_block = m.group(1)
    body = text[m.end():]
    meta: dict[str, Any] = {}
    lines = fm_block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        km = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", line)
        if not km:
            i += 1
            continue
        key = km.group(1).strip().casefold()
        val = km.group(2).strip()
        if val in (">", "|", ">-", "|-"):
            buf: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                if lines[i].strip():
                    buf.append(lines[i].strip())
                i += 1
            meta[key] = " ".join(buf)
            continue
        elif not val:
            items: list[str] = []
            i += 1
            while i < len(lines) and re.match(r"^\s*-\s+", lines[i]):
                items.append(re.sub(r"^\s*-\s+", "", lines[i]).strip().strip("\"'"))
                i += 1
            meta[key] = items if items else ""
            continue
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            items = [x.strip().strip("\"'") for x in inner.split(",") if x.strip().strip("\"'")]
            meta[key] = items
        else:
            meta[key] = val.strip("\"'")
        i += 1
    return meta, body


_MDN_DROP_MACROS: frozenset[str] = frozenset({
    "deprecated_header", "deprecated_inline", "obsolete_header", "obsolete_inline",
    "non-standard_header", "non-standard_inline", "experimental_inline",
    "glossarysidebar", "seecompattable", "specifications", "browsercompat",
    "previousmenunext", "quicklinkswithsubpages", "compat", "availableinworkers",
    "securecontext_header", "securecontext_inline", "readonlyinline",
    "optional_inline", "defaultapisidebar", "cssref", "htmlsidebar", "jssidebar",
    "httpheader_sidebar", "httpsidebar", "previousnext", "previous", "next",
    "embedlivesample", "embedghlivesample", "anch", "draft", "page",
})


def _resolve_mdn_macro(match: re.Match[str]) -> str:
    name = (match.group(1) or "").strip()
    args_str = (match.group(2) or "").strip()
    low = name.casefold()
    if low in _MDN_DROP_MACROS or low.endswith(("sidebar", "_header", "_inline")):
        return ""
    # Parse quoted string arguments cleanly
    args = [m.group(1) or m.group(2) or "" for m in re.finditer(r'"([^"]*)"|\'([^\']*)\'', args_str)]
    if not args and args_str:
        args = [a.strip() for a in args_str.split(",") if a.strip()]
    if low == "rfc":
        if len(args) >= 2 and args[1].strip():
            return f"RFC {args[0].strip()} ({args[1].strip()})"
        return f"RFC {args[0].strip()}" if args else ""
    if low == "interwiki":
        if len(args) >= 3 and args[2].strip():
            return args[2].strip()
        if len(args) >= 2 and args[1].strip():
            return args[1].strip().replace("_", " ")
        return ""
    if low == "glossary":
        if len(args) >= 2 and args[1].strip():
            return args[1].strip()
        if args:
            return args[0].strip().split("/")[-1].replace("_", " ")
        return ""
    if low in (
        "domxref", "cssxref", "jsxref", "htmlelement", "svgelement", "mathmlelement",
        "httpheader", "httpmethod", "httpstatus", "ariarole", "svgattr", "htmlattrxref",
        "webextapiref", "specname", "ariastate", "event", "csssyntax", "cssinfo",
    ):
        if len(args) >= 2 and args[1].strip():
            return args[1].strip()
        if args:
            return args[0].strip()
        return ""
    # Generic fallback: pick most readable alphabetic arg if any
    for candidate in reversed(args):
        c = candidate.strip()
        if c and re.search(r"[A-Za-z]", c) and not c.startswith(("http", "/")):
            return c
    return ""


def _clean_mdn_macros(text: str) -> str:
    # Two passes handle nested macros cleanly
    out = text
    for _ in range(2):
        out = re.sub(r"\{\{\s*([A-Za-z0-9_\-]+)\s*(?:\(([^}]*)\))?\s*\}\}", _resolve_mdn_macro, out)
    out = re.sub(r"\{\{[^}]*\}\}", "", out)
    return out


def _clean_hugo_shortcodes(text: str) -> str:
    def _repl_tooltip(m: re.Match[str]) -> str:
        inner = m.group(1)
        tm = re.search(r'text\s*=\s*"([^"]+)"', inner)
        if tm:
            return tm.group(1)
        im = re.search(r'term_id\s*=\s*"([^"]+)"', inner)
        if im:
            return im.group(1).replace("-", " ").replace("_", " ")
        return ""

    out = re.sub(r"\{\{<\s*glossary_tooltip\s+([^>]+?)>\}\}", _repl_tooltip, text, flags=re.IGNORECASE)
    out = re.sub(r"\{\{[<%][^}]*[>%]\}\}", " ", out)
    return out


def _clean_markdown_to_paragraphs(raw: str, *, is_mdn: bool = False, is_hugo: bool = False) -> list[str]:
    """Strip Markdown, HTML, code blocks, tables, citations, badges & macros into clean paragraphs."""
    txt = _normalize_unicode((raw or "")[:12000])
    # Remove HTML comments
    txt = re.sub(r"<!--[\s\S]*?-->", " ", txt)
    # Remove fenced code blocks completely
    txt = re.sub(r"```[\s\S]*?```", "\n\n", txt)
    txt = re.sub(r"~~~[\s\S]*?~~~", "\n\n", txt)
    if is_mdn:
        txt = _clean_mdn_macros(txt)
    if is_hugo:
        txt = _clean_hugo_shortcodes(txt)
    # Strip reference-style link definitions: [label]: https://...
    txt = re.sub(r"(?m)^\s*\[[^\]]+\]:\s+\S+.*$", "", txt)
    # Strip markdown image/badge syntax [![alt](img)](url) and ![alt](img)
    txt = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", txt)
    txt = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", txt)
    # Convert markdown inline links [label](url) -> label (linear O(N), zero catastrophic backtracking)
    txt = re.sub(r"\[([^\]]+)\]\(([^()\s]*(?:\([^()\s]*\)[^()\s]*)*)(?:\s+\"[^\"]*\")?\)", r"\1", txt)
    # Convert reference links [label][ref] -> label and [label][] -> label
    txt = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", txt)
    # Strip MITRE citations (Citation: ...) and numeric/standard bracket refs [1], [I7498-2]
    txt = re.sub(r"\(Citation:[^)]*\)", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\[\s*(?:\d+|[A-Z]{1,4}\d{2,5}(?:-\d+)?)\s*(?:,\s*(?:\d+|[A-Z]{1,4}\d{2,5}))*\s*\]", "", txt)
    # Clean HTML anchor / code / emphasis tags preserving inner text
    txt = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", txt, flags=re.IGNORECASE | re.DOTALL)
    txt = re.sub(r"<code\b[^>]*>(.*?)</code>", r"\1", txt, flags=re.IGNORECASE | re.DOTALL)
    txt = re.sub(r"<(?:br|p|div|li)\b[^>]*>", "\n", txt, flags=re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    # Strip bare URLs and angle-bracket URLs
    txt = re.sub(r"https?://[^\s)\]>\"']+", "", txt)
    txt = re.sub(r"\(\s*\)", "", txt)

    # Split into blank-line-delimited blocks, discarding tables, headings, horizontal rules, and code lines
    blocks: list[str] = []
    cur: list[str] = []

    def _flush() -> None:
        if not cur:
            return
        joined = " ".join(cur)
        joined = _clean_inline_markdown(joined)
        if len(joined) >= 25:
            blocks.append(joined)
        cur.clear()

    for line in txt.splitlines():
        s = line.strip()
        if not s:
            _flush()
            continue
        # Skip markdown headings, horizontal rules, table lines, Rust rule anchors r[...], admonitions
        if (
            s.startswith("#")
            or re.match(r"^[-*_]{3,}\s*$", s)
            or s.startswith("|")
            or (s.count("|") >= 2)
            or re.match(r"^r\[[a-z0-9._\-]+\]$", s)
            or re.match(r"^\[!(?:NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]$", s, re.IGNORECASE)
            or re.match(r"^\.\.\s+[a-z0-9_\-]+::", s)
        ):
            _flush()
            continue
        # Skip indented code lines (4+ spaces) when not a bullet continuation
        if line.startswith("    ") and not s.startswith(("-", "*", "+")):
            if re.search(r"[;{}()=<>]", s):
                _flush()
                continue
        # Strip blockquote and list prefixes
        s = re.sub(r"^>\s*", "", s)
        # Drop GitHub / MDN blockquote alert tags ([!NOTE], [!WARNING], [!CALLOUT], etc.)
        if re.match(r"^\[![A-Z]+\]$", s, re.IGNORECASE):
            _flush()
            continue
        s = re.sub(r"\[![A-Z]+\]\s*", "", s)
        s = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", s)
        if s:
            cur.append(s)
    _flush()
    return blocks


def _clean_inline_markdown(text: str) -> str:
    s = _normalize_unicode(text)
    # Strip backticks cleanly
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = s.replace("`", "")
    # Strip bold/italic wrappers
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)
    s = re.sub(r"__([^_\n]+)__", r"\1", s)
    s = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", s)
    s = re.sub(r"(?:(?<=^)|(?<=[\s(\"'/]))_([^_\n]+)_(?=$|[\s.,;:!?)\"'/])", r"\1", s)
    # Clean stray empty brackets/parens left by stripped links or citations
    s = re.sub(r"\(\s*[,;:]?\s*\)", "", s)
    s = re.sub(r"\[\s*[,;:]?\s*\]", "", s)
    s = re.sub(r"\s+([.,;:])(?=\s|$|[)\"'])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_rst_text(raw: str) -> str:
    """Clean Python Sphinx/reStructuredText glossary prose into crisp plain text."""
    s = _normalize_unicode(raw)
    # Drop directive blocks like `.. index:: ...` or `.. versionadded:: ...`
    s = re.sub(r"(?m)^\s*\.\.\s+[a-z0-9_\-]+::.*$", "", s)
    # Expand `:pep:`8`` / `:rfc:`7231``
    s = re.sub(r":pep:`(\d+)(?:[^`]*)`", r"PEP \1", s, flags=re.IGNORECASE)
    s = re.sub(r":rfc:`(\d+)(?:[^`]*)`", r"RFC \1", s, flags=re.IGNORECASE)
    # Expand `:role:`display <target>`` -> display
    s = re.sub(r":[a-z0-9_:.\-]+:`([^`<]+?)\s*<[^>]+>`", r"\1", s, flags=re.IGNORECASE)
    # Expand `:role:`~pkg.name`` or `:role:`!name`` -> name
    s = re.sub(r":[a-z0-9_:.\-]+:`[~!]?([^`]+)`", r"\1", s, flags=re.IGNORECASE)
    # Strip double-backtick inline literals
    s = re.sub(r"``([^`]+)``", r"\1", s)
    # Strip single-backtick interpreted text & external reST links `label <url>`_
    s = re.sub(r"`([^`<]+?)\s*<https?://[^>]+>`_?", r"\1", s)
    s = re.sub(r"`([^`]+)`_?", r"\1", s)
    s = _clean_inline_markdown(s)
    return s


_SGML_ENTITIES: tuple[tuple[str, str], ...] = (
    ("&mdash;", " - "),
    ("&ndash;", "-"),
    ("&nbsp;", " "),
    ("&ldquo;", '"'),
    ("&rdquo;", '"'),
    ("&lsquo;", "'"),
    ("&rsquo;", "'"),
    ("&hellip;", "..."),
    ("&times;", "x"),
)


def _clean_sgml_text(raw: str) -> str:
    """Clean PostgreSQL DocBook SGML glossary prose into crisp plain text."""
    s = _normalize_unicode(raw)
    for ent, rep in _SGML_ENTITIES:
        s = s.replace(ent, rep)
    s = re.sub(r"<!--[\s\S]*?-->", " ", s)
    s = re.sub(r"<xref\b[^>]*/>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_CROSSREF_SENT_RE = re.compile(
    r"^(?:see\s+(?:also\b|[A-Z`\"']|the\s)|for\s+more\s+(?:information|details)\b|refer\s+to\s+|read\s+more\b)",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return []
    sentinel = "\x00"
    protected = _ABBREV_PROTECT_RE.sub(lambda m: m.group(1) + sentinel, clean)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", protected)
    out: list[str] = []
    for p in parts:
        s = p.replace(sentinel, ".").strip()
        if len(s) < 22:
            continue
        if _CROSSREF_SENT_RE.match(s) and len(s) < 140:
            continue
        out.append(s)
    return out


def _truncate_facet(text: str, limit: int = 220) -> str:
    """Ensure every facet stays strictly <= limit chars and ends cleanly without dangling clauses."""
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if not s:
        return ""
    needs_cut = len(s) > limit or (len(s) == limit and s[-1] not in ".!?")
    if needs_cut:
        # Prefer ending on a complete sentence boundary within [60, limit] when available
        sents = _split_sentences(s)
        acc = ""
        for sent in sents:
            cand = f"{acc} {sent}".strip() if acc else sent
            if len(cand) <= limit:
                acc = cand
            else:
                break
        if len(acc) >= 60:
            s = acc
        else:
            cut = s[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-([ ")
            if not cut:
                cut = s[: limit - 1].rstrip(",;:-([ ")
            # Prefer a clean clause break (semicolon or comma) if >= 75 chars
            for delim in ("; ", ", "):
                pos = cut.rfind(delim)
                if pos >= 75:
                    cut = cut[:pos].rstrip(",;:-([ ")
                    break
            words = cut.split(" ")
            while len(words) > 4 and (
                words[-1].rstrip(",;:-").casefold() in _TRAILING_DANGLING_WORDS
                or words[-1].endswith(("'s", "’s", "'"))
            ):
                words.pop()
            s = " ".join(words).rstrip(",;:-([ ")
    if s.count("(") > s.count(")"):
        last_open = s.rfind("(")
        if last_open >= 35:
            s = s[:last_open].rstrip(" ,;:-")
        else:
            s = s.replace("(", "").replace(")", "")
    s = s.rstrip(" ,;:-([")
    if s and s[-1] not in ".!?":
        if len(s) >= limit:
            s = s[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-([ ")
        s += "."
    return s


def _smart_title(raw: str) -> str:
    s = _clean_inline_markdown(raw)
    s = re.sub(r"\s+", " ", s).strip(" -:;.,")
    if not s:
        return ""
    if s[0].islower():
        return s[0].upper() + s[1:]
    return s


def _is_code_heavy(text: str) -> bool:
    if not text:
        return True
    s = text.strip()
    if re.match(r"^(?:\$\s|>>>\s|def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import|SELECT\s+|CREATE\s+TABLE|kubectl\s+|curl\s+|#include)", s):
        return True
    code_chars = sum(1 for c in s if c in "{};=<>\\$[]")
    return (code_chars / max(len(s), 1)) > 0.075


# ---------------------------------------------------------------------------
# Mirror 7 Domain Classification & Quality Validation
# ---------------------------------------------------------------------------

def _infer_tech_domain(title: str, text: str, default: str = "software_engineering") -> str:
    blob = f"{title} {text}".casefold()
    if any(k in blob for k in (
        "attack", "adversar", "exploit", "malware", "phish", "credential", "ransomware",
        "vulnerability", "cve", "cwe", "owasp", "xss", "csrf", "sql injection", "tls",
        "ssl", "https", "certificate", "cipher", "encrypt", "decrypt", "cryptograph",
        "hmac", "hash function", "zero trust", "firewall", "intrusion", "exfiltrat",
        "privilege escalation", "lateral movement", "authentication", "authorization",
        "oauth", "jwt", "saml", "mfa", "cors", "csp", "hsts", "dnssec", "bgp", "vpn",
        "ipsec", "kerberos", "pki", "signature", "sandbox", "mitigat", "security",
        "threat", "penetration", "spoofing", "hijacking", "tampering", "fuzzing",
        "backdoor", "rootkit", "botnet", "ddos", "tcp", "udp", "ipv4", "ipv6",
        "http", "quic", "websocket", "packet", "subnet", "routing", "dns", "smtp",
    )):
        return "cybersecurity_networking"
    if any(k in blob for k in (
        "machine learning", "neural network", "deep learning", "large language model",
        "artificial intelligence", "transformer", "reinforcement learning", "embedding",
        "gradient descent", "backpropagation", "adversarial machine learning", "ai agent",
        "prompt injection", "hallucination", "generative ai", "model training",
    )):
        return "artificial_intelligence"
    if any(k in blob for k in (
        "algorithm", "complexity", "big o", "data structure", "compiler", "interpreter",
        "abstract syntax", "turing", "automata", "recursion", "binary tree", "hash table",
        "graph traversal", "concurrency", "deadlock", "mutex", "semaphore", "virtual memory",
        "garbage collection", "type system", "lambda calculus", "character encoding",
        "unicode", "utf-8", "endianness", "operating system", "kernel", "thread", "cpu",
    )):
        return "computer_science"
    if any(k in blob for k in (
        "scada", "plc", "industrial control", "firmware", "hardware", "microcontroller",
        "embedded", "signal", "sensor", "actuator", "electrical", "telemetry", "radio",
    )):
        return "engineering"
    if any(k in blob for k in (
        "compliance", "regulation", "gdpr", "hipaa", "sanctions", "audit", "governance",
        "privacy law", "legal", "liability", "standardization", "fips",
    )):
        return "law_governance"
    return default if default in VALID_MIRROR_DOMAINS else "software_engineering"


def _score_and_validate_record(
    *,
    concept_id: str,
    title: str,
    category: str,
    aliases: list[str],
    definition: str,
    mechanism: str,
    applications: str,
    tradeoffs: str,
    provenance: str,
    reject_counter: Counter[str],
) -> RepoConceptRecord | None:
    """Enforce Mirror 7 Phase-3 Quality-Over-Quantity gate (`quality_score >= 0.80`)."""
    clean_title = _smart_title(title)
    if not clean_title or len(clean_title) < 3 or len(clean_title) > 110:
        reject_counter["bad_title"] += 1
        return None
    if _REJECT_TITLE_PREFIX_RE.match(clean_title) or _RAW_ID_TITLE_RE.match(clean_title):
        reject_counter["bad_title"] += 1
        return None
    if re.search(r"\b(?:deprecated|obsolete|revoked)\b", clean_title, re.IGNORECASE):
        reject_counter["deprecated_or_junk_cue"] += 1
        return None
    if _RESIDUAL_SYNTAX_RE.search(clean_title) or any(ch in clean_title for ch in ("<", ">", "`", "|", "{", "}", "[")):
        reject_counter["residual_syntax"] += 1
        return None

    defn = _truncate_facet(_clean_inline_markdown(definition), 220)
    mech = _truncate_facet(_clean_inline_markdown(mechanism), 220)
    apps = _truncate_facet(_clean_inline_markdown(applications), 220)
    trd = _truncate_facet(_clean_inline_markdown(tradeoffs), 220)

    if len(defn) < 55:
        reject_counter["stub_short_definition"] += 1
        return None
    if len(mech) < 30 or len(apps) < 30 or len(trd) < 30:
        reject_counter["stub_short_facet"] += 1
        return None

    # Reject interrogative / MCQ definitions
    if "?" in defn:
        reject_counter["question_or_mcq"] += 1
        return None

    # Check all facets for residual markdown/HTML/macro syntax, unbalanced parens, or banned cues
    all_facets = (defn, mech, apps, trd)
    for facet in all_facets:
        if _RESIDUAL_SYNTAX_RE.search(facet):
            reject_counter["residual_syntax"] += 1
            return None
        if facet.count("(") != facet.count(")"):
            reject_counter["unbalanced_parens"] += 1
            return None
        low = facet.casefold()
        if _DEPRECATED_PROSE_RE.search(facet) or any(cue in low for cue in _REJECT_PROSE_CUES):
            reject_counter["deprecated_or_junk_cue"] += 1
            return None

    # Reject code-heavy definitions/mechanisms
    if _is_code_heavy(defn) or _is_code_heavy(mech):
        reject_counter["code_heavy_fragment"] += 1
        return None

    # Reject broken sentence starts in definition
    if re.match(r"^(?:and|or|but|however|for example|e\.g\.|i\.e\.|see also|note:|also known as)\b", defn, re.IGNORECASE):
        reject_counter["broken_sentence"] += 1
        return None

    # Distinctiveness across facets
    norm_facets = [re.sub(r"[^a-z0-9]+", " ", f.casefold()).strip() for f in all_facets]
    if len(set(norm_facets)) < 4:
        reject_counter["duplicate_facets"] += 1
        return None

    # Compute deterministic quality score aligned with open_ended_concept_lattice.score_concept_quality
    score = 1.0
    if len(defn) < 70:
        score -= 0.08
    if len(mech) < 45:
        score -= 0.06
    if len(apps) < 45:
        score -= 0.05
    if len(trd) < 45:
        score -= 0.05
    if defn.count("(") != defn.count(")"):
        score -= 0.25
    if clean_title.count("(") != clean_title.count(")"):
        score -= 0.25
    if not re.search(r"[a-z]{3,}", clean_title.casefold()):
        score -= 0.30
    score = round(max(0.0, min(1.0, score)), 3)
    if score < 0.80:
        reject_counter["below_quality_threshold"] += 1
        return None

    cid = re.sub(r"\s+", " ", (concept_id or clean_title)).strip().casefold()
    cid = re.sub(r"[^\w\s\-:+/.]", "", cid).strip()
    if len(cid) < 3:
        reject_counter["bad_title"] += 1
        return None

    cat = category if category in VALID_MIRROR_DOMAINS else "software_engineering"
    ck_main = canonical_concept_key(cid)
    clean_aliases: list[str] = [cid]
    title_low = clean_title.casefold()
    if title_low != cid:
        clean_aliases.append(title_low)
    for raw_al in aliases:
        al = re.sub(r"\s+", " ", _clean_inline_markdown(raw_al)).strip().casefold()
        if not al or len(al) < 3 or len(al) > 85:
            continue
        if _RESIDUAL_SYNTAX_RE.search(al):
            continue
        ak = canonical_concept_key(al)
        if not ak or ak in _FORBIDDEN_SECONDARY_ALIASES:
            continue
        if ak != ck_main and al not in clean_aliases:
            clean_aliases.append(al)

    return RepoConceptRecord(
        concept_id=cid,
        title=clean_title,
        category=cat,
        aliases=tuple(dict.fromkeys(clean_aliases[:6])),
        definition=defn,
        mechanism=mech,
        applications=apps,
        tradeoffs=trd,
        provenance=provenance,
        quality_score=score,
    )


# ---------------------------------------------------------------------------
# Worker 1: MITRE ATT&CK STIX 2.0 (Enterprise + ICS + Mobile)
# ---------------------------------------------------------------------------

_MITRE_ATTACK_BUNDLES: tuple[tuple[str, str], ...] = (
    ("enterprise", "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"),
    ("ics", "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json"),
    ("mobile", "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json"),
)


def _extract_mitre_external_id(obj: dict[str, Any], source_prefix: str = "mitre") -> str:
    for ref in obj.get("external_references") or []:
        src = str(ref.get("source_name", "")).casefold()
        ext = str(ref.get("external_id", "")).strip()
        if ext and (source_prefix in src or not source_prefix):
            return ext
    return ""


def harvest_mitre_attack(reject_counter: Counter[str]) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest active, non-deprecated, non-revoked MITRE ATT&CK tactics, mitigations, and techniques."""
    records: list[RepoConceptRecord] = []
    raw_bytes = 0
    ok_domains: list[str] = []

    for domain_tag, url in _MITRE_ATTACK_BUNDLES:
        try:
            bundle, nbytes = _http_get_json(url, timeout=45.0)
            raw_bytes += nbytes
            ok_domains.append(domain_tag)
        except Exception:
            continue

        objects = bundle.get("objects") or []
        by_stix_id: dict[str, dict[str, Any]] = {}
        for obj in objects:
            sid = obj.get("id")
            if sid:
                by_stix_id[sid] = obj

        # Build relationship maps: subtechnique-of & mitigates
        parent_of: dict[str, str] = {}
        mitigations_for_tech: dict[str, list[str]] = defaultdict(list)
        techs_for_mitigation: dict[str, list[str]] = defaultdict(list)

        for obj in objects:
            if obj.get("type") != "relationship" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue
            rel = obj.get("relationship_type", "")
            src_id = obj.get("source_ref", "")
            tgt_id = obj.get("target_ref", "")
            if rel == "subtechnique-of" and src_id and tgt_id:
                parent_obj = by_stix_id.get(tgt_id)
                if parent_obj and parent_obj.get("name"):
                    parent_of[src_id] = str(parent_obj["name"]).strip()
            elif rel == "mitigates" and src_id and tgt_id:
                coa = by_stix_id.get(src_id)
                ap = by_stix_id.get(tgt_id)
                if coa and ap and not coa.get("revoked") and not ap.get("revoked"):
                    coa_name = str(coa.get("name", "")).strip()
                    ap_name = str(ap.get("name", "")).strip()
                    if coa_name and coa_name not in mitigations_for_tech[tgt_id]:
                        mitigations_for_tech[tgt_id].append(coa_name)
                    if ap_name and ap_name not in techs_for_mitigation[src_id]:
                        techs_for_mitigation[src_id].append(ap_name)

        domain_label = {"enterprise": "Enterprise", "ics": "ICS", "mobile": "Mobile"}[domain_tag]

        # Process tactics, mitigations, and techniques
        for obj in objects:
            otype = obj.get("type")
            if otype not in ("x-mitre-tactic", "course-of-action", "attack-pattern"):
                continue
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                reject_counter["mitre_deprecated_or_revoked"] += 1
                continue

            ext_id = _extract_mitre_external_id(obj, "mitre")
            raw_name = str(obj.get("name", "")).strip()
            raw_desc = str(obj.get("description", "")).strip()
            if not ext_id or not raw_name or not raw_desc:
                continue

            paras = _clean_markdown_to_paragraphs(raw_desc)
            if not paras:
                reject_counter["stub_short_definition"] += 1
                continue
            sents = _split_sentences(" ".join(paras[:3]))
            if not sents:
                reject_counter["stub_short_definition"] += 1
                continue

            defn = sents[0]
            if len(defn) < 55 and len(sents) >= 2:
                defn = f"{sents[0]} {sents[1]}"

            sid = str(obj.get("id", ""))

            if otype == "x-mitre-tactic":
                title = f"{raw_name} (ATT&CK {domain_label} Tactic)" if domain_tag != "enterprise" else f"{raw_name} (ATT&CK Tactic)"
                cid = f"{raw_name.casefold()} attack tactic" if domain_tag == "enterprise" else f"{raw_name.casefold()} {domain_tag} attack tactic"
                aliases = [ext_id, f"mitre {ext_id}", f"{raw_name} tactic"]
                mech = sents[1] if len(sents) >= 2 else f"Represents a core adversary tactical objective in the MITRE ATT&CK {domain_label} kill chain ({ext_id})."
                apps = f"Used by SOC teams, threat hunters, and purple teams to structure detection coverage, threat modeling, and adversary emulation for {raw_name}."
                trd = f"Mapping controls solely to {raw_name} is insufficient without layered prevention and telemetry across adjacent ATT&CK kill-chain phases."
                cat = "cybersecurity_networking"

            elif otype == "course-of-action":
                title = raw_name
                if len(raw_name.split()) <= 2:
                    title = f"{raw_name} (ATT&CK Mitigation)"
                cid = title.casefold()
                aliases = [ext_id, f"mitre {ext_id}"]
                mitigated = techs_for_mitigation.get(sid, [])[:4]
                mech = sents[1] if len(sents) >= 2 else f"Enforces defensive architecture and configuration controls standardized under MITRE ATT&CK mitigation {ext_id}."
                if mitigated:
                    apps = f"Directly mitigates adversary techniques including {', '.join(mitigated)} across {domain_label} environments."
                else:
                    apps = f"Deployed in enterprise security baselines and defense-in-depth hardening aligned with MITRE ATT&CK {domain_label} ({ext_id})."
                trd = f"Requires careful operational testing and policy governance to prevent workflow disruption or bypass via misconfigured exceptions in {raw_name}."
                cat = "cybersecurity_networking"

            else:  # attack-pattern
                is_sub = bool(obj.get("x_mitre_is_subtechnique"))
                parent_name = parent_of.get(sid, "")
                if is_sub and parent_name and parent_name.casefold() not in raw_name.casefold():
                    title = f"{parent_name}: {raw_name}"
                else:
                    title = raw_name
                if domain_tag in ("ics", "mobile") and len(title.split()) <= 2:
                    title = f"{title} ({domain_label} ATT&CK)"
                cid = title.casefold()
                aliases = [ext_id, f"mitre {ext_id}", f"att&ck {ext_id}"]

                phases = [
                    str(kp.get("phase_name", "")).replace("-", " ").title()
                    for kp in (obj.get("kill_chain_phases") or [])
                    if kp.get("phase_name")
                ]
                platforms = [str(p) for p in (obj.get("x_mitre_platforms") or []) if p][:5]
                det_paras = _clean_markdown_to_paragraphs(str(obj.get("x_mitre_detection", "") or ""))
                det_sents = _split_sentences(" ".join(det_paras[:2])) if det_paras else []
                mits = mitigations_for_tech.get(sid, [])[:4]

                if len(sents) >= 2:
                    mech = sents[1]
                elif phases:
                    mech = f"Operates during the {', '.join(phases[:2])} phase(s) in MITRE ATT&CK {domain_label} ({ext_id}), targeting {', '.join(platforms) if platforms else 'enterprise systems'}."
                else:
                    mech = f"Executed by adversaries targeting {', '.join(platforms) if platforms else domain_label + ' systems'} under MITRE ATT&CK technique {ext_id}."

                if det_sents:
                    apps = f"Detection & threat hunting: {det_sents[0]}"
                else:
                    plat_str = ", ".join(platforms) if platforms else f"{domain_label} environments"
                    phase_str = ", ".join(phases[:2]) if phases else "adversary operations"
                    apps = f"Analyzed in SOC telemetry, SIEM rule engineering, and red-team emulation covering {phase_str} on {plat_str}."

                if mits:
                    trd = f"Defended against via MITRE ATT&CK mitigations including {', '.join(mits)}, balancing security hardening against administrative overhead."
                else:
                    trd = f"Difficult to eliminate completely because {title} often abuses legitimate system utilities, requiring behavioral telemetry and least-privilege containment."
                cat = "cybersecurity_networking"

            rec = _score_and_validate_record(
                concept_id=cid,
                title=title,
                category=cat,
                aliases=aliases,
                definition=defn,
                mechanism=mech,
                applications=apps,
                tradeoffs=trd,
                provenance=f"mitre:attack:{ext_id}",
                reject_counter=reject_counter,
            )
            if rec is not None:
                records.append(rec)

    status = f"ok ({','.join(ok_domains)}: {len(records)} candidates)" if ok_domains else "failed"
    return records, raw_bytes, status


# ---------------------------------------------------------------------------
# Worker 2: MITRE CAPEC STIX 2.1 (`mitre/cti`)
# ---------------------------------------------------------------------------

def harvest_mitre_capec(reject_counter: Counter[str]) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest active MITRE CAPEC common attack patterns enriched with severity & mitigations."""
    url = "https://raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json"
    try:
        bundle, raw_bytes = _http_get_json(url, timeout=35.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    objects = bundle.get("objects") or []
    coa_by_id: dict[str, str] = {}
    for obj in objects:
        if obj.get("type") == "course-of-action" and not obj.get("revoked"):
            raw_coa = str(obj.get("description", "") or obj.get("name", ""))
            raw_coa = re.sub(r"<[^>]+>", " ", raw_coa)
            raw_coa = html.unescape(raw_coa)
            desc = _clean_inline_markdown(raw_coa)
            if desc:
                coa_by_id[str(obj.get("id", ""))] = desc

    mits_for_ap: dict[str, list[str]] = defaultdict(list)
    for obj in objects:
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "mitigates":
            src = str(obj.get("source_ref", ""))
            tgt = str(obj.get("target_ref", ""))
            if src in coa_by_id and tgt:
                mits_for_ap[tgt].append(coa_by_id[src])

    records: list[RepoConceptRecord] = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        capec_status = str(obj.get("x_capec_status", "")).strip().casefold()
        if obj.get("revoked") or capec_status in ("deprecated", "obsolete"):
            reject_counter["capec_deprecated"] += 1
            continue

        ext_id = _extract_mitre_external_id(obj, "capec")
        raw_name = str(obj.get("name", "")).strip()
        raw_desc = str(obj.get("description", "")).strip()
        if not ext_id or not raw_name or not raw_desc:
            continue

        paras = _clean_markdown_to_paragraphs(raw_desc)
        if not paras:
            continue
        sents = _split_sentences(" ".join(paras[:3]))
        if not sents:
            continue

        defn = sents[0]
        if len(defn) < 55 and len(sents) >= 2:
            defn = f"{sents[0]} {sents[1]}"

        domains = [str(d) for d in (obj.get("x_capec_domains") or []) if d][:3]
        prereqs = [str(p) for p in (obj.get("x_capec_prerequisites") or []) if p][:2]
        severity = str(obj.get("x_capec_typical_severity", "") or "").strip()
        likelihood = str(obj.get("x_capec_likelihood_of_attack", "") or "").strip()
        consequences = obj.get("x_capec_consequences") or {}
        impacts = [f"{k}: {', '.join(v[:2])}" for k, v in list(consequences.items())[:2] if isinstance(v, list) and v]

        if len(sents) >= 2:
            mech = sents[1]
        elif prereqs:
            mech = f"Requires prerequisite conditions such as {_clean_inline_markdown(prereqs[0])}."
        else:
            mech = f"Exploits design or implementation weaknesses classified under MITRE {ext_id} across {', '.join(domains) if domains else 'software systems'}."

        dom_txt = ", ".join(domains) if domains else "application security"
        sev_txt = f" (typical severity: {severity}" + (f", likelihood: {likelihood})" if likelihood else ")") if severity else ""
        apps = f"Used in secure design reviews, threat modeling, and CWE weakness mapping for {dom_txt}{sev_txt}."

        sid = str(obj.get("id", ""))
        linked_mits = mits_for_ap.get(sid, [])
        if linked_mits:
            mit_sents = _split_sentences(linked_mits[0])
            trd = f"Mitigation: {mit_sents[0] if mit_sents else linked_mits[0]}"
        elif impacts:
            trd = f"Security impact spans {'; '.join(impacts)}, requiring defense-in-depth validation and least-privilege isolation."
        else:
            trd = f"Mitigating {raw_name} requires input validation, strict authorization checks, and continuous security testing."

        rec = _score_and_validate_record(
            concept_id=raw_name.casefold(),
            title=raw_name,
            category="cybersecurity_networking",
            aliases=[ext_id, f"mitre {ext_id}"],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"mitre:capec:{ext_id}",
            reject_counter=reject_counter,
        )
        if rec is not None:
            records.append(rec)

    return records, raw_bytes, f"ok ({len(records)} candidates)"


# ---------------------------------------------------------------------------
# Worker 3: CNCF Cloud Native Glossary (`github.com/cncf/glossary`)
# ---------------------------------------------------------------------------

def _split_md_sections(body: str) -> dict[str, str]:
    """Split markdown body into heading-keyed sections ('_intro' + lowercased H2/H3 titles)."""
    sections: dict[str, list[str]] = defaultdict(list)
    current = "_intro"
    for line in body.splitlines():
        hm = re.match(r"^#{1,3}\s+(.+?)\s*$", line.strip())
        if hm:
            current = _clean_inline_markdown(hm.group(1)).casefold()
            continue
        sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def harvest_cncf_glossary(reject_counter: Counter[str], *, max_workers: int = 10) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest CNCF Cloud Native Glossary (`content/en/*.md`) via GitHub Tree + raw.githubusercontent.com."""
    tree_url = "https://api.github.com/repos/cncf/glossary/git/trees/main?recursive=1"
    try:
        tree_data, raw_bytes = _http_get_json(tree_url, timeout=20.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    md_paths = [
        item["path"]
        for item in (tree_data.get("tree") or [])
        if item.get("type") == "blob"
        and str(item.get("path", "")).startswith("content/en/")
        and str(item.get("path", "")).endswith(".md")
        and not Path(item["path"]).name.startswith("_")
    ]

    def _fetch_one(rel_path: str) -> tuple[RepoConceptRecord | None, int]:
        url = f"https://raw.githubusercontent.com/cncf/glossary/main/{urllib.parse.quote(rel_path)}"
        try:
            text, nb = _http_get_text(url, timeout=15.0)
        except Exception:
            return None, 0
        meta, body = _parse_yaml_frontmatter(text)
        status_str = str(meta.get("status", "")).casefold()
        if "deprecated" in status_str or "obsolete" in status_str:
            reject_counter["cncf_deprecated"] += 1
            return None, nb
        slug = Path(rel_path).stem
        title = str(meta.get("title", "")).strip() or slug.replace("-", " ").title()
        sections = _split_md_sections(body)

        intro_paras = _clean_markdown_to_paragraphs(sections.get("_intro", "") + "\n\n" + sections.get("what it is", ""))
        prob_paras = _clean_markdown_to_paragraphs(sections.get("problem it addresses", ""))
        help_paras = _clean_markdown_to_paragraphs(sections.get("how it helps", ""))

        intro_sents = _split_sentences(" ".join(intro_paras[:2]))
        if not intro_sents:
            return None, nb
        defn = intro_sents[0]
        if len(defn) < 55 and len(intro_sents) >= 2:
            defn = f"{intro_sents[0]} {intro_sents[1]}"

        help_sents = _split_sentences(" ".join(help_paras[:2]))
        prob_sents = _split_sentences(" ".join(prob_paras[:2]))

        mech = help_sents[0] if help_sents else (
            intro_sents[1] if len(intro_sents) >= 2 else f"Operates within cloud-native architectures to standardize orchestration, observability, and deployment for {title}."
        )
        apps = (
            f"Addresses cloud-native operational needs: {prob_sents[0]}"
            if prob_sents
            else f"Applied in Kubernetes clusters, microservices platforms, and DevOps delivery pipelines leveraging {title}."
        )
        trd = (
            help_sents[1]
            if len(help_sents) >= 2
            else f"Adopting {title} introduces platform configuration complexity, observability overhead, and team operational learning curves."
        )

        tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
        cat = _infer_tech_domain(title, f"{' '.join(str(t) for t in tags)} {defn} {mech}", default="software_engineering")
        rec = _score_and_validate_record(
            concept_id=title.casefold(),
            title=title,
            category=cat,
            aliases=[slug.replace("-", " ")],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"cncf:glossary:{slug}",
            reject_counter=reject_counter,
        )
        return rec, nb

    records: list[RepoConceptRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_one, p) for p in md_paths]
        for fut in as_completed(futs):
            rec, nb = fut.result()
            raw_bytes += nb
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r.concept_id)
    return records, raw_bytes, f"ok ({len(records)}/{len(md_paths)} files)"


# ---------------------------------------------------------------------------
# Worker 4: CNCF Kubernetes Official Glossary (`github.com/kubernetes/website`)
# ---------------------------------------------------------------------------

def harvest_k8s_glossary(reject_counter: Counter[str], *, max_workers: int = 12) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest Kubernetes Official Glossary (`content/en/docs/reference/glossary/*.md`)."""
    list_url = "https://api.github.com/repos/kubernetes/website/contents/content/en/docs/reference/glossary"
    try:
        items, raw_bytes = _http_get_json(list_url, timeout=20.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    files = [
        item["name"]
        for item in (items if isinstance(items, list) else [])
        if str(item.get("name", "")).endswith(".md") and not str(item.get("name", "")).startswith("_")
    ]

    def _fetch_one(fname: str) -> tuple[RepoConceptRecord | None, int]:
        url = f"https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/reference/glossary/{urllib.parse.quote(fname)}"
        try:
            text, nb = _http_get_text(url, timeout=15.0)
        except Exception:
            return None, 0
        meta, body = _parse_yaml_frontmatter(text)
        raw_title = str(meta.get("title", "")).strip()
        slug = str(meta.get("id", "")).strip() or Path(fname).stem
        if not raw_title or "deprecated" in raw_title.casefold():
            reject_counter["k8s_deprecated"] += 1
            return None, nb

        paras = _clean_markdown_to_paragraphs(body, is_hugo=True)
        short_desc = _clean_inline_markdown(_clean_hugo_shortcodes(str(meta.get("short_description", "") or "")))
        all_sents = _split_sentences(" ".join(paras[:4]))
        if not all_sents and short_desc:
            all_sents = _split_sentences(short_desc)
        if not all_sents:
            return None, nb

        # Disambiguate generic single-word English nouns cleanly with "Kubernetes "
        t_low = raw_title.casefold()
        if (
            len(raw_title.split()) == 1
            and t_low in _GENERIC_K8S_SINGLETONS
            and not t_low.startswith(("kube", "k8s"))
        ):
            title = f"Kubernetes {raw_title}"
            aliases = [f"k8s {t_low}"]
        else:
            title = raw_title
            aliases = [f"kubernetes {t_low}"] if not t_low.startswith(("kube", "k8s")) else []

        defn = all_sents[0]
        if len(defn) < 55 and len(all_sents) >= 2:
            defn = f"{all_sents[0]} {all_sents[1]}"
        elif len(defn) < 55 and short_desc and short_desc.casefold() != defn.casefold():
            defn = f"{defn.rstrip('.')} - {short_desc}"

        mech = all_sents[1] if len(all_sents) >= 2 and all_sents[1] not in defn else (
            all_sents[2] if len(all_sents) >= 3 else f"Managed declaratively by the Kubernetes control plane and API server as part of cluster state reconciliation for {title}."
        )
        apps = all_sents[2] if len(all_sents) >= 3 and all_sents[2] not in (defn, mech) else (
            f"Used in production Kubernetes container orchestration for declarative workload scheduling, scaling, and lifecycle management."
        )
        trd = f"Misconfiguring {title} in Kubernetes manifests can impact cluster scheduling, security isolation, or resource utilization under load."

        cat = _infer_tech_domain(title, f"{defn} {mech}", default="software_engineering")
        rec = _score_and_validate_record(
            concept_id=title.casefold(),
            title=title,
            category=cat,
            aliases=aliases,
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"cncf:k8s_glossary:{slug}",
            reject_counter=reject_counter,
        )
        return rec, nb

    records: list[RepoConceptRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_one, f) for f in files]
        for fut in as_completed(futs):
            rec, nb = fut.result()
            raw_bytes += nb
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r.concept_id)
    return records, raw_bytes, f"ok ({len(records)}/{len(files)} files)"


# ---------------------------------------------------------------------------
# Worker 5: OWASP Cheat Sheet Series (`github.com/OWASP/CheatSheetSeries`)
# ---------------------------------------------------------------------------

def harvest_owasp_cheatsheets(reject_counter: Counter[str], *, max_workers: int = 10) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest authoritative application-security defense patterns from OWASP Cheat Sheet Series."""
    tree_url = "https://api.github.com/repos/OWASP/CheatSheetSeries/git/trees/master?recursive=1"
    try:
        tree_data, raw_bytes = _http_get_json(tree_url, timeout=20.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    cs_paths = [
        item["path"]
        for item in (tree_data.get("tree") or [])
        if item.get("type") == "blob"
        and str(item.get("path", "")).startswith("cheatsheets/")
        and str(item.get("path", "")).endswith(".md")
        and not any(skip in Path(item["path"]).name.casefold() for skip in ("index", "template", "preface", "glossary"))
    ]

    def _fetch_one(rel_path: str) -> tuple[RepoConceptRecord | None, int]:
        url = f"https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/{urllib.parse.quote(rel_path)}"
        try:
            text, nb = _http_get_text(url, timeout=15.0)
        except Exception:
            return None, 0

        slug = Path(rel_path).stem
        # Extract H1 title
        h1_m = re.search(r"(?m)^#\s+(.+?)\s*$", text)
        raw_title = _clean_inline_markdown(h1_m.group(1)) if h1_m else slug.replace("_", " ")
        # Strip trailing "Cheat Sheet" cleanly
        clean_t = re.sub(r"\s+Cheat\s*Sheet\s*$", "", raw_title, flags=re.IGNORECASE).strip()
        if not clean_t or len(clean_t) < 4:
            return None, nb
        # Contextualize generic single-word OWASP sheets cleanly (e.g. "Logging" -> "OWASP Logging Security")
        if len(clean_t.split()) == 1 and "security" not in clean_t.casefold():
            title = f"OWASP {clean_t} Security"
        else:
            title = clean_t

        sections = _split_md_sections(text)
        intro_chunks: list[str] = []
        mech_chunks: list[str] = []
        app_chunks: list[str] = []
        trd_chunks: list[str] = []

        for sec_name, sec_body in sections.items():
            if sec_name in ("references", "related articles", "authors", "acknowledgments", "credits"):
                continue
            paras = _clean_markdown_to_paragraphs(sec_body)
            if not paras:
                continue
            if sec_name in ("_intro", "introduction", "overview", "context", "background", "about", "objective") or sec_name == raw_title.casefold():
                intro_chunks.extend(paras[:3])
            elif any(k in sec_name for k in ("how", "architecture", "mechanism", "defense", "prevent", "control", "mitigat", "protect", "guidance", "recommend", "best practice")):
                mech_chunks.extend(paras[:2])
            elif any(k in sec_name for k in ("risk", "threat", "attack", "pitfall", "mistake", "caveat", "limitation", "vulnerabilit", "common")):
                trd_chunks.extend(paras[:2])
            else:
                app_chunks.extend(paras[:2])

        intro_sents = _split_sentences(" ".join(intro_chunks))
        if not intro_sents:
            # Fallback to first substantive non-reference section
            intro_sents = _split_sentences(" ".join(mech_chunks[:2] or app_chunks[:2]))
        if not intro_sents:
            return None, nb

        # Skip meta lead-in sentences ("This cheat sheet provides...") if a stronger substantive sentence follows
        defn = intro_sents[0]
        if re.match(r"^this\s+cheat\s+sheet\b", defn, re.IGNORECASE) and len(intro_sents) >= 2 and len(intro_sents[1]) >= 55:
            defn = intro_sents[1]
        if len(defn) < 55 and len(intro_sents) >= 2:
            defn = f"{intro_sents[0]} {intro_sents[1]}"

        mech_sents = _split_sentences(" ".join(mech_chunks))
        trd_sents = _split_sentences(" ".join(trd_chunks))
        app_sents = _split_sentences(" ".join(app_chunks))

        mech = mech_sents[0] if mech_sents and mech_sents[0] != defn else (
            intro_sents[1] if len(intro_sents) >= 2 and intro_sents[1] != defn else f"Enforces OWASP defense-in-depth engineering controls, validation boundaries, and secure defaults for {title}."
        )
        apps = app_sents[0] if app_sents and app_sents[0] not in (defn, mech) else (
            f"Applied during secure architecture design, code review, and AppSec hardening checklists for {title}."
        )
        trd = trd_sents[0] if trd_sents and trd_sents[0] not in (defn, mech, apps) else (
            f"Overly rigid security controls in {title} can increase engineering friction or false positives unless tailored to threat models and tested continuously."
        )

        rec = _score_and_validate_record(
            concept_id=title.casefold(),
            title=title,
            category="cybersecurity_networking",
            aliases=[f"owasp {clean_t.casefold()}"],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"owasp:cheatsheet:{_slugify(slug)}",
            reject_counter=reject_counter,
        )
        return rec, nb

    records: list[RepoConceptRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_one, p) for p in cs_paths]
        for fut in as_completed(futs):
            rec, nb = fut.result()
            raw_bytes += nb
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r.concept_id)
    return records, raw_bytes, f"ok ({len(records)}/{len(cs_paths)} sheets)"


# ---------------------------------------------------------------------------
# Worker 6: MDN Web Docs Glossary (`github.com/mdn/content`)
# ---------------------------------------------------------------------------

def harvest_mdn_glossary(reject_counter: Counter[str], *, max_workers: int = 14) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest MDN Web Docs Glossary (`files/en-us/glossary/*/index.md`) with full KumaScript macro cleaning."""
    list_url = "https://api.github.com/repos/mdn/content/contents/files/en-us/glossary"
    try:
        items, raw_bytes = _http_get_json(list_url, timeout=25.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    dirs = [item["name"] for item in (items if isinstance(items, list) else []) if item.get("type") == "dir"]

    def _fetch_one(slug: str) -> tuple[RepoConceptRecord | None, int]:
        url = f"https://raw.githubusercontent.com/mdn/content/main/files/en-us/glossary/{urllib.parse.quote(slug)}/index.md"
        try:
            text, nb = _http_get_text(url, timeout=15.0)
        except Exception:
            return None, 0
        # Reject explicitly deprecated/obsolete pages immediately
        if re.search(r"\{\{\s*(?:Deprecated_Header|Obsolete_Header)\s*\}\}", text, re.IGNORECASE):
            reject_counter["mdn_deprecated"] += 1
            return None, nb
        meta, body = _parse_yaml_frontmatter(text)
        status_val = str(meta.get("status", "")).casefold()
        if "deprecated" in status_val or "obsolete" in status_val:
            reject_counter["mdn_deprecated"] += 1
            return None, nb

        title = str(meta.get("title", "")).strip() or slug.replace("_", " ").title()
        # Strip everything from "## See also" / "## Learn more" onward
        body_main = re.split(r"(?m)^##\s+(?:See also|Learn more|External links|References)\b", body, maxsplit=1, flags=re.IGNORECASE)[0]
        paras = _clean_markdown_to_paragraphs(body_main, is_mdn=True)
        if not paras:
            return None, nb

        sents = _split_sentences(" ".join(paras[:4]))
        if not sents:
            return None, nb

        defn = sents[0]
        if len(defn) < 55 and len(sents) >= 2:
            defn = f"{sents[0]} {sents[1]}"

        mech = sents[1] if len(sents) >= 2 and sents[1] not in defn else (
            sents[2] if len(sents) >= 3 else f"Defined by open web platform specifications and browser engine standards governing {title}."
        )
        apps = sents[2] if len(sents) >= 3 and sents[2] not in (defn, mech) else (
            f"Used across web applications, HTTP/DOM APIs, browser interoperability, and front-end/network engineering for {title}."
        )
        trd = sents[3] if len(sents) >= 4 and sents[3] not in (defn, mech, apps) else (
            f"Browser support differences, specification evolution, and security boundary constraints affect how {title} behaves in production web systems."
        )

        cat = _infer_tech_domain(title, f"{defn} {mech}", default="software_engineering")
        rec = _score_and_validate_record(
            concept_id=title.casefold(),
            title=title,
            category=cat,
            aliases=[slug.replace("_", " ")],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"mdn:glossary:{slug}",
            reject_counter=reject_counter,
        )
        return rec, nb

    records: list[RepoConceptRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_one, d) for d in dirs]
        for fut in as_completed(futs):
            rec, nb = fut.result()
            raw_bytes += nb
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r.concept_id)
    return records, raw_bytes, f"ok ({len(records)}/{len(dirs)} entries)"


# ---------------------------------------------------------------------------
# Worker 7: NIST CSRC Official Glossary (`csrc.nist.gov`)
# ---------------------------------------------------------------------------

def harvest_nist_csrc_glossary(
    reject_counter: Counter[str],
    *,
    max_records: int = 2000,
) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest official NIST CSRC Glossary (FIPS, SP 800-series, AI RMF, CSF 2.0, NISTIR 7298r3)."""
    url = "https://csrc.nist.gov/csrc/media/glossary/glossary-export.zip"
    try:
        zbytes = _http_get_bytes(url, timeout=30.0)
        zf = zipfile.ZipFile(io.BytesIO(zbytes))
        json_name = next((n for n in zf.namelist() if n.endswith(".json")), None)
        if not json_name:
            return [], len(zbytes), "fallback_skipped (no json in zip)"
        raw_json = zf.read(json_name)
        data = json.loads(raw_json.decode("utf-8-sig", errors="replace"))
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    parent_terms = data.get("parentTerms") or []
    scored_candidates: list[tuple[int, RepoConceptRecord]] = []

    for entry in parent_terms:
        term = _normalize_unicode(str(entry.get("term", "") or "")).strip()
        defs = entry.get("definitions") or []
        if not term or not defs:
            continue
        # Reject cryptic symbols, raw acronyms without expansion, overly terse single words
        if len(term) < 4 or len(term) > 90:
            reject_counter["nist_bad_title"] += 1
            continue
        if re.match(r"^[0-9(\[\-_]", term) or re.match(r"^[A-Z0-9\-/]{2,6}$", term):
            reject_counter["nist_raw_abbr_or_symbol"] += 1
            continue
        # Require at least 2 words OR a rich specific technical term (>= 8 alpha chars)
        if len(term.split()) == 1 and len(re.sub(r"[^a-zA-Z]", "", term)) < 8:
            reject_counter["nist_generic_singleton"] += 1
            continue

        # Select the best substantive, non-deprecated definition with official NIST citation
        best_def = ""
        best_sources: list[str] = []
        second_def = ""
        for d in defs:
            raw_dtxt = html.unescape(re.sub(r"<[^>]+>", " ", str(d.get("text", "") or "")))
            dtxt = _clean_inline_markdown(_normalize_unicode(raw_dtxt))
            if not dtxt or len(dtxt) < 65:
                continue
            dlow = dtxt.casefold()
            if dlow.startswith(("see ", "synonym", "superseded", "withdrawn", "obsolete", "deprecated")):
                continue
            if any(bad in dlow for bad in ("superseded", "withdrawn", "deprecated", "obsolete")):
                continue
            srcs = [str(s.get("text", "")).strip() for s in (d.get("sources") or []) if s.get("text")]
            if not best_def or (len(dtxt) > len(best_def) and len(dtxt) <= 320):
                if best_def and not second_def:
                    second_def = best_def
                best_def = dtxt
                best_sources = srcs[:3]
            elif not second_def and dtxt.casefold() != best_def.casefold():
                second_def = dtxt

        if not best_def:
            reject_counter["nist_stub_or_see_ref"] += 1
            continue

        # Reject circular "X is the X of..." definitions
        if best_def.casefold().startswith(term.casefold() + " means " + term.casefold()):
            reject_counter["nist_circular"] += 1
            continue

        sents = _split_sentences(best_def)
        if not sents:
            continue
        defn = sents[0]
        if len(defn) < 60 and len(sents) >= 2:
            defn = f"{sents[0]} {sents[1]}"

        src_str = ", ".join(best_sources) if best_sources else "NIST CSRC"
        note_clean = html.unescape(re.sub(r"<[^>]+>", " ", str(entry.get("note", "") or "")))
        note_raw = _clean_inline_markdown(_normalize_unicode(note_clean))
        note_sents = _split_sentences(note_raw) if note_raw else []
        see_also_raw = [
            _clean_inline_markdown(html.unescape(re.sub(r"<[^>]+>", " ", str(x.get("text", "")))))
            for x in (entry.get("seeAlso") or [])
            if isinstance(x, dict) and x.get("text")
        ]
        see_also: list[str] = []
        seen_sa: set[str] = set()
        for sa in see_also_raw:
            if sa and sa.casefold() not in seen_sa:
                seen_sa.add(sa.casefold())
                see_also.append(sa)
            if len(see_also) >= 3:
                break
        abbr_syns = [
            _clean_inline_markdown(html.unescape(re.sub(r"<[^>]+>", " ", str(x.get("text", "")))))
            for x in (entry.get("abbrSyn") or [])
            if isinstance(x, dict) and x.get("text")
        ][:3]

        if len(sents) >= 2 and sents[1] not in defn:
            mech = sents[1]
        elif second_def:
            sec_sents = _split_sentences(second_def)
            mech = sec_sents[0] if sec_sents else f"Standardized in {src_str} to establish unambiguous technical criteria for {term}."
        elif note_sents:
            mech = note_sents[0]
        else:
            mech = f"Defined and standardized in {src_str} as a formal security, cryptographic, or risk-management construct."

        if see_also:
            apps = f"Applied in NIST compliance frameworks ({src_str}), security engineering, and governance alongside related controls such as {', '.join(see_also)}."
        else:
            apps = f"Applied in federal and enterprise cybersecurity architecture, risk assessments, and technical standards referencing {src_str}."

        trd = f"Implementing {term} per {src_str} requires rigorous verification, lifecycle maintenance, and alignment with organizational threat models to avoid compliance gaps."

        cat = _infer_tech_domain(term, f"{defn} {mech} {src_str}", default="cybersecurity_networking")
        slug = _slugify(term)
        rec = _score_and_validate_record(
            concept_id=term.casefold(),
            title=term,
            category=cat,
            aliases=[a for a in abbr_syns if len(a) >= 3],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"nist:csrc:{slug}",
            reject_counter=reject_counter,
        )
        if rec is not None:
            # Rank higher when cited by core FIPS / SP 800 / NIST AI / CSF standards and rich multi-sentence prose
            priority = len(best_sources) * 10 + (15 if any(p in src_str for p in ("FIPS", "SP 800", "NIST AI", "CSF")) else 0) + min(len(best_def), 200) // 10
            scored_candidates.append((priority, rec))

    scored_candidates.sort(key=lambda pair: (-pair[0], pair[1].concept_id))
    selected = [rec for _, rec in scored_candidates[:max_records]]
    return selected, len(raw_json), f"ok ({len(selected)}/{len(parent_terms)} terms)"


# ---------------------------------------------------------------------------
# Worker 8: IETF RFC 4949 Internet Security Glossary (`rfc-editor.org`)
# ---------------------------------------------------------------------------

def harvest_ietf_rfc4949(
    reject_counter: Counter[str],
    *,
    max_records: int = 550,
) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest Internet-standard `(I)` security & cryptography definitions from IETF RFC 4949."""
    url = "https://www.rfc-editor.org/rfc/rfc4949.txt"
    try:
        text, raw_bytes = _http_get_text(url, timeout=25.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    # Strip RFC page headers/footers ("Shirey Informational [Page N]", form-feed, "RFC 4949 ...")
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if "\x0c" in line:
            continue
        if re.match(r"^\s*Shirey\s+Informational\s+\[Page\s+\d+\]\s*$", line):
            continue
        if re.match(r"^\s*RFC\s+4949\s+Internet\s+Security\s+Glossary", line):
            continue
        cleaned_lines.append(line)
    body = "\n".join(cleaned_lines)

    # Entries start with indented `$ term` (RFC 4949 indents 3 spaces)
    parts = re.split(r"(?m)^ {0,4}\$\s+", body)
    records: list[RepoConceptRecord] = []

    for block in parts[1:]:
        lines = block.splitlines()
        if not lines:
            continue
        raw_term = lines[0].strip()
        # Reject single-letter headers, symbols, overly short or parenthetical-only terms
        if not raw_term or len(raw_term) < 4 or len(raw_term) > 80:
            continue
        if re.match(r"^[^A-Za-z]|^[A-Z0-9\-/]{1,5}$", raw_term):
            continue
        entry_body = "\n".join(lines[1:])

        # Look for Internet-standard `(I)` definitions ONLY (reject `(D)` deprecated, `(N)` not recommended, `(C)` colloquial)
        i_defs: list[str] = []
        # Split numbered/lettered items inside the entry
        sub_blocks = re.split(r"\n\s*\n", entry_body)
        for sb in sub_blocks:
            flat = re.sub(r"\s+", " ", sb).strip()
            # Must explicitly have `(I)` recommended Internet marker
            m = re.match(r"^(?:\d+[a-z]?\.?\s*)?\(I\)\s*(?:/[^/]{1,40}/\s*)?(.*)$", flat)
            if not m:
                continue
            cand = m.group(1).strip()
            # Drop bare cross-references (`See: ...`, `Compare: ...`, `Deprecated ...`)
            if re.match(r"^(?:See:|Compare:|Deprecated|Abbreviation|Acronym|Synonym)", cand, re.IGNORECASE):
                continue
            # Strip trailing `(See: ...)` / `(Compare: ...)` / `[RFC...]` / `[I7498-2]`
            cand = re.sub(r"\((?:See|Compare|Tutorial|Usage|Deprecated)[^)]*\)", "", cand, flags=re.IGNORECASE)
            cand = re.sub(r"\[[A-Za-z0-9\-.:/ ]{2,25}\]", "", cand)
            cand = _clean_inline_markdown(cand)
            if len(cand) >= 65:
                i_defs.append(cand)

        if not i_defs:
            reject_counter["rfc4949_non_standard_or_stub"] += 1
            continue

        sents = _split_sentences(i_defs[0])
        if not sents:
            continue
        defn = sents[0]
        if len(defn) < 60 and len(sents) >= 2:
            defn = f"{sents[0]} {sents[1]}"

        mech = (
            _split_sentences(i_defs[1])[0]
            if len(i_defs) >= 2 and _split_sentences(i_defs[1])
            else (sents[1] if len(sents) >= 2 and sents[1] not in defn else f"Formalized in IETF RFC 4949 for Internet protocol security, cryptographic assurance, and network trust architectures.")
        )
        apps = f"Referenced in IETF RFC specifications, TLS/IPsec/PKI protocol design, and network security audits involving {raw_term}."
        trd = f"Real-world deployments of {raw_term} must balance cryptographic rigor and protocol compliance against latency, key lifecycle, and interoperability constraints."

        rec = _score_and_validate_record(
            concept_id=raw_term.casefold(),
            title=raw_term,
            category="cybersecurity_networking",
            aliases=[],
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"ietf:rfc4949:{_slugify(raw_term)}",
            reject_counter=reject_counter,
        )
        if rec is not None:
            records.append(rec)
            if len(records) >= max_records:
                break

    return records, raw_bytes, f"ok ({len(records)} entries)"


# ---------------------------------------------------------------------------
# Worker 9: PostgreSQL Official Database Glossary (`github.com/postgres/postgres`)
# ---------------------------------------------------------------------------

def harvest_postgresql_glossary(reject_counter: Counter[str]) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest PostgreSQL official relational database & storage engine glossary (`glossary.sgml`)."""
    url = "https://raw.githubusercontent.com/postgres/postgres/master/doc/src/sgml/glossary.sgml"
    try:
        sgml, raw_bytes = _http_get_text(url, timeout=20.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    records: list[RepoConceptRecord] = []
    for entry_m in re.finditer(r"<glossentry\b([^>]*)>([\s\S]*?)</glossentry>", sgml, flags=re.IGNORECASE):
        attrs = entry_m.group(1)
        inner = entry_m.group(2)
        id_m = re.search(r'id="([^"]+)"', attrs)
        slug = (id_m.group(1) if id_m else "entry").replace("glossary-", "")

        term_m = re.search(r"<glossterm\b[^>]*>([\s\S]*?)</glossterm>", inner, flags=re.IGNORECASE)
        def_m = re.search(r"<glossdef\b[^>]*>([\s\S]*?)</glossdef>", inner, flags=re.IGNORECASE)
        if not term_m or not def_m:
            continue

        raw_term = _clean_sgml_text(term_m.group(1))
        # Remove trailing parenthetical "(routine)" etc. for cleaner canonical title
        clean_term = re.sub(r"\s*\([^)]{1,30}\)\s*$", "", raw_term).strip()
        if not clean_term or len(clean_term) < 3:
            continue

        paras = [
            _clean_sgml_text(pm.group(1))
            for pm in re.finditer(r"<para\b[^>]*>([\s\S]*?)</para>", def_m.group(1), flags=re.IGNORECASE)
        ]
        paras = [p for p in paras if len(p) >= 25 and not p.casefold().startswith("for more information")]
        sents = _split_sentences(" ".join(paras[:3]))
        if not sents:
            continue

        t_low = clean_term.casefold()
        if len(clean_term.split()) == 1 and t_low in _GENERIC_PG_SINGLETONS:
            title = f"PostgreSQL {clean_term}"
            aliases = [f"postgres {t_low}"]
        else:
            title = clean_term
            aliases = [f"postgresql {t_low}"] if "postgres" not in t_low else []

        defn = sents[0]
        if len(defn) < 55 and len(sents) >= 2:
            defn = f"{sents[0]} {sents[1]}"

        mech = sents[1] if len(sents) >= 2 and sents[1] not in defn else (
            f"Implemented inside PostgreSQL's transactional MVCC storage, query planner, and write-ahead logging (WAL) architecture."
        )
        apps = sents[2] if len(sents) >= 3 and sents[2] not in (defn, mech) else (
            f"Applied in relational database schema design, SQL query optimization, replication, and crash-safe data management."
        )
        trd = f"Tuning {title} in relational databases involves balancing transactional consistency, disk I/O amplification, lock contention, and memory overhead."

        rec = _score_and_validate_record(
            concept_id=title.casefold(),
            title=title,
            category="software_engineering",
            aliases=aliases,
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"docs:postgresql_glossary:{_slugify(slug)}",
            reject_counter=reject_counter,
        )
        if rec is not None:
            records.append(rec)

    return records, raw_bytes, f"ok ({len(records)} entries)"


# ---------------------------------------------------------------------------
# Worker 10: Python Official Documentation Glossary (`github.com/python/cpython`)
# ---------------------------------------------------------------------------

def harvest_python_glossary(reject_counter: Counter[str]) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest Python Official Documentation Glossary (`Doc/glossary.rst`)."""
    url = "https://raw.githubusercontent.com/python/cpython/main/Doc/glossary.rst"
    try:
        rst, raw_bytes = _http_get_text(url, timeout=20.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    # Locate `.. glossary::` block
    m = re.search(r"\.\.\s+glossary::\s*\n", rst)
    if not m:
        return [], raw_bytes, "fallback_skipped (glossary directive not found)"
    body = rst[m.end():]

    entries: list[tuple[str, str]] = []
    cur_term: str | None = None
    cur_lines: list[str] = []

    for line in body.splitlines():
        # Term lines are indented by exactly 3 spaces (not 6) and non-empty
        if re.match(r"^   [^\s]", line) and not line.lstrip().startswith(".."):
            if cur_term and cur_lines:
                entries.append((cur_term, "\n".join(cur_lines)))
            cur_term = line.strip()
            cur_lines = []
        elif cur_term is not None:
            if line.startswith("      ") or not line.strip():
                cur_lines.append(line[6:] if line.startswith("      ") else "")
    if cur_term and cur_lines:
        entries.append((cur_term, "\n".join(cur_lines)))

    records: list[RepoConceptRecord] = []
    for raw_term, raw_def in entries:
        clean_term = _clean_rst_text(raw_term).strip(" `\"'")
        if not clean_term or len(clean_term) < 3 or not re.search(r"[A-Za-z]{3,}", clean_term):
            reject_counter["python_symbol_term"] += 1
            continue
        # Remove indented code blocks (`::\n\n   code`) from reST definition
        def_no_code = re.split(r"::\s*\n\s*\n", raw_def, maxsplit=1)[0]
        cleaned = _clean_rst_text(def_no_code)
        sents = _split_sentences(cleaned)
        if not sents:
            continue

        t_low = clean_term.casefold()
        if len(clean_term.split()) == 1 and t_low in _GENERIC_PY_SINGLETONS:
            title = f"Python {_smart_title(clean_term)}"
            aliases = []
        else:
            title = _smart_title(clean_term)
            aliases = [f"python {t_low}"] if "python" not in t_low else []

        defn = sents[0]
        if len(defn) < 55 and len(sents) >= 2:
            defn = f"{sents[0]} {sents[1]}"

        mech = sents[1] if len(sents) >= 2 and sents[1] not in defn else (
            f"Defined by the CPython language specification and Python data model semantics for {title}."
        )
        apps = sents[2] if len(sents) >= 3 and sents[2] not in (defn, mech) else (
            f"Used across Python software engineering, standard-library APIs, type checking, and runtime performance tuning."
        )
        trd = f"Using {title} effectively in Python requires balancing readability and dynamic flexibility against runtime overhead and static type guarantees."

        cat = _infer_tech_domain(title, f"{defn} {mech}", default="software_engineering")
        rec = _score_and_validate_record(
            concept_id=title.casefold(),
            title=title,
            category=cat,
            aliases=aliases,
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"docs:python_glossary:{_slugify(clean_term)}",
            reject_counter=reject_counter,
        )
        if rec is not None:
            records.append(rec)

    return records, raw_bytes, f"ok ({len(records)} entries)"


# ---------------------------------------------------------------------------
# Worker 11: Rust Language Reference Glossary (`github.com/rust-lang/reference`)
# ---------------------------------------------------------------------------

def harvest_rust_glossary(reject_counter: Counter[str]) -> tuple[list[RepoConceptRecord], int, str]:
    """Harvest official Rust Language Reference Glossary (`src/glossary.md`)."""
    url = "https://raw.githubusercontent.com/rust-lang/reference/master/src/glossary.md"
    try:
        md, raw_bytes = _http_get_text(url, timeout=20.0)
    except Exception as exc:
        return [], 0, f"fallback_skipped ({exc})"

    # Split by `### Heading`
    sections = re.split(r"(?m)^###\s+", md)
    records: list[RepoConceptRecord] = []

    for sec in sections[1:]:
        lines = sec.splitlines()
        if not lines:
            continue
        raw_title = _clean_inline_markdown(lines[0].strip())
        if not raw_title or len(raw_title) < 3:
            continue
        body = "\n".join(lines[1:])
        paras = _clean_markdown_to_paragraphs(body)
        sents = _split_sentences(" ".join(paras[:3]))
        if not sents:
            continue

        t_low = raw_title.casefold()
        if len(raw_title.split()) == 1 and t_low in _GENERIC_RUST_SINGLETONS:
            title = f"Rust {_smart_title(raw_title)}"
            aliases = []
        else:
            title = _smart_title(raw_title)
            aliases = [f"rust {t_low}"] if "rust" not in t_low else []

        defn = sents[0]
        if len(defn) < 55 and len(sents) >= 2:
            defn = f"{sents[0]} {sents[1]}"

        mech = sents[1] if len(sents) >= 2 and sents[1] not in defn else (
            f"Enforced statically by the Rust compiler's type system, borrow checker, and LLVM code generation pipeline."
        )
        apps = f"Applied in systems programming, memory-safe concurrent services, WebAssembly modules, and zero-cost abstractions in Rust."
        trd = f"Strict compile-time rules around {title} eliminate classes of memory-safety bugs at the cost of steeper initial design and lifetime annotations."

        cat = _infer_tech_domain(title, f"{defn} {mech}", default="computer_science")
        rec = _score_and_validate_record(
            concept_id=title.casefold(),
            title=title,
            category=cat,
            aliases=aliases,
            definition=defn,
            mechanism=mech,
            applications=apps,
            tradeoffs=trd,
            provenance=f"docs:rust_glossary:{_slugify(raw_title)}",
            reject_counter=reject_counter,
        )
        if rec is not None:
            records.append(rec)

    return records, raw_bytes, f"ok ({len(records)} entries)"


# ---------------------------------------------------------------------------
# Read-Only Primary Vault & Optional Pre-Staged JSONL Skip-Set Loader
# ---------------------------------------------------------------------------

def read_primary_vault_dedup_sets(
    vault_path: str | Path | None,
    *,
    extra_jsonl_paths: tuple[str | Path, ...] = (),
) -> tuple[set[str], set[str], set[str]]:
    """Open `artifacts/mirror_uol_skills.dcrystal` strictly READ-ONLY (`?mode=ro`) to load existing dedup keys.

    Returns `(existing_cids, existing_canonical_and_alias_keys, existing_content_fps)`.
    """
    existing_cids: set[str] = set()
    existing_keys: set[str] = set()
    existing_fps: set[str] = set()

    if vault_path is not None:
        vpath = Path(vault_path)
        if vpath.exists():
            uri = f"file:{vpath.resolve().as_posix()}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=10.0)
            try:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                if "mirror_concept_lattice" in tables:
                    cols = {r[1] for r in con.execute("PRAGMA table_info(mirror_concept_lattice)").fetchall()}
                    if {"canonical_key", "content_fp"}.issubset(cols):
                        rows = con.execute(
                            "SELECT concept_id, title, aliases_csv, canonical_key, content_fp FROM mirror_concept_lattice"
                        ).fetchall()
                        for cid, title, aliases_csv, ckey, cfp in rows:
                            cid_s = str(cid or "").strip().casefold()
                            if cid_s:
                                existing_cids.add(cid_s)
                            for raw in (cid_s, str(title or ""), str(ckey or "")):
                                ck = canonical_concept_key(raw)
                                if ck:
                                    existing_keys.add(ck)
                            for al in str(aliases_csv or "").split("|"):
                                ak = canonical_concept_key(al)
                                if ak:
                                    existing_keys.add(ak)
                            if cfp:
                                existing_fps.add(str(cfp))
                    else:
                        rows = con.execute(
                            "SELECT concept_id, title, aliases_csv, definition FROM mirror_concept_lattice"
                        ).fetchall()
                        for cid, title, aliases_csv, defn in rows:
                            cid_s = str(cid or "").strip().casefold()
                            if cid_s:
                                existing_cids.add(cid_s)
                            for raw in (cid_s, str(title or "")):
                                ck = canonical_concept_key(raw)
                                if ck:
                                    existing_keys.add(ck)
                            for al in str(aliases_csv or "").split("|"):
                                ak = canonical_concept_key(al)
                                if ak:
                                    existing_keys.add(ak)
                            if defn:
                                existing_fps.add(content_fingerprint(str(defn)))
                if "mirror_concept_aliases" in tables:
                    for (ak,) in con.execute("SELECT alias_ck FROM mirror_concept_aliases").fetchall():
                        if ak:
                            existing_keys.add(str(ak))
            finally:
                con.close()

    for jpath in extra_jsonl_paths:
        p = Path(jpath)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cid = str(rec.get("concept_id", "")).strip().casefold()
                title = str(rec.get("title", "")).strip()
                if cid:
                    existing_cids.add(cid)
                for raw in (cid, title):
                    ck = canonical_concept_key(raw)
                    if ck:
                        existing_keys.add(ck)
                for al in rec.get("aliases") or []:
                    ak = canonical_concept_key(str(al))
                    if ak:
                        existing_keys.add(ak)
                defn = str(rec.get("definition", ""))
                if defn:
                    existing_fps.add(content_fingerprint(defn))

    return existing_cids, existing_keys, existing_fps


# ---------------------------------------------------------------------------
# Unified Orchestrator & 4-Layer Canonical Deduplication
# ---------------------------------------------------------------------------

def deduplicate_repo_records(
    ordered_source_batches: list[tuple[str, list[RepoConceptRecord]]],
    *,
    primary_cids: set[str],
    primary_keys: set[str],
    primary_fps: set[str],
    reject_counter: Counter[str],
    target_max: int = 8000,
) -> tuple[list[RepoConceptRecord], dict[str, int]]:
    """Enforce global 4-layer deduplication against primary vault + internal batch (`0` duplicates)."""
    seen_cids: set[str] = set()
    seen_keys: set[str] = set()
    seen_fps: set[str] = set()

    final_records: list[RepoConceptRecord] = []
    source_counts: Counter[str] = Counter()

    for src_name, batch in ordered_source_batches:
        for rec in batch:
            if len(final_records) >= target_max:
                break
            cid = rec.concept_id.casefold()
            ck_id = canonical_concept_key(cid)
            ck_title = canonical_concept_key(rec.title)
            fp = content_fingerprint(rec.definition)

            if not ck_id:
                reject_counter["bad_title"] += 1
                continue

            # Layer 1: Primary vault collision check (read-only peek)
            if (
                cid in primary_cids
                or ck_id in primary_keys
                or (ck_title and ck_title in primary_keys)
                or fp in primary_fps
            ):
                reject_counter["duplicate_existing_primary"] += 1
                continue

            # Layer 2: Internal batch deduplication check
            if (
                cid in seen_cids
                or ck_id in seen_keys
                or (ck_title and ck_title in seen_keys)
                or fp in seen_fps
            ):
                reject_counter["duplicate_internal"] += 1
                continue

            # Filter secondary aliases so they NEVER collide with primary or already-claimed keys
            clean_aliases: list[str] = [cid]
            claimed_new_keys: set[str] = {ck_id}
            if ck_title:
                claimed_new_keys.add(ck_title)
            for al in rec.aliases:
                if al == cid:
                    continue
                ak = canonical_concept_key(al)
                if (
                    ak
                    and ak not in _FORBIDDEN_SECONDARY_ALIASES
                    and ak not in primary_keys
                    and ak not in seen_keys
                    and ak not in claimed_new_keys
                ):
                    clean_aliases.append(al)
                    claimed_new_keys.add(ak)

            final_rec = RepoConceptRecord(
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
            seen_keys.update(claimed_new_keys)
            seen_fps.add(fp)
            final_records.append(final_rec)
            source_counts[src_name] += 1

    return final_records, dict(source_counts)


def harvest_all_github_repo_concepts(
    output_path: str | Path,
    *,
    vault_path: str | Path | None = None,
    extra_skip_jsonl: tuple[str | Path, ...] = (),
    target_max: int = 8000,
    nist_max: int = 2000,
    rfc4949_max: int = 550,
    seed: int = 42,
) -> RepoHarvestReport:
    """Harvest all public GitHub repos & standards in parallel, strictly quality-filter, dedup, and write JSONL."""
    t0 = time.perf_counter()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    primary_cids, primary_keys, primary_fps = read_primary_vault_dedup_sets(
        vault_path,
        extra_jsonl_paths=extra_skip_jsonl,
    )

    per_worker_rejects: dict[str, Counter[str]] = defaultdict(Counter)
    worker_results: dict[str, tuple[list[RepoConceptRecord], int, str]] = {}

    tasks = {
        "mitre_attack": lambda rc: harvest_mitre_attack(rc),
        "mitre_capec": lambda rc: harvest_mitre_capec(rc),
        "cncf_glossary": lambda rc: harvest_cncf_glossary(rc, max_workers=8),
        "k8s_glossary": lambda rc: harvest_k8s_glossary(rc, max_workers=8),
        "owasp_cheatsheets": lambda rc: harvest_owasp_cheatsheets(rc, max_workers=8),
        "mdn_glossary": lambda rc: harvest_mdn_glossary(rc, max_workers=10),
        "nist_csrc_glossary": lambda rc: harvest_nist_csrc_glossary(rc, max_records=nist_max),
        "ietf_rfc4949": lambda rc: harvest_ietf_rfc4949(rc, max_records=rfc4949_max),
        "postgresql_glossary": lambda rc: harvest_postgresql_glossary(rc),
        "python_glossary": lambda rc: harvest_python_glossary(rc),
        "rust_glossary": lambda rc: harvest_rust_glossary(rc),
    }

    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {
            pool.submit(fn, per_worker_rejects[name]): name
            for name, fn in tasks.items()
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                res = fut.result()
                worker_results[name] = res
                print(
                    f"[repo-harvester] {name}: {res[2]} ({res[1] / 1_000_000:.2f} MB, {time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                worker_results[name] = ([], 0, f"error ({exc})")
                print(f"[repo-harvester] {name}: ERROR {exc}", flush=True)

    total_rejects: Counter[str] = Counter()
    for rc in per_worker_rejects.values():
        total_rejects.update(rc)

    # Strict priority order: official security/cloud/web standards first, then language/DB glossaries
    priority_order = (
        "mitre_attack",
        "mitre_capec",
        "cncf_glossary",
        "k8s_glossary",
        "owasp_cheatsheets",
        "mdn_glossary",
        "nist_csrc_glossary",
        "ietf_rfc4949",
        "postgresql_glossary",
        "python_glossary",
        "rust_glossary",
    )

    ordered_batches: list[tuple[str, list[RepoConceptRecord]]] = []
    source_status: dict[str, str] = {}
    raw_bytes_total = 0
    raw_candidates_total = 0

    for src_name in priority_order:
        recs, nb, st = worker_results.get(src_name, ([], 0, "skipped"))
        ordered_batches.append((src_name, recs))
        source_status[src_name] = st
        raw_bytes_total += nb
        raw_candidates_total += len(recs)

    final_records, source_counts = deduplicate_repo_records(
        ordered_batches,
        primary_cids=primary_cids,
        primary_keys=primary_keys,
        primary_fps=primary_fps,
        reject_counter=total_rejects,
        target_max=target_max,
    )

    # Atomic JSONL write
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
    sample_records = [
        r.to_json_dict()
        for r in rng.sample(final_records, min(8, len(final_records)))
    ]

    report = RepoHarvestReport(
        total_emitted=len(final_records),
        raw_candidates_collected=raw_candidates_total,
        raw_bytes_fetched=raw_bytes_total,
        rejected_breakdown=dict(total_rejects.most_common()),
        source_counts=source_counts,
        source_status=source_status,
        domain_counts=domain_counts,
        avg_quality_score=avg_q,
        min_quality_score=min_q,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
        output_path=str(out_file),
        sample_records=sample_records,
    )

    # Write sidecar meta JSON for Phase-B raw-byte accounting
    meta_path = out_file.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror 7 Phase-3 Public GitHub Repositories & Open Standards Harvester"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for deduplicated, quality-gated repository concepts.",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "mirror_uol_skills.dcrystal",
        help="Read-only path to primary .dcrystal vault for zero-duplication pre-filtering.",
    )
    parser.add_argument(
        "--extra-skip-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Optional higher-tier JSONL staging file(s) (e.g. browser_phase3.jsonl) to avoid duplicating.",
    )
    parser.add_argument(
        "--target-max",
        type=int,
        default=8000,
        help="Maximum unique verified ConceptNodes to emit (default: 8000).",
    )
    parser.add_argument(
        "--nist-max",
        type=int,
        default=2000,
        help="Maximum NIST CSRC glossary concepts to retain (default: 2000).",
    )
    parser.add_argument(
        "--rfc4949-max",
        type=int,
        default=550,
        help="Maximum IETF RFC 4949 concepts to retain (default: 550).",
    )
    args = parser.parse_args()

    report = harvest_all_github_repo_concepts(
        output_path=args.output,
        vault_path=args.vault,
        extra_skip_jsonl=tuple(args.extra_skip_jsonl),
        target_max=args.target_max,
        nist_max=args.nist_max,
        rfc4949_max=args.rfc4949_max,
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
