"""
Mirror 7 — Pure-Python Permissive Internet Dataset Curator & Six-Gate Hygiene Pipeline.
100% Python 3.11+ standard library (0 PyTorch, 0 GPU, 0 external packages).

Extracts strictly permissive (MIT / Apache-2.0) `train`-split datasets from verified
open sources via direct raw CDN Range streaming, GitHub archives, and polite API pulls,
cleans formatting, scrubs third-party chatbot persona boilerplate, executes doctest
assertions inside ZeroTrustSandboxJudge for Python algorithms, enforces all six hygiene
gates (SPDX+PII, MinHash+LSH dedup, Shannon entropy, pedagogical density, AST compiler,
benchmark decontamination), and exports crystal-ready JSONL (`split="train"`).
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import doctest
import io
import json
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile

from .grammar_and_sandbox import ZeroTrustSandboxJudge
from .phase7_qwen_to_opus import SixGateDataHygienePipeline

USER_AGENT = "Mirror7-DatasetCurator/1.1 (+pure-python-independent)"
FORBIDDEN_SPLITS = frozenset({"test", "eval", "evaluation", "eval_holdout", "dev_test"})

EXTRA_DECONTAM_SIGNATURES: tuple[str, ...] = (
    "def has_close_elements",
    "def separate_paren_groups",
    "def truncate_number",
    "def below_zero",
    "def mean_absolute_deviation",
    "def intersperse",
    "def parse_nested_parens",
    "def filter_by_substring",
    "def sum_product",
    "def rolling_max",
    "def make_palindrome",
    "natalia sold clips to 48 of her friends",  # guard only exact holdout test phrasing if any
    "janet’s ducks lay 16 eggs",
    "janet's ducks lay 16 eggs",
)

# Reject third-party chatbot identity boilerplate & canned disclaimers so Mirror 7 keeps an honest independent identity
PERSONA_LEAK_SIGNATURES: tuple[str, ...] = (
    "i am open assistant",
    "i'm open assistant",
    "i am open-assistant",
    "as open assistant",
    "as openassistant",
    "openassistant model",
    "open assistant works as",
    "open assistant is an open",
    "open assistant, a language model",
    "trained by laion",
    "i am chatgpt",
    "i'm chatgpt",
    "as chatgpt",
    "developed by openai",
    "created by openai",
    "trained by openai",
    "i am gpt-",
    "i am claude",
    "created by anthropic",
    "developed by anthropic",
    "as an ai language model",
    "as a large language model",
    "as an artificial intelligence language model",
    "i am an ai language model",
    "i am a large language model",
    "i'm an ai language model",
    "i'm a large language model",
    "i am an artificial intelligence",
    "as an ai developed by",
    "as an ai assistant created by",
    "my knowledge cutoff",
)

# Strictly permissive sources vetted for commercial & open weights independence
CURATED_SOURCE_CATALOG: tuple[dict[str, str], ...] = (
    {
        "name": "thealgorithms_python",
        "url": "https://github.com/TheAlgorithms/Python/archive/refs/heads/master.zip",
        "license": "MIT",
        "split": "train",
        "domain": "coding",
    },
    {
        "name": "qiskit_textbook",
        "repo": "Qiskit/textbook",
        "license": "Apache-2.0",
        "split": "train",
        "domain": "quantum_computing",
    },
    {
        "name": "qiskit_documentation",
        "repo": "Qiskit/documentation",
        "license": "Apache-2.0",
        "split": "train",
        "domain": "quantum_computing",
    },
    {
        "name": "pennylane_qml",
        "repo": "PennyLaneAI/qml",
        "license": "Apache-2.0",
        "split": "train",
        "domain": "quantum_computing",
    },
    {
        "name": "codefeedback",
        "dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
        "config": "default",
        "license": "Apache-2.0",
        "split": "train",
        "domain": "coding",
    },
    {
        "name": "magicoder_python",
        "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "config": "default",
        "license": "MIT",
        "split": "train",
        "domain": "coding",
    },
    {
        "name": "gsm8k",
        "dataset": "openai/gsm8k",
        "config": "main",
        "license": "MIT",
        "split": "train",
        "domain": "math",
    },
    {
        "name": "mathinstruct",
        "dataset": "TIGER-Lab/MathInstruct",
        "config": "default",
        "license": "MIT",
        "split": "train",
        "domain": "math",
    },
    {
        "name": "slimorca",
        "dataset": "Open-Orca/SlimOrca",
        "config": "default",
        "license": "MIT",
        "split": "train",
        "domain": "general_chat",
    },
    {
        "name": "oasst2",
        "dataset": "OpenAssistant/oasst2",
        "config": "default",
        "license": "Apache-2.0",
        "split": "train",
        "domain": "general_chat",
    },
    {
        "name": "cosmopedia",
        "dataset": "HuggingFaceTB/cosmopedia",
        "config": "openstax",
        "license": "Apache-2.0",
        "split": "train",
        "domain": "science_reasoning",
    },
    {
        "name": "commonsense_qa",
        "dataset": "tau/commonsense_qa",
        "config": "default",
        "license": "MIT",
        "split": "train",
        "domain": "general_chat",
    },
)


@dataclass
class CurationConfig:
    output_jsonl_path: Path = Path("training/data/mirror_internet_curated_v1.jsonl")
    report_json_path: Path = Path("artifacts/mirror_internet_curation_report.json")
    thealgorithms_max_rules: int = 1200
    qiskit_textbook_max_files: int = 45
    qiskit_docs_max_files: int = 55
    qiskit_max_rules: int = 600
    pennylane_max_files: int = 40
    pennylane_max_rules: int = 350
    codefeedback_target: int = 1000
    magicoder_target: int = 800
    gsm8k_target: int = 1200
    mathinstruct_target: int = 800
    slimorca_target: int = 800
    oasst2_target: int = 400
    cosmopedia_target: int = 250
    commonsense_target: int = 250
    request_timeout_sec: int = 35
    verbose: bool = False


@dataclass
class SourceAuditStats:
    source: str
    spdx_license: str
    split: str
    domain: str
    candidates_seen: int = 0
    prefilter_rejected: int = 0
    sandbox_verified: int = 0
    sandbox_rejected: int = 0
    gate_passed: int = 0
    gate_rejections: dict[str, int] = field(default_factory=dict)
    fetch_errors: list[str] = field(default_factory=list)

    def record_gate_reject(self, gate_name: str | None) -> None:
        key = gate_name or "Unknown"
        self.gate_rejections[key] = self.gate_rejections.get(key, 0) + 1


def _assert_train_split(split: str) -> None:
    if split.strip().lower() in FORBIDDEN_SPLITS or split.strip().lower() != "train":
        raise ValueError(f"Split '{split}' is forbidden; only 'train' split extraction is permitted.")


def _parse_ast_quiet(code: str, mode: str = "exec") -> ast.AST:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(code, mode=mode)


def _http_get_bytes(
    url: str,
    timeout: int = 35,
    retries: int = 4,
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (400, 404, 416):
                raise
            wait_s = min(8.0, 1.0 * (2**attempt))
            if exc.code == 429:
                ra = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    wait_s = max(wait_s, min(15.0, float(ra))) if ra else min(10.0, 2.5 * (attempt + 1))
                except ValueError:
                    wait_s = min(10.0, 2.5 * (attempt + 1))
            time.sleep(wait_s)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            time.sleep(min(6.0, 0.8 * (2**attempt)))
    raise RuntimeError(f"HTTP fetch failed after {retries} attempts for {url}: {last_exc}")


def _http_get_json(url: str, timeout: int = 35, retries: int = 4) -> Any:
    raw = _http_get_bytes(url, timeout=timeout, retries=retries)
    return json.loads(raw.decode("utf-8", errors="replace"))


def _stream_hf_jsonl_ranges(
    st_obj: SourceAuditStats,
    url: str,
    offsets: tuple[int, ...],
    chunk_bytes: int = 2_400_000,
    timeout: int = 35,
) -> list[dict[str, Any]]:
    """Stream stratified windows from a HuggingFace CDN `.jsonl` file via HTTP Range requests."""
    rows: list[dict[str, Any]] = []
    for off in offsets:
        try:
            raw = _http_get_bytes(
                url,
                timeout=timeout,
                extra_headers={"Range": f"bytes={off}-{off + chunk_bytes - 1}"},
            )
        except Exception as exc:
            st_obj.fetch_errors.append(f"range={off}: {exc}")
            continue
        txt = raw.decode("utf-8", errors="replace")
        lines = txt.splitlines()
        if not lines:
            continue
        start_idx = 0 if off == 0 else 1
        end_idx = len(lines) - 1 if len(raw) >= chunk_bytes - 64 else len(lines)
        for line in lines[start_idx:end_idx]:
            s = line.strip()
            if not s or not s.startswith("{"):
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows


def _stream_hf_json_array_ranges(
    st_obj: SourceAuditStats,
    url: str,
    offsets: tuple[int, ...],
    chunk_bytes: int = 3_200_000,
    timeout: int = 35,
) -> list[dict[str, Any]]:
    """Incrementally parse JSON objects from Range slices of a formatted JSON array (e.g. MathInstruct.json)."""
    decoder = json.JSONDecoder()
    items: list[dict[str, Any]] = []
    for off in offsets:
        try:
            raw = _http_get_bytes(
                url,
                timeout=timeout,
                extra_headers={"Range": f"bytes={off}-{off + chunk_bytes - 1}"},
            )
        except Exception as exc:
            st_obj.fetch_errors.append(f"range={off}: {exc}")
            continue
        txt = raw.decode("utf-8", errors="replace")
        pos = 0
        if off == 0:
            brk = txt.find("[")
            pos = brk + 1 if brk >= 0 else 0
        else:
            m_obj = re.search(r'(?:^|\n)\s*\{\s*\n\s*"source"\s*:', txt)
            if not m_obj:
                m_obj = re.search(r'(?:^|\n)\s*\{', txt)
            if not m_obj:
                continue
            pos = txt.find("{", m_obj.start())
        n = len(txt)
        while pos < n:
            while pos < n and txt[pos] in " \t\r\n,":
                pos += 1
            if pos >= n or txt[pos] == "]":
                break
            if txt[pos] != "{":
                nxt = txt.find("{", pos)
                if nxt < 0:
                    break
                pos = nxt
            try:
                obj, next_pos = decoder.raw_decode(txt, pos)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                items.append(obj)
            pos = next_pos
    return items


def _fetch_hf_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int = 100,
    timeout: int = 35,
) -> tuple[list[dict[str, Any]], int]:
    _assert_train_split(split)
    qs = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": max(0, offset), "length": min(100, max(1, length))}
    )
    data = _http_get_json(f"https://datasets-server.huggingface.co/rows?{qs}", timeout=timeout)
    total = int(data.get("num_rows_total") or 0)
    rows = [item.get("row", {}) for item in data.get("rows", []) if isinstance(item, dict)]
    return rows, total


def _has_noise_or_mojibake(text: str) -> bool:
    if not text or "\ufffd" in text or "\x00" in text:
        return True
    if "data:image/" in text.lower() or "base64," in text.lower():
        return True
    return False


def _has_persona_leakage(user_text: str, target_text: str) -> bool:
    """Reject third-party chatbot identity boilerplate (e.g., Open Assistant / OpenAI / ChatGPT)."""
    target_low = target_text.lower()
    user_low = user_text.lower()
    if any(sig in target_low for sig in PERSONA_LEAK_SIGNATURES):
        return True
    if re.search(r"\b(?:open[- ]?assistant|chatgpt|openai)\b", user_low) and re.search(
        r"\b(?:i am|i'm|as an ai|language model|assistant)\b", target_low
    ):
        return True
    return False


def _passes_extra_decontam(user_text: str, target_text: str) -> bool:
    combined = f"{user_text}\n{target_text}".lower()
    return not any(sig in combined for sig in EXTRA_DECONTAM_SIGNATURES)


def _strip_think_tags(text: str) -> str:
    return re.sub(r"</?think>", "", text).strip()


def _sanitize_markdown_body(raw_md: str) -> str:
    txt = raw_md.replace("\r\n", "\n").replace("&nbsp;", " ")
    # Remove YAML frontmatter
    txt = re.sub(r"^\s*---\n.*?\n---\s*\n?", "", txt, flags=re.DOTALL)
    # Remove HTML anchor tags & MDX imports/exports/components
    txt = re.sub(r"<a\s+id=[^>]*>\s*</a>", "", txt, flags=re.I)
    txt = re.sub(r"^\s*(?:import|export)\s+.*$", "", txt, flags=re.M)
    txt = re.sub(r"</?(?:Admonition|Tabs|TabItem|details|summary|span|div|figure|figcaption)[^>]*>", "", txt, flags=re.I)
    # Remove image markdown tags
    txt = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", txt)
    # Remove Qiskit video promo lines
    txt = re.sub(r"In the following video,[^\n]*\n?", "", txt, flags=re.I)
    # Collapse excess newlines
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt


def _clean_heading_title(raw_title: str) -> str:
    t = re.sub(r"<[^>]+>", "", raw_title).strip()
    t = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", t).strip()
    return re.sub(r"\s+", " ", t)


def clean_gsm8k_answer(raw_answer: str) -> tuple[str, str]:
    """Strip GSM8K `<<calc=val>>` calculator markup and split reasoning from `#### final`."""
    cleaned = re.sub(r"<<[^<>]*>>", "", raw_answer).strip()
    final_ans = ""
    if "####" in cleaned:
        body, tail = cleaned.rsplit("####", 1)
        cleaned = body.strip()
        final_ans = tail.strip()
    else:
        body = cleaned
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    step_lines = [f"Step {idx}: {ln}" if not ln.lower().startswith("step ") else ln for idx, ln in enumerate(lines, 1)]
    if final_ans:
        formatted = "Step-by-Step Derivation:\n" + "\n".join(step_lines) + f"\n\nFinal Answer: {final_ans}"
    else:
        formatted = "Step-by-Step Derivation:\n" + "\n".join(step_lines)
    return formatted, final_ans


def _extract_node_names_used(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


def _build_thealgorithms_unit_tests(fn_name: str, docstring: str, max_asserts: int = 8) -> str:
    if not docstring.strip():
        return ""
    parser = doctest.DocTestParser()
    try:
        examples = parser.get_examples(docstring)
    except Exception:
        return ""
    asserts: list[str] = []
    for ex in examples:
        src_ex = ex.source.strip()
        want_ex = ex.want.strip()
        if not src_ex or not want_ex:
            continue
        if "\n" in src_ex or "\n" in want_ex or "..." in want_ex or "Traceback" in want_ex:
            continue
        if fn_name not in src_ex:
            continue
        try:
            _parse_ast_quiet(src_ex, mode="eval")
            ast.literal_eval(want_ex)
        except Exception:
            continue
        asserts.append(f"assert ({src_ex}) == ({want_ex})")
        if len(asserts) >= max_asserts:
            break
    return "\n".join(asserts)


def _strip_function_docstring(fn_node: ast.FunctionDef) -> str:
    """Return clean Python source for a FunctionDef with optional doctest docstring preserved or cleaned."""
    return ast.unparse(fn_node)


def extract_thealgorithms_from_zip_bytes(
    zip_bytes: bytes,
    pipeline: SixGateDataHygienePipeline,
    stats: SourceAuditStats,
    max_rules: int = 1200,
) -> list[dict[str, Any]]:
    """Extract and sandbox-verify MIT-licensed Python algorithms from a GitHub zip archive."""
    records: list[dict[str, Any]] = []
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    py_paths = sorted(
        p
        for p in zf.namelist()
        if p.endswith(".py")
        and "/scripts/" not in p
        and "/tests/" not in p
        and not p.endswith("/__init__.py")
    )

    for rel_zip_path in py_paths:
        if len(records) >= max_rules:
            break
        parts = rel_zip_path.split("/")
        if len(parts) < 2:
            continue
        topic_rel = "/".join(parts[1:])
        topic_area = parts[1].replace("_", " ")

        try:
            src = zf.read(rel_zip_path).decode("utf-8", errors="replace")
            tree = _parse_ast_quiet(src)
        except Exception:
            continue

        # Collect safe module imports and top-level helper functions
        safe_imports: list[str] = []
        top_funcs: dict[str, ast.FunctionDef] = {}
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imp_src = ast.unparse(node)
                ok_imp, _ = ZeroTrustSandboxJudge.is_ast_safe(imp_src)
                if ok_imp:
                    safe_imports.append(imp_src)
            elif isinstance(node, ast.FunctionDef):
                top_funcs[node.name] = node

        for node in tree.body:
            if len(records) >= max_rules:
                break
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            stats.candidates_seen += 1

            doc = ast.get_docstring(node) or ""
            test_code = _build_thealgorithms_unit_tests(node.name, doc)
            if not test_code:
                stats.prefilter_rejected += 1
                continue

            # Pull any transitive top-level sibling helper functions called by node
            used_names = _extract_node_names_used(node)
            helper_defs: list[str] = []
            seen_helpers: set[str] = {node.name}
            frontier = [n for n in used_names if n in top_funcs and n != node.name]
            while frontier:
                h_name = frontier.pop()
                if h_name in seen_helpers:
                    continue
                seen_helpers.add(h_name)
                h_node = top_funcs[h_name]
                helper_defs.append(ast.unparse(h_node))
                for nxt in _extract_node_names_used(h_node):
                    if nxt in top_funcs and nxt not in seen_helpers:
                        frontier.append(nxt)

            code_parts: list[str] = []
            if safe_imports:
                all_used = used_names.copy()
                for h_src_item in helper_defs:
                    try:
                        all_used |= _extract_node_names_used(_parse_ast_quiet(h_src_item))
                    except Exception:
                        pass
                for imp_s in safe_imports:
                    imp_toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", imp_s))
                    if imp_toks & all_used:
                        code_parts.append(imp_s)
            code_parts.extend(reversed(helper_defs))
            code_parts.append(_strip_function_docstring(node))
            candidate_code = "\n\n".join(code_parts).strip()

            if len(candidate_code) > 2600 or len(candidate_code.splitlines()) > 95:
                stats.prefilter_rejected += 1
                continue

            safe_ok, _ = ZeroTrustSandboxJudge.is_ast_safe(candidate_code)
            if not safe_ok:
                stats.sandbox_rejected += 1
                continue

            response_md = f"Algorithm Topic: {topic_area} ({topic_rel})\nExample Implementation (`{node.name}`):\n```python\n{candidate_code}\n```"
            ok_sb, _ = ZeroTrustSandboxJudge.verify_code_and_test(response_md, test_code=test_code, max_loop_steps=50_000)
            if not ok_sb:
                stats.sandbox_rejected += 1
                continue
            stats.sandbox_verified += 1

            first_doc_line = ""
            for ln in doc.splitlines():
                s_ln = ln.strip()
                if s_ln and not s_ln.startswith(">>>") and not s_ln.startswith("..."):
                    first_doc_line = s_ln
                    break
            prompt = (
                f"Write a Python function `{node.name}` for {topic_area}: {first_doc_line}".strip()
                if first_doc_line
                else f"Write a verified Python function `{node.name}` in {topic_area} (`{topic_rel}`)."
            )

            combined_audit = f"Problem: {prompt}\n{response_md}"
            if _has_noise_or_mojibake(combined_audit) or not _passes_extra_decontam(prompt, response_md):
                stats.prefilter_rejected += 1
                continue

            gate_res = pipeline.audit_record(combined_audit, spdx_license="MIT")
            if not gate_res.passed:
                stats.record_gate_reject(gate_res.rejection_gate)
                continue

            domain = "quantum_computing" if "/quantum/" in rel_zip_path else "coding"
            stats.gate_passed += 1
            records.append(
                {
                    "id": f"thealg_{len(records):05d}_{node.name}",
                    "source": "TheAlgorithms/Python",
                    "spdx_license": "MIT",
                    "split": "train",
                    "domain": domain,
                    "trust_tier": 1,
                    "user_text": prompt,
                    "target_text": response_md,
                    "think": f"Verified {node.name} from {topic_rel} against doctest unit assertions in ZeroTrustSandboxJudge.",
                    "test_code": test_code,
                    "hygiene_metrics": gate_res.metrics,
                }
            )
    return records


def extract_qiskit_sections(raw_content: str, path: str, source_name: str) -> list[tuple[str, str]]:
    """Extract clean (prompt, structured_answer) pairs from an English Qiskit .ipynb/.md/.mdx file."""
    doc_title = _clean_heading_title(Path(path).stem.replace("-", " ").replace("_", " ").title())
    segments: list[tuple[str, list[str]]] = []

    if path.endswith(".ipynb"):
        try:
            nb = json.loads(raw_content)
        except json.JSONDecodeError:
            return []
        curr_heading = doc_title
        curr_chunks: list[str] = []
        for cell in nb.get("cells", []):
            ctype = cell.get("cell_type")
            src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
            if not src.strip():
                continue
            if ctype == "markdown":
                cleaned = _sanitize_markdown_body(src)
                if not cleaned:
                    continue
                for line in cleaned.splitlines():
                    hm = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
                    if hm:
                        if curr_chunks:
                            segments.append((curr_heading, curr_chunks))
                            curr_chunks = []
                        cand_h = _clean_heading_title(hm.group(2))
                        if cand_h and cand_h.lower() not in {"references", "exercise", "exercises", "further reading"}:
                            curr_heading = cand_h
                    else:
                        curr_chunks.append(line)
            elif ctype == "code":
                code_lines = [
                    ln for ln in src.splitlines()
                    if not ln.lstrip().startswith(("!", "%", "from qiskit_ibm_runtime", "QiskitRuntimeService.save_account"))
                ]
                code_body = "\n".join(code_lines).strip()
                if 20 <= len(code_body) <= 750:
                    try:
                        _parse_ast_quiet(code_body)
                        curr_chunks.append(f"```python\n{code_body}\n```")
                    except SyntaxError:
                        pass
        if curr_chunks:
            segments.append((curr_heading, curr_chunks))
    else:
        cleaned = _sanitize_markdown_body(raw_content)
        curr_heading = doc_title
        curr_chunks = []
        in_code = False
        for line in cleaned.splitlines():
            if line.strip().startswith("```"):
                in_code = not in_code
                curr_chunks.append(line)
                continue
            if not in_code:
                hm = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
                if hm:
                    if curr_chunks:
                        segments.append((curr_heading, curr_chunks))
                        curr_chunks = []
                    cand_h = _clean_heading_title(hm.group(2))
                    if cand_h and cand_h.lower() not in {"references", "exercise", "exercises", "further reading", "next steps"}:
                        curr_heading = cand_h
                    continue
            curr_chunks.append(line)
        if curr_chunks:
            segments.append((curr_heading, curr_chunks))

    pairs: list[tuple[str, str]] = []
    for heading, lines in segments:
        if not heading or len(heading) < 4:
            continue
        if heading.lower() in {"introduction", "overview", "summary", "prerequisites"}:
            topic = f"{doc_title} ({heading})"
        else:
            topic = heading if doc_title.lower() in heading.lower() else f"{heading} ({doc_title})"
        body = "\n".join(lines).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if len(body) < 140:
            continue
        if len(body) > 1750:
            cut = body[:1750].rsplit("\n\n", 1)[0].strip()
            if cut.count("```") % 2 != 0:
                cut = cut.rsplit("```", 1)[0].strip()
            body = cut if len(cut) >= 140 else body[:1600].strip()
        if body.count("```") % 2 != 0:
            body = body.rsplit("```", 1)[0].strip()
        if len(body) < 140:
            continue
        prompt = f"Explain {topic} in quantum computing."
        answer = f"Analysis ({source_name} — {topic}):\n{body}"
        pairs.append((prompt, answer))
    return pairs


def extract_pennylane_sections(raw_py: str, path: str) -> list[tuple[str, str]]:
    """Extract clean (prompt, structured_answer) pairs from an Apache-2.0 PennyLane Sphinx-gallery demo.py."""
    m_doc = re.search(r'^\s*r?"""(.*?)"""', raw_py, flags=re.DOTALL)
    if not m_doc:
        return []
    top_doc = m_doc.group(1).strip()
    rest_src = raw_py[m_doc.end() :].strip()

    # Extract RST/Markdown title
    lines = top_doc.splitlines()
    title = ""
    if len(lines) >= 2 and re.match(r"^[=\-~]{4,}$", lines[1].strip()):
        title = lines[0].strip()
        lines = lines[2:]
    if not title:
        slug = Path(path).parent.name.replace("tutorial_", "").replace("_", " ").replace("-", " ").title()
        title = slug or "Quantum Computing Demonstration"

    # Strip `.. meta::`, `.. related::`, `.. rst-class::`, `.. figure::` directive blocks
    cleaned_lines: list[str] = []
    skip_indent_block = False
    for ln in lines:
        s = ln.strip()
        if re.match(r"^\.\.\s+(?:meta|related|rst-class|figure|raw)::", s):
            skip_indent_block = True
            continue
        if skip_indent_block:
            if not s or ln.startswith((" ", "\t")):
                continue
            skip_indent_block = False
        cleaned_lines.append(ln)

    narrative = "\n".join(cleaned_lines).strip()
    narrative = re.sub(r":(?:math|doc|func|class):`([^`]+)`", r"`\1`", narrative)
    narrative = re.sub(r"\[#[^\]]+\]_", "", narrative)
    narrative = re.sub(r"\n{3,}", "\n\n", narrative).strip()

    # Extract first clean code block from `# %%` segments if concise & valid AST
    code_snippets: list[str] = []
    for seg in re.split(r"(?m)^#\s*%%\s*$", rest_src):
        code_lines = [ln for ln in seg.splitlines() if not ln.strip().startswith("#")]
        code_txt = "\n".join(code_lines).strip()
        if 40 <= len(code_txt) <= 720 and ("qml." in code_txt or "def " in code_txt):
            try:
                _parse_ast_quiet(code_txt)
                code_snippets.append(f"```python\n{code_txt}\n```")
                break
            except SyntaxError:
                pass

    if len(narrative) < 150:
        return []
    body = narrative[:1350].rsplit("\n\n", 1)[0].strip() or narrative[:1200].strip()
    if code_snippets:
        body = f"{body}\n\n{code_snippets[0]}"
    prompt = f"Explain {title} in quantum computing and quantum machine learning."
    answer = f"Analysis (PennyLane/qml — {title}):\n{body}"
    return [(prompt, answer)]


def _format_cosmopedia_prompt(raw_prompt: str, text_body: str) -> str:
    m_unit = re.search(r"titled\s+['\"]([^'\"]+)['\"]", raw_prompt, re.I)
    m_book = re.search(r"textbook\s+on\s+['\"]([^'\"]+)['\"]", raw_prompt, re.I)
    m_chap = re.search(r"chapter\s+['\"]([^'\"]+)['\"]", raw_prompt, re.I)
    if m_unit and m_book:
        return f"Explain {m_unit.group(1).strip()} in {m_book.group(1).strip()}."
    if m_unit and m_chap:
        return f"Explain {m_unit.group(1).strip()} ({m_chap.group(1).strip()})."
    first_line = text_body.strip().splitlines()[0] if text_body.strip() else ""
    first_clean = re.sub(r"[*#`]+", "", first_line).strip(" :.-")
    if 6 <= len(first_clean) <= 95:
        return f"Explain the concept of {first_clean}."
    return ""


def _fetch_github_files_parallel(urls: list[tuple[str, str]], timeout: int, max_workers: int = 8) -> list[tuple[str, str]]:
    """Fetch a list of (rel_path, raw_url) pairs concurrently using stdlib ThreadPoolExecutor."""
    results: list[tuple[str, str]] = []

    def _job(item: tuple[str, str]) -> tuple[str, str] | None:
        rel_p, raw_url = item
        try:
            txt = _http_get_bytes(raw_url, timeout=timeout, retries=2).decode("utf-8", errors="replace")
            return rel_p, txt
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_job, u) for u in urls]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                results.append(res)
    results.sort(key=lambda pair: pair[0])
    return results


def curate_all_datasets(
    config: CurationConfig | None = None,
    offline_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run complete internet/offline permissive dataset curation and write curated JSONL + report."""
    cfg = config or CurationConfig()
    t0 = time.perf_counter()
    pipeline = SixGateDataHygienePipeline(min_entropy=3.2, max_entropy=5.8, min_pedagogy=0.55, max_seen_cache=100_000)

    all_records: list[dict[str, Any]] = []
    source_stats: list[SourceAuditStats] = []

    def _log(msg: str) -> None:
        if cfg.verbose:
            print(msg, flush=True)

    def _audit_and_append(
        *,
        st: SourceAuditStats,
        rec_id: str,
        source_label: str,
        user_text: str,
        target_text: str,
        spdx_license: str,
        domain: str,
        trust_tier: int = 2,
        think: str = "",
        test_code: str = "",
    ) -> bool:
        st.candidates_seen += 1
        u = _strip_think_tags(user_text).strip()
        r = _strip_think_tags(target_text).strip()
        r = re.sub(r"\n+Output:\s*```\s*$", "", r).strip()
        if not u or not r or len(u) < 8 or len(r) < 60:
            st.prefilter_rejected += 1
            return False
        if r.count("```") % 2 != 0:
            st.prefilter_rejected += 1
            return False
        if _has_noise_or_mojibake(u) or _has_noise_or_mojibake(r) or _has_persona_leakage(u, r):
            st.prefilter_rejected += 1
            return False
        if not _passes_extra_decontam(u, r):
            st.record_gate_reject("Gate 6 (Decontamination)")
            return False

        audit_text = f"Problem: {u}\n{r}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            gate_res = pipeline.audit_record(audit_text, spdx_license=spdx_license)
        if not gate_res.passed:
            st.record_gate_reject(gate_res.rejection_gate)
            return False

        st.gate_passed += 1
        all_records.append(
            {
                "id": rec_id,
                "source": source_label,
                "spdx_license": spdx_license,
                "split": "train",
                "domain": domain,
                "trust_tier": int(trust_tier),
                "user_text": u,
                "target_text": r,
                "think": think,
                "test_code": test_code,
                "hygiene_metrics": gate_res.metrics,
            }
        )
        return True

    # ------------------------------------------------------------------
    # 1. GitHub TheAlgorithms/Python (MIT, Sandbox + Doctest verified)
    # ------------------------------------------------------------------
    st_thealg = SourceAuditStats("TheAlgorithms/Python", "MIT", "train", "coding")
    source_stats.append(st_thealg)
    try:
        zip_bytes = (
            offline_bundle.get("thealgorithms_zip")
            if offline_bundle is not None
            else _http_get_bytes("https://github.com/TheAlgorithms/Python/archive/refs/heads/master.zip", timeout=cfg.request_timeout_sec)
        )
    except Exception as exc:
        st_thealg.fetch_errors.append(str(exc))
        zip_bytes = None
    if zip_bytes:
        ta_records = extract_thealgorithms_from_zip_bytes(
            zip_bytes=zip_bytes,
            pipeline=pipeline,
            stats=st_thealg,
            max_rules=cfg.thealgorithms_max_rules,
        )
        all_records.extend(ta_records)
        _log(f"[1/11] TheAlgorithms/Python (MIT): kept={len(ta_records)} sandbox_ok={st_thealg.sandbox_verified} sandbox_rej={st_thealg.sandbox_rejected}")

    # ------------------------------------------------------------------
    # 2. GitHub Qiskit/textbook & Qiskit/documentation (Apache-2.0)
    # ------------------------------------------------------------------
    st_qiskit = SourceAuditStats("Qiskit/textbook+documentation", "Apache-2.0", "train", "quantum_computing")
    source_stats.append(st_qiskit)
    qiskit_added = 0
    if offline_bundle is not None:
        for path, raw_txt in offline_bundle.get("qiskit_files", []):
            if qiskit_added >= cfg.qiskit_max_rules:
                break
            for u_q, a_q in extract_qiskit_sections(raw_txt, path, "Qiskit"):
                if qiskit_added >= cfg.qiskit_max_rules:
                    break
                if _audit_and_append(
                    st=st_qiskit,
                    rec_id=f"qiskit_{qiskit_added:05d}",
                    source_label="Qiskit/Apache-2.0",
                    user_text=u_q,
                    target_text=a_q,
                    spdx_license="Apache-2.0",
                    domain="quantum_computing",
                ):
                    qiskit_added += 1
    else:
        for repo_name, prefix_filter, max_files in (
            ("Qiskit/textbook", "notebooks/ch-", cfg.qiskit_textbook_max_files),
            ("Qiskit/documentation", "learning/courses/", cfg.qiskit_docs_max_files),
        ):
            if qiskit_added >= cfg.qiskit_max_rules or max_files <= 0:
                break
            try:
                tree = _http_get_json(f"https://api.github.com/repos/{repo_name}/git/trees/main?recursive=1", timeout=cfg.request_timeout_sec)
            except Exception as exc:
                st_qiskit.fetch_errors.append(f"{repo_name}: {exc}")
                continue
            cand_paths = [
                e["path"]
                for e in tree.get("tree", [])
                if e.get("type") == "blob"
                and e["path"].startswith(prefix_filter)
                and e["path"].endswith((".ipynb", ".md", ".mdx"))
                and "/translations/" not in e["path"]
                and not e["path"].endswith(("index.md", "index.mdx", "README.md", "exam.mdx"))
            ]
            if len(cand_paths) > max_files > 0:
                step = max(1, len(cand_paths) // max_files)
                cand_paths = cand_paths[::step][:max_files]
            else:
                cand_paths = cand_paths[:max_files]

            url_list = [
                (rel_p, f"https://raw.githubusercontent.com/{repo_name}/main/{urllib.parse.quote(rel_p)}")
                for rel_p in cand_paths
            ]
            for rel_p, content in _fetch_github_files_parallel(url_list, timeout=cfg.request_timeout_sec):
                if qiskit_added >= cfg.qiskit_max_rules:
                    break
                for u_q, a_q in extract_qiskit_sections(content, rel_p, repo_name):
                    if qiskit_added >= cfg.qiskit_max_rules:
                        break
                    if _audit_and_append(
                        st=st_qiskit,
                        rec_id=f"qiskit_{qiskit_added:05d}",
                        source_label=repo_name,
                        user_text=u_q,
                        target_text=a_q,
                        spdx_license="Apache-2.0",
                        domain="quantum_computing",
                    ):
                        qiskit_added += 1
        _log(f"[2/11] Qiskit textbook+docs (Apache-2.0): kept={qiskit_added} seen={st_qiskit.candidates_seen}")

    # ------------------------------------------------------------------
    # 3. GitHub PennyLaneAI/qml (Apache-2.0, Quantum ML & Circuits)
    # ------------------------------------------------------------------
    st_pl = SourceAuditStats("PennyLaneAI/qml", "Apache-2.0", "train", "quantum_computing")
    source_stats.append(st_pl)
    pl_added = 0
    if offline_bundle is not None:
        for path, raw_py in offline_bundle.get("pennylane_files", []):
            if pl_added >= cfg.pennylane_max_rules:
                break
            for u_p, a_p in extract_pennylane_sections(raw_py, path):
                if pl_added >= cfg.pennylane_max_rules:
                    break
                if _audit_and_append(
                    st=st_pl,
                    rec_id=f"pennylane_{pl_added:05d}",
                    source_label="PennyLaneAI/qml",
                    user_text=u_p,
                    target_text=a_p,
                    spdx_license="Apache-2.0",
                    domain="quantum_computing",
                ):
                    pl_added += 1
    elif cfg.pennylane_max_rules > 0 and cfg.pennylane_max_files > 0:
        try:
            pl_tree = _http_get_json("https://api.github.com/repos/PennyLaneAI/qml/git/trees/master?recursive=1", timeout=cfg.request_timeout_sec)
            demo_paths = [
                e["path"]
                for e in pl_tree.get("tree", [])
                if e.get("type") == "blob" and e["path"].startswith("demonstrations_v2/") and e["path"].endswith("/demo.py")
            ]
            if len(demo_paths) > cfg.pennylane_max_files:
                step = max(1, len(demo_paths) // cfg.pennylane_max_files)
                demo_paths = demo_paths[::step][: cfg.pennylane_max_files]
            url_list = [
                (rel_p, f"https://raw.githubusercontent.com/PennyLaneAI/qml/master/{urllib.parse.quote(rel_p)}")
                for rel_p in demo_paths
            ]
            for rel_p, raw_py in _fetch_github_files_parallel(url_list, timeout=cfg.request_timeout_sec):
                if pl_added >= cfg.pennylane_max_rules:
                    break
                for u_p, a_p in extract_pennylane_sections(raw_py, rel_p):
                    if pl_added >= cfg.pennylane_max_rules:
                        break
                    if _audit_and_append(
                        st=st_pl,
                        rec_id=f"pennylane_{pl_added:05d}",
                        source_label="PennyLaneAI/qml",
                        user_text=u_p,
                        target_text=a_p,
                        spdx_license="Apache-2.0",
                        domain="quantum_computing",
                    ):
                        pl_added += 1
        except Exception as exc:
            st_pl.fetch_errors.append(str(exc))
        _log(f"[3/11] PennyLaneAI/qml (Apache-2.0): kept={pl_added} seen={st_pl.candidates_seen}")

    # ------------------------------------------------------------------
    # 4. CodeFeedback-Filtered-Instruction (Apache-2.0, train JSONL Range)
    # ------------------------------------------------------------------
    st_cf = SourceAuditStats("m-a-p/CodeFeedback-Filtered-Instruction", "Apache-2.0", "train", "coding")
    source_stats.append(st_cf)
    cf_added = 0
    if offline_bundle is None and cfg.codefeedback_target > 0:
        cf_url = "https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction/resolve/main/CodeFeedback-Filtered-Instruction.jsonl"
        cf_rows = _stream_hf_jsonl_ranges(
            st_cf,
            cf_url,
            offsets=(0, 35_000_000, 85_000_000, 145_000_000, 210_000_000, 285_000_000),
            chunk_bytes=2_600_000,
            timeout=cfg.request_timeout_sec,
        )
        for row in cf_rows:
            if cf_added >= cfg.codefeedback_target:
                break
            if str(row.get("lang", "")).lower() != "python":
                continue
            q = str(row.get("query") or "").strip()
            a = str(row.get("answer") or "").strip()
            if len(q) > 450 or len(a) < 160 or len(a) > 2200:
                st_cf.candidates_seen += 1
                st_cf.prefilter_rejected += 1
                continue
            low_head = a[:220].lower()
            if any(bad in low_head for bad in ("contains an error", "buggy version", "incorrect code")):
                st_cf.candidates_seen += 1
                st_cf.prefilter_rejected += 1
                continue
            formatted_a = a if a.startswith(("Example", "Step", "Analysis")) else f"Example Solution:\n{a}"
            if _audit_and_append(
                st=st_cf,
                rec_id=f"codefb_{cf_added:05d}",
                source_label="m-a-p/CodeFeedback-Filtered-Instruction",
                user_text=q,
                target_text=formatted_a,
                spdx_license="Apache-2.0",
                domain="coding",
            ):
                cf_added += 1
        _log(f"[4/11] CodeFeedback-Filtered-Instruction (Apache-2.0): kept={cf_added} seen={st_cf.candidates_seen}")

    # ------------------------------------------------------------------
    # 5. ise-uiuc/Magicoder-OSS-Instruct-75K (MIT, train decontaminated Python)
    # ------------------------------------------------------------------
    st_mag = SourceAuditStats("ise-uiuc/Magicoder-OSS-Instruct-75K", "MIT", "train", "coding")
    source_stats.append(st_mag)
    mag_added = 0
    if offline_bundle is None and cfg.magicoder_target > 0:
        mag_url = "https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K/resolve/main/data-oss_instruct-decontaminated.jsonl"
        mag_rows = _stream_hf_jsonl_ranges(
            st_mag,
            mag_url,
            offsets=(0, 22_000_000, 50_000_000, 82_000_000, 120_000_000, 160_000_000),
            chunk_bytes=2_600_000,
            timeout=cfg.request_timeout_sec,
        )
        for row in mag_rows:
            if mag_added >= cfg.magicoder_target:
                break
            if str(row.get("lang", "")).lower() != "python":
                continue
            q = str(row.get("problem") or "").strip()
            a = str(row.get("solution") or "").strip()
            if len(q) > 1300 or len(a) < 160 or len(a) > 2200:
                st_mag.candidates_seen += 1
                st_mag.prefilter_rejected += 1
                continue
            formatted_a = a if a.startswith(("Example", "Step", "Analysis")) else f"Example Implementation:\n{a}"
            if _audit_and_append(
                st=st_mag,
                rec_id=f"magicoder_{mag_added:05d}",
                source_label="ise-uiuc/Magicoder-OSS-Instruct-75K",
                user_text=q,
                target_text=formatted_a,
                spdx_license="MIT",
                domain="coding",
            ):
                mag_added += 1
        _log(f"[5/11] Magicoder Python (MIT): kept={mag_added} seen={st_mag.candidates_seen}")

    # ------------------------------------------------------------------
    # 6. openai/gsm8k (MIT, direct GitHub grade_school_math train.jsonl ONLY!)
    # ------------------------------------------------------------------
    st_gsm = SourceAuditStats("openai/gsm8k", "MIT", "train", "math")
    source_stats.append(st_gsm)
    _assert_train_split("train")
    gsm_rows: list[dict[str, Any]] = []
    if offline_bundle is not None:
        gsm_rows = list(offline_bundle.get("gsm8k_rows", []))
    elif cfg.gsm8k_target > 0:
        try:
            raw_gsm = _http_get_bytes(
                "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
                timeout=cfg.request_timeout_sec,
            ).decode("utf-8", errors="replace")
            all_gsm = [json.loads(ln) for ln in raw_gsm.splitlines() if ln.strip().startswith("{")]
            # Uniformly stride through 7,473 train rows so diverse word-problem templates are covered
            step = max(1, len(all_gsm) // max(1, cfg.gsm8k_target + 250))
            gsm_rows = all_gsm[::step]
        except Exception as exc:
            st_gsm.fetch_errors.append(str(exc))

    gsm_added = 0
    for row in gsm_rows:
        if gsm_added >= cfg.gsm8k_target:
            break
        q = str(row.get("question") or "").strip()
        formatted_a, final_ans = clean_gsm8k_answer(str(row.get("answer") or ""))
        if not q or not formatted_a:
            st_gsm.candidates_seen += 1
            st_gsm.prefilter_rejected += 1
            continue
        if _audit_and_append(
            st=st_gsm,
            rec_id=f"gsm8k_{gsm_added:05d}",
            source_label="openai/gsm8k",
            user_text=q,
            target_text=formatted_a,
            spdx_license="MIT",
            domain="math",
            think=f"GSM8K exact train derivation -> {final_ans}",
        ):
            gsm_added += 1
    _log(f"[6/11] openai/gsm8k train.jsonl (MIT): kept={gsm_added} seen={st_gsm.candidates_seen}")

    # ------------------------------------------------------------------
    # 7. TIGER-Lab/MathInstruct (MIT, train incremental Range slices)
    # ------------------------------------------------------------------
    st_mi = SourceAuditStats("TIGER-Lab/MathInstruct", "MIT", "train", "math")
    source_stats.append(st_mi)
    mi_added = 0
    if offline_bundle is None and cfg.mathinstruct_target > 0:
        mi_url = "https://huggingface.co/datasets/TIGER-Lab/MathInstruct/resolve/main/MathInstruct.json"
        mi_rows = _stream_hf_json_array_ranges(
            st_mi,
            mi_url,
            offsets=(0, 45_000_000, 95_000_000, 140_000_000, 175_000_000),
            chunk_bytes=3_000_000,
            timeout=cfg.request_timeout_sec,
        )
        for row in mi_rows:
            if mi_added >= cfg.mathinstruct_target:
                break
            inst = str(row.get("instruction") or "").strip()
            out = str(row.get("output") or "").strip()
            if len(inst) > 600 or len(out) < 110 or len(out) > 1800:
                st_mi.candidates_seen += 1
                st_mi.prefilter_rejected += 1
                continue
            if "```python" in out or "def " in out:
                code_blk = ZeroTrustSandboxJudge.extract_python_code(out)
                if not code_blk:
                    st_mi.candidates_seen += 1
                    st_mi.prefilter_rejected += 1
                    continue
            if "Step" not in out and "Derivation" not in out:
                out = f"Step-by-Step Derivation:\n{out}"
            if _audit_and_append(
                st=st_mi,
                rec_id=f"mathinst_{mi_added:05d}",
                source_label="TIGER-Lab/MathInstruct",
                user_text=inst,
                target_text=out,
                spdx_license="MIT",
                domain="math",
            ):
                mi_added += 1
        _log(f"[7/11] TIGER-Lab/MathInstruct (MIT): kept={mi_added} seen={st_mi.candidates_seen}")

    # ------------------------------------------------------------------
    # 8. Open-Orca/SlimOrca (MIT, train ShareGPT Range slices)
    # ------------------------------------------------------------------
    st_orca = SourceAuditStats("Open-Orca/SlimOrca", "MIT", "train", "general_chat")
    source_stats.append(st_orca)
    orca_added = 0
    if offline_bundle is None and cfg.slimorca_target > 0:
        orca_url = "https://huggingface.co/datasets/Open-Orca/SlimOrca/resolve/main/oo-labeled_correct.gpt4.sharegpt.jsonl"
        orca_rows = _stream_hf_jsonl_ranges(
            st_orca,
            orca_url,
            offsets=(0, 80_000_000, 220_000_000, 410_000_000, 620_000_000, 810_000_000),
            chunk_bytes=2_600_000,
            timeout=cfg.request_timeout_sec,
        )
        for row in orca_rows:
            if orca_added >= cfg.slimorca_target:
                break
            convs = row.get("conversations") or []
            if not isinstance(convs, list):
                continue
            human_msg = next((str(c.get("value") or "").strip() for c in convs if c.get("from") == "human"), "")
            gpt_msg = next((str(c.get("value") or "").strip() for c in convs if c.get("from") == "gpt"), "")
            if len(human_msg) < 20 or len(human_msg) > 550 or len(gpt_msg) < 120 or len(gpt_msg) > 1900:
                st_orca.candidates_seen += 1
                st_orca.prefilter_rejected += 1
                continue
            h_low = human_msg.lower()
            if (
                any(
                    bad in h_low
                    for bad in (
                        "hateful post",
                        "hate speech",
                        "offensive language",
                        "<br>",
                        "which language",
                        "short movie review",
                        "definition: in this task",
                        "determine if the hypothesis",
                        "entails the hypothesis",
                    )
                )
                or "<br>" in gpt_msg.lower()
            ):
                st_orca.candidates_seen += 1
                st_orca.prefilter_rejected += 1
                continue
            if not gpt_msg.startswith(("Detailed Analysis", "Analysis", "Step", "Example", "Explanation", "#")):
                gpt_msg = f"Detailed Analysis:\n{gpt_msg}"
            if _audit_and_append(
                st=st_orca,
                rec_id=f"slimorca_{orca_added:05d}",
                source_label="Open-Orca/SlimOrca",
                user_text=human_msg,
                target_text=gpt_msg,
                spdx_license="MIT",
                domain="general_chat",
            ):
                orca_added += 1
        _log(f"[8/11] Open-Orca/SlimOrca (MIT): kept={orca_added} seen={st_orca.candidates_seen}")

    # ------------------------------------------------------------------
    # Polite HuggingFace /rows helper (bounded <= 14 total requests, split="train" only)
    # ------------------------------------------------------------------
    def _polite_hf_batches(st_obj: SourceAuditStats, dataset: str, config_name: str, offsets: tuple[int, ...]):
        for off in offsets:
            try:
                rows, _ = _fetch_hf_rows(dataset, config_name, "train", offset=off, length=100, timeout=cfg.request_timeout_sec)
                if rows:
                    yield rows
            except Exception as exc:
                st_obj.fetch_errors.append(f"offset={off}: {exc}")
            time.sleep(0.85)

    # ------------------------------------------------------------------
    # 9. OpenAssistant/oasst2 (Apache-2.0, strict train split)
    # ------------------------------------------------------------------
    st_oasst = SourceAuditStats("OpenAssistant/oasst2", "Apache-2.0", "train", "general_chat")
    source_stats.append(st_oasst)
    oasst_added = 0
    if offline_bundle is None and cfg.oasst2_target > 0:
        # Four contiguous 200-row train windows keep parent_id trees intact with only 8 total HTTP calls
        for win_start in (0, 1600, 3800, 6400):
            if oasst_added >= cfg.oasst2_target:
                break
            msgs_by_id: dict[str, dict[str, Any]] = {}
            rows_win: list[dict[str, Any]] = []
            for batch in _polite_hf_batches(st_oasst, "OpenAssistant/oasst2", "default", (win_start, win_start + 100)):
                rows_win.extend(batch)
            for msg in rows_win:
                mid = str(msg.get("message_id") or "")
                if mid:
                    msgs_by_id[mid] = msg
            for msg in rows_win:
                if oasst_added >= cfg.oasst2_target:
                    break
                if (
                    msg.get("role") != "assistant"
                    or msg.get("lang") != "en"
                    or msg.get("deleted") is True
                    or msg.get("review_result") is False
                ):
                    continue
                if msg.get("rank") not in (0, 1, None):
                    continue
                parent = msgs_by_id.get(str(msg.get("parent_id") or ""))
                if not parent or parent.get("role") != "prompter" or parent.get("lang") != "en":
                    continue
                q = str(parent.get("text") or "").strip()
                a = str(msg.get("text") or "").strip()
                if len(q) > 450 or len(a) < 110 or len(a) > 1900:
                    st_oasst.candidates_seen += 1
                    st_oasst.prefilter_rejected += 1
                    continue
                if not a.startswith(("Assistant", "Analysis", "Explanation", "Step", "Example", "#")):
                    a = f"Assistant Explanation:\n{a}"
                if _audit_and_append(
                    st=st_oasst,
                    rec_id=f"oasst2_{oasst_added:05d}",
                    source_label="OpenAssistant/oasst2",
                    user_text=q,
                    target_text=a,
                    spdx_license="Apache-2.0",
                    domain="general_chat",
                ):
                    oasst_added += 1
        _log(f"[9/11] OpenAssistant/oasst2 (Apache-2.0): kept={oasst_added} seen={st_oasst.candidates_seen}")

    # ------------------------------------------------------------------
    # 10. HuggingFaceTB/cosmopedia (Apache-2.0, openstax + khanacademy, train)
    # ------------------------------------------------------------------
    st_cosmo = SourceAuditStats("HuggingFaceTB/cosmopedia", "Apache-2.0", "train", "science_reasoning")
    source_stats.append(st_cosmo)
    cosmo_added = 0
    if offline_bundle is None and cfg.cosmopedia_target > 0:
        half_target = cfg.cosmopedia_target // 2
        for sub_cfg, offsets, cap in (
            ("openstax", (0, 1500), half_target),
            ("khanacademy", (0, 800), cfg.cosmopedia_target),
        ):
            for batch in _polite_hf_batches(st_cosmo, "HuggingFaceTB/cosmopedia", sub_cfg, offsets):
                if cosmo_added >= cap:
                    break
                for row in batch:
                    if cosmo_added >= cap:
                        break
                    if row.get("audience") == "young_children":
                        continue
                    body = str(row.get("text") or "").strip()
                    prompt_q = _format_cosmopedia_prompt(str(row.get("prompt") or ""), body)
                    if not prompt_q or len(body) < 180:
                        st_cosmo.candidates_seen += 1
                        st_cosmo.prefilter_rejected += 1
                        continue
                    body_trim = body[:1700].rsplit("\n\n", 1)[0].strip() or body[:1400].strip()
                    if "\n" not in body_trim:
                        body_trim = f"Analysis:\n{body_trim}"
                    if _audit_and_append(
                        st=st_cosmo,
                        rec_id=f"cosmo_{cosmo_added:05d}",
                        source_label=f"HuggingFaceTB/cosmopedia/{sub_cfg}",
                        user_text=prompt_q,
                        target_text=body_trim,
                        spdx_license="Apache-2.0",
                        domain="science_reasoning",
                    ):
                        cosmo_added += 1
        _log(f"[10/11] Cosmopedia (Apache-2.0): kept={cosmo_added} seen={st_cosmo.candidates_seen}")

    # ------------------------------------------------------------------
    # 11. tau/commonsense_qa (MIT, train)
    # ------------------------------------------------------------------
    st_csqa = SourceAuditStats("tau/commonsense_qa", "MIT", "train", "general_chat")
    source_stats.append(st_csqa)
    csqa_added = 0
    if offline_bundle is None and cfg.commonsense_target > 0:
        for batch in _polite_hf_batches(st_csqa, "tau/commonsense_qa", "default", (0, 900, 1800)):
            if csqa_added >= cfg.commonsense_target:
                break
            for row in batch:
                if csqa_added >= cfg.commonsense_target:
                    break
                q = str(row.get("question") or "").strip()
                concept = str(row.get("question_concept") or "").strip()
                choices = row.get("choices") or {}
                labels = choices.get("label") or []
                texts = choices.get("text") or []
                ans_key = str(row.get("answerKey") or "").strip()
                choice_map = dict(zip(labels, texts))
                ans_text = str(choice_map.get(ans_key) or "").strip()
                if not q or not ans_text:
                    continue
                opts_str = ", ".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
                resp = (
                    f"Commonsense Analysis (concept: {concept}):\n"
                    f"Evaluating options {opts_str}:\n"
                    f"Step 1: Match the physical and situational context of '{concept}'.\n"
                    f"Final Answer: ({ans_key}) {ans_text}."
                )
                if _audit_and_append(
                    st=st_csqa,
                    rec_id=f"csqa_{csqa_added:05d}",
                    source_label="tau/commonsense_qa",
                    user_text=q,
                    target_text=resp,
                    spdx_license="MIT",
                    domain="general_chat",
                ):
                    csqa_added += 1
        _log(f"[11/11] CommonsenseQA (MIT): kept={csqa_added} seen={st_csqa.candidates_seen}")

    # Export JSONL + comprehensive report
    cfg.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    domain_counts = dict(Counter(r["domain"] for r in all_records))
    license_counts = dict(Counter(r["spdx_license"] for r in all_records))
    trust_tier_counts = dict(Counter(str(r["trust_tier"]) for r in all_records))

    gate_reject_totals: dict[str, int] = defaultdict(int)
    for st in source_stats:
        for gk, gv in st.gate_rejections.items():
            gate_reject_totals[gk] += gv

    elapsed_s = round(time.perf_counter() - t0, 2)
    report: dict[str, Any] = {
        "pipeline": "SixGateDataHygienePipeline + ZeroTrustSandboxJudge",
        "split_enforced": "train",
        "allowed_licenses": sorted(SixGateDataHygienePipeline.ALLOWED_SPDX),
        "output_jsonl": str(cfg.output_jsonl_path.as_posix()),
        "total_curated_records": len(all_records),
        "total_candidates_seen": sum(s.candidates_seen for s in source_stats),
        "total_prefilter_rejected": sum(s.prefilter_rejected for s in source_stats),
        "total_sandbox_verified": sum(s.sandbox_verified for s in source_stats),
        "total_sandbox_rejected": sum(s.sandbox_rejected for s in source_stats),
        "gate_rejections_total": dict(gate_reject_totals),
        "records_by_domain": domain_counts,
        "records_by_license": license_counts,
        "records_by_trust_tier": trust_tier_counts,
        "elapsed_sec": elapsed_s,
        "per_source": [asdict(s) for s in source_stats],
    }

    cfg.report_json_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"[DONE] Total curated records={len(all_records)} in {elapsed_s}s -> {cfg.output_jsonl_path}")
    return report


if __name__ == "__main__":
    summary = curate_all_datasets(CurationConfig(verbose=True))
    print(json.dumps(summary, indent=2))
