"""1-Click Document Auto-Crystallizer.

Parses arbitrary Markdown, documentation, textbooks, or Python code files,
extracts verified knowledge triples and structured QA rule pairs, runs 6-gate hygiene
and AST security audits, and commits them directly into the local Crystal vault.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .knowledge_graph import KnowledgeGraphEngine
from .text import clean_entity, normalize


class DocumentAutoCrystallizerEngine:
    """Extracts facts, rules, and code blocks from raw documents into the crystal vault."""

    def __init__(self, kg: KnowledgeGraphEngine | None = None, vault: Any | None = None) -> None:
        self.kg = kg or KnowledgeGraphEngine()
        self.vault = vault

    def crystallize_text(
        self,
        text: str,
        title: str = "Uploaded Document",
        domain: str = "doc_crystallized",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        clean = normalize(text)
        if not clean:
            return {"ok": False, "error": "empty document", "triples_added": 0, "rules_added": 0}

        # 1. Six-Gate PII and Secret Guard
        from mirror_crystal import SixGateDataHygienePipeline
        has_pii, reason = SixGateDataHygienePipeline.scan_for_pii(text)
        if has_pii:
            return {"ok": False, "error": f"PII or secret detected: {reason}", "triples_added": 0, "rules_added": 0}

        triples_added = 0
        rules_added = 0
        code_verified = 0

        # 2. Extract Relational Knowledge Triples
        # Pattern A: "The [predicate] of [Subject] is [Object]."
        for m in re.finditer(
            r"\bthe\s+([a-zA-Z\s]{2,20}?)\s+of\s+([a-zA-Z\s]{2,30}?)\s+is\s+([a-zA-Z0-9\s.,'()$-]{1,40}?)(?=[.!,;]|$)",
            clean,
            re.I,
        ):
            pred = clean_entity(m.group(1)).casefold()
            subj = clean_entity(m.group(2)).casefold()
            obj = clean_entity(m.group(3))
            if pred and subj and obj and len(subj.split()) <= 4:
                if self.kg.insert_triple(subj, pred, obj, category=domain, provenance=title):
                    triples_added += 1

        # Pattern B: "[Subject] is the [predicate] of [Object]."
        for m in re.finditer(
            r"\b([a-zA-Z\s]{2,30}?)\s+is\s+the\s+([a-zA-Z\s]{2,20}?)\s+of\s+([a-zA-Z0-9\s.,'()$-]{1,40}?)(?=[.!,;]|$)",
            clean,
            re.I,
        ):
            obj = clean_entity(m.group(1))
            pred = clean_entity(m.group(2)).casefold()
            subj = clean_entity(m.group(3)).casefold()
            if pred and subj and obj and len(subj.split()) <= 4:
                if self.kg.insert_triple(subj, pred, obj, category=domain, provenance=title):
                    triples_added += 1

        # Pattern C: "[Author] wrote [Work]."
        for m in re.finditer(
            r"\b([A-Z][a-zA-Z\s.'-]{2,30}?)\s+wrote\s+([A-Za-z0-9\s.'-]{2,40}?)(?=[.!,;]|$)",
            clean,
        ):
            author = clean_entity(m.group(1))
            work = clean_entity(m.group(2)).casefold()
            if author and work:
                if self.kg.insert_triple(work, "author", author, category=domain, provenance=title):
                    triples_added += 1

        # 3. Extract Python Code Blocks and verify with AST Sandbox
        from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge

        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
        for code in code_blocks:
            code_strip = code.strip()
            if code_strip:
                is_safe, s_reason = ZeroTrustSandboxJudge.is_ast_safe(code_strip)
                if is_safe:
                    code_verified += 1
                    if self.vault is not None and hasattr(self.vault, "ingest_pair"):
                        # Extract first function name as question
                        fn = re.search(r"def\s+([a-zA-Z0-9_]+)\(", code_strip)
                        prompt = f"Write Python code for {fn.group(1)}" if fn else f"Code snippet from {title}"
                        self.vault.ingest_pair(prompt, f"```python\n{code_strip}\n```", domain=domain)
                        rules_added += 1

        # 4. Extract Markdown Section QA Rules
        # Header -> content block
        sections = re.findall(r"(?:^|\n)#{1,4}\s+([^\n]+)\n+([^#\n][^\n]+(?:\n[^#\n]+)*)", text)
        for header, content in sections:
            h = header.strip()
            c = content.strip()
            if 3 <= len(h) <= 100 and 10 <= len(c) <= 1000:
                if self.vault is not None and hasattr(self.vault, "ingest_pair"):
                    self.vault.ingest_pair(h, c, domain=domain)
                    rules_added += 1

        latency = (time.perf_counter() - t0) * 1000
        return {
            "ok": True,
            "document_title": title,
            "triples_added": triples_added,
            "rules_added": rules_added,
            "code_snippets_verified": code_verified,
            "latency_ms": round(latency, 2),
            "total_knowledge_triples": self.kg.total_triples(),
        }

    def crystallize_file(self, file_path: Path | str, domain: str = "doc_crystallized") -> dict[str, Any]:
        p = Path(file_path)
        if not p.is_file():
            return {"ok": False, "error": f"file not found: {file_path}", "triples_added": 0, "rules_added": 0}
        content = p.read_text(encoding="utf-8", errors="ignore")
        return self.crystallize_text(content, title=p.name, domain=domain)
