"""Scoring and reporting for the NovelGen generalization benchmark.

A *system under test* (SUT) is any callable ``prompt -> Response``. Responses carry
an optional ``source`` label so the report can attribute answers to

* ``learned``    - produced by a trained model (e.g. a neural LM);
* ``reasoner``   - produced by a general in-context inference engine;
* ``handler``    - produced by a topic/pattern-specific handler;
* ``abstain``    - the system declined to answer.

Metrics (per family x split, and overall):

* accuracy        - correct / all variants;
* coverage        - answered / all variants (non-abstentions);
* precision       - correct / answered (how trustworthy a given answer is);
* consistency     - fraction of items whose *every* surface variant is answered
                    correctly (robustness to paraphrase, order, distractors).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from .generator import Item

_ABSTAIN_MARKERS = (
    "i don't know", "i do not know", "not sure", "cannot answer", "can't answer",
    "unable to", "i'm not able", "no answer", "insufficient", "abstain",
)


@dataclass(frozen=True)
class Response:
    text: str
    source: str = "unknown"  # learned | reasoner | handler | abstain | unknown

    @property
    def abstained(self) -> bool:
        if self.source == "abstain" or not self.text.strip():
            return True
        low = self.text.casefold()
        return any(m in low for m in _ABSTAIN_MARKERS) and not _yesno(low)


SUT = Callable[[str], Response]


def _yesno(text: str) -> str | None:
    low = text.casefold()
    # "cannot be determined" / "unknown" first, since they can co-occur with "not".
    if re.search(r"\b(unknown|cannot be determined|can't be determined|undetermined|not enough information|cannot tell|can't tell|not necessarily)\b", low):
        return "unknown"
    m = re.search(r"\b(yes|no)\b", low)
    if m:
        return m.group(1)
    if re.search(r"\b(true|correct)\b", low):
        return "yes"
    if re.search(r"\b(false|incorrect)\b", low):
        return "no"
    return None


def _number(text: str) -> str | None:
    # Prefer an explicit final answer marker, else the last number in the text.
    m = re.search(r"(?:answer|result|=|is)\s*:?\s*(-?\d+(?:\.\d+)?)\s*\.?\s*$", text.strip(), re.I)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    val = m.group(1) if m else (nums[-1] if nums else None)
    if val is None:
        return None
    f = float(val)
    return str(int(f)) if f == int(f) else str(f)


_NON_NAMES = {"The", "Answer", "So", "Therefore", "Yes", "No", "It", "Who", "Is", "A", "An", "Final", "Result", "Thus", "Hence"}


def answer_region(text: str) -> str:
    """The part of a response that states the answer: after 'Answer:' if present, else line 1."""
    m = re.search(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", text, re.I | re.S)
    region = m.group(1) if m else text
    return region.strip().split("\n", 1)[0]


def extract(answer_type: str, text: str, gold: str | None = None) -> str | None:
    """Normalize a free-text answer into the comparable form for ``answer_type``."""
    region = answer_region(text)
    if answer_type == "yesno":
        return _yesno(region) or _yesno(text)
    if answer_type == "number":
        return _number(region)
    if answer_type == "name":
        names = [n for n in re.findall(r"\b[A-Z][a-z]+\b", region) if n not in _NON_NAMES]
        return names[0] if names else None
    if answer_type == "string":
        quoted = re.findall(r"['\"`]([^'\"`]+)['\"`]", region)
        if quoted:
            return quoted[-1].strip()
        toks = re.findall(r"[A-Za-z]+", region)
        return toks[-1] if toks else None
    raise ValueError(answer_type)


def score_variant(item: Item, resp: Response) -> bool:
    if resp.abstained:
        return False
    return extract(item.answer_type, resp.text, item.gold) == item.gold


def _cell() -> dict:
    return {"variants": 0, "answered": 0, "correct": 0, "items": 0, "consistent": 0,
            "sources": defaultdict(int)}


def evaluate(items: list[Item], sut: SUT, keep_transcript: bool = False) -> dict:
    cells: dict[tuple[str, str], dict] = defaultdict(_cell)
    transcript = []
    for it in items:
        cell = cells[(it.family, it.split)]
        cell["items"] += 1
        all_ok = True
        for v in it.variants:
            try:
                resp = sut(v)
            except Exception as exc:  # a crashing SUT is scored as an abstention
                resp = Response(text="", source="abstain")
                if keep_transcript:
                    transcript.append({"item_id": it.item_id, "error": repr(exc)})
            ok = score_variant(it, resp)
            cell["variants"] += 1
            cell["answered"] += 0 if resp.abstained else 1
            cell["correct"] += int(ok)
            cell["sources"]["abstain" if resp.abstained else resp.source] += 1
            all_ok &= ok
            if keep_transcript:
                transcript.append({"item_id": it.item_id, "family": it.family, "depth": it.depth,
                                   "prompt": v, "gold": it.gold, "answer": resp.text,
                                   "source": resp.source, "correct": ok})
        cell["consistent"] += int(all_ok)
    return summarize(cells, transcript if keep_transcript else None)


def _rates(c: dict) -> dict:
    v = max(c["variants"], 1)
    return {
        "items": c["items"],
        "variants": c["variants"],
        "accuracy": round(c["correct"] / v, 4),
        "coverage": round(c["answered"] / v, 4),
        "precision": round(c["correct"] / c["answered"], 4) if c["answered"] else None,
        "consistency": round(c["consistent"] / max(c["items"], 1), 4),
        "sources": dict(c["sources"]),
    }


def summarize(cells: dict, transcript: list | None) -> dict:
    by_split: dict[str, dict] = defaultdict(_cell)
    overall = _cell()
    for (fam, split), c in cells.items():
        for agg in (by_split[split], overall):
            for k in ("variants", "answered", "correct", "items", "consistent"):
                agg[k] += c[k]
            for s, n in c["sources"].items():
                agg["sources"][s] += n
    report = {
        "overall": _rates(overall),
        "by_split": {s: _rates(c) for s, c in sorted(by_split.items())},
        "by_family": {f"{fam}/{split}": _rates(c) for (fam, split), c in sorted(cells.items())},
    }
    # Generalization gap: how much accuracy drops from in-distribution to OOD depths.
    if "id" in report["by_split"] and "ood" in report["by_split"]:
        report["id_to_ood_drop"] = round(report["by_split"]["id"]["accuracy"] - report["by_split"]["ood"]["accuracy"], 4)
    if transcript is not None:
        report["transcript"] = transcript
    return report


def markdown_table(report: dict, title: str) -> str:
    lines = [f"### {title}", "", "| Slice | Acc | Coverage | Precision | Consistency | Sources |", "|---|---:|---:|---:|---:|---|"]

    def row(name: str, r: dict) -> str:
        prec = "-" if r["precision"] is None else f"{r['precision']:.2f}"
        src = ", ".join(f"{k}:{v}" for k, v in sorted(r["sources"].items()))
        return f"| {name} | {r['accuracy']:.2f} | {r['coverage']:.2f} | {prec} | {r['consistency']:.2f} | {src} |"

    lines.append(row("**overall**", report["overall"]))
    for s, r in report["by_split"].items():
        lines.append(row(f"**{s}**", r))
    for f, r in report["by_family"].items():
        lines.append(row(f, r))
    if "id_to_ood_drop" in report:
        lines += ["", f"ID→OOD accuracy drop: {report['id_to_ood_drop']:+.2f}"]
    return "\n".join(lines)
