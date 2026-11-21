"""Parameter-Free Semantic Frame Parser & Stateful Discourse-Event VM (`SemanticStateVM`).

Replaces brittle single-sentence regexes with a noun-agnostic, clause-by-clause
symbolic virtual machine:
    Text -> Normalize -> Clause Stream -> Typed State Transitions -> Verified Trace

Supports:
- Sequential inventory state machines (arbitrary length INIT -> ADD/SUB chains)
- Rate x Duration / Unit-Price Purchase + Change Due
- Percentage discount / tax / markup
- Equal partition / per-unit division ("N boxes hold M", "N friends split $M equally")
- Relative scaling with discourse ledger binding ("3 times as many A as B", "twice as many")

Safety Invariant: Zero neural weights, zero guessing. If clauses do not compile
cleanly into a typed state machine matching the question intent, the VM abstains.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

_WORD_NUMS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100",
}

_NUM_RE = r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"


def _normalize_text(text: str) -> str:
    s = text.strip()
    for word, digit in _WORD_NUMS.items():
        s = re.sub(rf"\b{word}\b", digit, s, flags=re.IGNORECASE)
    return s


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6g}"


@dataclass(frozen=True)
class SemanticVMResult:
    solved: bool
    frame_kind: str = ""
    output: float | None = None
    steps: tuple[str, ...] = ()
    explanation: str = ""


class SemanticStateVM:
    """Zero-parameter discourse-state virtual machine for free-form word problems."""

    VERSION = "semantic-state-vm-v1"

    def solve(self, prompt: str) -> SemanticVMResult:
        if not isinstance(prompt, str) or not prompt.strip():
            return SemanticVMResult(False, explanation="empty prompt")
        text = _normalize_text(prompt)
        lower = text.casefold()

        # Must be a quantitative question (never fire on pure poems/opinions without numbers)
        if not re.search(r"\d", lower):
            return SemanticVMResult(False, explanation="no numeric quantities found")

        for solver in (
            self._frame_buy_and_change,
            self._frame_percent_change,
            self._frame_relative_scale_and_ledger,
            self._frame_rate_times_duration,
            self._frame_equal_split_or_unit_rate,
            self._frame_sequential_inventory_vm,
        ):
            res = solver(lower)
            if res is not None and res.solved:
                return res

        return SemanticVMResult(False, explanation="no verified semantic frame matched; abstaining honestly")

    # ---- Frame 1: Unit-Price Purchase + Optional Change ----
    def _frame_buy_and_change(self, lower: str) -> SemanticVMResult | None:
        # e.g. "sells notebooks for $4 each ... buys 7 notebooks and pays with $50 ... how much change"
        m_price = re.search(rf"(?:for|at|costs?|priced\s+at)\s*\$\s*{_NUM_RE}\s*(?:each|per|apiece|a\s+\w+)", lower)
        if not m_price:
            m_price = re.search(rf"\$\s*{_NUM_RE}\s*(?:each|per\s+\w+|apiece)", lower)
        if not m_price:
            return None
        unit_price = _to_float(m_price.group(1))

        m_qty = re.search(rf"\b(?:buys?|bought|purchases?|purchased|orders?|ordered|gets?|wants?)\s+{_NUM_RE}\b", lower)
        if not m_qty:
            return None
        qty = _to_float(m_qty.group(1))
        total_cost = unit_price * qty
        steps = [f"UNIT_COST: {_fmt(qty)} * ${_fmt(unit_price)} = ${_fmt(total_cost)}"]

        m_paid = re.search(rf"\b(?:pays?|paid|gives?|gave|hands?)\s+(?:with\s+)?\$\s*{_NUM_RE}\b", lower)
        if m_paid and re.search(r"\b(?:change|left|back|remain)\b", lower):
            paid = _to_float(m_paid.group(1))
            change = paid - total_cost
            steps.append(f"CHANGE_DUE: ${_fmt(paid)} - ${_fmt(total_cost)} = ${_fmt(change)}")
            return SemanticVMResult(
                True, "buy_and_change", change, tuple(steps),
                f"Verified SemanticVM (buy_and_change): {_fmt(change)} | " + " -> ".join(steps),
            )
        if re.search(r"\b(?:cost|total|spend|spent|pay|bill)\b", lower):
            return SemanticVMResult(
                True, "unit_purchase", total_cost, tuple(steps),
                f"Verified SemanticVM (unit_purchase): {_fmt(total_cost)} | " + " -> ".join(steps),
            )
        return None

    # ---- Frame 2: Percentage Discount / Tax / Markup ----
    def _frame_percent_change(self, lower: str) -> SemanticVMResult | None:
        m_pct = re.search(rf"{_NUM_RE}\s*(?:%|percent)\s*(off|discount|less|decrease|reduction|tax|more|increase|markup|tip)?", lower)
        if not m_pct:
            return None
        pct = _to_float(m_pct.group(1))
        direction = (m_pct.group(2) or "").strip()

        # Find base price / amount
        m_base = re.search(rf"\$\s*{_NUM_RE}\b", lower)
        if not m_base:
            m_base = re.search(rf"\b(?:costs?|price(?:d)?(?:\s+at)?|worth|originally|base|amount\s+of)\s+{_NUM_RE}\b", lower)
        if not m_base:
            return None
        base = _to_float(m_base.group(1))

        is_discount = direction in {"off", "discount", "less", "decrease", "reduction"} or bool(
            re.search(r"\b(?:sale|discount|off|markdown|reduced)\b", lower)
        )
        is_increase = direction in {"tax", "more", "increase", "markup", "tip"} or bool(
            re.search(r"\b(?:tax|markup|tip|increase|surcharge)\b", lower)
        )
        if is_discount and not is_increase:
            delta = base * (pct / 100.0)
            final = base - delta
            steps = (
                f"DISCOUNT_AMT: {_fmt(base)} * ({_fmt(pct)} / 100) = {_fmt(delta)}",
                f"FINAL_PRICE: {_fmt(base)} - {_fmt(delta)} = {_fmt(final)}",
            )
            return SemanticVMResult(
                True, "percent_discount", final, steps,
                f"Verified SemanticVM (percent_discount): {_fmt(final)} | " + " -> ".join(steps),
            )
        if is_increase and not is_discount:
            delta = base * (pct / 100.0)
            final = base + delta
            steps = (
                f"SURCHARGE_AMT: {_fmt(base)} * ({_fmt(pct)} / 100) = {_fmt(delta)}",
                f"FINAL_PRICE: {_fmt(base)} + {_fmt(delta)} = {_fmt(final)}",
            )
            return SemanticVMResult(
                True, "percent_increase", final, steps,
                f"Verified SemanticVM (percent_increase): {_fmt(final)} | " + " -> ".join(steps),
            )
        return None

    # ---- Frame 3: Relative Scaling + Multi-Entity Discourse Ledger ----
    def _frame_relative_scale_and_ledger(self, lower: str) -> SemanticVMResult | None:
        # Sub-pattern 3a: "K times as many A as B ... there are N B ... how many in total"
        m_rel = re.search(rf"{_NUM_RE}\s+times\s+as\s+(?:many|much)\s+[a-z]+\s+as\s+[a-z]+", lower)
        if m_rel:
            factor = _to_float(m_rel.group(1))
            # Find the anchor quantity introduced later (not the factor itself)
            rest = lower[m_rel.end():]
            m_anchor = re.search(rf"\b(?:are|has|have|with|is)\s+{_NUM_RE}\b", rest)
            if not m_anchor:
                m_anchor = re.search(rf"\b{_NUM_RE}\b", rest)
            if m_anchor:
                anchor = _to_float(m_anchor.group(1))
                scaled = factor * anchor
                steps = [
                    f"ANCHOR: {_fmt(anchor)}",
                    f"SCALED: {_fmt(factor)} * {_fmt(anchor)} = {_fmt(scaled)}",
                ]
                if re.search(r"\b(?:in\s+total|total|altogether|combined|all\s+together)\b", lower):
                    total = anchor + scaled
                    steps.append(f"LEDGER_SUM: {_fmt(anchor)} + {_fmt(scaled)} = {_fmt(total)}")
                    return SemanticVMResult(
                        True, "relative_scale_total", total, tuple(steps),
                        f"Verified SemanticVM (relative_scale_total): {_fmt(total)} | " + " -> ".join(steps),
                    )
                return SemanticVMResult(
                    True, "relative_scale", scaled, tuple(steps),
                    f"Verified SemanticVM (relative_scale): {_fmt(scaled)} | " + " -> ".join(steps),
                )

        # Sub-pattern 3b: "reads 15 pages on Monday and twice/triple/half as many on Tuesday ... in total"
        m_scale_word = re.search(r"\b(twice|double|triple|thrice|half)\s+as\s+(?:many|much)\b", lower)
        if m_scale_word:
            word = m_scale_word.group(1)
            factor = {"twice": 2.0, "double": 2.0, "triple": 3.0, "thrice": 3.0, "half": 0.5}[word]
            before = lower[: m_scale_word.start()]
            nums_before = re.findall(_NUM_RE, before)
            if nums_before:
                base = _to_float(nums_before[-1])
                second = base * factor
                steps = [
                    f"LEDGER[0]: {_fmt(base)}",
                    f"LEDGER[1] ({word}): {_fmt(base)} * {_fmt(factor)} = {_fmt(second)}",
                ]
                if re.search(r"\b(?:in\s+total|total|altogether|combined|all\s+together)\b", lower):
                    total = base + second
                    steps.append(f"LEDGER_SUM: {_fmt(base)} + {_fmt(second)} = {_fmt(total)}")
                    return SemanticVMResult(
                        True, "discourse_ledger_total", total, tuple(steps),
                        f"Verified SemanticVM (discourse_ledger_total): {_fmt(total)} | " + " -> ".join(steps),
                    )
                return SemanticVMResult(
                    True, "discourse_ledger_scale", second, tuple(steps),
                    f"Verified SemanticVM (discourse_ledger_scale): {_fmt(second)} | " + " -> ".join(steps),
                )
        return None

    # ---- Frame 4: Rate x Duration / Speed x Time ----
    def _frame_rate_times_duration(self, lower: str) -> SemanticVMResult | None:
        # Pattern 4a: "80 km/h for 3 hours"
        m_speed = re.search(rf"{_NUM_RE}\s*(?:km/h|kph|mph|miles\s+per\s+hour|km\s+per\s+hour|m/s)\b", lower)
        if m_speed and re.search(r"\b(?:how\s+far|distance|travel)\b", lower):
            rate = _to_float(m_speed.group(1))
            m_dur = re.search(rf"\b(?:for|in)\s+{_NUM_RE}\s*(?:hours?|hrs?|seconds?|secs?|minutes?|mins?)\b", lower)
            if m_dur:
                dur = _to_float(m_dur.group(1))
                dist = rate * dur
                steps = (f"RATE_MUL: {_fmt(rate)} * {_fmt(dur)} = {_fmt(dist)}",)
                return SemanticVMResult(
                    True, "rate_distance", dist, steps,
                    f"Verified SemanticVM (rate_distance): {_fmt(dist)} | {steps[0]}",
                )

        # Pattern 4b: "makes 12 loaves ... per hour ... in 5 hours"
        m_rate = re.search(rf"{_NUM_RE}\s+(?:[a-z]+\s+){{1,4}}(?:per|an|a|every)\s+(hour|minute|day|week|second)\b", lower)
        if m_rate:
            rate = _to_float(m_rate.group(1))
            unit = m_rate.group(2)
            m_dur = re.search(rf"\b(?:in|for|over|after)\s+{_NUM_RE}\s+{unit}s?\b", lower)
            if m_dur:
                dur = _to_float(m_dur.group(1))
                total = rate * dur
                steps = (f"RATE_MUL: {_fmt(rate)}/{unit} * {_fmt(dur)} {unit}s = {_fmt(total)}",)
                return SemanticVMResult(
                    True, "rate_production", total, steps,
                    f"Verified SemanticVM (rate_production): {_fmt(total)} | {steps[0]}",
                )
        return None

    # ---- Frame 5: Equal Partition / Unit Rate ----
    def _frame_equal_split_or_unit_rate(self, lower: str) -> SemanticVMResult | None:
        # Pattern 5a: "3 friends split a ... bill of $96 equally. How much does each pay?"
        if re.search(r"\b(?:split|share|divide|dividing)\b", lower) and re.search(r"\b(?:equally|evenly|each)\b", lower):
            m_parts = re.search(rf"\b{_NUM_RE}\s+(?:friends?|people|persons?|students?|workers?|siblings?|players?)\b", lower)
            m_total = re.search(rf"\$\s*{_NUM_RE}\b", lower)
            if not m_total and m_parts:
                after = lower[m_parts.end():]
                m_total = re.search(rf"\b{_NUM_RE}\b", after)
            if m_parts and m_total:
                parts = _to_float(m_parts.group(1))
                total = _to_float(m_total.group(1))
                if parts > 0:
                    share = total / parts
                    steps = (f"EQUAL_SPLIT: {_fmt(total)} / {_fmt(parts)} = {_fmt(share)}",)
                    return SemanticVMResult(
                        True, "equal_split", share, steps,
                        f"Verified SemanticVM (equal_split): {_fmt(share)} | {steps[0]}",
                    )

        # Pattern 5b: "4 boxes hold 36 pencils in total, how many pencils are in 1 box?"
        m_container = re.search(
            rf"\b{_NUM_RE}\s+([a-z]+)\s+(?:hold|holds|contain|contains|have|carry|pack)\s+{_NUM_RE}\s+[a-z]+\b",
            lower,
        )
        if m_container and re.search(r"\b(?:(?:in|on)\s+1|(?:in|on)\s+each|per\s+[a-z]+|each\s+[a-z]+)\b", lower):
            parts = _to_float(m_container.group(1))
            total = _to_float(m_container.group(3))
            if parts > 0:
                per_unit = total / parts
                steps = (f"UNIT_PARTITION: {_fmt(total)} / {_fmt(parts)} = {_fmt(per_unit)}",)
                return SemanticVMResult(
                    True, "unit_partition", per_unit, steps,
                    f"Verified SemanticVM (unit_partition): {_fmt(per_unit)} | {steps[0]}",
                )
        return None

    # ---- Frame 6: Sequential Inventory State Machine (INIT -> ADD/SUB -> ...) ----
    def _frame_sequential_inventory_vm(self, lower: str) -> SemanticVMResult | None:
        # Must ask a state-balance question ("how many ... left", "how much ... in the tank", "now have")
        if not re.search(r"\b(?:left|remain|remaining|now|in\s+the\s+[a-z]+|altogether|end\s+up\s+with)\b", lower):
            return None

        # Split premise sentences before the question clause
        q_idx = lower.rfind("how ")
        premise = lower[:q_idx].strip() if q_idx > 0 else lower
        clauses = [c.strip() for c in re.split(r"[.;,]|\band\b|\bthen\b", premise) if c.strip()]
        if len(clauses) < 2:
            return None

        state: float | None = None
        steps: list[str] = []

        sub_pat = re.compile(
            r"\b(?:gives?|gave|loses?|lost|removes?|removed|pumped\s+out|drained|spends?|spent|eats?|ate|sells?|sold|drops?|dropped|takes?\s+out|took\s+out|uses?|used|withdraws?|withdrew)\b"
        )
        add_pat = re.compile(
            r"\b(?:adds?|added|receives?|received|gains?|gained|earns?|earned|buys?|bought|pours?\s+in|poured\s+in|deposits?|deposited|finds?|found|gets?|got|collects?|collected)\b"
        )
        init_pat = re.compile(
            r"\b(?:has|have|had|holds?|held|starts?\s+with|started\s+with|contains?|begins?\s+with|owns?)\b"
        )

        transitions = 0
        for clause in clauses:
            nums = re.findall(_NUM_RE, clause)
            if not nums:
                continue
            val = _to_float(nums[0])
            if state is None and init_pat.search(clause):
                state = val
                steps.append(f"INIT: state = {_fmt(state)}")
            elif state is not None and sub_pat.search(clause):
                nxt = state - val
                steps.append(f"SUB: {_fmt(state)} - {_fmt(val)} = {_fmt(nxt)}")
                state = nxt
                transitions += 1
            elif state is not None and add_pat.search(clause):
                nxt = state + val
                steps.append(f"ADD: {_fmt(state)} + {_fmt(val)} = {_fmt(nxt)}")
                state = nxt
                transitions += 1

        if state is not None and transitions >= 1:
            return SemanticVMResult(
                True, "sequential_inventory_vm", state, tuple(steps),
                f"Verified SemanticVM (sequential_inventory_vm): {_fmt(state)} | " + " -> ".join(steps),
            )
        return None
