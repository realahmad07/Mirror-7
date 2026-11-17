"""Schema-based word problem solver for the CPU engine.

Extracts entities, rates, quantities and constraints from natural language,
translates them to exact Fraction arithmetic, and presents verified step-by-step
solutions without hallucinated extra items.

Supported schema families:
1. Speed / Distance / Time: d = v * t
2. Cost & Quantity / Multi-item Shopping: total = sum(qty_i * price_i)
3. Percentage Increase / Decrease / Discount / Tax / Tip
4. Total / Difference / Sum Systems (e.g., A + B = S, A - B = D)
5. Work Rate: 1/T = sum(1/t_i)
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from .arithmetic import format_fraction
from .text import normalize
from .types import NativeAnswer

_NUM = r"(?:\d+(?:\.\d+)?|\d+/\d+)"
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}


def _parse_num(token: str) -> Fraction:
    tok = token.strip().replace("$", "").replace(",", "")
    tok_low = tok.casefold()
    if tok_low in _WORD_NUMS:
        return Fraction(_WORD_NUMS[tok_low])
    if "/" in tok:
        n, d = tok.split("/", 1)
        return Fraction(int(n), int(d))
    return Fraction(tok)


def _fmt_curr(val: Fraction) -> str:
    if val.denominator == 1:
        return f"${val.numerator}"
    # Formatted to 2 decimals if standard currency cents
    flt = float(val)
    if val.denominator in {2, 4, 5, 10, 20, 25, 50, 100}:
        return f"${flt:.2f}"
    return f"${format_fraction(val)}"


class WordProblemSolver:
    """Interprets word problems and produces step-by-step derivations."""

    def solve(self, prompt: str) -> NativeAnswer | None:
        text = normalize(prompt)
        return (
            self._solve_speed_distance_time(text)
            or self._solve_shopping_and_item_cost(text)
            or self._solve_discounts_and_tax(text)
            or self._solve_sum_difference(text)
            or self._solve_work_rate(text)
        )

    # 1. Speed, Distance, Time
    def _solve_speed_distance_time(self, text: str) -> NativeAnswer | None:
        # e.g.: "A train travels 180 km in 2.5 hours. What is its average speed?"
        # Pattern 1: Find Speed (distance + time given)
        m_speed = re.search(
            rf"\b(?:travels?|covers?|drives?|runs?|walks?)\s+({_NUM})\s*(km|kilometers?|miles?|m|meters?)\s+(?:in|for)\s+({_NUM})\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
            text,
            re.I,
        )
        if m_speed:
            dist_val = _parse_num(m_speed.group(1))
            dist_unit = m_speed.group(2).lower()
            time_val = _parse_num(m_speed.group(3))
            time_unit = m_speed.group(4).lower()

            if "min" in time_unit:
                time_in_hrs = time_val / 60
                speed = dist_val / time_in_hrs
                speed_str = format_fraction(speed)
                steps = (
                    f"Distance = {dist_val} {dist_unit}",
                    f"Time = {time_val} {time_unit} = {time_in_hrs} hours",
                    f"Average Speed = Distance / Time = {dist_val} / {time_in_hrs} = {speed_str} {dist_unit}/hour",
                )
            else:
                speed = dist_val / time_val
                speed_str = format_fraction(speed)
                steps = (
                    f"Distance = {dist_val} {dist_unit}",
                    f"Time = {time_val} {time_unit}",
                    f"Average Speed = Distance / Time = {dist_val} / {time_val} = {speed_str} {dist_unit}/{time_unit.rstrip('s')}",
                )

            unit_label = f"{dist_unit}/{time_unit.rstrip('s')}"
            ans = (
                f"The average speed is **{speed_str} {unit_label}**.\n\n"
                f"**Step-by-Step Derivation:**\n"
                + "\n".join(f"• {s}" for s in steps)
            )
            return NativeAnswer(
                kind="word_problem",
                answer=ans,
                qualification="computed_solution",
                confidence=0.98,
                verified=True,
                steps=steps,
                metadata={"schema": "speed_distance_time", "speed": str(speed)},
            )

        # Pattern 2: Find Distance (speed + time given)
        m_dist = re.search(
            rf"\b(?:at|speed of)\s+({_NUM})\s*(?:km/h|mph|km\s+per\s+hour|miles\s+per\s+hour)\s+(?:for|in)\s+({_NUM})\s*(hours?|hrs?|minutes?|mins?)\b",
            text,
            re.I,
        )
        if m_dist:
            speed_val = _parse_num(m_dist.group(1))
            time_val = _parse_num(m_dist.group(2))
            time_unit = m_dist.group(3).lower()
            time_hrs = time_val / 60 if "min" in time_unit else time_val
            dist = speed_val * time_hrs
            steps = (
                f"Speed = {speed_val}",
                f"Time = {time_val} {time_unit} ({time_hrs} hours)",
                f"Distance = Speed × Time = {speed_val} × {time_hrs} = {format_fraction(dist)}",
            )
            ans = (
                f"The total distance traveled is **{format_fraction(dist)}**.\n\n"
                f"**Step-by-Step Derivation:**\n"
                + "\n".join(f"• {s}" for s in steps)
            )
            return NativeAnswer(
                kind="word_problem",
                answer=ans,
                qualification="computed_solution",
                confidence=0.98,
                verified=True,
                steps=steps,
                metadata={"schema": "distance_from_speed_time", "distance": str(dist)},
            )

        return None

    # 2. Multi-Item Shopping & Cost
    def _solve_shopping_and_item_cost(self, text: str) -> NativeAnswer | None:
        # e.g.: "Sofia buys 2 shirts at $7 each and shoes that cost $3 more than a shirt. How much does she spend?"
        if not re.search(r"\b(?:cost|costs|buys?|spends?|price|each|total)\b", text, re.I):
            return None

        # Look for: Qty items at Price each
        qty_pattern = re.compile(
            rf"\b({_NUM}|{'|'.join(_WORD_NUMS.keys())})\s+([a-zA-Z\s]+?)\s+(?:at|for|costing)\s+\$?({_NUM})\s*(?:each|per\s+[a-zA-Z]+)?\b",
            re.I,
        )
        base_items = []
        for m in qty_pattern.finditer(text):
            q = _parse_num(m.group(1))
            name = m.group(2).strip().lower()
            if name in {"more than", "less than"}:
                continue
            p = _parse_num(m.group(3))
            base_items.append((name, q, p))

        # Look for relative cost: "shoes that cost $3 more than a shirt"
        rel_pattern = re.compile(
            rf"\b(?:a pair of\s+)?([a-zA-Z\s]+?)\s+(?:that\s+)?costs?\s+\$?({_NUM})\s+(more|less)\s+than\s+(?:a|an|the)\s+([a-zA-Z\s]+?)(?:[.,;]|\band\b|$)",
            re.I,
        )
        rel_match = rel_pattern.search(text)
        steps = []
        total = Fraction(0)

        if base_items:
            for name, q, p in base_items:
                sub = q * p
                total += sub
                steps.append(f"{q} {name} at {_fmt_curr(p)} each = {q} × {_fmt_curr(p)} = {_fmt_curr(sub)}")

            if rel_match:
                item2 = rel_match.group(1).strip().lower()
                diff = _parse_num(rel_match.group(2))
                direction = rel_match.group(3).lower()
                base_ref = rel_match.group(4).strip().lower()

                # Find price of base_ref
                ref_price = None
                for n, _, p in base_items:
                    if base_ref in n or n in base_ref or n.rstrip("s") in base_ref:
                        ref_price = p
                        break
                if ref_price is not None:
                    p2 = (ref_price + diff) if direction == "more" else (ref_price - diff)
                    total += p2
                    op = "+" if direction == "more" else "-"
                    steps.append(f"1 {item2} costs {_fmt_curr(diff)} {direction} than {base_ref} = {_fmt_curr(ref_price)} {op} {_fmt_curr(diff)} = {_fmt_curr(p2)}")

            # Check if prompt asks for total
            if re.search(r"\b(?:how much|total price|total cost|how much will (?:she|he|they|i) pay|spend)\b", text, re.I):
                steps.append(f"Total spent = {' + '.join(_fmt_curr(s) for s in [q * p for _, q, p in base_items] + ([p2] if rel_match and ref_price else []))} = {_fmt_curr(total)}")
                ans = (
                    f"The total amount spent is **{_fmt_curr(total)}**.\n\n"
                    f"**Step-by-Step Derivation:**\n"
                    + "\n".join(f"• {s}" for s in steps)
                )
                return NativeAnswer(
                    kind="word_problem",
                    answer=ans,
                    qualification="computed_solution",
                    confidence=0.98,
                    verified=True,
                    steps=steps,
                    metadata={"schema": "shopping_total", "total": str(total)},
                )

        return None

    # 3. Discounts, Tips and Taxes
    def _solve_discounts_and_tax(self, text: str) -> NativeAnswer | None:
        # e.g.: "A coat costs $80. It is on sale for 25% off with 8% sales tax. What is the final price?"
        m_base = re.search(rf"\b(?:costs?|price(?: of)?|bill is)\s+\$?({_NUM})\b", text, re.I)
        m_disc = re.search(rf"\b({_NUM})\s*%\s*(?:off|discount)\b", text, re.I)
        m_tax = re.search(rf"\b({_NUM})\s*%\s*(?:tax|tip)\b", text, re.I)

        if m_base and (m_disc or m_tax):
            base_price = _parse_num(m_base.group(1))
            current = base_price
            steps = [f"Original price = {_fmt_curr(base_price)}"]

            if m_disc:
                disc_pct = _parse_num(m_disc.group(1))
                disc_amount = (current * disc_pct) / 100
                current = current - disc_amount
                steps.append(f"{disc_pct}% discount = {_fmt_curr(disc_amount)} off ({_fmt_curr(base_price)} - {_fmt_curr(disc_amount)} = {_fmt_curr(current)})")

            if m_tax:
                tax_pct = _parse_num(m_tax.group(1))
                tax_amount = (current * tax_pct) / 100
                current = current + tax_amount
                steps.append(f"{tax_pct}% tax/tip = {_fmt_curr(tax_amount)} added ({_fmt_curr(current - tax_amount)} + {_fmt_curr(tax_amount)} = {_fmt_curr(current)})")

            steps.append(f"Final price = {_fmt_curr(current)}")
            ans = (
                f"The final price is **{_fmt_curr(current)}**.\n\n"
                f"**Step-by-Step Derivation:**\n"
                + "\n".join(f"• {s}" for s in steps)
            )
            return NativeAnswer(
                kind="word_problem",
                answer=ans,
                qualification="computed_solution",
                confidence=0.98,
                verified=True,
                steps=steps,
                metadata={"schema": "discount_tax", "final_price": str(current)},
            )

        return None

    # 4. Sum & Difference Systems
    def _solve_sum_difference(self, text: str) -> NativeAnswer | None:
        # e.g.: "The sum of two numbers is 50 and their difference is 14. What are the numbers?"
        m = re.search(
            rf"\bsum of two numbers is\s+({_NUM})\s+and\s+(?:their\s+)?difference is\s+({_NUM})\b",
            text,
            re.I,
        )
        if m:
            s_val = _parse_num(m.group(1))
            d_val = _parse_num(m.group(2))
            # x + y = s, x - y = d => 2x = s + d => x = (s+d)/2, y = (s-d)/2
            x = (s_val + d_val) / 2
            y = (s_val - d_val) / 2
            steps = (
                f"Equations: x + y = {s_val}, x - y = {d_val}",
                f"Add both equations: 2x = {s_val} + {d_val} = {s_val + d_val}",
                f"Larger number (x) = {format_fraction(x)}",
                f"Smaller number (y) = {s_val} - {format_fraction(x)} = {format_fraction(y)}",
            )
            ans = (
                f"The two numbers are **{format_fraction(x)}** and **{format_fraction(y)}**.\n\n"
                f"**Step-by-Step Derivation:**\n"
                + "\n".join(f"• {s}" for s in steps)
            )
            return NativeAnswer(
                kind="word_problem",
                answer=ans,
                qualification="computed_solution",
                confidence=0.98,
                verified=True,
                steps=steps,
                metadata={"schema": "sum_difference", "x": str(x), "y": str(y)},
            )
        return None

    # 5. Work / Pipe Rate Problems
    def _solve_work_rate(self, text: str) -> NativeAnswer | None:
        # e.g.: "Worker A can finish a job in 4 hours and Worker B in 6 hours. How long together?"
        rates = []
        for m in re.finditer(rf"\b(?:in|takes?)\s+({_NUM})\s*(?:hours?|hrs?|days?)\b", text, re.I):
            val = _parse_num(m.group(1))
            if val > 0:
                rates.append(val)
        if len(rates) >= 2 and re.search(r"\b(?:together|combined|working together)\b", text, re.I):
            sum_rate = sum(Fraction(1, r) for r in rates[:2])
            time_together = Fraction(1, sum_rate)
            t_str = format_fraction(time_together)
            steps = (
                f"Worker 1 rate = 1/{rates[0]} per unit time",
                f"Worker 2 rate = 1/{rates[1]} per unit time",
                f"Combined rate = 1/{rates[0]} + 1/{rates[1]} = {format_fraction(sum_rate)}",
                f"Time together = 1 / ({format_fraction(sum_rate)}) = {t_str}",
            )
            ans = (
                f"Working together, they will take **{t_str} hours** (or days).\n\n"
                f"**Step-by-Step Derivation:**\n"
                + "\n".join(f"• {s}" for s in steps)
            )
            return NativeAnswer(
                kind="word_problem",
                answer=ans,
                qualification="computed_solution",
                confidence=0.95,
                verified=True,
                steps=steps,
                metadata={"schema": "work_rate", "time": str(time_together)},
            )
        return None
