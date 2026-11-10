"""
SlotAbstractor — Extracts reusable {v0}, {v1} variable slots and exact arithmetic/modular formulas
from single training examples so 1 example generalizes to infinite unseen numbers without backprop.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_INT_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class AbstractedPair:
    abstract_prompt: str
    abstract_response: str
    abstract_tool: str
    num_slots: int
    derived_expr: str  # Optional synthesized exact formula (e.g., "pow({v0}, {v1}, {v2})")


class SlotAbstractor:
    """Turns concrete numeric/symbolic training pairs into slot-parameterized rules in 1 CPU pass."""

    @staticmethod
    def extract_numbers(text: str) -> list[str]:
        return [m.group(0) for m in _INT_RE.finditer(text)]

    @classmethod
    def abstract_pair(cls, prompt: str, response: str, tool: str = "") -> AbstractedPair:
        p_nums = cls.extract_numbers(prompt)
        if not p_nums:
            return AbstractedPair(prompt, response, tool, 0, "")

        # Deduplicate prompt numbers preserving first-seen order; skip tiny structural bullets 0/1/2 unless <=3 nums
        unique_p: list[str] = []
        for n_str in p_nums:
            if n_str not in unique_p:
                unique_p.append(n_str)

        if len(unique_p) > 6:
            return AbstractedPair(prompt, response, tool, 0, "")

        # Check if prompt numbers appear verbatim in response/tool OR explain a computed result in response!
        r_nums = cls.extract_numbers(response)
        derived_expr = ""
        computed_val_str = ""

        ints = [int(x) for x in unique_p]
        r_ints = [int(x) for x in r_nums if x not in unique_p]

        # Discover exact 1-shot mathematical law connecting prompt numbers -> output number!
        # Require domain cue + non-trivial result (> 9) so unrelated word problems never collide by accident.
        p_low = prompt.lower()
        has_mod_cue = any(k in p_low for k in ("mod", "remainder", "modulo"))
        if len(ints) == 3 and has_mod_cue and ints[2] > 1 and 0 <= ints[1] <= 10_000_000:
            mod_cand = pow(ints[0], ints[1], ints[2])
            if mod_cand in r_ints and mod_cand > 9:
                derived_expr = "pow({v0}, {v1}, {v2})"
                computed_val_str = str(mod_cand)
        if not derived_expr and len(ints) == 2:
            a, b = ints[0], ints[1]
            candidates: list[tuple[int, str]] = []
            if any(k in p_low for k in ("sum", "add", "+")):
                candidates.append((a + b, "({v0} + {v1})"))
            if any(k in p_low for k in ("subtract", "minus", "difference", "-")):
                candidates.append((a - b, "({v0} - {v1})"))
            if any(k in p_low for k in ("multiply", "product", "times", "*")):
                candidates.append((a * b, "({v0} * {v1})"))
            if b != 0 and has_mod_cue:
                candidates.append((a % b, "({v0} % {v1})"))
            if b != 0 and any(k in p_low for k in ("gcd", "greatest common divisor", "hcf")):
                candidates.append((math.gcd(abs(a), abs(b)), "math.gcd({v0}, {v1})"))
            if 0 <= b <= 20 and any(k in p_low for k in ("^", "**", "power")):
                candidates.append((a**b, "({v0} ** {v1})"))
            for val, expr in candidates:
                if val in r_ints and abs(val) > 9:
                    derived_expr = expr
                    computed_val_str = str(val)
                    break

        # Strip markdown numbered list prefixes ("1. ", "2. ") before checking unexplained factual numbers
        resp_no_bullets = re.sub(r"(?:^|\n)\s*\d+\.\s+", "\n", response)
        r_all_digits = re.findall(r"-?\d+", resp_no_bullets)
        unexplained_r = [
            x for x in r_all_digits if x not in unique_p and x != computed_val_str
        ]
        has_tool_binding = bool(tool) and any(x in tool and len(x) >= 2 for x in unique_p)
        if (not derived_expr and not has_tool_binding) or unexplained_r:
            return AbstractedPair(prompt, response, tool, 0, "")

        abs_p = prompt
        abs_r = response
        abs_t = tool
        slot_targets = unique_p if derived_expr else [x for x in unique_p if x in tool and len(x) >= 2]

        for idx, lit in enumerate(slot_targets):
            pat = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(lit)}(?![A-Za-z0-9_])")
            abs_p = pat.sub(f"{{v{idx}}}", abs_p)
            abs_r = pat.sub(f"{{v{idx}}}", abs_r)
            if abs_t:
                abs_t = pat.sub(f"{{v{idx}}}", abs_t)

        if derived_expr and computed_val_str:
            res_pat = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(computed_val_str)}(?![A-Za-z0-9_])")
            abs_r = res_pat.sub("{result}", abs_r)

        return AbstractedPair(
            abstract_prompt=abs_p,
            abstract_response=abs_r,
            abstract_tool=abs_t,
            num_slots=len(slot_targets),
            derived_expr=derived_expr,
        )

    @classmethod
    def mask_query_numbers(cls, query: str) -> str:
        """Replace numbers in a user query with {v0}, {v1}, ... for matching parameterized formula rules."""
        unique_q: list[str] = []
        for n_str in cls.extract_numbers(query):
            if n_str not in unique_q:
                unique_q.append(n_str)
        out = query
        for idx, lit in enumerate(unique_q[:6]):
            pat = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(lit)}(?![A-Za-z0-9_])")
            out = pat.sub(f"{{v{idx}}}", out)
        return out

    @classmethod
    def instantiate(cls, query: str, num_slots: int, abstract_response: str, derived_expr: str) -> str | None:
        """Bind numbers from a new user query into a slot-generalized rule and evaluate exact math on CPU."""
        if num_slots <= 0:
            return abstract_response
        q_nums: list[str] = []
        for x in cls.extract_numbers(query):
            if x not in q_nums:
                q_nums.append(x)
        if len(q_nums) != num_slots:
            return None

        out = abstract_response
        expr = derived_expr
        for i in range(num_slots):
            tok = f"{{v{i}}}"
            out = out.replace(tok, q_nums[i])
            if expr:
                expr = expr.replace(tok, q_nums[i])

        if expr and "{result}" in out:
            try:
                vals = [int(q_nums[i]) for i in range(num_slots)]
                if derived_expr == "pow({v0}, {v1}, {v2})":
                    val = pow(vals[0], vals[1], vals[2])
                elif derived_expr == "({v0} + {v1})":
                    val = vals[0] + vals[1]
                elif derived_expr == "({v0} - {v1})":
                    val = vals[0] - vals[1]
                elif derived_expr == "({v0} * {v1})":
                    val = vals[0] * vals[1]
                elif derived_expr == "({v0} % {v1})":
                    val = vals[0] % vals[1]
                elif derived_expr == "math.gcd({v0}, {v1})":
                    val = math.gcd(abs(vals[0]), abs(vals[1]))
                elif derived_expr == "({v0} ** {v1})" and 0 <= vals[1] <= 1000:
                    val = vals[0] ** vals[1]
                else:
                    return None
                out = out.replace("{result}", str(val))
            except Exception:
                return None
        return out
