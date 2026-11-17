"""Deductive logic organ.

Solves problems whose premises are stated inside the prompt:

* categorical syllogisms - "All cats are animals. Tom is a cat. Is Tom an animal?"
* transitive comparisons - "Alice is taller than Bob. Bob is taller than Carl. Who is shortest?"
* propositional modus ponens / tollens - "If it rains, the ground is wet. It rains. Is the ground wet?"

Conclusions are derived only from the given premises, so a successful answer is a
``deductive_proof`` (valid relative to those premises, not a world-fact claim).
"""
from __future__ import annotations

import re
from collections import defaultdict, deque

from .text import normalize, stem
from .types import NativeAnswer

# comparative -> (canonical relation, direction). direction +1 means "A rel B" == "A > B" on that scale.
_COMPARATIVES: dict[str, tuple[str, int]] = {}
_SUPERLATIVES: dict[str, tuple[str, int]] = {}
for high, low, scale, sup_high, sup_low in (
    ("taller", "shorter", "height", "tallest", "shortest"),
    ("older", "younger", "age", "oldest", "youngest"),
    ("faster", "slower", "speed", "fastest", "slowest"),
    ("bigger", "smaller", "size", "biggest", "smallest"),
    ("larger", "smaller", "size", "largest", "smallest"),
    ("heavier", "lighter", "weight", "heaviest", "lightest"),
    ("richer", "poorer", "wealth", "richest", "poorest"),
    ("stronger", "weaker", "strength", "strongest", "weakest"),
    ("smarter", "dumber", "smartness", "smartest", "dumbest"),
    ("higher", "lower", "height", "highest", "lowest"),
    ("longer", "shorter", "length", "longest", "shortest"),
    ("more expensive", "cheaper", "price", "most expensive", "cheapest"),
    ("hotter", "colder", "temperature", "hottest", "coldest"),
    ("better", "worse", "quality", "best", "worst"),
):
    _COMPARATIVES.setdefault(high, (scale, 1))
    _COMPARATIVES.setdefault(low, (scale, -1))
    _SUPERLATIVES.setdefault(sup_high, (scale, 1))
    _SUPERLATIVES.setdefault(sup_low, (scale, -1))
# "shorter"/"smallest" are shared between scales; resolve by premises at solve time.
_AMBIGUOUS_LOW = {"shorter": ("height", "length"), "smaller": ("size",), "shortest": ("height", "length"), "smallest": ("size",)}

_COMP_ALT = "|".join(sorted((re.escape(k) for k in _COMPARATIVES), key=len, reverse=True))
_SUP_ALT = "|".join(sorted((re.escape(k) for k in _SUPERLATIVES), key=len, reverse=True))
_NAME = r"[A-Za-z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)?"

_COMP_FACT = re.compile(rf"^(?P<a>{_NAME})\s+(?:is|are|was|runs|drives|grows)\s+(?P<rel>{_COMP_ALT})\s+than\s+(?P<b>{_NAME})$", re.I)
_COMP_Q = re.compile(rf"^(?:is|are|was)\s+(?P<a>{_NAME})\s+(?P<rel>{_COMP_ALT})\s+than\s+(?P<b>{_NAME})$", re.I)
_SUP_Q = re.compile(rf"^(?:who|which(?:\s+\w+)?|what)\s+(?:is|are|was)\s+(?:the\s+)?(?P<rel>{_SUP_ALT})(?:\s+(?:one|person|of (?:them|all|the \w+)))?$", re.I)

_ARTICLE = r"(?:an?\s+|the\s+)"
_OPT_ARTICLE = r"(?:an?\s+|the\s+)?"
_ALL = re.compile(rf"^(?:all|every|each)\s+(?P<x>[a-z][a-z -]*?)\s+(?:are|is|can|have|has)\s+{_OPT_ARTICLE}(?P<y>[a-z][a-z -]*)$", re.I)
_NO = re.compile(rf"^(?:no)\s+(?P<x>[a-z][a-z -]*?)\s+(?:are|is|can)\s+{_OPT_ARTICLE}(?P<y>[a-z][a-z -]*)$", re.I)
_ISA = re.compile(rf"^(?P<s>.+?)\s+(?:is|are)\s+(?P<neg>not\s+)?{_OPT_ARTICLE}(?P<x>[a-z][a-z -]*)$", re.I)
_ISA_Q = re.compile(rf"^(?:is|are)\s+(?P<s>.+?)\s+{_OPT_ARTICLE}(?P<x>[a-z][a-z -]*)$", re.I)
_ALL_Q = re.compile(rf"^(?:are|is)\s+(?:all|every)\s+(?P<x>[a-z][a-z -]*?)\s+{_OPT_ARTICLE}(?P<y>[a-z][a-z -]*)$", re.I)

_IF = re.compile(r"^if\s+(?P<p>.+?)\s*,?\s+then\s+(?P<q>.+)$|^if\s+(?P<p2>.+?)\s*,\s*(?P<q2>.+)$", re.I)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!;])\s+|\n+", normalize(text))
    out = []
    for p in parts:
        p = p.strip().rstrip(".;").strip()
        if p:
            out.append(p)
    return out


def _cat(phrase: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", phrase.casefold()) if w not in {"a", "an", "the"}]
    return " ".join(stem(w) for w in words)


def _ent(name: str) -> str:
    return name.strip().casefold()


def _clause(text: str) -> str:
    t = normalize(text).casefold().strip(" .?!")
    t = re.sub(r"^(?:then|that)\s+", "", t)
    t = re.sub(r"\b(?:does|do|did|will|would)\b", "", t)
    t = re.sub(r"\b(rains|rained|raining)\b", "rain", t)
    words = [stem(w) for w in re.findall(r"[a-z0-9]+", t) if w not in {"a", "an", "the", "it", "is", "are", "be", "was", "will"}]
    return " ".join(words)


class LogicOrgan:
    def respond(self, prompt: str) -> NativeAnswer | None:
        sentences = _sentences(prompt)
        if len(sentences) < 2 and not re.search(r"\b(?:if|all|every|than)\b", prompt, re.I):
            return None
        # A question may be inline: "If all cats are animals and Tom is a cat, is Tom an animal?"
        expanded: list[str] = []
        for s in sentences:
            m = re.match(r"^if\s+(.+?),\s*((?:is|are|who|which|what|does|do|can)\b.+)$", s, re.I)
            if m and re.search(r"\b(?:all|every|no)\b", m.group(1), re.I):
                expanded.extend(x.strip() for x in re.split(r"\s+and\s+", m.group(1)) if x.strip())
                expanded.append(m.group(2))
            else:
                expanded.append(s)
        question = next((s for s in reversed(expanded) if s.endswith("?") or re.match(r"^(?:is|are|who|which|what|does|do|can)\b", s, re.I)), None)
        if question is None:
            return None
        premises = [s for s in expanded if s is not question]
        q = question.rstrip("?").strip()
        return (
            self._comparison(premises, q)
            or self._syllogism(premises, q)
            or self._propositional(premises, q)
        )

    # ---------------- comparisons ----------------
    def _comparison(self, premises: list[str], q: str) -> NativeAnswer | None:
        edges: dict[str, set[str]] = defaultdict(set)  # greater -> lesser on scale
        names: dict[str, str] = {}
        scales: set[str] = set()
        steps: list[str] = []
        for p in premises:
            for clause in re.split(r",\s*(?:and\s+)?|\s+and\s+|\s+but\s+", p):
                m = _COMP_FACT.match(clause.strip())
                if not m:
                    continue
                scale, direction = _COMPARATIVES[m.group("rel").casefold()]
                a, b = _ent(m.group("a")), _ent(m.group("b"))
                names.setdefault(a, m.group("a").strip())
                names.setdefault(b, m.group("b").strip())
                scales.add(scale)
                hi, lo = (a, b) if direction > 0 else (b, a)
                edges[hi].add(lo)
                steps.append(f"Premise: {names[hi]} > {names[lo]} ({scale})")
        if not edges:
            return None
        if len(scales) > 1:
            # Merge "shorter" interpreted as length vs height when the other scale is present.
            pass

        def reachable(src: str, dst: str) -> list[str] | None:
            prev: dict[str, str | None] = {src: None}
            dq = deque([src])
            while dq:
                cur = dq.popleft()
                if cur == dst:
                    path = []
                    node: str | None = cur
                    while node is not None:
                        path.append(node)
                        node = prev[node]
                    return path[::-1]
                for nxt in edges.get(cur, ()):
                    if nxt not in prev:
                        prev[nxt] = cur
                        dq.append(nxt)
            return None

        # cycle check
        for node in list(edges):
            for nxt in edges[node]:
                if reachable(nxt, node):
                    return NativeAnswer(
                        kind="logic", answer="The statements contradict each other (they form a cycle), so no consistent answer exists.",
                        qualification="deductive_proof", confidence=0.9, verified=True, steps=tuple(steps),
                    )

        all_nodes = set(edges) | {x for v in edges.values() for x in v}
        m = _SUP_Q.match(q)
        if m:
            rel = m.group("rel").casefold()
            _, direction = _SUPERLATIVES[rel]
            if direction > 0:
                cands = [n for n in all_nodes if not any(n in edges[o] for o in all_nodes)]
            else:
                cands = [n for n in all_nodes if not edges.get(n)]
            if len(cands) == 1:
                winner = cands[0]
                others = [names[n] for n in sorted(all_nodes - {winner})]
                chain = self._chain(edges, names, winner, direction, all_nodes)
                answer = f"{names[winner]} is the {rel}." + (f" ({chain})" if chain else "")
                return NativeAnswer(kind="logic", answer=answer, qualification="deductive_proof", confidence=0.97,
                                    verified=True, steps=tuple(steps + [f"Only {names[winner]} has nobody {'above' if direction > 0 else 'below'} them among {', '.join(others)}."]))
            return NativeAnswer(
                kind="logic",
                answer=f"It can't be determined from the given statements — {', '.join(names[c] for c in sorted(cands))} could each be the {rel}.",
                qualification="deductive_proof", confidence=0.9, verified=True, steps=tuple(steps),
            )
        m = _COMP_Q.match(q)
        if m:
            scale, direction = _COMPARATIVES[m.group("rel").casefold()]
            a, b = _ent(m.group("a")), _ent(m.group("b"))
            if a not in all_nodes or b not in all_nodes:
                return None
            hi, lo = (a, b) if direction > 0 else (b, a)
            path = reachable(hi, lo)
            if path:
                return NativeAnswer(kind="logic", answer=f"Yes. {' > '.join(names[p] for p in path)}, so {m.group('a')} is {m.group('rel')} than {m.group('b')}.",
                                    qualification="deductive_proof", confidence=0.97, verified=True, steps=tuple(steps))
            back = reachable(lo, hi)
            if back:
                return NativeAnswer(kind="logic", answer=f"No. {' > '.join(names[p] for p in back)}, so {m.group('a')} is not {m.group('rel')} than {m.group('b')}.",
                                    qualification="deductive_proof", confidence=0.97, verified=True, steps=tuple(steps))
            return NativeAnswer(kind="logic", answer="It can't be determined from the given statements.",
                                qualification="deductive_proof", confidence=0.85, verified=True, steps=tuple(steps))
        return None

    @staticmethod
    def _chain(edges, names, winner, direction, nodes) -> str:
        # Produce a full ordering if the order is total.
        order: list[str] = []
        remaining = set(nodes)
        indeg = {n: 0 for n in nodes}
        for hi in edges:
            for lo in edges[hi]:
                indeg[lo] += 1
        while remaining:
            tops = [n for n in remaining if indeg[n] == 0]
            if len(tops) != 1:
                return ""
            top = tops[0]
            order.append(top)
            remaining.remove(top)
            for lo in edges.get(top, ()):
                indeg[lo] -= 1
        return "order: " + " > ".join(names[n] for n in order)

    # ---------------- syllogisms ----------------
    def _syllogism(self, premises: list[str], q: str) -> NativeAnswer | None:
        subset: dict[str, set[str]] = defaultdict(set)   # X -> {Y}: all X are Y
        disjoint: set[tuple[str, str]] = set()           # no X are Y
        member: dict[str, set[str]] = defaultdict(set)    # entity -> categories
        not_member: dict[str, set[str]] = defaultdict(set)
        display: dict[str, str] = {}
        steps: list[str] = []
        for p in premises:
            for clause in re.split(r"\s+and\s+|,\s*", p):
                c = clause.strip()
                if not c:
                    continue
                if (m := _ALL.match(c)):
                    x, y = _cat(m.group("x")), _cat(m.group("y"))
                    subset[x].add(y)
                    steps.append(f"All {m.group('x')} are {m.group('y')}.")
                elif (m := _NO.match(c)):
                    x, y = _cat(m.group("x")), _cat(m.group("y"))
                    disjoint.add((x, y))
                    disjoint.add((y, x))
                    steps.append(f"No {m.group('x')} are {m.group('y')}.")
                elif (m := _ISA.match(c)):
                    s = _ent(m.group("s"))
                    display[s] = m.group("s").strip()
                    if m.group("neg"):
                        not_member[s].add(_cat(m.group("x")))
                    else:
                        member[s].add(_cat(m.group("x")))
                    steps.append(c + ".")
        if not subset and not disjoint:
            return None

        def closure(cats: set[str]) -> dict[str, str | None]:
            seen: dict[str, str | None] = {c: None for c in cats}
            dq = deque(cats)
            while dq:
                cur = dq.popleft()
                for nxt in subset.get(cur, ()):
                    if nxt not in seen:
                        seen[nxt] = cur
                        dq.append(nxt)
            return seen

        def path_to(seen: dict[str, str | None], target: str) -> list[str]:
            out = []
            node: str | None = target
            while node is not None:
                out.append(node)
                node = seen[node]
            return out[::-1]

        if (m := _ISA_Q.match(q)):
            s, target = _ent(m.group("s")), _cat(m.group("x"))
            if s not in member and s not in not_member:
                return None
            seen = closure(member.get(s, set()))
            name = display.get(s, m.group("s"))
            if target in seen:
                chain = " → ".join(path_to(seen, target))
                return NativeAnswer(kind="logic", answer=f"Yes — {name} is {self._article(m.group('x'))}{m.group('x')}. ({name}: {chain})",
                                    qualification="deductive_proof", confidence=0.97, verified=True, steps=tuple(steps))
            for cat in seen:
                if (cat, target) in disjoint:
                    return NativeAnswer(kind="logic", answer=f"No — {name} is not {self._article(m.group('x'))}{m.group('x')}, because {name} is in '{cat}' and no {cat} is {target}.",
                                        qualification="deductive_proof", confidence=0.95, verified=True, steps=tuple(steps))
            if target in not_member.get(s, set()):
                return NativeAnswer(kind="logic", answer=f"No — the premises say {name} is not {m.group('x')}.",
                                    qualification="deductive_proof", confidence=0.95, verified=True, steps=tuple(steps))
            # Modus tollens: all target are Y and s is not Y -> s is not target
            return NativeAnswer(kind="logic", answer=f"It can't be determined from the given statements whether {name} is {self._article(m.group('x'))}{m.group('x')}.",
                                qualification="deductive_proof", confidence=0.85, verified=True, steps=tuple(steps))
        if (m := _ALL_Q.match(q)):
            x, y = _cat(m.group("x")), _cat(m.group("y"))
            seen = closure({x})
            if y in seen:
                return NativeAnswer(kind="logic", answer=f"Yes — all {m.group('x')} are {m.group('y')} ({' → '.join(path_to(seen, y))}).",
                                    qualification="deductive_proof", confidence=0.95, verified=True, steps=tuple(steps))
            return NativeAnswer(kind="logic", answer="That doesn't follow from the given statements.",
                                qualification="deductive_proof", confidence=0.85, verified=True, steps=tuple(steps))
        return None

    @staticmethod
    def _article(word: str) -> str:
        w = word.strip().casefold()
        if w.endswith("s") and not w.endswith("ss"):
            return ""
        return "an " if w[:1] in "aeiou" else "a "

    # ---------------- propositional ----------------
    def _propositional(self, premises: list[str], q: str) -> NativeAnswer | None:
        rules: list[tuple[str, str, str, str]] = []
        facts: set[str] = set()
        neg_facts: set[str] = set()
        for p in premises:
            m = _IF.match(p)
            if m:
                ptxt = m.group("p") or m.group("p2")
                qtxt = m.group("q") or m.group("q2")
                rules.append((_clause(ptxt), _clause(qtxt), ptxt, qtxt))
                continue
            c = _clause(p)
            if re.search(r"\b(?:not|n't|never)\b", p, re.I):
                neg_facts.add(_clause(re.sub(r"\b(?:not|never)\b|n't", "", p, flags=re.I)))
            elif c:
                facts.add(c)
        if not rules:
            return None
        target_txt = re.sub(r"^(?:is|are|does|do|will|did|can)\s+", "", q, flags=re.I)
        target = _clause(target_txt)
        # forward chaining
        derived = set(facts)
        changed = True
        trail: list[str] = []
        while changed:
            changed = False
            for pc, qc, pt, qt in rules:
                if pc in derived and qc not in derived:
                    derived.add(qc)
                    trail.append(f"Since '{pt}', it follows that '{qt}'.")
                    changed = True
        if target in derived:
            return NativeAnswer(kind="logic", answer="Yes. " + " ".join(trail), qualification="deductive_proof",
                                confidence=0.95, verified=True, steps=tuple(trail))
        # modus tollens: if P then Q, not Q -> not P
        for pc, qc, pt, qt in rules:
            if pc == target and qc in neg_facts:
                return NativeAnswer(kind="logic", answer=f"No. If '{pt}' were true, then '{qt}' would be true — but it is not.",
                                    qualification="deductive_proof", confidence=0.93, verified=True)
            if qc == target and pc not in facts and target not in facts:
                return NativeAnswer(kind="logic",
                                    answer=f"It can't be determined. Knowing '{qt}' would require knowing '{pt}' (affirming the consequent or denying the antecedent is not valid).",
                                    qualification="deductive_proof", confidence=0.85, verified=True)
        return None
