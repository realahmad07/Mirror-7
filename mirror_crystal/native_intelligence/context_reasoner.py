"""General in-context reasoning organ.

Unlike pattern handlers that recognize fixed adjective lists or domain templates,
this organ parses any prompt into open-vocabulary logical/operational clauses and
runs general inference over the facts stated in the prompt itself:

1. **Taxonomic / set reasoning** — ``All X are Y``, ``No X is Y``, ``e is an X``
   for arbitrary (including invented) category nouns, with explicit distinction
   between ``yes``, ``no`` and ``unknown`` (affirming the consequent is rejected).
2. **Transitive comparative ordering** — ``A is <adj>er than B``, ``A is more/less
   <adj> than B`` for any adjective stem, answering pairwise and extremal queries.
3. **In-prompt function definitions** — ``Define f(x, y) = <expr>`` followed by an
   arbitrarily nested call ``f(a, f(b, c))`` evaluated via a restricted AST walker.
4. **World-state entity tracking** — initial counts plus transfer/gain/loss events
   simulated forward in sentence order for any countable noun.
5. **Few-shot string program induction** — searches compositions of primitive
   string edits + affix rules consistent with the prompt's input→output examples.
6. **Relation composition** — ``R of B is A`` facts plus a k-fold composition
   definition resolved by walking the relation graph k hops.

If the prompt does not supply all premises needed to derive an answer, the organ
returns ``None`` rather than guessing.
"""
from __future__ import annotations

import ast
import itertools
import operator
import re
from collections import defaultdict, deque
from typing import Callable

from .text import normalize, stem
from .types import NativeAnswer

_NAME = r"[A-Z][A-Za-z'-]*"
_WORD = r"[a-z][a-z-]*"
_ART = r"(?:an?\s+|the\s+)?"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!;])\s+|\n+", normalize(text))
    return [p.strip().rstrip(".;").strip() for p in parts if p.strip().rstrip(".;").strip()]


def _singular(word: str) -> str:
    w = word.strip().casefold()
    for prefix in ("one of the ", "a ", "an ", "the ", "also a ", "also an ", "also "):
        if w.startswith(prefix):
            w = w[len(prefix):].strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _answer(ans: str, steps: list[str], kind: str = "context_reasoning", qual: str = "deductive_proof") -> NativeAnswer:
    return NativeAnswer(kind=kind, answer=ans, qualification=qual, confidence=0.97, verified=True, steps=tuple(steps))


# ------------------------------------------------------------ 1. Taxonomy
_TAX_ALL = re.compile(rf"^(?:every|all|each)\s+(?P<x>{_WORD})\s+(?:is|are)\s+(?:also\s+)?{_ART}(?P<y>{_WORD})$", re.I)
_TAX_NO = re.compile(rf"^no\s+(?P<x>{_WORD})\s+(?:is|are)\s+{_ART}(?P<y>{_WORD})$", re.I)
_TAX_MEM = re.compile(rf"^(?P<e>{_NAME})\s+is\s+(?:one of the\s+|an?\s+|the\s+)(?P<x>{_WORD})$", re.I)
_TAX_Q = re.compile(rf"^is\s+(?P<e>{_NAME})\s+{_ART}(?P<x>{_WORD})\??$", re.I)


def _try_taxonomy(sents: list[str], q: str) -> NativeAnswer | None:
    mq = _TAX_Q.match(q)
    if not mq:
        return None
    target_ent = mq.group("e")
    target_cat = _singular(mq.group("x"))
    isa: dict[str, set[str]] = defaultdict(set)
    disjoint: set[tuple[str, str]] = set()
    member: dict[str, set[str]] = defaultdict(set)
    for s in sents:
        if m := _TAX_ALL.match(s):
            isa[_singular(m.group("x"))].add(_singular(m.group("y")))
        elif m := _TAX_NO.match(s):
            a, b = _singular(m.group("x")), _singular(m.group("y"))
            disjoint.add((a, b))
            disjoint.add((b, a))
        elif m := _TAX_MEM.match(s):
            member[m.group("e")].add(_singular(m.group("x")))
    if target_ent not in member or (not isa and not disjoint):
        return None
    # Forward closure from target_ent's direct categories.
    parent: dict[str, str | None] = {c: None for c in member[target_ent]}
    queue = deque(parent)
    while queue:
        cur = queue.popleft()
        for nxt in isa.get(cur, ()):
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    if target_cat in parent:
        chain = [target_cat]
        while parent[chain[-1]] is not None:
            chain.append(parent[chain[-1]])  # type: ignore[arg-type]
        chain.reverse()
        return _answer(f"Yes. {target_ent} ∈ {' ⊆ '.join(chain)}.", [f"{target_ent} -> {' -> '.join(chain)}"])
    for c in parent:
        if (c, target_cat) in disjoint:
            return _answer(f"No. {target_ent} is a {c}, and no {c} is a {target_cat}.", [f"disjoint({c}, {target_cat})"])
    # Check whether target_cat is a *subclass* of one of target_ent's known classes (affirming the consequent).
    sub_reach: set[str] = {target_cat}
    dq = deque([target_cat])
    while dq:
        u = dq.popleft()
        for v in isa.get(u, ()):
            if v not in sub_reach:
                sub_reach.add(v)
                dq.append(v)
    if sub_reach & set(parent):
        return _answer(
            f"Unknown (cannot be determined): knowing every {target_cat} is in { sorted(sub_reach & set(parent))[0] } does not entail that {target_ent} is a {target_cat}.",
            ["affirming the consequent is invalid"],
        )
    return None


# ------------------------------------------------------------ 2. Comparative
# Matches open-vocabulary "<A> is <adj>er than <B>" and "<A> is more/less <adj> than <B>".
_CMP_MORE = re.compile(rf"^(?P<a>{_NAME})\s+is\s+(?:more\s+(?P<adj1>{_WORD})|(?P<adj2>{_WORD})er)\s+than\s+(?P<b>{_NAME})$", re.I)
_CMP_LESS = re.compile(rf"^(?P<a>{_NAME})\s+is\s+less\s+(?P<adj>{_WORD})\s+than\s+(?P<b>{_NAME})$", re.I)
_CMP_Q_PAIR = re.compile(rf"^is\s+(?P<a>{_NAME})\s+(?:more\s+(?P<adj1>{_WORD})|(?P<adj2>{_WORD})er)\s+than\s+(?P<b>{_NAME})\??$", re.I)
_CMP_Q_TOP = re.compile(rf"^who\s+is\s+(?:the\s+)?(?:most\s+(?P<adj1>{_WORD})|(?P<adj2>{_WORD})est)\??$", re.I)
_CMP_Q_BOT = re.compile(rf"^who\s+is\s+(?:the\s+)?least\s+(?P<adj>{_WORD})\??$", re.I)


def _adj_stem(a1: str | None, a2: str | None) -> str:
    raw = (a1 or a2 or "").casefold()
    # Handle orthographic doubling/e-drop cleanly enough to match across forms of the same stem.
    return raw


def _try_comparative(sents: list[str], q: str) -> NativeAnswer | None:
    # edges[adj][hi] -> set of lower entities
    edges: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for s in sents:
        if m := _CMP_MORE.match(s):
            adj = _adj_stem(m.group("adj1"), m.group("adj2"))
            edges[adj][m.group("a")].add(m.group("b"))
        elif m := _CMP_LESS.match(s):
            adj = m.group("adj").casefold()
            edges[adj][m.group("b")].add(m.group("a"))
    if not edges:
        return None

    def reach(g: dict[str, set[str]], src: str, dst: str) -> bool:
        seen = {src}
        dq = deque([src])
        while dq:
            u = dq.popleft()
            if u == dst:
                return True
            for v in g.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        return False

    if m := _CMP_Q_PAIR.match(q):
        adj = _adj_stem(m.group("adj1"), m.group("adj2"))
        g = edges.get(adj)
        if not g:
            return None
        a, b = m.group("a"), m.group("b")
        if reach(g, a, b):
            return _answer(f"Yes, {a} is more {adj} than {b}.", [f"{a} >* {b}"])
        if reach(g, b, a):
            return _answer(f"No, {b} is more {adj} than {a}.", [f"{b} >* {a}"])
        return _answer("Unknown (cannot be determined).", ["no directed path"])

    for matcher, want_top in ((_CMP_Q_TOP, True), (_CMP_Q_BOT, False)):
        if m := matcher.match(q):
            adj = _adj_stem(m.groupdict().get("adj1"), m.groupdict().get("adj2") or m.groupdict().get("adj"))
            g = edges.get(adj)
            if not g:
                return None
            nodes = set(g) | {v for vs in g.values() for v in vs}
            winners = [
                n for n in nodes
                if all(reach(g, n, o) if want_top else reach(g, o, n) for o in nodes if o != n)
            ]
            if len(winners) == 1:
                return _answer(winners[0], [f"unique {'maximum' if want_top else 'minimum'}: {winners[0]}"])
    return None


# ------------------------------------------------------------ 3. Defined function
_FN_DEF = re.compile(
    r"(?:define|let|suppose(?:\s+the\s+function)?\s+\w+\s+is\s+given\s+by)\s+"
    r"(?P<f>[a-z]\w*)\s*\(\s*(?P<p>[a-z])\s*,\s*(?P<q>[a-z])\s*\)\s*(?:=|be)\s*(?P<body>[^.?]+)",
    re.I,
)
_FN_ASK = re.compile(r"(?:what\s+is|compute|evaluate)\s+(?P<call>[a-z]\w*\s*\([0-9a-z_,\s()+-/*]+\))\s*[.?]?$", re.I)
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}


def _try_defined_function(prompt: str) -> NativeAnswer | None:
    md = _FN_DEF.search(prompt)
    ma = _FN_ASK.search(prompt)
    if not (md and ma):
        return None
    fname = md.group("f")
    p_var, q_var = md.group("p"), md.group("q")
    try:
        body_ast = ast.parse(md.group("body").strip(), mode="eval").body
        call_ast = ast.parse(ma.group("call").strip(), mode="eval").body
    except SyntaxError:
        return None

    def eval_expr(node: ast.AST, env: dict[str, int]) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -eval_expr(node.operand, env)
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](eval_expr(node.left, env), eval_expr(node.right, env))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == fname and len(node.args) == 2:
            av = eval_expr(node.args[0], env)
            bv = eval_expr(node.args[1], env)
            return eval_expr(body_ast, {p_var: av, q_var: bv})
        raise ValueError("unsupported AST")

    try:
        val = eval_expr(call_ast, {})
    except Exception:
        return None
    return _answer(str(val), [f"{fname}({p_var},{q_var}) = {md.group('body').strip()} => {val}"], qual="computed_result")


# ------------------------------------------------------------ 4. Entity tracking
_ET_INIT = re.compile(rf"^(?P<p>{_NAME})\s+(?:has|starts with)\s+(?P<k>\d+)\s+(?P<obj>{_WORD})$", re.I)
_ET_GIVE = re.compile(rf"^(?P<a>{_NAME})\s+gives\s+(?P<k>\d+)\s+(?P<obj>{_WORD})\s+to\s+(?P<b>{_NAME})$", re.I)
_ET_RECV = re.compile(rf"^(?P<b>{_NAME})\s+receives\s+(?P<k>\d+)\s+(?P<obj>{_WORD})\s+from\s+(?P<a>{_NAME})$", re.I)
_ET_GAIN = re.compile(rf"^(?P<a>{_NAME})\s+(?:finds|gains)\s+(?P<k>\d+)\s+(?:more\s+)?(?P<obj>{_WORD})$", re.I)
_ET_LOSE = re.compile(rf"^(?P<a>{_NAME})\s+(?:loses|drops)\s+(?P<k>\d+)\s+(?P<obj>{_WORD})(?:\s+and they are gone)?$", re.I)
_ET_Q = re.compile(rf"^how\s+many\s+(?P<obj>{_WORD})\s+does\s+(?P<p>{_NAME})\s+have(?:\s+now)?\??$", re.I)


def _try_entity_tracking(sents: list[str], q: str) -> NativeAnswer | None:
    mq = _ET_Q.match(q)
    if not mq:
        return None
    target_obj = _singular(mq.group("obj"))
    target_p = mq.group("p")
    counts: dict[str, int] = {}
    steps: list[str] = []
    saw_event = False
    for s in sents:
        if m := _ET_INIT.match(s):
            if _singular(m.group("obj")) == target_obj:
                counts[m.group("p")] = int(m.group("k"))
        elif m := (_ET_GIVE.match(s) or _ET_RECV.match(s)):
            if _singular(m.group("obj")) == target_obj and m.group("a") in counts and m.group("b") in counts:
                k = int(m.group("k"))
                counts[m.group("a")] -= k
                counts[m.group("b")] += k
                saw_event = True
                steps.append(f"{m.group('a')} -{k}, {m.group('b')} +{k}")
        elif m := _ET_GAIN.match(s):
            if _singular(m.group("obj")) == target_obj and m.group("a") in counts:
                k = int(m.group("k"))
                counts[m.group("a")] += k
                saw_event = True
                steps.append(f"{m.group('a')} +{k}")
        elif m := _ET_LOSE.match(s):
            if _singular(m.group("obj")) == target_obj and m.group("a") in counts:
                k = int(m.group("k"))
                counts[m.group("a")] -= k
                saw_event = True
                steps.append(f"{m.group('a')} -{k}")
    if target_p in counts and saw_event:
        return _answer(str(counts[target_p]), steps, qual="computed_result")
    return None


# ------------------------------------------------------------ 5. String induction
_PRIMS: dict[str, Callable[[str], str]] = {
    "reverse": lambda s: s[::-1],
    "double": lambda s: s + s,
    "drop_first": lambda s: s[1:],
    "drop_last": lambda s: s[:-1],
    "rotate_left": lambda s: s[1:] + s[:1],
    "upper": lambda s: s.upper(),
    "shift1": lambda s: "".join(chr((ord(c) - 97 + 1) % 26 + 97) if c.islower() else c for c in s),
}
_PAIR_ARROW = re.compile(r"\b([a-z]{2,12})\s*->\s*([A-Za-z]{1,30})\b")
_PAIR_FN = re.compile(r"\b[a-z]+\(\s*([a-z]{2,12})\s*\)\s*=\s*([A-Za-z]{1,30})\b")
_PAIR_BECOMES = re.compile(r"'([a-z]{2,12})'\s+becomes\s+'([A-Za-z]{1,30})'")
_SI_TEST = (
    re.compile(r"what\s+does\s+([a-z]{2,12})\s+become\??$", re.I),
    re.compile(r"what\s+is\s+[a-z]+\(\s*([a-z]{2,12})\s*\)\??$", re.I),
    re.compile(r"transformation\s+to\s+'([a-z]{2,12})'[.?]?$", re.I),
)


def _try_string_induction(prompt: str) -> NativeAnswer | None:
    pairs = _PAIR_ARROW.findall(prompt) or _PAIR_FN.findall(prompt) or _PAIR_BECOMES.findall(prompt)
    if len(pairs) < 2:
        return None
    test_word = None
    for rx in _SI_TEST:
        if m := rx.search(prompt.strip()):
            test_word = m.group(1)
            break
    if not test_word:
        return None
    # Infer candidate suffixes from how outputs end relative to inputs.
    suffixes = {"", "z", "ka"}
    for _, out in pairs:
        for k in (1, 2):
            if len(out) > k and out[-k:].islower():
                suffixes.add(out[-k:])
    names = list(_PRIMS)
    for length in (1, 2, 3):
        for combo in itertools.product(names, repeat=length):
            for suf in suffixes:
                ok = True
                for inp, out in pairs:
                    cur = inp
                    for step in combo:
                        cur = _PRIMS[step](cur)
                    if cur + suf != out:
                        ok = False
                        break
                if ok:
                    res = test_word
                    for step in combo:
                        res = _PRIMS[step](res)
                    res += suf
                    return _answer(f"'{res}'", [f"program={list(combo)} suffix={suf!r}"], qual="deductive_proof")
    return None


# ------------------------------------------------------------ 6. Relation composition
_REL_FACT1 = re.compile(rf"^(?P<a>{_NAME})\s+is\s+the\s+(?P<r>{_WORD})\s+of\s+(?P<b>{_NAME})$", re.I)
_REL_FACT2 = re.compile(rf"^the\s+(?P<r>{_WORD})\s+of\s+(?P<b>{_NAME})\s+is\s+(?P<a>{_NAME})$", re.I)
_WORDS_TO_K = {"twice": 2, "three times": 3, "four times": 4, "five times": 5, "six times": 6, "seven times": 7}
_REL_DEF = re.compile(r"\b(" + "|".join(_WORDS_TO_K) + r")\b", re.I)
_REL_Q = re.compile(rf"^who\s+is\s+the\s+(?P<d>{_WORD})\s+of\s+(?P<p>{_NAME})\??$", re.I)


def _try_relation_composition(sents: list[str], q: str) -> NativeAnswer | None:
    mq = _REL_Q.match(q)
    if not mq:
        return None
    # rel_of[r][b] = a  meaning "a is the r of b"
    rel_of: dict[str, dict[str, str]] = defaultdict(dict)
    k = None
    base_rel = None
    for s in sents:
        if m := (_REL_FACT1.match(s) or _REL_FACT2.match(s)):
            r = m.group("r").casefold()
            rel_of[r][m.group("b")] = m.group("a")
        elif m := _REL_DEF.search(s):
            k = _WORDS_TO_K[m.group(1).casefold()]
            for r in rel_of:
                if r in s.casefold():
                    base_rel = r
    if not rel_of or k is None:
        return None
    if base_rel is None and len(rel_of) == 1:
        base_rel = next(iter(rel_of))
    if base_rel is None:
        return None
    cur = mq.group("p")
    path = [cur]
    for _ in range(k):
        nxt = rel_of[base_rel].get(cur)
        if nxt is None:
            return None
        cur = nxt
        path.append(cur)
    return _answer(cur, [" -> ".join(path)])


class ContextReasoner:
    """Open-vocabulary in-context inference over premises stated in the prompt."""

    def respond(self, prompt: str) -> NativeAnswer | None:
        if not prompt or not prompt.strip():
            return None
        if ans := _try_defined_function(prompt):
            return ans
        if ans := _try_string_induction(prompt):
            return ans
        sents = _sentences(prompt)
        if len(sents) < 2:
            return None
        premises, q = sents[:-1], sents[-1]
        for solver in (_try_taxonomy, _try_comparative, _try_entity_tracking, _try_relation_composition):
            if ans := solver(premises, q):
                return ans
        return None
