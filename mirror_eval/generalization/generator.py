"""Procedural novel-concept generalization benchmark (NovelGen).

Every item is generated from a seed using *invented* words (nonce names,
categories, adjectives, operators and relations). Nothing in an item can be
memorized from a training corpus or matched by a topic-specific handler: the
only way to answer is to understand the definitions stated in the prompt and
reason over them.

Each family has a difficulty knob (``depth``). The split convention is

* ``id``  - in-distribution depths (1-2): what a system is typically tuned on;
* ``ood`` - out-of-distribution depths (3-6): longer chains / deeper nesting.

Every item is rendered in several surface forms (paraphrase templates, entity
order shuffles, irrelevant distractor sentences). A system that truly
understands the content answers all variants identically; surface pattern
matchers do not. Gold answers are computed programmatically, never by hand.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

GENERATOR_VERSION = "novelgen-1.0"

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "bl", "gr", "kr", "pl", "sn", "tr", "dr", "fl", "gl", "sk", "st", "zw")
_VOWELS = ("a", "e", "i", "o", "u", "ai", "ou", "ee", "oo")
_CODAS = ("", "", "b", "d", "k", "m", "n", "p", "rp", "sh", "t", "x", "z", "nk", "lt")

# Real words that must never be produced (would leak meaning or collide with grammar words).
_BLOCKLIST = frozenset("""
a an the is are was be not no yes all every each some any if then than and or of to in on at by for
as it its he she they we you me my his her who what which how many much more most less few has have
had does did do can will may might shall this that these those there here one two three four five six
seven eight nine ten zero bad bat bed bet bit bin big bud bun but dad den did dig dim din dip dot dub
fan fat fed fin fit fix fun gap gas gum gun kit lab lad lap led let lid lip lit log lot mad man map mat
men met mix mob mom mop mud mug nap net nod not nut pad pan pat pen pet pig pin pit pod pop pot pub pun
pup rag ram ran rap rat red rib rid rim rip rob rod rot rub rug run rut sad sat set sin sip sit sob son
sun tab tan tap ten tin tip ton top tub tug van vet zip zoo boot book cook cool deed doom door feed feel
food fool foot good keen keep kook look loop mood moon need noon peek peel pool poop reed reef roof room
root seed seek seem seen soon tool toot tree free flee glee knee bait bail fail gait laid maid mail nail
paid pail raid rail sail tail vain zest stun stub stop step star snip snap slip slot skin skip trap trip
trim drum drip drop flag flap flat flip flop glad glum grab grad grin grip plan plot plug plum
""".split())


def _syllable(rng: random.Random) -> str:
    return rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)


class NonceLexicon:
    """Draws unique invented words from a seeded RNG."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._used: set[str] = set()

    def word(self, syllables: int | None = None) -> str:
        for _ in range(10_000):
            n = syllables or self._rng.choice((1, 2, 2))
            w = "".join(_syllable(self._rng) for _ in range(n))
            if 3 <= len(w) <= 9 and w not in self._used and w not in _BLOCKLIST and not w.endswith(("s", "er", "est", "y")):
                self._used.add(w)
                return w
        raise RuntimeError("nonce lexicon exhausted")

    def name(self) -> str:
        return self.word().capitalize()

    def adjective(self) -> str:
        # Adjective stem whose comparative/superlative is regular: stem+"er" / stem+"est".
        return self.word(1)


@dataclass(frozen=True)
class Item:
    item_id: str
    family: str
    split: str
    depth: int
    answer_type: str  # "yesno" | "number" | "name" | "string"
    gold: str
    variants: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["variants"] = list(self.variants)
        return d

    def public(self) -> dict:
        """What a system under test may see (no gold, no meta)."""
        return {"item_id": self.item_id, "variants": list(self.variants)}


_DISTRACTORS = (
    "The {w} festival happens every spring.",
    "Nobody remembers who painted the old {w}.",
    "A {w} was mentioned in an unrelated story.",
    "{N} likes quiet mornings.",
    "The road to {N}'s village is long.",
)


def _distract(rng: random.Random, lex: NonceLexicon, k: int) -> list[str]:
    return [rng.choice(_DISTRACTORS).format(w=lex.word(), N=lex.name()) for _ in range(k)]


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


# ---------------------------------------------------------------- families

def gen_taxonomy(rng: random.Random, depth: int) -> tuple[str, str, list[str], dict]:
    """Category inheritance with invented categories; answers yes / no / unknown."""
    lex = NonceLexicon(rng)
    cats = [lex.word() for _ in range(depth + 1)]
    ent = lex.name()
    other = lex.word()
    premises_sets = []
    mode = rng.choice(("yes", "yes", "no", "unknown"))
    for tmpl in (("Every {x} is {y}.", "{e} is {x}."), ("All {xs} are {ys}.", "{e} is {x}."), ("Each {x} is also {y}.", "{e} is one of the {xs}.")):
        chain = []
        for i in range(depth):
            chain.append(tmpl[0].format(x=cats[i], y=_a(cats[i + 1]), xs=cats[i] + "s", ys=cats[i + 1] + "s"))
        if mode == "no":
            chain.append(f"No {cats[-1]} is {_a(other)}.")
        rng.shuffle(chain)
        chain.insert(rng.randrange(len(chain) + 1), tmpl[1].format(e=ent, x=_a(cats[0]), xs=cats[0] + "s"))
        premises_sets.append(chain)
    if mode == "yes":
        q, gold = f"Is {ent} {_a(cats[-1])}?", "yes"
    elif mode == "no":
        q, gold = f"Is {ent} {_a(other)}?", "no"
    else:
        # Reverse direction: knowing every A is B says nothing about whether a B-member is an A.
        ent2 = lex.name()
        for chain in premises_sets:
            chain.append(f"{ent2} is {_a(cats[-1])}.")
        q, gold = f"Is {ent2} {_a(cats[0])}?", "unknown"
    return "yesno", gold, [" ".join(p) + " " + q for p in premises_sets], {"mode": mode}


def gen_comparative(rng: random.Random, depth: int) -> tuple[str, str, list[str], dict]:
    """Transitive ordering over an invented adjective (glorp -> glorper -> glorpest)."""
    lex = NonceLexicon(rng)
    adj = lex.adjective()
    n = depth + 1
    names = [lex.name() for _ in range(n + 1)]  # names[0] is the top
    facts = [(names[i], names[i + 1]) for i in range(n)]
    ask = rng.choice(("top", "bottom", "pair_yes", "pair_no"))
    variants = []
    for style in range(3):
        sents = []
        for a, b in facts:
            if style == 0:
                sents.append(f"{a} is {adj}er than {b}.")
            elif style == 1:
                sents.append(f"{b} is less {adj} than {a}.")
            else:
                sents.append(f"{a} is more {adj} than {b}.")
        rng.shuffle(sents)
        variants.append(sents)
    if ask == "top":
        q, gold, atype = f"Who is the {adj}est?", names[0], "name"
    elif ask == "bottom":
        q, gold, atype = f"Who is the least {adj}?", names[-1], "name"
    else:
        i, j = sorted(rng.sample(range(n + 1), 2))
        if ask == "pair_yes":
            q, gold = f"Is {names[i]} {adj}er than {names[j]}?", "yes"
        else:
            q, gold = f"Is {names[j]} {adj}er than {names[i]}?", "no"
        atype = "yesno"
    return atype, gold, [" ".join(v) + " " + q for v in variants], {"adjective": adj, "ask": ask}


_OPS = (
    ("{a} + {b}", lambda a, b: a + b),
    ("{a} * {b}", lambda a, b: a * b),
    ("2 * {a} + {b}", lambda a, b: 2 * a + b),
    ("{a} * {b} - 1", lambda a, b: a * b - 1),
    ("{a} + 3 * {b}", lambda a, b: a + 3 * b),
    ("{a} - {b}", lambda a, b: a - b),
)


def gen_defined_function(rng: random.Random, depth: int) -> tuple[str, str, list[str], dict]:
    """An invented binary function defined in the prompt, applied with nesting ``depth``."""
    lex = NonceLexicon(rng)
    fname = lex.word()
    body, fn = rng.choice(_OPS)
    p, q = rng.sample(("x", "y", "m", "n", "p", "q"), 2)
    body_txt = body.format(a=p, b=q)

    def build(d: int) -> tuple[str, int]:
        a = rng.randint(1, 5)
        if d <= 1:
            b = rng.randint(1, 5)
            return f"{fname}({a}, {b})", fn(a, b)
        inner_s, inner_v = build(d - 1)
        if rng.random() < 0.5:
            return f"{fname}({a}, {inner_s})", fn(a, inner_v)
        return f"{fname}({inner_s}, {a})", fn(inner_v, a)

    expr, val = build(depth)
    variants = [
        f"Define {fname}({p}, {q}) = {body_txt}. What is {expr}?",
        f"Let {fname}({p}, {q}) be {body_txt}. Compute {expr}.",
        f"Suppose the function {fname} is given by {fname}({p}, {q}) = {body_txt}. Evaluate {expr}.",
    ]
    return "number", str(val), variants, {"function": fname, "body": body_txt}


def gen_entity_tracking(rng: random.Random, depth: int) -> tuple[str, str, list[str], dict]:
    """Track invented-object counts across ``depth`` transfer events."""
    lex = NonceLexicon(rng)
    obj = lex.word()
    people = [lex.name() for _ in range(3)]
    counts = {p: rng.randint(3, 9) for p in people}
    events = []
    for _ in range(depth):
        kind = rng.choice(("give", "find", "lose"))
        a = rng.choice(people)
        if kind == "give":
            b = rng.choice([p for p in people if p != a])
            if counts[a] == 0:
                kind = "find"
            else:
                k = rng.randint(1, counts[a])
                counts[a] -= k
                counts[b] += k
                events.append(("give", a, b, k))
                continue
        if kind == "find":
            k = rng.randint(1, 4)
            counts[a] += k
            events.append(("find", a, None, k))
        else:
            if counts[a] == 0:
                k = rng.randint(1, 4)
                counts[a] += k
                events.append(("find", a, None, k))
            else:
                k = rng.randint(1, counts[a])
                counts[a] -= k
                events.append(("lose", a, None, k))
    target = rng.choice(people)
    variants = []
    for style in range(3):
        start = [f"{p} has {c} {obj}s." if style != 1 else f"{p} starts with {c} {obj}s." for p, c in zip(people, [counts_init for counts_init in _initial(people, events, counts)])]
        evs = []
        for kind, a, b, k in events:
            if kind == "give":
                evs.append(f"{a} gives {k} {obj}s to {b}." if style != 2 else f"{b} receives {k} {obj}s from {a}.")
            elif kind == "find":
                evs.append(f"{a} finds {k} more {obj}s." if style != 2 else f"{a} gains {k} {obj}s.")
            else:
                evs.append(f"{a} loses {k} {obj}s." if style != 2 else f"{a} drops {k} {obj}s and they are gone.")
        q = f"How many {obj}s does {target} have now?"
        variants.append(" ".join(start + evs) + " " + q)
    return "number", str(counts[target]), variants, {"object": obj}


def _initial(people: list[str], events: list, final: dict[str, int]) -> list[int]:
    cur = dict(final)
    for kind, a, b, k in reversed(events):
        if kind == "give":
            cur[a] += k
            cur[b] -= k
        elif kind == "find":
            cur[a] -= k
        else:
            cur[a] += k
    return [cur[p] for p in people]


_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "reverse": lambda s: s[::-1],
    "double": lambda s: s + s,
    "drop_first": lambda s: s[1:],
    "drop_last": lambda s: s[:-1],
    "rotate_left": lambda s: s[1:] + s[:1],
    "upper": lambda s: s.upper(),
    "shift1": lambda s: "".join(chr((ord(c) - 97 + 1) % 26 + 97) if c.islower() else c for c in s),
}


def _all_programs(max_len: int = 3) -> list[tuple[tuple[str, ...], str]]:
    import itertools
    progs = []
    for n in range(1, max_len + 1):
        for combo in itertools.product(_TRANSFORMS, repeat=n):
            for suf in _SUFFIXES:
                progs.append((combo, suf))
    return progs


_SUFFIXES = ("", "z", "ka")
_PROGRAM_SPACE: list[tuple[tuple[str, ...], str]] | None = None


def _run_program(prog: tuple[str, ...], suffix: str, s: str) -> str:
    for st in prog:
        s = _TRANSFORMS[st](s)
    return s + suffix


def gen_string_induction(rng: random.Random, depth: int) -> tuple[str, str, list[str], dict]:
    """Induce an unnamed string program from examples; ``depth`` = number of composed steps.

    Items are rejected unless every program in the (transform^<=3 x suffix) space that
    fits the examples produces the same output on the test word, so gold is unique.
    """
    global _PROGRAM_SPACE
    if _PROGRAM_SPACE is None:
        _PROGRAM_SPACE = _all_programs()
    lex = NonceLexicon(rng)
    names = list(_TRANSFORMS)
    for _attempt in range(200):
        steps = tuple(rng.choice(names) for _ in range(min(depth, 3)))
        suffix = rng.choice(("", "", "z", "ka")) if depth >= 2 else ""
        shots = max(3, depth + 2)
        words = [lex.word(1) for _ in range(shots + 1)]
        examples = [(w, _run_program(steps, suffix, w)) for w in words[:-1]]
        test = words[-1]
        gold = _run_program(steps, suffix, test)
        consistent = {
            _run_program(p, s, test)
            for p, s in _PROGRAM_SPACE
            if all(_run_program(p, s, a) == b for a, b in examples)
        }
        if consistent == {gold}:
            break
    else:  # pragma: no cover - extremely unlikely
        raise RuntimeError("could not generate an unambiguous string-induction item")
    verb = lex.word()
    v1 = " ".join(f"{a} -> {b}." for a, b in examples) + f" Following the same rule, what does {test} become?"
    v2 = f"To {verb} a word: " + "; ".join(f"{verb}({a}) = {b}" for a, b in examples) + f". What is {verb}({test})?"
    v3 = "Examples: " + ", ".join(f"'{a}' becomes '{b}'" for a, b in examples) + f". Apply the same transformation to '{test}'."
    return "string", gold, [v1, v2, v3], {"program": list(steps), "suffix": suffix}


def gen_relation_composition(rng: random.Random, depth: int) -> tuple[str, str, list[str], dict]:
    """An invented relation defined as the ``depth``-fold composition of a given relation."""
    lex = NonceLexicon(rng)
    base = lex.word()
    derived = lex.word()
    n = depth + 2
    people = [lex.name() for _ in range(n + 2)]
    facts = [(people[i], people[i + 1]) for i in range(len(people) - 1)]  # people[i] is the {base} of people[i+1]
    k = depth + 1
    start = rng.randrange(k, len(people))
    gold = people[start - k]
    word_k = {2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times", 7: "seven times"}[k]
    variants = []
    for style in range(3):
        sents = [f"{a} is the {base} of {b}." if style != 1 else f"The {base} of {b} is {a}." for a, b in facts]
        rng.shuffle(sents)
        if style == 2:
            definition = f"Applying '{base} of' {word_k} gives the {derived}."
        else:
            definition = f"The {derived} of someone is their {base} applied {word_k} ({' of '.join([base] * k)} of them)."
        variants.append(" ".join(sents) + " " + definition + f" Who is the {derived} of {people[start]}?")
    return "name", gold, variants, {"base": base, "derived": derived, "k": k}


FAMILIES: dict[str, Callable[[random.Random, int], tuple[str, str, list[str], dict]]] = {
    "taxonomy": gen_taxonomy,
    "comparative": gen_comparative,
    "defined_function": gen_defined_function,
    "entity_tracking": gen_entity_tracking,
    "string_induction": gen_string_induction,
    "relation_composition": gen_relation_composition,
}

ID_DEPTHS = (1, 2)
OOD_DEPTHS = (3, 4, 5, 6)


def generate(seed: int | str, per_cell: int = 5, families: Iterable[str] | None = None, distractors: bool = True) -> list[Item]:
    """Generate the benchmark deterministically from ``seed``.

    ``per_cell`` items are produced for every (family, depth) cell.
    """
    seed_bytes = str(seed).encode()
    items: list[Item] = []
    for fam in families or FAMILIES:
        gen = FAMILIES[fam]
        for depth in ID_DEPTHS + OOD_DEPTHS:
            for i in range(per_cell):
                sub = int.from_bytes(hashlib.sha256(seed_bytes + f"|{fam}|{depth}|{i}".encode()).digest()[:8], "big")
                rng = random.Random(sub)
                atype, gold, variants, meta = gen(rng, depth)
                if distractors and len(variants) > 1:
                    # Variant 0 stays clean; variant 1 is prefixed with irrelevant sentences.
                    lex = NonceLexicon(rng)
                    variants = list(variants)
                    variants[1] = " ".join(_distract(rng, lex, 1 + depth // 3)) + " " + variants[1]
                item_id = hashlib.sha256(f"{GENERATOR_VERSION}|{seed}|{fam}|{depth}|{i}".encode()).hexdigest()[:16]
                items.append(Item(
                    item_id=item_id,
                    family=fam,
                    split="id" if depth in ID_DEPTHS else "ood",
                    depth=depth,
                    answer_type=atype,
                    gold=gold,
                    variants=tuple(v.strip() for v in variants),
                    meta=meta,
                ))
    return items


def items_digest(items: list[Item]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps(it.to_dict(), sort_keys=True).encode())
    return h.hexdigest()
