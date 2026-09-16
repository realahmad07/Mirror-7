"""Generic artifact-driven MIRR compiler.

The engine contains only schema mechanics: it does not name MIRR operations and
has no opcode-specific lowering table. Language semantics come from the artifact.
"""
import re

MAX_SOURCE_BYTES = 1_000_000
MAX_LINES = 100_000
MAX_INSTRUCTIONS = 100_000
_INT_RE = re.compile(r"^[+-]?\d+$")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_int(text):
    if not isinstance(text, str) or not _INT_RE.fullmatch(text):
        raise ValueError(f"invalid integer: {text!r}")
    return int(text, 10)


def _render(parts, rule, line_no):
    kind = rule.get("kind")
    out = rule.get("emit")
    if not isinstance(kind, str) or not isinstance(out, list) or not all(isinstance(x, str) for x in out):
        raise ValueError(f"line {line_no}: malformed compiler rule")
    argc = len(parts) - 1
    if kind == "NOARG":
        if argc != 0:
            raise ValueError(f"line {line_no}: wrong operand count")
    elif kind == "IMM":
        if argc != 1:
            raise ValueError(f"line {line_no}: expected one integer")
        _parse_int(parts[1])
    elif kind == "LET_IMM":
        if argc != 3 or parts[2] != "=":
            raise ValueError(f"line {line_no}: expected name = integer")
        if not _NAME_RE.fullmatch(parts[1]):
            raise ValueError(f"line {line_no}: invalid variable name")
        _parse_int(parts[3])
    else:
        raise ValueError(f"line {line_no}: unsupported rule kind")
    rendered = []
    for item in out:
        if item.startswith("${") and item.endswith("}"):
            key = item[2:-1]
            if key == "arg1":
                if len(parts) < 2:
                    raise ValueError(f"line {line_no}: missing operand")
                rendered.append(parts[1])
            elif key == "value":
                if len(parts) < 4:
                    raise ValueError(f"line {line_no}: missing value")
                rendered.append(parts[3])
            else:
                raise ValueError(f"line {line_no}: unknown placeholder {key!r}")
        else:
            rendered.append(item)
    return " ".join(rendered)


def validate_artifact(artifact):
    if not isinstance(artifact, dict):
        raise TypeError("artifact must be an object")
    if artifact.get("format") != "MIRROR_SOURCE_ARTIFACT_V1":
        raise ValueError("unsupported artifact format")
    rules = artifact.get("rules")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("artifact missing rules")
    for key, rule in rules.items():
        if not isinstance(key, str) or not key:
            raise ValueError("invalid rule name")
        if not isinstance(rule, dict):
            raise ValueError(f"invalid rule {key!r}")
        if rule.get("kind") not in ("NOARG", "IMM", "LET_IMM"):
            raise ValueError(f"invalid rule kind for {key!r}")
        if not isinstance(rule.get("emit"), list) or not rule["emit"] or not all(isinstance(x, str) for x in rule["emit"]):
            raise ValueError(f"invalid emission for {key!r}")
    default = artifact.get("default_terminator")
    if default is not None and (not isinstance(default, list) or not all(isinstance(x, str) for x in default)):
        raise ValueError("invalid default terminator")
    return True


def compile_from_artifact(artifact, src):
    validate_artifact(artifact)
    if not isinstance(src, str):
        raise TypeError("source must be str")
    if len(src.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds size limit")
    lines = src.splitlines()
    if len(lines) > MAX_LINES:
        raise ValueError("source exceeds line limit")
    out = []
    rules = artifact["rules"]
    for lineno, raw in enumerate(lines, 1):
        s = raw.split('#', 1)[0].strip()
        if not s:
            continue
        parts = s.split()
        key = parts[0].lower()
        rule = rules.get(key)
        if rule is None:
            raise ValueError(f"line {lineno}: unknown syntax: {key}")
        try:
            out.append(_render(parts, rule, lineno))
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        if len(out) > MAX_INSTRUCTIONS:
            raise ValueError("instruction limit exceeded")
    if not out:
        raise ValueError("source has no instructions")
    term = artifact.get("default_terminator")
    if term and not any(x.strip() == " ".join(term) for x in out):
        out.extend(term)
    return "\n".join(out)
