"""Stage 11.7: Adversarial input and static-policy checks.

IMPORTANT: AST inspection is not a Python execution sandbox. This module does not
execute submitted code. Untrusted code must run in a separate OS/container boundary
with resource limits, no ambient credentials, and a restricted filesystem/network.
"""
from __future__ import annotations

import ast
import base64
import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Optional, Tuple


class PromptInjectionDefenseGateway:
    """Heuristic detector; a clean result is not proof that a prompt is safe."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|rules|prompts)", re.I),
        re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts)", re.I),
        re.compile(r"you\s+are\s+now\s+(?:an?\s+)?(?:dan|unrestricted|jailbroken|evil)", re.I),
        re.compile(r"(?:print|output|reveal|dump|leak)\s+(?:your\s+)?(?:system\s+prompt|initial\s+instructions|secret\s+key)", re.I),
        re.compile(r"repeat\s+the\s+words\s+above\s+verbatim", re.I),
        re.compile(r"###\s*(?:system|instruction|admin)\s*:", re.I),
    ]
    BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
    MAX_PROMPT_CHARS = 200_000
    MAX_DECODED_CHARS = 32_768

    @classmethod
    def scan_prompt(cls, prompt_text: str) -> Tuple[bool, Optional[str]]:
        if not isinstance(prompt_text, str):
            return False, "Prompt must be text"
        if len(prompt_text) > cls.MAX_PROMPT_CHARS:
            return False, "Prompt exceeds inspection size limit"
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(prompt_text):
                return False, "Prompt injection heuristic matched"
        for match in cls.BASE64_PATTERN.finditer(prompt_text):
            try:
                decoded_bytes = base64.b64decode(match.group(0), validate=True)
                if len(decoded_bytes) > cls.MAX_DECODED_CHARS:
                    continue
                decoded = decoded_bytes.decode("utf-8", errors="strict")
            except (ValueError, UnicodeDecodeError):
                continue
            for pattern in cls.INJECTION_PATTERNS:
                if pattern.search(decoded):
                    return False, "Obfuscated base64 prompt injection heuristic matched"
        return True, None


class HardenedProgramSandbox:
    """Static AST policy checker only; never treats AST approval as safe execution."""

    FORBIDDEN_CALLS = {
        "eval", "exec", "open", "compile", "__import__", "globals", "locals",
        "vars", "breakpoint", "getattr", "setattr", "delattr", "input",
    }
    FORBIDDEN_MODULES = {
        "os", "sys", "subprocess", "shutil", "socket", "pty", "posix",
        "resource", "ctypes", "pathlib", "importlib", "pickle", "builtins",
    }

    @classmethod
    def is_ast_safe(cls, code_snippet: str) -> Tuple[bool, str]:
        """Reject obvious dangerous syntax; success is not an execution guarantee."""
        if not isinstance(code_snippet, str):
            return False, "Program must be text"
        if len(code_snippet) > 100_000:
            return False, "Program exceeds static inspection size limit"
        try:
            tree = ast.parse(code_snippet)
        except (SyntaxError, ValueError, MemoryError, RecursionError) as err:
            return False, f"Invalid or excessively complex syntax: {err}"

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for name in names:
                    if name.split(".")[0] in cls.FORBIDDEN_MODULES:
                        return False, f"Forbidden import: '{name}'"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in cls.FORBIDDEN_CALLS:
                    return False, f"Forbidden builtin function call: '{node.func.id}'"
                if isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen", "spawn", "exec", "eval"}:
                    return False, f"Forbidden method call: '{node.func.attr}'"
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return False, f"Forbidden dunder attribute access: '{node.attr}'"
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                return False, f"Forbidden dunder name: '{node.id}'"
        return True, "No prohibited AST patterns found; execution remains untrusted"


class ResourceExhaustionGuard:
    """Heuristic checks only; not a process-level CPU or memory quota."""

    REDOS_PATTERN = re.compile(r"(\([^\)]*[+*][^\)]*\)[+*])|([+*]{2,})")

    @classmethod
    def check_regex_safety(cls, pattern_str: str) -> Tuple[bool, Optional[str]]:
        if not isinstance(pattern_str, str) or len(pattern_str) > 10_000:
            return False, "Regex must be text within the inspection size limit"
        if cls.REDOS_PATTERN.search(pattern_str):
            return False, "Possible ReDoS risk: nested quantifier heuristic matched"
        try:
            re.compile(pattern_str)
        except re.error as err:
            return False, f"Invalid regex pattern: {err}"
        return True, None

    @classmethod
    def execute_bounded_loop(cls, max_steps: int = 1000) -> bool:
        """Demonstrates a bounded internal loop; does not constrain external code."""
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        for _ in range(max_steps + 1):
            if _ >= max_steps:
                return False
        return True


class MalformedInputSanitizer:
    """Bounds text/JSON inspection before expensive parsing or recursive traversal."""

    MAX_JSON_CHARS = 1_000_000

    @classmethod
    def sanitize_text(cls, text: str) -> Tuple[bool, str, Optional[str]]:
        if not isinstance(text, str):
            return False, "", "Input must be text"
        if "\x00" in text:
            return False, "", "Null byte detected in input payload"
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as err:
            return False, "", f"Malformed Unicode surrogates: {err}"
        cleaned = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or ord(ch) >= 32)
        return True, cleaned, None

    @classmethod
    def validate_json_depth(cls, json_str: str, max_depth: int = 20) -> Tuple[bool, Optional[str]]:
        if not isinstance(json_str, str):
            return False, "JSON input must be text"
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
            return False, "max_depth must be a positive integer"
        if len(json_str) > cls.MAX_JSON_CHARS:
            return False, "JSON input exceeds size limit"

        # Count structural nesting before json.loads; ignore braces inside strings.
        depth = 0
        in_string = False
        escaped = False
        for char in json_str:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
                if depth > max_depth:
                    return False, f"JSON nesting depth exceeds safety threshold ({depth} > {max_depth})"
            elif char in "]}":
                depth -= 1
                if depth < 0:
                    return False, "Invalid JSON structure"
        if in_string or depth != 0:
            return False, "Invalid JSON structure"
        try:
            json.loads(json_str)
        except (ValueError, RecursionError, MemoryError) as err:
            return False, f"Invalid or excessively complex JSON: {err}"
        return True, None


class ToolMisuseGuard:
    """Validates workspace containment and rejects shell metacharacters by policy."""

    SHELL_INJECTION_PATTERN = re.compile(r"[;&|\x60$><]")

    @classmethod
    def validate_file_path(cls, path_str: str, allowed_base_dir: str = ".") -> Tuple[bool, Optional[str]]:
        if not isinstance(path_str, str) or not path_str or "\x00" in path_str:
            return False, "Invalid empty or malformed path"
        # Reject Windows drive/UNC paths even when running on POSIX.
        win_path = PureWindowsPath(path_str)
        if win_path.drive or path_str.startswith(("\\\\", "//")):
            return False, "Absolute or network paths are not allowed"
        try:
            base = Path(allowed_base_dir).resolve()
            candidate = Path(path_str)
            if not candidate.is_absolute():
                candidate = base / candidate
            resolved = candidate.resolve()
            if not resolved.is_relative_to(base):
                return False, f"Path escapes allowed workspace: '{path_str}'"
        except (OSError, RuntimeError, ValueError) as err:
            return False, f"Path could not be safely resolved: {err}"
        return True, None

    @classmethod
    def validate_command_argument(cls, arg_str: str) -> Tuple[bool, Optional[str]]:
        if not isinstance(arg_str, str):
            return False, "Command argument must be text"
        if cls.SHELL_INJECTION_PATTERN.search(arg_str):
            return False, "Disallowed shell metacharacter in command argument"
        return True, None
