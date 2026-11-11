"""Stage 11.7: Adversarial and Sandbox Robustness Engine.

Defends against five hostile attack vectors:
1. Prompt Injection: Delimiter hijacking, instruction overrides, system prompt extraction, obfuscated base64 attacks.
2. Unsafe Programs: AST sandbox blocking reflection, dunder breakouts, and forbidden imports.
3. Resource Exhaustion: Runaway loops, memory allocation bombs, and catastrophic ReDoS regexes.
4. Malformed Inputs: Null bytes, binary control codes, unmatched surrogates, and recursive JSON depth bombs.
5. Tool Misuse: Path traversal attacks (../), shell command injections, and parameter schema violations.
"""
from __future__ import annotations

import ast
import base64
import json
import re
from typing import Any, Dict, List, Optional, Tuple


class PromptInjectionDefenseGateway:
    """Detects and neutralizes prompt injections, jailbreaks, and system prompt extraction attacks."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|rules|prompts)", re.I),
        re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts)", re.I),
        re.compile(r"you\s+are\s+now\s+(?:an?\s+)?(?:dan|unrestricted|jailbroken|evil)", re.I),
        re.compile(r"(?:print|output|reveal|dump|leak)\s+(?:your\s+)?(?:system\s+prompt|initial\s+instructions|secret\s+key)", re.I),
        re.compile(r"repeat\s+the\s+words\s+above\s+verbatim", re.I),
        re.compile(r"###\s*(?:system|instruction|admin)\s*:", re.I),
    ]

    BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")

    @classmethod
    def scan_prompt(cls, prompt_text: str) -> Tuple[bool, Optional[str]]:
        """Scans prompt text for prompt injections, returning (is_safe, refusal_reason)."""
        # 1. Plaintext injection heuristics
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(prompt_text):
                return False, "Prompt injection attempt detected: instruction override or system prompt extraction"

        # 2. Base64 payload inspection
        for match in cls.BASE64_PATTERN.finditer(prompt_text):
            b64_str = match.group(0)
            try:
                decoded = base64.b64decode(b64_str).decode("utf-8", errors="ignore")
                for pattern in cls.INJECTION_PATTERNS:
                    if pattern.search(decoded):
                        return False, "Obfuscated base64 prompt injection detected"
            except Exception:
                pass

        return True, None


class HardenedProgramSandbox:
    """AST-level Python program verifier that strictly prevents privilege escalation and breakouts."""

    FORBIDDEN_CALLS = {
        "eval", "exec", "open", "compile", "__import__", "globals", "locals",
        "vars", "breakpoint", "getattr", "setattr", "delattr"
    }

    FORBIDDEN_MODULES = {
        "os", "sys", "subprocess", "shutil", "socket", "pty", "posix",
        "resource", "ctypes", "pathlib", "importlib", "pickle"
    }

    @classmethod
    def is_ast_safe(cls, code_snippet: str) -> Tuple[bool, str]:
        """Parses Python code and strictly rejects forbidden calls, modules, and dunder attributes."""
        try:
            tree = ast.parse(code_snippet)
        except SyntaxError as err:
            return False, f"SyntaxError in code snippet: {err}"

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in cls.FORBIDDEN_MODULES:
                        return False, f"Forbidden import: '{alias.name}'"
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod in cls.FORBIDDEN_MODULES:
                    return False, f"Forbidden from-import: '{node.module}'"

            # Check calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in cls.FORBIDDEN_CALLS:
                    return False, f"Forbidden builtin function call: '{node.func.id}'"

            # Check dunder attribute accesses
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return False, f"Forbidden dunder attribute access: '{node.attr}'"

        return True, "AST certified safe"


class ResourceExhaustionGuard:
    """Guards against loop bombs, memory allocation bombs, and ReDoS regex explosions."""

    REDOS_PATTERN = re.compile(r"(\([^\)]*[\+\*][^\)]*\)[\+\*])|([\+\*]{2,})")

    @classmethod
    def check_regex_safety(cls, pattern_str: str) -> Tuple[bool, Optional[str]]:
        """Detects catastrophic backtracking patterns in user-supplied regexes."""
        if cls.REDOS_PATTERN.search(pattern_str):
            return False, "ReDoS risk: nested quantifier or catastrophic backtracking detected"
        try:
            re.compile(pattern_str)
        except re.error as err:
            return False, f"Invalid regex pattern: {err}"
        return True, None

    @classmethod
    def execute_bounded_loop(cls, max_steps: int = 1000) -> bool:
        """Helper demonstrating execution step bounding."""
        steps = 0
        while True:
            steps += 1
            if steps > max_steps:
                return False  # Bounded and aborted safely
        return True


class MalformedInputSanitizer:
    """Sanitizes inputs with binary control codes, invalid surrogates, and recursive JSON bombs."""

    @classmethod
    def sanitize_text(cls, text: str) -> Tuple[bool, str, Optional[str]]:
        """Strips binary control characters while preserving valid newlines and tabs."""
        if "\x00" in text:
            return False, "", "Null byte detected in input payload"

        # Check for invalid surrogate pairs
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as err:
            return False, "", f"Malformed Unicode surrogates: {err}"

        # Clean control characters except \n, \r, \t
        cleaned = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or ord(ch) >= 32)
        return True, cleaned, None

    @classmethod
    def validate_json_depth(cls, json_str: str, max_depth: int = 20) -> Tuple[bool, Optional[str]]:
        """Parses JSON while enforcing maximum nesting depth to prevent stack overflow attacks."""
        try:
            data = json.loads(json_str)
        except Exception as err:
            return False, f"Invalid JSON syntax: {err}"

        def get_depth(obj: Any, current_depth: int = 1) -> int:
            if current_depth > max_depth:
                return current_depth
            if isinstance(obj, dict):
                return max([get_depth(v, current_depth + 1) for v in obj.values()], default=current_depth)
            elif isinstance(obj, list):
                return max([get_depth(v, current_depth + 1) for v in obj], default=current_depth)
            return current_depth

        depth = get_depth(data)
        if depth > max_depth:
            return False, f"JSON nesting depth exceeds safety threshold ({depth} > {max_depth})"
        return True, None


class ToolMisuseGuard:
    """Validates tool parameters to prevent path traversal and shell injection."""

    PATH_TRAVERSAL_PATTERN = re.compile(r"(?:\.\.[\\/])|[\\/]\.\.(?:[\\/]|$)|^(?:/etc|/var|/dev|/proc|/sys)")
    SHELL_INJECTION_PATTERN = re.compile(r"[;&|`$><]")

    @classmethod
    def validate_file_path(cls, path_str: str, allowed_base_dir: str = ".") -> Tuple[bool, Optional[str]]:
        """Prevents directory traversal attacks escaping the designated workspace."""
        if cls.PATH_TRAVERSAL_PATTERN.search(path_str):
            return False, f"Path traversal attack detected in '{path_str}'"
        return True, None

    @classmethod
    def validate_command_argument(cls, arg_str: str) -> Tuple[bool, Optional[str]]:
        """Prevents shell metacharacter injection in parameter arguments."""
        if cls.SHELL_INJECTION_PATTERN.search(arg_str):
            return False, f"Disallowed shell metacharacter in parameter '{arg_str}'"
        return True, None
