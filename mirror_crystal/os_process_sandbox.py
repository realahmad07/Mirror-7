"""Resource-limited child-process execution for untrusted Mirror code candidates.

This provides process isolation and OS resource limits on POSIX. It is not a container,
seccomp profile, network namespace, or guarantee against every Python sandbox escape.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
import tempfile
from typing import Any


@dataclass(frozen=True)
class SandboxExecutionResult:
    ok: bool
    output: str
    error: str
    pid: int | None
    returncode: int | None


_CHILD_RUNNER = r'''
import ast, bisect, builtins, cmath, collections, copy, csv, dataclasses, decimal
import enum, fractions, functools, heapq, itertools, json, math, operator, queue
import random, re, statistics, string, sys, io, contextlib, os, subprocess, time
payload = json.loads(sys.stdin.read())
source = payload.get("source")
candidate_source = payload.get("candidate_source")
stage19_mode = candidate_source is not None
if stage19_mode:
    if not isinstance(candidate_source, str):
        raise RuntimeError("Stage 19 candidate source must be text")
    sources_to_check = (candidate_source,)
else:
    if not isinstance(source, str):
        raise RuntimeError("source must be text")
    sources_to_check = (source,)
step_cap = max(100, int(payload.get("max_loop_steps", 50000)))
modules = {
    "math": math, "cmath": cmath, "decimal": decimal, "operator": operator,
    "queue": queue, "enum": enum, "heapq": heapq, "collections": collections,
    "bisect": bisect, "itertools": itertools, "functools": functools,
    "typing": __import__("typing"), "fractions": fractions, "statistics": statistics,
    "re": re, "json": json, "csv": csv, "string": string, "random": random,
    "copy": copy, "dataclasses": dataclasses,
}
allowed = set(modules)
if stage19_mode:
    allowed.difference_update({"typing", "dataclasses", "enum"})
for checked_source in sources_to_check:
    for node in ast.walk(ast.parse(checked_source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    raise RuntimeError("blocked import: " + alias.name)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in allowed:
                raise RuntimeError("blocked from-import: " + str(node.module))
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            raise RuntimeError("blocked private or mutating attribute: " + node.attr)
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in {"eval","exec","compile","open","__import__","input","globals","locals","vars","getattr","setattr","delattr","breakpoint","exit","quit"}:
                raise RuntimeError("blocked name: " + node.id)

steps = [0]
def tick():
    steps[0] += 1
    if steps[0] > step_cap:
        raise RuntimeError("LoopStepLimitExceeded (%d steps)" % step_cap)
def safe_range(*args):
    value = range(*args)
    if len(value) > step_cap:
        raise RuntimeError("RangeSpanLimitExceeded (%d > %d)" % (len(value), step_cap))
    return value
class Guard(ast.NodeTransformer):
    def inject(self, body, doc=False):
        stmt = ast.Expr(value=ast.Call(func=ast.Name(id="_mc_step_tick", ctx=ast.Load()), args=[], keywords=[]))
        if doc and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            return [body[0], stmt] + body[1:]
        return [stmt] + body
    def visit_While(self, node):
        self.generic_visit(node); node.body = self.inject(node.body); return node
    def visit_For(self, node):
        self.generic_visit(node); node.body = self.inject(node.body); return node
    def visit_FunctionDef(self, node):
        self.generic_visit(node); node.body = self.inject(node.body, True); return node
def safe_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in modules:
        raise ImportError("blocked import: " + name)
    return modules[root]
safe_builtins = {
    "abs":abs,"all":all,"any":any,"bool":bool,"callable":callable,"complex":complex,"dict":dict,"divmod":divmod,
    "enumerate":enumerate,"filter":filter,"float":float,"int":int,"isinstance":isinstance,
    "len":len,"list":list,"map":map,"max":max,"min":min,"pow":pow,"print":print,
    "range":safe_range,"reversed":reversed,"round":round,"set":set,"sorted":sorted,"str":str,
    "sum":sum,"tuple":tuple,"zip":zip,"Exception":Exception,"ValueError":ValueError,
    "TypeError":TypeError,"ZeroDivisionError":ZeroDivisionError,"IndexError":IndexError,
    "KeyError":KeyError,"RuntimeError":RuntimeError,"AssertionError":AssertionError,
    "object":object,"__build_class__":builtins.__build_class__,"__import__":safe_import,
}
_CANDIDATE_CHILD = r"""
import ast, bisect, builtins, cmath, collections, copy, csv, dataclasses, decimal
import enum, fractions, functools, heapq, itertools, json, math, operator, queue
import random, re, statistics, string, sys, types
try:
    data = json.loads(sys.stdin.read())
    candidate_source = data["candidate_source"]
    inputs = data["inputs"]
    step_cap = max(100, int(data.get("max_loop_steps", 50000)))
    modules = {
        "math": math, "cmath": cmath, "decimal": decimal, "operator": operator,
        "queue": queue, "enum": enum, "heapq": heapq, "collections": collections,
        "bisect": bisect, "itertools": itertools, "functools": functools,
        "fractions": fractions, "statistics": statistics,
        "re": re, "json": json, "csv": csv, "string": string, "random": random,
        "copy": copy,
    }
    allowed = set(modules)
    for node in ast.walk(ast.parse(candidate_source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    raise RuntimeError("blocked import: " + alias.name)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in allowed:
                raise RuntimeError("blocked from-import: " + str(node.module))
        elif isinstance(node, ast.Attribute) and (
                node.attr.startswith("_") or isinstance(node.ctx, (ast.Store, ast.Del))
            ):
            raise RuntimeError("blocked private or mutating attribute: " + node.attr)
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in {"eval","exec","compile","open","__import__","input","globals","locals","vars","getattr","setattr","delattr","breakpoint","exit","quit"}:
                raise RuntimeError("blocked name: " + node.id)
    steps = [0]
    def tick():
        steps[0] += 1
        if steps[0] > step_cap:
            raise RuntimeError("LoopStepLimitExceeded (%d steps)" % step_cap)
    def safe_range(*args):
        value = range(*args)
        if len(value) > step_cap:
            raise RuntimeError("RangeSpanLimitExceeded (%d > %d)" % (len(value), step_cap))
        return value
    class Guard(ast.NodeTransformer):
        def inject(self, body, doc=False):
            stmt = ast.Expr(value=ast.Call(func=ast.Name(id="_mc_step_tick", ctx=ast.Load()), args=[], keywords=[]))
            if doc and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                return [body[0], stmt] + body[1:]
            return [stmt] + body
        def visit_While(self, node):
            self.generic_visit(node); node.body = self.inject(node.body); return node
        def visit_For(self, node):
            self.generic_visit(node); node.body = self.inject(node.body); return node
        def visit_FunctionDef(self, node):
            self.generic_visit(node); node.body = self.inject(node.body, True); return node
    def build_builtins():
        cloned_modules = {}
        for module_name, module in modules.items():
            cloned = types.ModuleType(module.__name__)
            cloned.__dict__.update(module.__dict__)
            cloned_modules[module_name] = cloned
        def namespace_import(name, *args, **kwargs):
            root = name.split(".")[0]
            if root not in cloned_modules:
                raise ImportError("blocked import: " + name)
            return cloned_modules[root]
        return {
            "abs":abs,"all":all,"any":any,"bool":bool,"callable":callable,"complex":complex,"dict":dict,"divmod":divmod,
            "enumerate":enumerate,"filter":filter,"float":float,"int":int,"isinstance":isinstance,
            "len":len,"list":list,"map":map,"max":max,"min":min,"pow":pow,"print":print,
            "range":safe_range,"reversed":reversed,"round":round,"set":set,"sorted":sorted,"str":str,
            "sum":sum,"tuple":tuple,"zip":zip,"Exception":Exception,"ValueError":ValueError,
            "TypeError":TypeError,"ZeroDivisionError":ZeroDivisionError,"IndexError":IndexError,
            "KeyError":KeyError,"RuntimeError":RuntimeError,"AssertionError":AssertionError,
            "object":object,"__build_class__":builtins.__build_class__,"__import__":namespace_import,
        }
    tree = Guard().visit(ast.parse(candidate_source))
    ast.fix_missing_locations(tree)
    compiled = compile(tree, "<mirror-stage19-candidate>", "exec")
    namespace = {
        "__builtins__": build_builtins(),
        "__name__": "mirror_candidate",
        "_mc_step_tick": tick,
    }
    exec(compiled, namespace)
    entrypoint = namespace.get("run")
    if not callable(entrypoint):
        raise TypeError("candidate must expose callable run(inputs)")
    actual = entrypoint(inputs)
    encoded = json.dumps({"ok": True, "actual": actual}, allow_nan=False, sort_keys=True, separators=(",", ":"))
except BaseException as exc:
    encoded = json.dumps({"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:1000]})
sys.stdout.write(encoded + "\n")
"""
def execute_checked(source_text, namespace, filename):
    tree = Guard().visit(ast.parse(source_text))
    ast.fix_missing_locations(tree)
    compiled = compile(tree, filename, "exec")
    exec(compiled, namespace)

try:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        if stage19_mode:
            regression_cases = payload.get("regression_cases", [])
            expected_key = payload.get("expected_key", "candidate_expected")
            if not isinstance(regression_cases, list) or not regression_cases or len(regression_cases) > 250:
                raise RuntimeError("Stage 19 requires between 1 and 250 regression cases")
            if expected_key not in {"baseline_expected", "candidate_expected"}:
                raise RuntimeError("invalid Stage 19 expected-output key")
            deadline = time.monotonic() + float(payload.get("candidate_timeout_seconds", 8.0))
            results = []
            for case in regression_cases:
                case_id = str(case.get("case_id") or "")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    results.append({"case_id": case_id, "passed": False, "actual_json": "", "error": "suite execution deadline exceeded"})
                    continue
                child_payload = json.dumps({
                    "candidate_source": candidate_source,
                    "inputs": case["inputs"],
                    "max_loop_steps": step_cap,
                }, allow_nan=False, separators=(",", ":"))
                try:
                    completed = subprocess.run(
                        [sys.executable, "-I", "-S", "-c", _CANDIDATE_CHILD],
                        input=child_payload,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=max(0.05, remaining),
                        check=False,
                        close_fds=True,
                        env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                    )
                    data = json.loads(completed.stdout.strip().splitlines()[-1])
                    if completed.returncode != 0 or not data.get("ok"):
                        results.append({
                            "case_id": case_id, "passed": False, "actual_json": "",
                            "error": str(data.get("error") or completed.stderr.strip() or "candidate child failed")[:1000],
                        })
                        continue
                    actual_json = json.dumps(data["actual"], allow_nan=False, sort_keys=True, separators=(",", ":"))
                    expected_json = json.dumps(case[expected_key], allow_nan=False, sort_keys=True, separators=(",", ":"))
                    results.append({
                        "case_id": case_id,
                        "passed": actual_json == expected_json,
                        "actual_json": actual_json[:2000],
                        "error": "",
                    })
                except subprocess.TimeoutExpired:
                    results.append({"case_id": case_id, "passed": False, "actual_json": "", "error": "candidate case timeout"})
                except (ValueError, KeyError, TypeError) as exc:
                    results.append({"case_id": case_id, "passed": False, "actual_json": "", "error": str(exc)[:1000]})
            print("__MIRROR_STAGE19_RESULTS__" + json.dumps(
                {"results": results}, allow_nan=False, sort_keys=True, separators=(",", ":")
            ))
        else:
            namespace = {
                "__builtins__":dict(safe_builtins),
                "__name__":"__main__",
                "_mc_step_tick":tick,
            }
            execute_checked(source, namespace, "<mirror-isolated-child>")
    result = {"ok":True,"output":captured.getvalue().strip(),"pid":os.getpid()}
except BaseException as exc:
    result = {"ok":False,"error":type(exc).__name__ + ": " + str(exc),"pid":os.getpid()}
sys.stdout.write(json.dumps(result))
'''


def run_python_isolated(
    source: str,
    *,
    max_loop_steps: int = 50000,
    timeout_seconds: float = 5.0,
    memory_limit_mb: int = 256,
) -> SandboxExecutionResult:
    """Run a single candidate in a distinct, time/memory-limited interpreter process."""
    if not isinstance(source, str) or not source.strip():
        return SandboxExecutionResult(False, "", "empty source", None, None)
    if len(source.encode("utf-8", errors="replace")) > 512_000:
        return SandboxExecutionResult(False, "", "source exceeds 512 KB limit", None, None)

    payload = json.dumps({"source": source, "max_loop_steps": int(max_loop_steps)})
    with tempfile.TemporaryDirectory(prefix="mirror-sandbox-") as working_dir:
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        if os.name == "nt" and os.environ.get("SYSTEMROOT"):
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

        preexec_fn = None
        if os.name == "posix":
            def apply_limits() -> None:
                import resource
                cpu = max(1, int(timeout_seconds))
                memory = max(64, int(memory_limit_mb)) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
                resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
            preexec_fn = apply_limits

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _CHILD_RUNNER],
                input=payload,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                env=env,
                timeout=max(1.0, float(timeout_seconds)),
                check=False,
                preexec_fn=preexec_fn,
                close_fds=True,
            )
        except subprocess.TimeoutExpired:
            return SandboxExecutionResult(False, "", "sandbox timeout", None, None)
        except OSError as exc:
            return SandboxExecutionResult(False, "", f"sandbox process error: {exc}", None, None)

        try:
            data: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            detail = completed.stderr.strip()[-1000:] or completed.stdout.strip()[-1000:]
            return SandboxExecutionResult(
                False, "", f"sandbox child returned invalid output: {detail}",
                None, completed.returncode,
            )
        ok = bool(data.get("ok")) and completed.returncode == 0
        return SandboxExecutionResult(
            ok=ok,
            output=str(data.get("output") or ""),
            error=str(data.get("error") or (completed.stderr.strip()[-1000:] if not ok else "")),
            pid=int(data["pid"]) if data.get("pid") is not None else None,
            returncode=completed.returncode,
        )