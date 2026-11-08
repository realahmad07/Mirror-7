"""
Mirror 7 — CPU-Native Sparse Compute, Parallel State Scan, Wake-Sleep Self-Trainer & Verified Skill Forge
Author: Ahmad Hassan | Repository: https://github.com/realahmad07/Mirror-7

Implements the four CPU-native architectural capabilities designed to minimize GPU dependence:
1. Sparse Conditional Compute:
   - ``TernaryBitLinear``: 1.58-bit {-1, 0, +1} AbsMean quantization with Straight-Through Estimator (STE)
     and packed 2-bit storage metrics for CPU L2/L3 cache residency.
   - ``ProductKeyMemoryBank``: O(sqrt(N) * d) sub-linear Cartesian key lookup over large CPU-RAM slots
     with instant 0-backprop episodic skill writes (< 1 ms on CPU).
2. Fast Sequence Recurrence on CPU:
   - ``BytePatchCompressor`` + ``DiagonalGatedLinearScan``: Replaces sequential O(T * d^2) byte-by-byte
     GRUCell matrix multiplications with an O(T * d) vectorized parallel associative prefix scan in
     ceil(log2(T)) SIMD tensor steps.
3. Autonomous Verifier-Grounded CPU Self-Training ("Wake-Sleep Loop"):
   - ``AutonomousWakeSleepTrainer``: Synthesizes parametric coding/math challenges with executable CPU
     unit-test oracles, filters K candidate rollouts via AST + unit-test rewards, writes passing skills
     instantly into episodic memory (0 backprop), and runs fast state-cached LoRA micro-consolidation
     guarded by an automatic anchor-regression rollback gate.
4. Autonomous Verified Code Synthesis & Dynamic Skill Recruitment:
   - ``ASTSafetyAuditor`` + ``VerifiedSkillForge``: Audits candidate Python helper modules against a
     strict zero-trust AST allowlist, verifies unit-test correctness + CPU execution speed, hot-registers
     verified functions into a persistent ``DynamicSkillRegistry``, self-upgrades slower O(n^2) routines
     when verified O(n log n) candidates win on latency, and automatically recruits registered tools
     inside sandboxed execution.
"""
from __future__ import annotations

import ast
import io
import json
import math
import re
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.architecture_v6_rlvr_dpo_hub import compute_code_verifiable_reward
from training.native_conversation_v5 import LoRALinear


# ============================================================================
# 1. SPARSE & L3-CACHE-FRIENDLY CPU COMPUTE (1.58-BIT TERNARY + O(sqrt(N)) PKM)
# ============================================================================


class TernaryBitLinear(nn.Module):
    """1.58-bit {-1, 0, +1} linear projection with Straight-Through Estimator (STE).

    Compresses active weight footprint by ~8x vs FP16 (~16x vs FP32) when packed at 4 ternary
    weights per byte, allowing active specialist projections to stay inside CPU L2/L3 cache.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, eps: float = 1e-6):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("frozen_ternary_int8", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("frozen_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    @staticmethod
    def quantize_ternary_ste(w: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        scale = w.abs().mean().clamp(min=eps)
        w_scaled = w / scale
        w_quant = w_scaled.round().clamp(-1.0, 1.0)
        # Straight-Through Estimator: exact ternary in forward, smooth gradient in backward
        w_ste = w_scaled + (w_quant - w_scaled).detach()
        return w_ste, scale

    @torch.no_grad()
    def freeze_for_cpu_inference(self) -> dict[str, float]:
        """Freeze master FP32 weights into compact int8 {-1, 0, +1} + scalar scale."""
        scale = self.weight.abs().mean().clamp(min=self.eps)
        q = (self.weight / scale).round().clamp(-1.0, 1.0).to(torch.int8)
        self.frozen_ternary_int8 = q
        self.frozen_scale = scale.detach().to(torch.float32)
        fp32_bytes = self.weight.numel() * 4
        packed_2bit_bytes = math.ceil(self.weight.numel() / 4) + 4
        return {
            "num_weights": float(self.weight.numel()),
            "fp32_bytes": float(fp32_bytes),
            "packed_2bit_bytes": float(packed_2bit_bytes),
            "compression_vs_fp32": round(fp32_bytes / max(1, packed_2bit_bytes), 2),
            "zero_sparsity_fraction": round(float((q == 0).float().mean().item()), 4),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self.frozen_ternary_int8.numel() == self.weight.numel():
            w_eff = self.frozen_ternary_int8.to(dtype=x.dtype) * self.frozen_scale.to(dtype=x.dtype)
            return F.linear(x, w_eff, self.bias)
        w_ste, scale = self.quantize_ternary_ste(self.weight, eps=self.eps)
        return F.linear(x, w_ste * scale, self.bias)


@dataclass
class EpisodicSkillRecord:
    prompt_signature: str
    response_text: str
    reward: float
    timestamp: float


class ProductKeyMemoryBank(nn.Module):
    """Sub-linear O(sqrt(N) * d) Cartesian Product-Key Memory with instant 0-backprop episodic writes.

    For N = C^2 slots (e.g., C=256 -> N=65,536 slots), lookup evaluates only 2*C = 512 dot products
    instead of 65,536 dense dot products (128x fewer lookup FLOPs on CPU).
    """

    def __init__(self, state_dim: int, codebook_size: int = 128, top_k: int = 8):
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError(f"state_dim must be even for ProductKeyMemoryBank, got {state_dim}")
        self.state_dim = int(state_dim)
        self.half_dim = self.state_dim // 2
        self.codebook_size = int(codebook_size)
        self.total_slots = self.codebook_size * self.codebook_size
        self.top_k = min(max(1, int(top_k)), self.codebook_size)

        scale = 1.0 / math.sqrt(self.half_dim)
        self.subkeys_1 = nn.Parameter(torch.randn(self.codebook_size, self.half_dim) * scale)
        self.subkeys_2 = nn.Parameter(torch.randn(self.codebook_size, self.half_dim) * scale)
        self.values = nn.Embedding(self.total_slots, self.state_dim)
        nn.init.normal_(self.values.weight, mean=0.0, std=0.02)
        self.mem_gate = nn.Linear(self.state_dim * 2, self.state_dim)
        nn.init.constant_(self.mem_gate.bias, -2.0)

        # Zero-backprop episodic store indexed by Product-Key slot OR cosine-verified fast state table
        self._episodic_states: list[torch.Tensor] = []
        self._episodic_records: list[EpisodicSkillRecord] = []

    def _query_product_keys(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1, q2 = state[:, : self.half_dim], state[:, self.half_dim :]
        k_sub = min(self.top_k, self.codebook_size)
        s1, idx1 = torch.topk(q1 @ self.subkeys_1.t(), k=k_sub, dim=-1)
        s2, idx2 = torch.topk(q2 @ self.subkeys_2.t(), k=k_sub, dim=-1)

        # Cartesian sum of top sub-scores: shape (B, k_sub, k_sub) -> flatten (B, k_sub^2)
        grid_scores = (s1.unsqueeze(2) + s2.unsqueeze(1)).reshape(state.shape[0], -1)
        grid_indices = (idx1.unsqueeze(2) * self.codebook_size + idx2.unsqueeze(1)).reshape(state.shape[0], -1)
        best_scores, best_pos = torch.topk(grid_scores, k=min(self.top_k, grid_scores.shape[-1]), dim=-1)
        best_slots = torch.gather(grid_indices, 1, best_pos)
        return best_scores / math.sqrt(self.state_dim), best_slots

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        top_scores, top_slots = self._query_product_keys(state)
        weights = torch.softmax(top_scores, dim=-1).unsqueeze(-1)
        retrieved = self.values(top_slots)
        summary = (weights * retrieved).sum(dim=1)
        gate = torch.sigmoid(self.mem_gate(torch.cat([state, summary], dim=-1)))
        return torch.tanh(state + gate * summary)

    @torch.no_grad()
    def write_verified_episode(
        self,
        state_vec: torch.Tensor,
        prompt_signature: str,
        response_text: str,
        reward: float = 1.0,
        hebbian_lr: float = 0.25,
    ) -> int:
        """Instant 0-backprop write (< 1 ms on CPU): updates addressed Product-Key value slot & episodic store."""
        vec = F.normalize(state_vec.detach().reshape(1, self.state_dim).float(), dim=-1)
        _, top_slots = self._query_product_keys(vec.to(self.subkeys_1.device))
        primary_slot = int(top_slots[0, 0].item())
        self.values.weight.data[primary_slot].lerp_(vec.squeeze(0).to(self.values.weight.device), hebbian_lr)

        rec = EpisodicSkillRecord(
            prompt_signature=prompt_signature.strip(),
            response_text=response_text,
            reward=float(reward),
            timestamp=time.time(),
        )
        # Deduplicate / upgrade if a closely matching episodic state already exists
        for idx, existing_vec in enumerate(self._episodic_states):
            cos = float((existing_vec @ vec.squeeze(0).cpu()).item())
            if cos >= 0.985 or self._episodic_records[idx].prompt_signature == rec.prompt_signature:
                if rec.reward >= self._episodic_records[idx].reward:
                    self._episodic_states[idx] = vec.squeeze(0).cpu()
                    self._episodic_records[idx] = rec
                return primary_slot

        self._episodic_states.append(vec.squeeze(0).cpu())
        self._episodic_records.append(rec)
        return primary_slot

    @torch.no_grad()
    def recall_verified_episode(self, state_vec: torch.Tensor, min_cosine: float = 0.92) -> EpisodicSkillRecord | None:
        if not self._episodic_states:
            return None
        q = F.normalize(state_vec.detach().reshape(1, self.state_dim).float().cpu(), dim=-1)
        mat = torch.stack(self._episodic_states, dim=0)
        sims = (mat @ q.t()).squeeze(-1)
        best_val, best_idx = torch.max(sims, dim=0)
        if float(best_val.item()) >= min_cosine:
            return self._episodic_records[int(best_idx.item())]
        return None


# ============================================================================
# 2. FAST CPU SEQUENCE RECURRENCE (PARALLEL ASSOCIATIVE SCAN + BYTE PATCHING)
# ============================================================================


def parallel_associative_linear_scan(gates: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Exact vectorized Blelloch/Hillis-Steele associative prefix scan for h_t = gates_t * h_{t-1} + values_t.

    Computes all T steps in ceil(log2(T)) vectorized SIMD tensor operations on CPU or GPU
    (e.g., 11 tensor ops for T=2048 instead of 2,048 sequential Python GRUCell matrix multiplies).
    """
    a = gates
    b = values
    seq_len = a.shape[1]
    stride = 1
    while stride < seq_len:
        a_prev = a[:, :-stride]
        b_prev = b[:, :-stride]
        a_curr = a[:, stride:]
        b_curr = b[:, stride:]
        new_b = a_curr * b_prev + b_curr
        new_a = a_curr * a_prev
        a = torch.cat([a[:, :stride], new_a], dim=1)
        b = torch.cat([b[:, :stride], new_b], dim=1)
        stride <<= 1
    return b


class BytePatchCompressor(nn.Module):
    """Groups raw UTF-8 byte embeddings (vocab=259) into K-byte local patches to cut CPU sequence length."""

    def __init__(self, vocab_size: int = 259, embed_dim: int = 128, patch_size: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.patch_size = max(1, int(patch_size))
        self.hidden_dim = int(hidden_dim)
        self.byte_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.patch_proj = TernaryBitLinear(self.embed_dim * self.patch_size, self.hidden_dim, bias=True)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = byte_ids.shape
        rem = seq_len % self.patch_size
        if rem != 0:
            pad_len = self.patch_size - rem
            byte_ids = F.pad(byte_ids, (0, pad_len), value=0)
        emb = self.byte_embed(byte_ids)
        patched = emb.reshape(bsz, -1, self.embed_dim * self.patch_size)
        return self.norm(F.gelu(self.patch_proj(patched)))


class DiagonalGatedLinearScanEncoder(nn.Module):
    """CPU-fast byte-to-state encoder combining 4-byte patching + O(T * d) associative prefix scan."""

    def __init__(
        self,
        vocab_size: int = 259,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        state_dim: int = 512,
        patch_size: int = 4,
    ):
        super().__init__()
        self.patcher = BytePatchCompressor(
            vocab_size=vocab_size, embed_dim=embed_dim, patch_size=patch_size, hidden_dim=hidden_dim
        )
        self.gate_proj = TernaryBitLinear(hidden_dim, hidden_dim, bias=True)
        self.val_proj = TernaryBitLinear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = TernaryBitLinear(hidden_dim, state_dim, bias=True)
        self.prev_state_gate = nn.Linear(state_dim * 2, state_dim)

    def forward(self, byte_ids: torch.Tensor, prev_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.patcher(byte_ids)
        gates = torch.sigmoid(self.gate_proj(u))
        vals = (1.0 - gates) * torch.tanh(self.val_proj(u))
        h_seq = parallel_associative_linear_scan(gates, vals)
        pooled = torch.tanh(self.out_proj(h_seq[:, -1, :]))
        if prev_state is not None:
            g = torch.sigmoid(self.prev_state_gate(torch.cat([pooled, prev_state], dim=-1)))
            pooled = torch.tanh(g * pooled + (1.0 - g) * prev_state)
        return h_seq, pooled


# ============================================================================
# 3. AUTONOMOUS SELF-CODE SYNTHESIS, AST SECURITY GATE & DYNAMIC RECRUITMENT
# ============================================================================

ALLOWED_SKILL_IMPORTS: frozenset[str] = frozenset(
    {"math", "heapq", "collections", "bisect", "itertools", "functools", "typing", "fractions", "statistics"}
)

FORBIDDEN_AST_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "exit",
        "quit",
    }
)


class ASTSafetyAuditor:
    """Zero-trust static AST verifier ensuring self-recruited Python modules are pure, deterministic & safe."""

    @staticmethod
    def audit(code_str: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code_str)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}"

        defined_functions: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod not in ALLOWED_SKILL_IMPORTS:
                        return False, f"Forbidden import: '{alias.name}'"
            elif isinstance(node, ast.ImportFrom):
                root_mod = (node.module or "").split(".")[0]
                if root_mod not in ALLOWED_SKILL_IMPORTS:
                    return False, f"Forbidden from-import: '{node.module}'"
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return False, f"Forbidden dunder attribute access: '{node.attr}'"
            elif isinstance(node, ast.Name):
                if node.id.startswith("__") and node.id.endswith("__"):
                    return False, f"Forbidden dunder identifier: '{node.id}'"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_AST_CALLS:
                    return False, f"Forbidden builtin call: '{node.func.id}()'"
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    defined_functions.append(node.name)

        if not defined_functions:
            return False, "No public top-level function defined in candidate module"
        return True, defined_functions[0]


@dataclass
class VerifiedSkillEntry:
    skill_name: str
    description: str
    code: str
    test_code: str
    benchmark_us: float
    version: int
    keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerifiedSkillForge:
    """Synthesizes, AST-audits, unit-test-verifies, benchmarks, and hot-recruits reusable Python skills on CPU.

    Supports self-upgrading: if a newly synthesized implementation passes 100% of unit tests AND runs
    faster on CPU than the existing version (e.g., O(n log n) replacing O(n^2)), it hot-swaps automatically.
    """

    def __init__(self, registry_path: Path | str | None = None):
        self.registry_path = Path(registry_path) if registry_path is not None else None
        self._skills: dict[str, VerifiedSkillEntry] = {}
        self._compiled_callables: dict[str, Callable[..., Any]] = {}
        if self.registry_path is not None and self.registry_path.is_file():
            self._load_registry()

    @staticmethod
    def _build_safe_exec_globals() -> dict[str, Any]:
        import bisect
        import collections
        import fractions
        import functools
        import heapq
        import itertools
        import statistics
        import typing

        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "frozenset": frozenset,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "ValueError": ValueError,
            "AssertionError": AssertionError,
            "zip": zip,
            "__import__": lambda name, *args, **kwargs: {
                "math": math,
                "heapq": heapq,
                "collections": collections,
                "bisect": bisect,
                "itertools": itertools,
                "functools": functools,
                "typing": typing,
                "fractions": fractions,
                "statistics": statistics,
            }[name.split(".")[0]],
        }
        return {"__builtins__": safe_builtins}

    def synthesize_and_recruit(
        self,
        skill_name: str,
        candidate_code: str,
        unit_test_code: str,
        description: str = "",
        keywords: list[str] | None = None,
        benchmark_expr: str | None = None,
    ) -> dict[str, Any]:
        """Audit, execute unit tests, benchmark latency, and hot-register (or upgrade) the skill."""
        clean_code = re.sub(r"<think>.*?</think>", "", candidate_code, flags=re.DOTALL).strip()
        is_safe, audit_msg = ASTSafetyAuditor.audit(clean_code)
        if not is_safe:
            return {"recruited": False, "reason": f"AST safety rejected: {audit_msg}", "skill_name": skill_name}

        ok_test, test_audit = ASTSafetyAuditor.audit(f"def _test_wrap():\n" + "\n".join("  " + ln for ln in unit_test_code.splitlines()))
        if not ok_test and "No public top-level" not in test_audit:
            return {"recruited": False, "reason": f"Test AST rejected: {test_audit}", "skill_name": skill_name}

        ns = self._build_safe_exec_globals()
        try:
            exec(clean_code, ns)
            if skill_name not in ns or not callable(ns[skill_name]):
                return {
                    "recruited": False,
                    "reason": f"Expected callable '{skill_name}' not found after execution",
                    "skill_name": skill_name,
                }
            exec(unit_test_code, ns)
        except Exception as exc:
            return {"recruited": False, "reason": f"Unit test verification failed: {exc}", "skill_name": skill_name}

        # Measure deterministic CPU execution latency in microseconds
        bench_stmt = benchmark_expr if benchmark_expr else unit_test_code
        t0 = time.perf_counter_ns()
        for _ in range(5):
            exec(bench_stmt, ns)
        bench_us = (time.perf_counter_ns() - t0) / 5_000.0

        existing = self._skills.get(skill_name)
        if existing is not None:
            # Self-upgrade rule: keep existing if new candidate is slower (> 1.05x latency)
            if bench_us > existing.benchmark_us * 1.05:
                return {
                    "recruited": False,
                    "upgraded": False,
                    "reason": f"Existing skill faster ({existing.benchmark_us:.1f} us vs {bench_us:.1f} us)",
                    "skill_name": skill_name,
                }
            version = existing.version + 1
            upgraded = True
        else:
            version = 1
            upgraded = False

        entry = VerifiedSkillEntry(
            skill_name=skill_name,
            description=description or f"Verified autonomous CPU skill: {skill_name}",
            code=clean_code,
            test_code=unit_test_code.strip(),
            benchmark_us=round(bench_us, 2),
            version=version,
            keywords=[k.lower() for k in (keywords or [skill_name])],
        )
        self._skills[skill_name] = entry
        self._compiled_callables[skill_name] = ns[skill_name]
        if self.registry_path is not None:
            self._save_registry()

        return {
            "recruited": True,
            "upgraded": upgraded,
            "version": version,
            "benchmark_us": entry.benchmark_us,
            "skill_name": skill_name,
        }

    def match_skills_for_query(self, query: str) -> list[VerifiedSkillEntry]:
        q_low = query.lower()
        matched: list[VerifiedSkillEntry] = []
        for entry in self._skills.values():
            if entry.skill_name.lower() in q_low or any(kw in q_low for kw in entry.keywords):
                matched.append(entry)
        return matched

    def export_callable_namespace(self) -> dict[str, Callable[..., Any]]:
        """Return verified functions ready to be injected into ``execute_sandboxed_python_tool``."""
        return dict(self._compiled_callables)

    def execute_with_recruited_skills(self, python_snippet: str) -> str:
        """Execute a sandboxed snippet with all verified self-recruited skills mounted in ``skills`` & globals."""
        clean_cmd = re.sub(r"^python\s*:\s*", "", python_snippet.strip(), flags=re.IGNORECASE)
        ok, msg = ASTSafetyAuditor.audit("def _snippet():\n" + "\n".join("  " + ln for ln in clean_cmd.splitlines()))
        if not ok and "No public top-level" not in msg:
            return f"ToolSecurityBlocked: {msg}"

        ns = self._build_safe_exec_globals()
        ns.update(self._compiled_callables)
        ns["skills"] = dict(self._compiled_callables)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(clean_cmd, ns)
            out = buf.getvalue().strip()
            return out if out else "OK"
        except Exception as exc:
            return f"ToolError: {exc}"

    def _save_registry(self) -> None:
        if self.registry_path is None:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.to_dict() for k, v in sorted(self._skills.items())}
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_registry(self) -> None:
        if self.registry_path is None or not self.registry_path.is_file():
            return
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        for name, item in raw.items():
            self.synthesize_and_recruit(
                skill_name=name,
                candidate_code=item["code"],
                unit_test_code=item["test_code"],
                description=item.get("description", ""),
                keywords=item.get("keywords", [name]),
            )


# ============================================================================
# 4. AUTONOMOUS VERIFIER-GROUNDED CPU SELF-TRAINER ("WAKE-SLEEP LOOP")
# ============================================================================


@dataclass(frozen=True)
class VerifiableChallenge:
    challenge_id: str
    prompt: str
    skill_name: str
    test_code: str
    candidate_rollouts: tuple[str, ...]
    keywords: tuple[str, ...]


class ParametricChallengeMutator:
    """Generates deterministic, oracle-verified algorithmic & number-theory challenges for CPU self-play."""

    @staticmethod
    def generate_curriculum(seed: int = 7) -> list[VerifiableChallenge]:
        mod_p = 1_000_000_007
        base_a = 17 + (seed % 13)
        exp_b = 23 + (seed % 11)
        expected_mod = pow(base_a, exp_b, mod_p)

        return [
            VerifiableChallenge(
                challenge_id=f"alg_lis_fast_s{seed}",
                prompt="Implement `lis_length(nums)` in O(n log n) returning the Longest Increasing Subsequence length.",
                skill_name="lis_length",
                keywords=("lis", "longest increasing subsequence", "subsequence"),
                test_code=(
                    "assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4\n"
                    "assert lis_length([0, 1, 0, 3, 2, 3]) == 4\n"
                    "assert lis_length([7, 7, 7, 7]) == 1\n"
                    "assert lis_length([]) == 0\n"
                ),
                candidate_rollouts=(
                    # Rollout 0: Flawed greedy candidate (fails unit test -> y_lose)
                    "<think> Greedy scan counting adjacent increases. </think>\n"
                    "def lis_length(nums):\n"
                    "  if not nums: return 0\n"
                    "  return sum(1 for i in range(1, len(nums)) if nums[i] > nums[i-1]) + 1\n",
                    # Rollout 1: Verified O(n log n) patience sorting candidate (passes -> y_win & recruited!)
                    "<think> Maintain monotonic `tails` via `bisect_left` in O(n log n). </think>\n"
                    "import bisect\n"
                    "def lis_length(nums):\n"
                    "  tails = []\n"
                    "  for x in nums:\n"
                    "    idx = bisect.bisect_left(tails, x)\n"
                    "    if idx == len(tails): tails.append(x)\n"
                    "    else: tails[idx] = x\n"
                    "  return len(tails)\n",
                ),
            ),
            VerifiableChallenge(
                challenge_id=f"nt_modexp_s{seed}",
                prompt=f"Write `fast_mod_exp(a, b, m)` via binary exponentiation and verify {base_a}^{exp_b} mod {mod_p}.",
                skill_name="fast_mod_exp",
                keywords=("modular exponentiation", "fast_mod_exp", "mod"),
                test_code=(
                    f"assert fast_mod_exp({base_a}, {exp_b}, {mod_p}) == {expected_mod}\n"
                    "assert fast_mod_exp(2, 10, 1000) == 24\n"
                    "assert fast_mod_exp(5, 0, 13) == 1\n"
                ),
                candidate_rollouts=(
                    # Rollout 0: Broken overflow/missing modulo on base -> fails or suboptimal (y_lose)
                    "<think> Naive multiplication loop without squaring. </think>\n"
                    "def fast_mod_exp(a, b, m):\n"
                    "  return (a * b) % m\n",
                    # Rollout 1: Verified O(log b) binary exponentiation -> passes (y_win & recruited!)
                    "<think> Binary square-and-multiply in O(log b) time and O(1) space. </think>\n"
                    "def fast_mod_exp(a, b, m):\n"
                    "  res, base = 1 % m, a % m\n"
                    "  while b > 0:\n"
                    "    if b & 1: res = (res * base) % m\n"
                    "    base = (base * base) % m\n"
                    "    b >>= 1\n"
                    "  return res\n",
                ),
            ),
            VerifiableChallenge(
                challenge_id=f"graph_kahn_s{seed}",
                prompt="Implement `topo_sort_kahn(n, edges)` returning a topological order or `[]` on cycle.",
                skill_name="topo_sort_kahn",
                keywords=("topological", "kahn", "cycle detection", "dag"),
                test_code=(
                    "assert topo_sort_kahn(4, [(0, 1), (0, 2), (1, 3), (2, 3)]) in ([0, 1, 2, 3], [0, 2, 1, 3])\n"
                    "assert topo_sort_kahn(3, [(0, 1), (1, 2), (2, 0)]) == []\n"
                ),
                candidate_rollouts=(
                    # Rollout 0: Ignores cycles -> fails cycle assertion (y_lose)
                    "<think> Return range(n) directly. </think>\n"
                    "def topo_sort_kahn(n, edges):\n"
                    "  return list(range(n))\n",
                    # Rollout 1: Verified Kahn's BFS indegree algorithm -> passes (y_win & recruited!)
                    "<think> Track indegrees and peel zero-indegree frontier; cycle exists iff len(order) < n. </think>\n"
                    "from collections import deque\n"
                    "def topo_sort_kahn(n, edges):\n"
                    "  adj = [[] for _ in range(n)]\n"
                    "  indeg = [0] * n\n"
                    "  for u, v in edges:\n"
                    "    adj[u].append(v); indeg[v] += 1\n"
                    "  q = deque(i for i in range(n) if indeg[i] == 0)\n"
                    "  order = []\n"
                    "  while q:\n"
                    "    u = q.popleft(); order.append(u)\n"
                    "    for v in adj[u]:\n"
                    "      indeg[v] -= 1\n"
                    "      if indeg[v] == 0: q.append(v)\n"
                    "  return order if len(order) == n else []\n",
                ),
            ),
        ]

    @staticmethod
    def generate_extended_curriculum(seed: int = 7) -> list[VerifiableChallenge]:
        """Extended 10-skill CPU curriculum spanning graphs, DP, number theory, search & quantum Born probabilities."""
        base = ParametricChallengeMutator.generate_curriculum(seed=seed)
        limit_p = 29 + (seed % 5)
        expected_primes = [x for x in range(2, limit_p + 1) if all(x % d != 0 for d in range(2, int(x**0.5) + 1))]

        extended = [
            VerifiableChallenge(
                challenge_id=f"graph_dijkstra_s{seed}",
                prompt="Implement `dijkstra_shortest_path(n, edges, src)` using a min-heap returning shortest distances or `-1` if unreachable.",
                skill_name="dijkstra_shortest_path",
                keywords=("dijkstra", "shortest path", "weighted graph", "heapq"),
                test_code=(
                    "assert dijkstra_shortest_path(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1)], 0) == [0, 3, 1, 4]\n"
                    "assert dijkstra_shortest_path(3, [(0,1,5)], 0) == [0, 5, -1]\n"
                    "assert dijkstra_shortest_path(1, [], 0) == [0]\n"
                ),
                candidate_rollouts=(
                    "<think> Unweighted BFS ignores edge weights. </think>\n"
                    "def dijkstra_shortest_path(n, edges, src):\n"
                    "  return [0 if i == src else 1 for i in range(n)]\n",
                    "<think> Maintain priority queue (dist, u) and relax edges with non-negative weights in O(E log V). </think>\n"
                    "import heapq\n"
                    "def dijkstra_shortest_path(n, edges, src):\n"
                    "  adj = [[] for _ in range(n)]\n"
                    "  for u, v, w in edges:\n"
                    "    adj[u].append((v, w))\n"
                    "  dist = [float('inf')] * n\n"
                    "  dist[src] = 0\n"
                    "  pq = [(0, src)]\n"
                    "  while pq:\n"
                    "    d, u = heapq.heappop(pq)\n"
                    "    if d > dist[u]: continue\n"
                    "    for v, w in adj[u]:\n"
                    "      if d + w < dist[v]:\n"
                    "        dist[v] = d + w\n"
                    "        heapq.heappush(pq, (dist[v], v))\n"
                    "  return [-1 if d == float('inf') else int(d) for d in dist]\n",
                ),
            ),
            VerifiableChallenge(
                challenge_id=f"dp_knapsack01_s{seed}",
                prompt="Implement `knapsack_01(weights, values, capacity)` returning maximum integer value in 0/1 Knapsack.",
                skill_name="knapsack_01",
                keywords=("knapsack", "0/1 knapsack", "dynamic programming"),
                test_code=(
                    "assert knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9\n"
                    "assert knapsack_01([2, 3], [10, 15], 0) == 0\n"
                    "assert knapsack_01([], [], 10) == 0\n"
                ),
                candidate_rollouts=(
                    "<think> Greedy value-per-weight heuristic fails 0/1 integrity constraint. </think>\n"
                    "def knapsack_01(weights, values, capacity):\n"
                    "  return sum(v for w, v in zip(weights, values) if w <= capacity)\n",
                    "<think> 1D reverse-capacity DP transitions dp[c] = max(dp[c], dp[c-w] + v). </think>\n"
                    "def knapsack_01(weights, values, capacity):\n"
                    "  if capacity <= 0 or not weights: return 0\n"
                    "  dp = [0] * (capacity + 1)\n"
                    "  for w, v in zip(weights, values):\n"
                    "    if w <= capacity:\n"
                    "      for c in range(capacity, w - 1, -1):\n"
                    "        cand = dp[c - w] + v\n"
                    "        if cand > dp[c]: dp[c] = cand\n"
                    "  return dp[capacity]\n",
                ),
            ),
            VerifiableChallenge(
                challenge_id=f"nt_sieve_s{seed}",
                prompt=f"Implement `sieve_of_eratosthenes(n)` returning all primes <= n (verify up to {limit_p}).",
                skill_name="sieve_of_eratosthenes",
                keywords=("sieve", "eratosthenes", "prime numbers"),
                test_code=(
                    f"assert sieve_of_eratosthenes({limit_p}) == {expected_primes!r}\n"
                    "assert sieve_of_eratosthenes(1) == []\n"
                    "assert sieve_of_eratosthenes(2) == [2]\n"
                ),
                candidate_rollouts=(
                    "<think> Include 1 by mistake. </think>\n"
                    "def sieve_of_eratosthenes(n):\n"
                    "  return [1] + [x for x in range(2, n + 1) if x % 2 == 1]\n",
                    "<think> Mark composite multiples starting at p*p in O(n log log n) time. </think>\n"
                    "import math\n"
                    "def sieve_of_eratosthenes(n):\n"
                    "  if n < 2: return []\n"
                    "  prime = [True] * (n + 1)\n"
                    "  prime[0] = prime[1] = False\n"
                    "  for p in range(2, math.isqrt(n) + 1):\n"
                    "    if prime[p]:\n"
                    "      for m in range(p * p, n + 1, p):\n"
                    "        prime[m] = False\n"
                    "  return [i for i, flag in enumerate(prime) if flag]\n",
                ),
            ),
            VerifiableChallenge(
                challenge_id=f"search_binary_s{seed}",
                prompt="Implement `binary_search(arr, target)` returning exact index or `-1` on a sorted array.",
                skill_name="binary_search",
                keywords=("binary search", "bisection", "sorted array"),
                test_code=(
                    "assert binary_search([], 5) == -1\n"
                    "assert binary_search([1, 3, 5, 7], 1) == 0\n"
                    "assert binary_search([1, 3, 5, 7], 7) == 3\n"
                    "assert binary_search([1, 3, 5, 7], 4) == -1\n"
                ),
                candidate_rollouts=(
                    "<think> Off-by-one upper bound skips last element. </think>\n"
                    "def binary_search(arr, target):\n"
                    "  lo, hi = 0, len(arr) - 1\n"
                    "  while lo < hi:\n"
                    "    mid = (lo + hi) // 2\n"
                    "    if arr[mid] == target: return mid\n"
                    "    if arr[mid] < target: lo = mid + 1\n"
                    "    else: hi = mid - 1\n"
                    "  return -1\n",
                    "<think> Inclusive [lo, hi] loop bounds handle singletons cleanly in O(log n). </think>\n"
                    "def binary_search(arr, target):\n"
                    "  lo, hi = 0, len(arr) - 1\n"
                    "  while lo <= hi:\n"
                    "    mid = (lo + hi) // 2\n"
                    "    if arr[mid] == target: return mid\n"
                    "    if arr[mid] < target: lo = mid + 1\n"
                    "    else: hi = mid - 1\n"
                    "  return -1\n",
                ),
            ),
            VerifiableChallenge(
                challenge_id=f"dp_edit_distance_s{seed}",
                prompt="Implement `edit_distance(a, b)` returning minimum Levenshtein insert/delete/replace steps.",
                skill_name="edit_distance",
                keywords=("edit distance", "levenshtein", "string dp"),
                test_code=(
                    "assert edit_distance('kitten', 'sitting') == 3\n"
                    "assert edit_distance('', 'abc') == 3\n"
                    "assert edit_distance('same', 'same') == 0\n"
                ),
                candidate_rollouts=(
                    "<think> Return length diff only. </think>\n"
                    "def edit_distance(a, b):\n"
                    "  return abs(len(a) - len(b))\n",
                    "<think> Wagner-Fischer rolling row DP over prefixes of b in O(min(m,n)) space. </think>\n"
                    "def edit_distance(a, b):\n"
                    "  prev = list(range(len(b) + 1))\n"
                    "  for i, ca in enumerate(a, 1):\n"
                    "    curr = [i]\n"
                    "    for j, cb in enumerate(b, 1):\n"
                    "      cost = 0 if ca == cb else 1\n"
                    "      curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))\n"
                    "    prev = curr\n"
                    "  return prev[-1]\n",
                ),
            ),
            VerifiableChallenge(
                challenge_id=f"dp_kadane_s{seed}",
                prompt="Implement `kadane_max_subarray(nums)` returning maximum non-empty contiguous subarray sum.",
                skill_name="kadane_max_subarray",
                keywords=("kadane", "maximum subarray", "contiguous sum"),
                test_code=(
                    "assert kadane_max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\n"
                    "assert kadane_max_subarray([-5, -2, -8]) == -2\n"
                    "assert kadane_max_subarray([42]) == 42\n"
                ),
                candidate_rollouts=(
                    "<think> Clamping running sum to 0 as overall max fails negative arrays. </think>\n"
                    "def kadane_max_subarray(nums):\n"
                    "  cur = best = 0\n"
                    "  for x in nums:\n"
                    "    cur = max(0, cur + x)\n"
                    "    best = max(best, cur)\n"
                    "  return best\n",
                    "<think> Track best ending here starting at nums[0] so negative arrays work cleanly. </think>\n"
                    "def kadane_max_subarray(nums):\n"
                    "  if not nums: return 0\n"
                    "  cur = best = nums[0]\n"
                    "  for x in nums[1:]:\n"
                    "    cur = x if x > cur + x else cur + x\n"
                    "    if cur > best: best = cur\n"
                    "  return best\n",
                ),
            ),
            VerifiableChallenge(
                challenge_id=f"qm_born_probs_s{seed}",
                prompt="Implement `bell_state_probabilities(amps)` computing Born rule measurement probabilities for a qubit statevector.",
                skill_name="bell_state_probabilities",
                keywords=("quantum", "born rule", "bell state", "amplitudes", "qubit probabilities"),
                test_code=(
                    "probs = bell_state_probabilities([0.7071067811865475, 0.0, 0.0, 0.7071067811865475])\n"
                    "assert abs(probs[0] - 0.5) < 1e-6 and abs(probs[1]) < 1e-9 and abs(probs[3] - 0.5) < 1e-6\n"
                    "assert abs(sum(probs) - 1.0) < 1e-6\n"
                ),
                candidate_rollouts=(
                    "<think> Return raw amplitudes without squaring magnitude. </think>\n"
                    "def bell_state_probabilities(amps):\n"
                    "  return list(amps)\n",
                    "<think> Born rule P(i) = |psi_i|^2 normalized by total state norm squared. </think>\n"
                    "def bell_state_probabilities(amps):\n"
                    "  mags = [abs(a) ** 2 for a in amps]\n"
                    "  tot = sum(mags)\n"
                    "  if tot == 0: raise ValueError('Zero statevector')\n"
                    "  return [round(m / tot, 12) for m in mags]\n",
                ),
            ),
        ]
        return base + extended


class StateCachedMicroAdapterHead(nn.Module):
    """Lightweight state-to-byte-preference projection using Rank-4 LoRA for < 50 ms CPU sleep updates."""

    def __init__(self, state_dim: int = 512, vocab_size: int = 259, lora_rank: int = 4):
        super().__init__()
        base_proj = nn.Linear(state_dim, state_dim, bias=False)
        for p in base_proj.parameters():
            p.requires_grad_(False)
        self.adapter = LoRALinear(base_proj, rank=lora_rank, alpha=8.0)
        self.byte_scorer = nn.Linear(state_dim, vocab_size, bias=False)
        for p in self.byte_scorer.parameters():
            p.requires_grad_(False)

    def sequence_logprob(self, state_vec: torch.Tensor, text: str) -> torch.Tensor:
        adapted = torch.tanh(self.adapter(state_vec))
        logits = self.byte_scorer(adapted)
        log_probs = F.log_softmax(logits, dim=-1)
        raw = list(text.encode("utf-8")[:512]) or [32]
        idx = torch.tensor(raw, dtype=torch.long, device=state_vec.device)
        return log_probs[0, idx].mean()


class AutonomousWakeSleepTrainer:
    """Closed-loop 0-GPU Autonomous CPU Self-Trainer + Self-Recruiting Code Forge.

    1. Wake Phase: Encodes challenge via fast Parallel Associative Scan, evaluates candidate rollouts
       in the CPU Python AST/unit-test verifier, writes verified winners into O(sqrt(N)) Product-Key
       Memory in < 1 ms (zero backprop), and hot-recruits verified functions into ``VerifiedSkillForge``.
    2. Sleep Phase: Consolidates verified (y_win, y_lose) pairs directly from cached 512-D state vectors
       into Rank-4 LoRA micro-adapters in milliseconds on CPU, guarded by an automatic anchor-cosine
       regression rollback check.
    """

    def __init__(
        self,
        state_dim: int = 512,
        lora_rank: int = 4,
        registry_path: Path | str | None = None,
        min_anchor_cosine: float = 0.985,
        crystal_vault: Any | None = None,
    ):
        self.device = torch.device("cpu")
        self.state_dim = int(state_dim)
        self.min_anchor_cosine = float(min_anchor_cosine)
        self.crystal_vault = crystal_vault

        self.encoder = DiagonalGatedLinearScanEncoder(
            vocab_size=259, embed_dim=128, hidden_dim=256, state_dim=self.state_dim, patch_size=4
        ).to(self.device)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.memory = ProductKeyMemoryBank(state_dim=self.state_dim, codebook_size=128, top_k=8).to(self.device)
        self.micro_head = StateCachedMicroAdapterHead(
            state_dim=self.state_dim, vocab_size=259, lora_rank=lora_rank
        ).to(self.device)
        self.skill_forge = VerifiedSkillForge(registry_path=registry_path)

        # Anchor states protecting prior representations against drift during autonomous sleep updates
        torch.manual_seed(42)
        self._anchor_states = F.normalize(torch.randn(8, self.state_dim, device=self.device), dim=-1)

    @torch.no_grad()
    def encode_prompt_state(self, prompt: str) -> torch.Tensor:
        raw = list(prompt.encode("utf-8")[:2048]) or [32]
        ids = torch.tensor([raw], dtype=torch.long, device=self.device)
        _, pooled = self.encoder(ids)
        return self.memory(pooled)

    def crystallize_verified_skills(self, vault: Any) -> dict[str, int]:
        """Bridge all sandbox-verified skills in ``VerifiedSkillForge`` into ``DiskCrystalVault`` / ``MirrorCrystalEngine``."""
        written = 0
        already_present = 0
        for entry in self.skill_forge._skills.values():
            target_md = (
                f"Verified CPU Skill (`{entry.skill_name}`, v{entry.version}, {entry.benchmark_us:.1f} us):\n"
                f"```python\n{entry.code.strip()}\n```"
            )
            ok = vault.ingest_pair(
                user_text=entry.description or f"Implement verified Python function `{entry.skill_name}`.",
                target_text=target_md,
                think=f"Verified by AutonomousWakeSleepTrainer + VerifiedSkillForge ({', '.join(entry.keywords)})",
                test_code=entry.test_code,
                domain="cpu_verified_skill",
                trust_tier=1,
            )
            if ok:
                written += 1
            else:
                already_present += 1
        return {"written": written, "reinforced_or_existing": already_present}

    def run_autonomous_cycle(
        self,
        challenges: list[VerifiableChallenge] | None = None,
        dpo_steps: int = 6,
        crystal_vault: Any | None = None,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        tasks = challenges if challenges is not None else ParametricChallengeMutator.generate_curriculum(seed=7)
        target_vault = crystal_vault if crystal_vault is not None else self.crystal_vault

        verified_pairs: list[tuple[torch.Tensor, str, str]] = []
        recruited_skills: list[str] = []
        episodic_writes = 0
        crystal_vault_writes = 0

        # ---------------- WAKE PHASE (0-Backprop Verification, Memory Write & Code Recruitment) ----------------
        for ch in tasks:
            state_vec = self.encode_prompt_state(ch.prompt)
            scored: list[tuple[float, str]] = [
                (compute_code_verifiable_reward(cand, ch.test_code), cand) for cand in ch.candidate_rollouts
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            best_r, best_cand = scored[0]
            worst_r, worst_cand = scored[-1]

            # Strict anti-collapse rule: NEVER learn unless best candidate passed 100% of unit tests (reward >= 0.80)
            if best_r >= 0.80:
                self.memory.write_verified_episode(
                    state_vec=state_vec,
                    prompt_signature=ch.prompt,
                    response_text=best_cand,
                    reward=best_r,
                )
                episodic_writes += 1

                forge_res = self.skill_forge.synthesize_and_recruit(
                    skill_name=ch.skill_name,
                    candidate_code=best_cand,
                    unit_test_code=ch.test_code,
                    description=ch.prompt,
                    keywords=list(ch.keywords),
                )
                if forge_res.get("recruited"):
                    recruited_skills.append(ch.skill_name)

                if target_vault is not None:
                    think_m = re.search(r"<think>(.*?)</think>", best_cand, flags=re.DOTALL)
                    think_txt = think_m.group(1).strip() if think_m else ""
                    clean_code = re.sub(r"<think>.*?</think>", "", best_cand, flags=re.DOTALL).strip()
                    wrote = target_vault.ingest_pair(
                        user_text=ch.prompt,
                        target_text=f"Verified Solution (`{ch.skill_name}`):\n```python\n{clean_code}\n```",
                        think=think_txt,
                        test_code=ch.test_code,
                        domain="cpu_verified_skill",
                        trust_tier=1,
                    )
                    if wrote:
                        crystal_vault_writes += 1

                if best_r > worst_r:
                    verified_pairs.append((state_vec.detach(), best_cand, worst_cand))

        # ---------------- SLEEP PHASE (State-Cached Rank-4 LoRA DPO + Regression Rollback Gate) ----------------
        t_sleep0 = time.perf_counter()
        with torch.no_grad():
            anchor_before = F.normalize(torch.tanh(self.micro_head.adapter(self._anchor_states)), dim=-1)
        adapter_snapshot = {k: v.detach().clone() for k, v in self.micro_head.adapter.state_dict().items()}

        trainable_params = [p for p in self.micro_head.adapter.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=5e-3, weight_decay=1e-4)

        initial_margin = 0.0
        final_margin = 0.0
        if verified_pairs:
            for step_i in range(max(1, dpo_steps)):
                opt.zero_grad()
                losses: list[torch.Tensor] = []
                margins: list[float] = []
                for s_vec, y_win, y_lose in verified_pairs:
                    lp_win = self.micro_head.sequence_logprob(s_vec, y_win)
                    lp_lose = self.micro_head.sequence_logprob(s_vec, y_lose)
                    diff = lp_win - lp_lose
                    margins.append(float(diff.detach().item()))
                    losses.append(-F.logsigmoid(4.0 * diff))
                curr_anchors = F.normalize(torch.tanh(self.micro_head.adapter(self._anchor_states)), dim=-1)
                anchor_reg = (1.0 - (anchor_before * curr_anchors).sum(dim=-1)).mean()
                loss = torch.stack(losses).mean() + 2.0 * anchor_reg
                loss.backward()
                opt.step()
                if step_i == 0:
                    initial_margin = sum(margins) / len(margins)
                final_margin = sum(margins) / len(margins)

        with torch.no_grad():
            anchor_after = F.normalize(torch.tanh(self.micro_head.adapter(self._anchor_states)), dim=-1)
            anchor_cos = float((anchor_before * anchor_after).sum(dim=-1).mean().item())

        rolled_back = False
        if anchor_cos < self.min_anchor_cosine:
            self.micro_head.adapter.load_state_dict(adapter_snapshot)
            rolled_back = True

        sleep_ms = (time.perf_counter() - t_sleep0) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "challenges_evaluated": len(tasks),
            "episodic_instant_writes": episodic_writes,
            "crystal_vault_writes": crystal_vault_writes,
            "recruited_skills": recruited_skills,
            "verified_preference_pairs": len(verified_pairs),
            "trainable_lora_params": sum(p.numel() for p in trainable_params),
            "initial_preference_margin": round(initial_margin, 5),
            "final_preference_margin": round(final_margin, 5),
            "anchor_retention_cosine": round(anchor_cos, 6),
            "regression_rolled_back": rolled_back,
            "sleep_consolidation_ms": round(sleep_ms, 2),
            "total_cycle_cpu_ms": round(total_ms, 2),
        }


def benchmark_cpu_scan_vs_sequential_gru(seq_len: int = 2048, hidden_dim: int = 512, repeats: int = 3) -> dict[str, float]:
    """Empirical head-to-head CPU benchmark: Sequential Byte GRUCell vs. Patched Parallel Associative Scan."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    byte_ids = torch.randint(0, 256, (1, seq_len), dtype=torch.long, device=device)

    emb = nn.Embedding(259, 128).to(device).eval()
    gru = nn.GRUCell(128, hidden_dim).to(device).eval()
    fast_enc = DiagonalGatedLinearScanEncoder(
        vocab_size=259, embed_dim=128, hidden_dim=hidden_dim, state_dim=hidden_dim, patch_size=4
    ).to(device).eval()

    with torch.no_grad():
        # Warm-up
        _ = fast_enc(byte_ids)
        h = torch.zeros(1, hidden_dim, device=device)
        x_e = emb(byte_ids)
        for t in range(min(32, seq_len)):
            h = gru(x_e[:, t], h)

        # Measure Sequential Byte GRUCell across all 2,048 bytes
        t0 = time.perf_counter()
        for _ in range(repeats):
            h = torch.zeros(1, hidden_dim, device=device)
            x_e = emb(byte_ids)
            for t in range(seq_len):
                h = gru(x_e[:, t], h)
        gru_ms = ((time.perf_counter() - t0) / repeats) * 1000.0

        # Measure Patched Parallel Associative Scan across all 2,048 bytes
        t1 = time.perf_counter()
        for _ in range(repeats):
            _ = fast_enc(byte_ids)
        scan_ms = ((time.perf_counter() - t1) / repeats) * 1000.0

    return {
        "seq_len_bytes": float(seq_len),
        "hidden_dim": float(hidden_dim),
        "sequential_gru_cpu_ms": round(gru_ms, 2),
        "parallel_patched_scan_cpu_ms": round(scan_ms, 2),
        "cpu_speedup_x": round(gru_ms / max(0.01, scan_ms), 2),
    }
