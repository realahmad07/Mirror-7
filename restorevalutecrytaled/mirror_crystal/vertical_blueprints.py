"""
Mirror 7 — Specialized Vertical Intelligence Blueprints.
Expands the Crystal Vault with 3 deep enterprise domains:
  1. Deep Systems Programming (Rust & C++ Memory Safety)
  2. Bioinformatics & Computational Genomics (Sequence Alignment & FM-Index)
  3. Quantitative Finance & Stochastic Calculus (Black-Scholes, Greeks, Risk & Portfolio Sizing)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List

from .disk_vault import DiskCrystalVault
from .grammar_and_sandbox import ZeroTrustSandboxJudge
from .phase7_qwen_to_opus import SixGateDataHygienePipeline


@dataclass
class VerticalRecord:
    domain: str
    prompt: str
    response: str
    test_code: str = ""
    spdx_license: str = "Apache-2.0"
    trust_tier: int = 1


def build_systems_programming_blueprints() -> List[VerticalRecord]:
    """Track 1: Deep Systems Programming (Rust & C++ Memory Safety)."""
    records: List[VerticalRecord] = [
        VerticalRecord(
            domain="systems_programming",
            prompt="Explain the Rust borrow checker invariants and how they guarantee data race freedom.",
            response=(
                "### Rust Borrow Checker Invariants & Data Race Freedom\n\n"
                "The Rust borrow checker enforces an affine type system governing memory access through two mutually exclusive invariants at compile time:\n\n"
                "1. **Aliasing XOR Mutability (The Core Invariant):**\n"
                "   - You may have any number of immutable references (`&T`) to a resource, OR\n"
                "   - You may have exactly one mutable reference (`&mut T`), but never both simultaneously.\n"
                "   - Formally: `&T` implements `Copy`, representing shared read-only access. `&mut T` does NOT implement `Copy`, representing exclusive linear ownership.\n\n"
                "2. **Lifetime Bound Guarantee:**\n"
                "   - Any reference `'a` must strictly outlive the scope of its pointee: `'pointee: 'a`.\n"
                "   - This statically eliminates dangling pointers and use-after-free (UAF) vulnerabilities without runtime garbage collection overhead.\n\n"
                "3. **Concurrency Safety via Marker Traits:**\n"
                "   - `Send`: Indicates ownership of type `T` can be transferred across thread boundaries safely.\n"
                "   - `Sync`: Indicates `&T` is thread-safe (`T: Sync <=> &T: Send`). A type is data-race-free because concurrent threads can only hold shared `&T` (immutable) or coordinate exclusive access via synchronization primitives (`Mutex<T>`)."
            ),
        ),
        VerticalRecord(
            domain="systems_programming",
            prompt="Compare C++ RAII and smart pointers with Rust ownership and Drop semantics.",
            response=(
                "### Systems Memory Management: C++ RAII vs Rust Ownership\n\n"
                "| Dimension | C++ (RAII & Smart Pointers) | Rust (Ownership & Drop) |\n"
                "|---|---|---|\n"
                "| **Resource Ownership** | Manual discipline; objects bound to stack scope (`unique_ptr`, `shared_ptr`) | Language-level affine type system; single owner by default |\n"
                "| **Move Semantics** | Move constructor (`T(T&&)`) leaves source in valid but unspecified state | Destructive bitwise move (`memcpy`); source is statically inaccessible |\n"
                "| **Destruction Trigger** | Destructor `~T()` invoked when exiting lexical scope | `Drop::drop(&mut self)` automatically called at end of owner scope |\n"
                "| **Dangling References** | Permitted by compiler; UB if dereferenced | Statically rejected at compile time via lifetime annotations (`'a`) |\n"
                "| **Nullability** | Raw pointers and uninitialized smart pointers can be `nullptr` | No null pointers; explicit `Option<&T>` zero-cost representation |"
            ),
        ),
        VerticalRecord(
            domain="systems_programming",
            prompt="Explain Pinning in Rust and why self-referential structs require Pin<&mut T> for async execution.",
            response=(
                "### Rust Pinning & Self-Referential Structs\n\n"
                "In standard Rust, all types are movable: passing a value by value bitwise-copies it (`memcpy`) to a new stack location.\n\n"
                "**The Self-Referential Problem:**\n"
                "- An `async fn` compiles into an anonymous compiler-generated state machine generator.\n"
                "- If a local variable holds a pointer or reference across an `.await` point to another field in that same stack frame, moving the future struct invalidates internal pointers, pointing to deallocated memory.\n\n"
                "**The `Pin` Invariant:**\n"
                "- `Pin<P<T>>` wraps a pointer type `P` and revokes the ability to obtain a `&mut T` unless `T: Unpin`.\n"
                "- If `T: !Unpin` (as generated futures are), the pointee's memory location is guaranteed to remain pinned and immobile until `drop()` is executed, preserving internal self-referential integrity safely."
            ),
        ),
        VerticalRecord(
            domain="systems_programming",
            prompt="What is False Sharing in multi-threaded CPU architectures and how do cache line paddings prevent it?",
            response=(
                "### False Sharing & Cache Coherency (MESI Protocol)\n\n"
                "Modern multi-core CPUs manage memory hierarchies in discrete **Cache Lines** (typically 64 bytes in x86-64 and ARM64).\n\n"
                "1. **Mechanism:** When two threads running on separate physical cores independently modify distinct variables that reside within the *same* 64-byte cache line, the hardware cache coherency protocol (MESI/MOESI) invalidates the entire cache line across cores.\n"
                "2. **Impact:** Even though there is no logical data conflict, memory buses experience high bus-traffic contention and stall cycles, collapsing throughput.\n"
                "3. **Mitigation:** Align critical multi-threaded variables to cache-line boundaries using structural padding:\n"
                "   - C++: `alignas(hardware_destructive_interference_size)`\n"
                "   - Rust: `#[repr(align(64))]`"
            ),
        ),
    ]
    return records


def build_bioinformatics_blueprints() -> List[VerticalRecord]:
    """Track 2: Bioinformatics & Computational Biology (Sequence Alignment & Genomic Indexing)."""
    needleman_code = """def needleman_wunsch(seq1: str, seq2: str, match: int = 1, mismatch: int = -1, gap: int = -2) -> tuple[int, str, str]:
    \"\"\"Compute optimal global sequence alignment using Needleman-Wunsch dynamic programming.\"\"\"
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * gap
    for j in range(n + 1):
        dp[0][j] = j * gap

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diag = dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            score_up = dp[i-1][j] + gap
            score_left = dp[i][j-1] + gap
            dp[i][j] = max(score_diag, score_up, score_left)

    align1, align2 = [], []
    i, j = m, n
    while i > 0 and j > 0:
        score_current = dp[i][j]
        diag_val = dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
        if score_current == diag_val:
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif score_current == dp[i-1][j] + gap:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1

    while i > 0:
        align1.append(seq1[i-1])
        align2.append('-')
        i -= 1
    while j > 0:
        align1.append('-')
        align2.append(seq2[j-1])
        j -= 1

    return dp[m][n], "".join(reversed(align1)), "".join(reversed(align2))
"""
    needleman_test = """score, a1, a2 = needleman_wunsch("GATTACA", "GCATGCU", match=1, mismatch=-1, gap=-1)
assert score == 0
assert len(a1) == len(a2)
score_id, _, _ = needleman_wunsch("ACTG", "ACTG", match=2, mismatch=-1, gap=-2)
assert score_id == 8
"""

    bwt_code = """def burrows_wheeler_transform(s: str) -> str:
    \"\"\"Compute Burrows-Wheeler Transform of string terminated by '$'.\"\"\"
    if not s.endswith('$'):
        s += '$'
    n = len(s)
    rotations = sorted([s[i:] + s[:i] for i in range(n)])
    return "".join(r[-1] for r in rotations)
"""
    bwt_test = """assert burrows_wheeler_transform("banana") == "annb$aa"
assert burrows_wheeler_transform("a") == "a$"
"""

    records: List[VerticalRecord] = [
        VerticalRecord(
            domain="bioinformatics",
            prompt="Write a Python function to compute the Needleman-Wunsch global sequence alignment score and traceback.",
            response=(
                "### Needleman-Wunsch Global Alignment Algorithm\n\n```python\n"
                + needleman_code
                + "```\n\n"
                + "- **Time Complexity:** O(M * N) where M and N are sequence lengths.\n"
                + "- **Space Complexity:** O(M * N) dynamic programming table recurrence."
            ),
            test_code=needleman_test,
        ),
        VerticalRecord(
            domain="bioinformatics",
            prompt="Write a Python function to compute the Burrows-Wheeler Transform (BWT) for genome string indexing.",
            response=(
                "### Burrows-Wheeler Transform (BWT)\n\n```python\n"
                + bwt_code
                + "```\n\n"
                + "- **Principle:** Reversibly permutes characters to group repeated contexts together, foundational for the FM-index used in modern aligners (Bowtie2, BWA)."
            ),
            test_code=bwt_test,
        ),
        VerticalRecord(
            domain="bioinformatics",
            prompt="Compare the Needleman-Wunsch and Smith-Waterman sequence alignment algorithms.",
            response=(
                "### Sequence Alignment: Needleman-Wunsch vs Smith-Waterman\n\n"
                "| Feature | Needleman-Wunsch (1970) | Smith-Waterman (1981) |\n"
                "|---|---|---|\n"
                "| **Alignment Scope** | Global alignment (entire length of both sequences) | Local alignment (highest scoring sub-regions) |\n"
                "| **DP Initialization** | $D[i, 0] = i \\cdot g$, $D[0, j] = j \\cdot g$ | $H[i, 0] = 0$, $H[0, j] = 0$ |\n"
                "| **Recurrence Floor** | Unbounded: $\\max(\\text{diag}, \\text{up}, \\text{left})$ | Zero-bounded: $\\max(0, \\text{diag}, \\text{up}, \\text{left})$ |\n"
                "| **Traceback Start** | Bottom-right corner $D[M, N]$ | Cell with maximum score anywhere in matrix |\n"
                "| **Traceback Termination**| Top-left origin $D[0, 0]$ | Any cell where score drops to $0$ |\n"
                "| **Primary Application** | Comparing homologous genes with similar overall length | Identifying conserved domains or motifs in divergent genomes |"
            ),
        ),
    ]
    return records


def build_quantitative_finance_blueprints() -> List[VerticalRecord]:
    """Track 3: Quantitative Finance & Stochastic Calculus."""
    black_scholes_code = """import math

def black_scholes_call_put(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    \"\"\"Compute analytical European Call and Put option prices via Black-Scholes-Merton formula.\"\"\"
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        call = max(0.0, S - K)
        put = max(0.0, K - S)
        return round(call, 4), round(put, 4)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    n_neg_d1 = _norm_cdf(-d1)
    n_neg_d2 = _norm_cdf(-d2)
    discount = math.exp(-r * T)
    call_price = S * nd1 - K * discount * nd2
    put_price = K * discount * n_neg_d2 - S * n_neg_d1
    return round(call_price, 4), round(put_price, 4)
"""
    black_scholes_test = """c, p = black_scholes_call_put(100.0, 100.0, 1.0, 0.05, 0.2)
assert 10.45 <= c <= 10.50
assert 5.55 <= p <= 5.60
# Verify Put-Call Parity: C - P = S - K * exp(-r*T)
assert abs((c - p) - (100.0 - 100.0 * math.exp(-0.05))) < 0.05
"""

    kelly_code = """def kelly_optimal_fraction(win_prob: float, win_loss_ratio: float) -> float:
    \"\"\"Compute Kelly Criterion optimal capital allocation fraction f*.\"\"\"
    if win_loss_ratio <= 0.0:
        return 0.0
    loss_prob = 1.0 - win_prob
    f_star = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
    return max(0.0, min(1.0, round(f_star, 4)))
"""
    kelly_test = """# 60% win rate, 1:1 payout -> f* = 0.20 (20%)
assert kelly_optimal_fraction(0.60, 1.0) == 0.20
# Even odds 50/50 -> f* = 0.0
assert kelly_optimal_fraction(0.50, 1.0) == 0.0
# Negative edge -> f* = 0.0
assert kelly_optimal_fraction(0.40, 1.0) == 0.0
"""

    records: List[VerticalRecord] = [
        VerticalRecord(
            domain="quantitative_finance",
            prompt="Write a Python function to compute European call and put option prices using the Black-Scholes-Merton formula.",
            response=(
                "### Black-Scholes European Option Pricing\n\n```python\n"
                + black_scholes_code
                + "```\n\n"
                "- **Put-Call Parity:** C - P = S_0 - K * exp(-r*T).\n"
                "- **Invariants:** Analytical solution to Black-Scholes PDE with risk-neutral discounting."
            ),
            test_code=black_scholes_test,
        ),
        VerticalRecord(
            domain="quantitative_finance",
            prompt="Write a Python function to calculate the Kelly Criterion optimal capital allocation fraction.",
            response=(
                "### Kelly Criterion Optimal Capital Allocation\n\n```python\n"
                + kelly_code
                + "```\n\n"
                "- **Formula:** f* = (p * b - q) / b where p is win probability, q = 1 - p, and b is net odds.\n"
                "- **Maximization:** Maximizes asymptotic log-wealth growth rate."
            ),
            test_code=kelly_test,
        ),
        VerticalRecord(
            domain="quantitative_finance",
            prompt="Derive the Black-Scholes Greeks: Delta, Gamma, Vega, Theta, and Rho.",
            response=(
                "### The Financial Greeks (Analytical Sensitivities)\n\n"
                "Given cumulative normal $N(x)$ and standard normal PDF $N'(x) = \\frac{1}{\\sqrt{2\\pi}} e^{-x^2/2}$:\n\n"
                "1. **Delta ($\\Delta = \\frac{\\partial V}{\\partial S}$):**\n"
                "   - Call: $\\Delta_{call} = N(d_1) \\in (0, 1)$\n"
                "   - Put: $\\Delta_{put} = N(d_1) - 1 \\in (-1, 0)$\n\n"
                "2. **Gamma ($\\Gamma = \\frac{\\partial^2 V}{\\partial S^2}$):**\n"
                "   - Identical for Calls and Puts: $\\Gamma = \\frac{N'(d_1)}{S \\sigma \\sqrt{T}} > 0$\n\n"
                "3. **Vega ($\\nu = \\frac{\\partial V}{\\partial \\sigma}$):**\n"
                "   - Sensitivity to volatility: $\\nu = S \\sqrt{T} N'(d_1) > 0$\n\n"
                "4. **Theta ($\\Theta = \\frac{\\partial V}{\\partial t}$):**\n"
                "   - Time decay (negative for long positions): $\\Theta_{call} = -\\frac{S N'(d_1) \\sigma}{2\\sqrt{T}} - r K e^{-rT} N(d_2)$\n\n"
                "5. **Rho ($\\rho = \\frac{\\partial V}{\\partial r}$):**\n"
                "   - Interest rate sensitivity: $\\rho_{call} = K T e^{-rT} N(d_2)$, $\\rho_{put} = -K T e^{-rT} N(-d_2)$"
            ),
        ),
        VerticalRecord(
            domain="quantitative_finance",
            prompt="Explain Ito's Lemma and derive the stochastic differential equation for Geometric Brownian Motion log-returns.",
            response=(
                "### Ito's Lemma & Geometric Brownian Motion\n\n"
                "Let asset price $S_t$ follow Geometric Brownian Motion (GBM):\n"
                "\\[ dS_t = \\mu S_t dt + \\sigma S_t dW_t \\]\n"
                "where $W_t$ is a standard Wiener process ($dW_t^2 = dt$, $dt \\cdot dW_t = 0$).\n\n"
                "1. **Ito's Formula for $f(S_t, t)$:**\n"
                "   \\[ df = \\frac{\\partial f}{\\partial t} dt + \\frac{\\partial f}{\\partial S} dS + \\frac{1}{2} \\frac{\\partial^2 f}{\\partial S^2} (dS)^2 \\]\n"
                "2. **Log Transformation:** Let $f(S_t) = \\ln(S_t)$.\n"
                "   - $\\frac{\\partial f}{\\partial S} = \\frac{1}{S}$\n"
                "   - $\\frac{\\partial^2 f}{\\partial S^2} = -\\frac{1}{S^2}$\n"
                "   - $(dS)^2 = (\\sigma S dW_t)^2 = \\sigma^2 S^2 dt$\n"
                "3. **Substitution:**\n"
                "   \\[ d(\\ln S_t) = \\frac{1}{S} (\\mu S dt + \\sigma S dW_t) + \\frac{1}{2} \\left(-\\frac{1}{S^2}\\right) (\\sigma^2 S^2 dt) \\]\n"
                "   \\[ d(\\ln S_t) = \\left(\\mu - \\frac{1}{2}\\sigma^2\\right) dt + \\sigma dW_t \\]\n\n"
                "**Conclusion:** The log-price increments are normally distributed: $\\ln(S_T / S_0) \\sim \\mathcal{N}\\left((\\mu - \\frac{1}{2}\\sigma^2)T, \\sigma^2 T\\right)$. $\\blacksquare$"
            ),
        ),
    ]
    return records


class VerticalBlueprintCurator:
    """Manages synthesis, 6-gate hygiene validation, and vault ingestion for vertical blueprints."""

    def __init__(self) -> None:
        self.hygiene = SixGateDataHygienePipeline()

    def curate_and_ingest(
        self,
        vault: DiskCrystalVault,
        report_path: Path = Path("artifacts/mirror_vertical_curation_report.json"),
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        candidates: List[VerticalRecord] = (
            build_systems_programming_blueprints()
            + build_bioinformatics_blueprints()
            + build_quantitative_finance_blueprints()
        )

        clean_records: List[Dict[str, Any]] = []
        rejections: Dict[str, int] = defaultdict(int)

        for rec in candidates:
            # Six-Gate Audit
            combined = f"{rec.prompt}\n{rec.response}"
            audit = self.hygiene.audit_record(combined, spdx_license=rec.spdx_license)
            if not audit.passed:
                rejections[audit.rejection_gate or "unknown"] += 1
                continue

            # AST sandbox execution if test_code present
            if rec.test_code.strip():
                ok_s, err = ZeroTrustSandboxJudge.verify_code_and_test(rec.response, test_code=rec.test_code)
                if not ok_s:
                    rejections["sandbox_test_failure"] += 1
                    continue

            clean_records.append({
                "prompt": rec.prompt,
                "response": rec.response,
                "tool": "",
                "test_code": rec.test_code,
                "domain": rec.domain,
                "trust_tier": rec.trust_tier,
                "spdx": rec.spdx_license,
            })

        # Ingest into vault
        stats = vault.ingest_batch(clean_records, batch_size=500)
        vault.optimize_disk()
        elapsed = time.perf_counter() - t0

        report = {
            "title": "Mirror 7 Vertical Blueprints Curation Report",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_candidates": len(candidates),
            "clean_passed": len(clean_records),
            "rejections": dict(rejections),
            "vault_stats": stats,
            "vault_total_rules": vault.rule_count(),
            "elapsed_seconds": round(elapsed, 3),
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
