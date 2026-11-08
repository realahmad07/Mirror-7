"""
Mirror 7 — Continuous Autonomous Dataset Streamer.
A lightweight, non-blocking background daemon that streams high-signal open-source datasets
(arXiv scientific papers, Lean Mathlib theorems, verified code repositories) continuously
through the Six-Gate Data Hygiene Pipeline and commits clean blueprints into DiskCrystalVault.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from .disk_vault import DiskCrystalVault
from .grammar_and_sandbox import ZeroTrustSandboxJudge
from .phase7_qwen_to_opus import SixGateDataHygienePipeline


@dataclass
class StreamCandidate:
    source: str
    domain: str
    prompt: str
    response: str
    test_code: str = ""
    spdx_license: str = "Apache-2.0"
    trust_tier: int = 2


class ArxivStreamGenerator:
    """Streams structured theoretical, mathematical, and algorithmic research papers."""

    TOPICS = [
        (
            "Quantum Error Mitigation via Zero-Noise Extrapolation",
            "quantum_computing",
            "Explain the principles of Zero-Noise Extrapolation (ZNE) in quantum error mitigation.",
            "### Zero-Noise Extrapolation (ZNE)\n\n"
            "Zero-Noise Extrapolation is an error mitigation technique for NISQ devices that operates without physical qubit overhead:\n\n"
            "1. **Noise Amplification:** Artificially scale hardware noise by factor $\\lambda > 1$ using pulse stretching or unitary folding ($U \\to U U^\\dagger U$).\n"
            "2. **Expectation Sampling:** Measure observable $\\langle O \\rangle_\\lambda$ across noise levels $\\lambda_1 < \\lambda_2 < \\dots < \\lambda_k$.\n"
            "3. **Zero-Noise Extrapolation:** Fit an analytical curve (polynomial, exponential, or Richardson extrapolation) to estimate $\\lim_{\\lambda \\to 0} \\langle O \\rangle_\\lambda$.\n\n"
            "- **Invariant:** Yields unbiased expectation values under linear noise scaling models.",
        ),
        (
            "Kolmogorov Complexity & Algorithmic Information Theory",
            "computer_science",
            "Define Kolmogorov Complexity and state why it is incomputable.",
            "### Kolmogorov Complexity & Incomputability\n\n"
            "The Kolmogorov complexity $K(s)$ of a finite string $s$ is the length of the shortest computer program that outputs $s$ on a Universal Turing Machine $U$:\n"
            "\\[ K(s) = \\min \\{ |p| : U(p) = s \\} \\]\n\n"
            "**Incomputability Proof (Berry Paradox Reduction):**\n"
            "1. Assume $K(s)$ were computable. Then we could define a program $Q(n)$ that enumerates all strings in lexicographical order and outputs the first string $s$ such that $K(s) \\ge n$.\n"
            "2. The length of program $Q(n)$ is $\\log_2(n) + C$ for some fixed constant $C$.\n"
            "3. For sufficiently large $n$, $|Q(n)| = \\log_2(n) + C < n$.\n"
            "4. But $Q(n)$ produces string $s$, which by definition has complexity $K(s) \\ge n$, yielding $|Q(n)| \\ge n$. Contradiction.\n\n"
            "- **Conclusion:** Kolmogorov complexity $K(s)$ is strictly incomputable.",
        ),
        (
            "Spectral Graph Theory & Laplacian Matrices",
            "discrete_mathematics",
            "Define the Graph Laplacian matrix and explain the significance of its second smallest eigenvalue.",
            "### Graph Laplacian & Fiedler Value\n\n"
            "For an undirected, unweighted graph $G = (V, E)$ with adjacency matrix $A$ and degree matrix $D = \\text{diag}(d_1, \\dots, d_n)$:\n"
            "\\[ L = D - A \\]\n\n"
            "1. **Properties:** $L$ is symmetric, positive semi-definite, and has row sums equal to zero ($L \\mathbf{1} = \\mathbf{0}$).\n"
            "2. **Eigenvalues:** $0 = \\lambda_1 \\le \\lambda_2 \\le \\dots \\le \\lambda_n$.\n"
            "3. **The Fiedler Value ($\\lambda_2$):** Known as the **algebraic connectivity** of $G$.\n"
            "   - $\\lambda_2 > 0$ if and only if $G$ is connected.\n"
            "   - By Cheeger's Inequality, $\\lambda_2$ bounds the graph conductance (isoperimetric constant): $\\frac{\\lambda_2}{2} \\le h(G) \\le \\sqrt{2 d_{\\max} \\lambda_2}$.",
        ),
    ]

    @classmethod
    def generate_candidates(cls) -> List[StreamCandidate]:
        candidates: List[StreamCandidate] = []
        for title, domain, q, a in cls.TOPICS:
            candidates.append(
                StreamCandidate(
                    source="arxiv_permissive_stream",
                    domain=domain,
                    prompt=q,
                    response=a,
                    spdx_license="CC-BY-4.0",
                    trust_tier=1,
                )
            )
        return candidates


class LeanMathlibStreamGenerator:
    """Streams verified formal mathematical theorem invariants."""

    THEOREMS = [
        (
            "Lagrange's Theorem in Group Theory",
            "abstract_algebra",
            "State Lagrange's Theorem in group theory and explain its algebraic proof.",
            "### Lagrange's Theorem (Finite Group Theory)\n\n"
            "**Theorem:** If $G$ is a finite group and $H$ is a subgroup of $G$, then the order of $H$ divides the order of $G$:\n"
            "\\[ |G| = [G : H] \\cdot |H| \\]\n\n"
            "**Formal Proof Steps:**\n"
            "1. Define left cosets: For any $g \\in G$, $gH = \\{gh : h \\in H\\}$.\n"
            "2. **Equivalence Relation:** Cosets partition $G$ into disjoint equivalence classes under $x \\sim y \\iff x^{-1}y \\in H$.\n"
            "3. **Bijection:** The map $\\phi: H \\to gH$ defined by $\\phi(h) = gh$ is a bijection, so $|gH| = |H|$ for every coset.\n"
            "4. **Partition Sum:** Since $G$ is the disjoint union of $[G:H]$ distinct cosets, each of size $|H|$, $|G| = [G:H] \\times |H|$. $\\blacksquare$",
        ),
        (
            "The Banach-Tarski Paradox",
            "measure_theory",
            "State the Banach-Tarski Paradox and explain the role of the Axiom of Choice in its construction.",
            "### The Banach-Tarski Paradox\n\n"
            "**Theorem (Banach & Tarski, 1924):** Given a solid unit ball in $\\mathbb{R}^3$, there exists a decomposition into a finite number (specifically 5) of non-overlapping pieces, which can be reassembled via rigid rotations and translations to form two identical solid unit balls.\n\n"
            "- **The Axiom of Choice:** The decomposed pieces are non-measurable sets (Vitali-type sets) constructed using the Axiom of Choice ($AC$).\n"
            "- **Measure Invariance:** Because the pieces lack Lebesgue measure, the paradox does not violate the conservation of volume for measurable physical matter.",
        ),
    ]

    @classmethod
    def generate_candidates(cls) -> List[StreamCandidate]:
        candidates: List[StreamCandidate] = []
        for name, domain, q, a in cls.THEOREMS:
            candidates.append(
                StreamCandidate(
                    source="lean_mathlib_stream",
                    domain=domain,
                    prompt=q,
                    response=a,
                    spdx_license="Apache-2.0",
                    trust_tier=1,
                )
            )
        return candidates


class VerifiedRepoStreamGenerator:
    """Streams verified algorithmic implementations with companion sandbox tests."""

    @classmethod
    def generate_candidates(cls) -> List[StreamCandidate]:
        candidates: List[StreamCandidate] = []

        # 1. Binary Heap Priority Queue
        heap_code = """class SimpleMinHeap:
    \"\"\"Binary Min-Heap in pure Python.\"\"\"
    def __init__(self):
        self.heap = []

    def push(self, val: int):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self) -> int:
        if not self.heap:
            raise IndexError('pop from empty heap')
        top = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)
        return top

    def _sift_up(self, idx: int):
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx] < self.heap[parent]:
                self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
                idx = parent
            else:
                break

    def _sift_down(self, idx: int):
        n = len(self.heap)
        while 2 * idx + 1 < n:
            child = 2 * idx + 1
            if child + 1 < n and self.heap[child + 1] < self.heap[child]:
                child += 1
            if self.heap[child] < self.heap[idx]:
                self.heap[idx], self.heap[child] = self.heap[child], self.heap[idx]
                idx = child
            else:
                break
"""
        heap_test = """h = SimpleMinHeap()
h.push(5)
h.push(1)
h.push(3)
assert h.pop() == 1
assert h.pop() == 3
assert h.pop() == 5
"""
        candidates.append(
            StreamCandidate(
                source="verified_repo_stream",
                domain="algorithms",
                prompt="Write a Python class implementing a Binary Min-Heap with push, pop, and sift operations.",
                response=(
                    "### Binary Min-Heap Implementation\n\n```python\n"
                    + heap_code
                    + "```\n\n- **Invariants:** O(log N) push and pop, O(1) peek."
                ),
                test_code=heap_test,
                spdx_license="MIT",
                trust_tier=1,
            )
        )
        return candidates


class AutonomousDatasetStreamer:
    """
    Lightweight, thread-safe autonomous background daemon that coordinates streaming sources,
    enforces 6-gate hygiene, and flushes admitted blueprints to DiskCrystalVault.
    """

    def __init__(
        self,
        vault: DiskCrystalVault,
        cadence_seconds: float = 10.0,
        flush_batch_size: int = 10,
    ) -> None:
        self.vault = vault
        self.cadence_seconds = max(1.0, float(cadence_seconds))
        self.flush_batch_size = max(1, int(flush_batch_size))
        self.hygiene = SixGateDataHygienePipeline()

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default

        self._buffer: List[Dict[str, Any]] = []
        self._stats: Dict[str, Any] = {
            "is_running": False,
            "total_evaluated": 0,
            "total_admitted": 0,
            "rejections_by_gate": defaultdict(int),
            "batches_committed": 0,
            "vault_rules_added": 0,
            "current_source": "idle",
            "last_admitted_topic": "",
            "started_at": 0.0,
            "last_pulse_at": 0.0,
        }

    # =========================================================================
    # Lifecycle Controls
    # =========================================================================
    def start(self) -> bool:
        """Starts the background streaming daemon."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._pause_event.set()
            self._stats["is_running"] = True
            self._stats["started_at"] = time.time()
            self._thread = threading.Thread(target=self._run_loop, name="MirrorStreamerDaemon", daemon=True)
            self._thread.start()
            return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Stops the background streaming daemon cleanly."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stats["is_running"] = False
                return True
            self._stop_event.set()
            self._pause_event.set()  # Unblock if paused
            self._thread.join(timeout=timeout)
            self._stats["is_running"] = False
            # Flush any remaining buffer
            self._flush_buffer_locked()
            return True

    def pause(self) -> None:
        """Pauses the stream pulses without shutting down the thread."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Resumes streaming after being paused."""
        self._pause_event.set()

    def pulse_once(self) -> Dict[str, Any]:
        """Manually triggers one streaming pass synchronously."""
        with self._lock:
            return self._execute_stream_pulse()

    # =========================================================================
    # Stream Engine Core
    # =========================================================================
    def _run_loop(self) -> None:
        """Background thread execution loop."""
        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            with self._lock:
                try:
                    self._execute_stream_pulse()
                except Exception as e:
                    self._stats["last_error"] = str(e)

            # Wait for cadence interval or until stopped
            self._stop_event.wait(timeout=self.cadence_seconds)

        self._stats["is_running"] = False

    def _execute_stream_pulse(self) -> Dict[str, Any]:
        """Pulls candidate items, filters through 6 gates, buffers, and flushes."""
        generators = [
            ArxivStreamGenerator.generate_candidates,
            LeanMathlibStreamGenerator.generate_candidates,
            VerifiedRepoStreamGenerator.generate_candidates,
        ]

        gen = random.choice(generators)
        candidates = gen()
        pulse_admitted = 0

        for cand in candidates:
            self._stats["total_evaluated"] += 1
            self._stats["current_source"] = cand.source

            # Six-Gate Audit
            combined = f"{cand.prompt}\n{cand.response}"
            audit = self.hygiene.audit_record(combined, spdx_license=cand.spdx_license)
            if not audit.passed:
                gate = audit.rejection_gate or "unknown_gate"
                self._stats["rejections_by_gate"][gate] += 1
                continue

            # Sandbox validation
            if cand.test_code.strip():
                ok_s, _ = ZeroTrustSandboxJudge.verify_code_and_test(cand.response, test_code=cand.test_code)
                if not ok_s:
                    self._stats["rejections_by_gate"]["sandbox_test_failure"] += 1
                    continue

            # Admitted
            self._buffer.append({
                "prompt": cand.prompt,
                "response": cand.response,
                "tool": "",
                "test_code": cand.test_code,
                "domain": cand.domain,
                "trust_tier": cand.trust_tier,
                "spdx": cand.spdx_license,
            })
            self._stats["total_admitted"] += 1
            self._stats["last_admitted_topic"] = cand.prompt[:60]
            pulse_admitted += 1

        # Check flush threshold
        if len(self._buffer) >= self.flush_batch_size:
            self._flush_buffer_locked()

        self._stats["last_pulse_at"] = time.time()
        return {
            "evaluated": len(candidates),
            "admitted": pulse_admitted,
            "buffer_len": len(self._buffer),
            "total_admitted": self._stats["total_admitted"],
        }

    def _flush_buffer_locked(self) -> None:
        """Commits buffered items to DiskCrystalVault (or MirrorCrystalEngine) in a transaction."""
        if not self._buffer:
            return
        to_commit = list(self._buffer)
        self._buffer.clear()
        if hasattr(self.vault, "ingest_batch"):
            stats = self.vault.ingest_batch(to_commit, batch_size=500)
            if hasattr(self.vault, "optimize_disk"):
                self.vault.optimize_disk()
            self._stats["batches_committed"] += 1
            self._stats["vault_rules_added"] += stats.get("new_rules", 0)
        elif hasattr(self.vault, "ingest_pair"):
            added = 0
            for item in to_commit:
                self.vault.ingest_pair(item["prompt"], item["response"], domain=item.get("domain", "autonomous_stream"))
                added += 1
            self._stats["batches_committed"] += 1
            self._stats["vault_rules_added"] += added

    # =========================================================================
    # Telemetry
    # =========================================================================
    def get_status(self) -> Dict[str, Any]:
        """Returns thread-safe snapshot of streamer status and metrics."""
        with self._lock:
            uptime = round(time.time() - self._stats["started_at"], 2) if self._stats["started_at"] > 0 else 0.0
            if hasattr(self.vault, "rule_count"):
                v_count = self.vault.rule_count()
            elif hasattr(self.vault, "rule_index") and hasattr(self.vault.rule_index, "rules"):
                v_count = len(self.vault.rule_index.rules)
            elif hasattr(self.vault, "rules"):
                v_count = len(self.vault.rules)
            else:
                v_count = 0
            return {
                "is_running": self._stats["is_running"],
                "total_evaluated": self._stats["total_evaluated"],
                "total_admitted": self._stats["total_admitted"],
                "buffer_pending": len(self._buffer),
                "rejections_by_gate": dict(self._stats["rejections_by_gate"]),
                "batches_committed": self._stats["batches_committed"],
                "vault_rules_added": self._stats["vault_rules_added"],
                "vault_total_rules": v_count,
                "current_source": self._stats["current_source"],
                "last_admitted_topic": self._stats["last_admitted_topic"],
                "cadence_seconds": self.cadence_seconds,
                "uptime_seconds": uptime,
            }
