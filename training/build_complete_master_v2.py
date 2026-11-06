"""
Build `mirror_complete_master_v2.jsonl` — The Complete, Self-Contained Master Dataset for Mirror 7 V5.

Fixes all three training bottlenecks observed in v1:
1. ZERO Boilerplate Header Dilution: `goal=""`, `state={}`, `evidence={}`, `actions=[]` so 100% of encoder
   attention focuses on the actual user prompt (`user_text`), preventing cross-prompt confusion.
2. ZERO Mid-Sentence Chunk Truncation: Every target is a complete, self-contained response (`110..900` bytes)
   ending on a clean sentence boundary (`.` / `!` / `?` / `]`), eliminating `[CONT k/n]` mid-sentence EOS cutoffs.
3. ZERO Calculator / Socratic Noise: Strips all `<<...>>` calculator tags and `Question? **` Socratic artifacts,
   and balances 4 pillars:
     - General English Fluency & Instruction Following (from OASST2 curated)
     - Quantum Computing & Quantum Mechanics (Gold Q&A + Quanta concise + QuantumLLMInstruct + Qiskit)
     - Mathematics: Linear Algebra, Eigenvalues, Calculus, Proofs & Clean Step-by-Step Math
     - Critical Thinking, Algorithms & Logical Problem Solving
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from training.curate_quality_dataset import (
    MATH_RE,
    QUANTUM_RE,
    REASONING_RE,
    REJECT_NOISE_RE,
    _assign_deterministic_split,
    _example_to_dict,
    _normalize_text,
    build_gold_anchor_examples,
)
from training.dataset import load_jsonl, summarize, validate_split_isolation
from training.schema import TrainingExample


def _clean_socratic_and_calc(text: str) -> str:
    """Remove GSM8K `<<...>>` calculator annotations and `Subquestion? **` Socratic prefixes."""
    # Remove <<...>> calculator tags
    text = re.sub(r"<<[^<>]*>>", "", text)
    # Remove Socratic sub-questions at the start of lines like "How many ...? ** "
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        line = re.sub(r"^[^?\n]{4,180}\?\s*\*\*\s*", "", line.strip())
        if line:
            cleaned_lines.append(line)
    out = "\n".join(cleaned_lines)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def _truncate_to_complete_sentences(text: str, max_bytes: int = 900) -> str:
    """Extract a complete, self-contained multi-sentence response <= max_bytes without mid-sentence cutoffs."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text.strip()
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    best_end = -1
    for match in re.finditer(r"[\.\!\?](?:\s|$)", clipped):
        pos = match.start() + 1
        if pos >= 160:
            best_end = pos
    if best_end > 0:
        cand = clipped[:best_end].strip()
        if cand and cand[-1] in ".!?)`]":
            return cand
    return ""


def _clean_prompt_to_concise(user_text: str, max_user_bytes: int = 420) -> str:
    """Keep prompts crisp and direct so encoder cross-attention locks onto the core question."""
    u = _normalize_text(user_text)
    u = re.sub(r"^Question:\s*", "", u, flags=re.I).strip()
    if len(u.encode("utf-8")) <= max_user_bytes:
        return u
    # If it's a long multi-part Quanta prompt, take the core problem statement + Part A
    clipped = u.encode("utf-8")[:max_user_bytes].decode("utf-8", errors="ignore")
    best_q = max(clipped.rfind("?"), clipped.rfind("."))
    if best_q >= 45:
        return clipped[: best_q + 1].strip()
    return ""


def _make_pure_example(
    example_id: str,
    user_text: str,
    target_text: str,
    split: str | None = None,
    quality: float = 0.95,
) -> TrainingExample:
    """Create a TrainingExample with ZERO artificial header boilerplate so 100% of attention hits user_text."""
    sp = split if split in ("train", "validation", "test") else _assign_deterministic_split(user_text)
    return TrainingExample(
        example_id=example_id,
        user_text=user_text,
        target_text=target_text,
        goal="",
        state={},
        evidence={},
        actions=[],
        split=sp,
        quality=float(quality),
    )


def _extra_gold_examples() -> list[tuple[str, str, str]]:
    """High-precision, self-contained Q&A pairs for Quantum Computing, Math, Critical Thinking & English."""
    return [
        (
            "What is a qubit and how does quantum superposition work mathematically?",
            "A qubit (quantum bit) is a two-level quantum system residing in a two-dimensional complex Hilbert space C^2 with orthonormal basis states |0> = [1, 0]^T and |1> = [0, 1]^T.\n"
            "1. Superposition State: Unlike a classical bit that is strictly 0 or 1, a pure qubit state is a linear combination |psi> = alpha|0> + beta|1>, where alpha, beta in C are complex probability amplitudes.\n"
            "2. Born Rule & Normalization: Measuring |psi> in the computational basis yields outcome 0 with probability P(0) = |alpha|^2 and outcome 1 with probability P(1) = |beta|^2, subject to the normalization constraint |alpha|^2 + |beta|^2 = 1.\n"
            "3. Quantum Interference: Because alpha and beta carry relative phase e^(i*phi), quantum gates such as the Hadamard gate H = (1/sqrt(2))*[[1, 1], [1, -1]] can constructively or destructively interfere amplitudes before measurement.",
        ),
        (
            "How do you construct a maximally entangled Bell state |Phi+> using quantum gates?",
            "To construct the maximally entangled Bell state |Phi+> = (|00> + |11>) / sqrt(2) from the two-qubit ground state |00>:\n"
            "1. Initialize Two Qubits: Start in the separable computational basis state |psi_0> = |0> (x) |0> = |00>.\n"
            "2. Apply Hadamard Gate (H) to Qubit 0: H|0> = (|0> + |1>) / sqrt(2), transforming the two-qubit state into |psi_1> = ((|0> + |1>) / sqrt(2)) (x) |0> = (|00> + |10>) / sqrt(2).\n"
            "3. Apply CNOT Gate (Control = Qubit 0, Target = Qubit 1): CNOT flips the target qubit iff the control qubit is |1>, mapping |00> -> |00> and |10> -> |11>.\n"
            "4. Final Entangled State: |Phi+> = (|00> + |11>) / sqrt(2). Neither qubit has a definite individual state (Tr_B(|Phi+><Phi+|) = I/2), and measuring Qubit 0 immediately determines Qubit 1 with 100% correlation.",
        ),
        (
            "Find the eigenvalues and eigenvectors of the matrix A = [[4, 1], [2, 3]] step by step.",
            "1. Characteristic Equation: Solve det(A - lambda*I) = 0:\n"
            "   det([[4 - lambda, 1], [2, 3 - lambda]]) = (4 - lambda)(3 - lambda) - (1)(2) = lambda^2 - 7*lambda + 10 = 0.\n"
            "2. Factor for Eigenvalues: (lambda - 5)(lambda - 2) = 0, so the eigenvalues are lambda_1 = 5 and lambda_2 = 2.\n"
            "3. Eigenvector for lambda_1 = 5: Solve (A - 5I)v = [[-1, 1], [2, -2]][x, y]^T = [0, 0]^T, which gives -x + y = 0, so x = y. Choosing x = 1 gives v_1 = [1, 1]^T.\n"
            "4. Eigenvector for lambda_2 = 2: Solve (A - 2I)v = [[2, 1], [2, 1]][x, y]^T = [0, 0]^T, which gives 2x + y = 0, so y = -2x. Choosing x = 1 gives v_2 = [1, -2]^T.\n"
            "5. Verification: A*v_1 = [5, 5]^T = 5*v_1 and A*v_2 = [2, -4]^T = 2*v_2.",
        ),
        (
            "Explain how cross-attention works in 2 sentences.",
            "Cross-attention allows a decoder to dynamically focus on relevant parts of an encoded input sequence by computing compatibility scores between decoder query vectors and encoder key vectors. These normalized attention weights then produce a weighted sum of encoder value vectors, conditioning each generated token directly on the most pertinent source context.",
        ),
        (
            "What are the Pauli matrices X, Y, and Z in quantum computing and how do they act on |0> and |1>?",
            "The single-qubit Pauli operators are 2x2 unitary, Hermitian, involutory matrices (P^2 = I):\n"
            "1. Pauli-X (Bit-Flip Gate): X = [[0, 1], [1, 0]], which maps X|0> = |1> and X|1> = |0>.\n"
            "2. Pauli-Z (Phase-Flip Gate): Z = [[1, 0], [0, -1]], which leaves |0> unchanged (Z|0> = |0>) and flips the phase of |1> (Z|1> = -|1>).\n"
            "3. Pauli-Y (Combined Bit & Phase Flip): Y = [[0, -i], [i, 0]] = i*X*Z, which maps Y|0> = i|1> and Y|1> = -i|0>.\n"
            "Together with the identity matrix I, {I, X, Y, Z} form an orthogonal basis for all 2x2 Hermitian operators.",
        ),
        (
            "What is the difference between Grover's algorithm and Shor's algorithm?",
            "1. Problem Solved: Grover's algorithm searches an unstructured database of N items for a marked solution, whereas Shor's algorithm factors large composite integers N = p*q and computes discrete logarithms.\n"
            "2. Quantum Speedup: Grover's algorithm achieves a quadratic speedup, reducing query complexity from O(N) classical checks to O(sqrt(N)) quantum oracle + diffusion iterations. Shor's algorithm achieves an exponential speedup over the best known classical sub-exponential factoring algorithms via Quantum Phase Estimation (QPE) and the Quantum Fourier Transform (QFT).\n"
            "3. Cryptographic Impact: Shor's algorithm breaks RSA and ECC public-key cryptography, while Grover's algorithm halves the effective bit-security of symmetric ciphers (making AES-256 equivalent to 128-bit security).",
        ),
        (
            "Solve the quadratic equation 2x^2 - 7x + 3 = 0 step by step.",
            "1. Identify Coefficients: For 2x^2 - 7x + 3 = 0, we have a = 2, b = -7, and c = 3.\n"
            "2. Compute the Discriminant: Delta = b^2 - 4ac = (-7)^2 - 4(2)(3) = 49 - 24 = 25.\n"
            "3. Apply the Quadratic Formula: x = (-b +/- sqrt(Delta)) / (2a) = (7 +/- 5) / 4.\n"
            "4. Find Both Roots:\n"
            "   - x_1 = (7 + 5) / 4 = 12 / 4 = 3.\n"
            "   - x_2 = (7 - 5) / 4 = 2 / 4 = 1/2 = 0.5.\n"
            "5. Verification: 2(3)^2 - 7(3) + 3 = 18 - 21 + 3 = 0, and 2(0.5)^2 - 7(0.5) + 3 = 0.5 - 3.5 + 3 = 0.",
        ),
        (
            "Find the derivative of f(x) = x^2 * sin(3x) using the product rule and chain rule.",
            "1. Product Rule Setup: Write f(x) = u(x) * v(x) where u(x) = x^2 and v(x) = sin(3x). The product rule states f'(x) = u'(x)v(x) + u(x)v'(x).\n"
            "2. Differentiate Each Factor:\n"
            "   - Power rule on u(x): u'(x) = d/dx(x^2) = 2x.\n"
            "   - Chain rule on v(x): v'(x) = cos(3x) * d/dx(3x) = 3*cos(3x).\n"
            "3. Combine and Factor: f'(x) = 2x*sin(3x) + 3x^2*cos(3x) = x*(2*sin(3x) + 3x*cos(3x)).",
        ),
        (
            "Evaluate the definite integral of f(x) = 3x^2 + 4x - 2 from x = 0 to x = 2.",
            "1. Find the Antiderivative F(x): Integrate term by term using the power rule:\n"
            "   F(x) = 3*(x^3 / 3) + 4*(x^2 / 2) - 2x = x^3 + 2x^2 - 2x.\n"
            "2. Apply the Fundamental Theorem of Calculus (F(2) - F(0)):\n"
            "   - F(2) = (2)^3 + 2(2)^2 - 2(2) = 8 + 8 - 4 = 12.\n"
            "   - F(0) = 0 + 0 - 0 = 0.\n"
            "3. Final Value: The definite integral equals 12 - 0 = 12.",
        ),
        (
            "How does quantum decoherence affect a qubit, and what is the difference between T1 and T2 relaxation times?",
            "Quantum decoherence occurs when a qubit unintentionally entangles with its noisy environment, causing pure superposition states to decay into classical statistical mixtures.\n"
            "1. Longitudinal Relaxation Time (T1 — Energy Decay): Measures the characteristic time for an excited state |1> to decay back to the ground state |0> by exchanging energy with the environment (damping diagonal density-matrix elements).\n"
            "2. Transverse Dephasing Time (T2 — Phase Loss): Measures the lifetime of phase coherence between |0> and |1> in a superposition like |+> = (|0> + |1>)/sqrt(2) (damping off-diagonal density-matrix elements rho_01), bounded physically by T2 <= 2*T1.",
        ),
    ]


def build_complete_master_v2(
    v1_master_path: str | Path = "training/data/mirror_extreme_quality_master_v1.jsonl",
    oasst_path: str | Path = "training/data/mirror_oasst2_curated_v1.jsonl",
    output_path: str | Path = "training/data/mirror_complete_master_v2.jsonl",
    desktop_copy_path: str | Path | None = r"C:\Users\Ali Computers\Desktop\mirror_complete_master_v2.jsonl",
) -> dict[str, object]:
    curated: list[TrainingExample] = []
    seen_prompts: set[str] = set()
    category_counts = {
        "gold_core": 0,
        "quantum_concise": 0,
        "math_clean": 0,
        "critical_thinking": 0,
        "english_general": 0,
    }

    def _add(ex_id: str, u: str, t: str, cat: str, split: str | None = None, quality: float = 0.95) -> bool:
        u_c = _clean_prompt_to_concise(u, max_user_bytes=420)
        t_c = _truncate_to_complete_sentences(_clean_socratic_and_calc(_normalize_text(t)), max_bytes=880)
        if not u_c or not t_c:
            return False
        if REJECT_NOISE_RE.search(f"{u_c}\n{t_c}"):
            return False
        u_bytes = len(u_c.encode("utf-8"))
        t_bytes = len(t_c.encode("utf-8"))
        if u_bytes < 14 or u_bytes > 420 or t_bytes < 90 or t_bytes > 900:
            return False
        if t_c[-1] not in ".!?)`]0123456789":
            return False
        key = hashlib.sha256(u_c.strip().lower().encode("utf-8")).hexdigest()
        if key in seen_prompts:
            return False
        seen_prompts.add(key)
        curated.append(_make_pure_example(ex_id, u_c, t_c, split=split, quality=quality))
        category_counts[cat] = category_counts.get(cat, 0) + 1
        return True

    # 1. Gold Anchor Examples (Handcrafted Quantum, Math, Critical Thinking & Core Q&A)
    for g in build_gold_anchor_examples():
        _add(g.example_id, g.user_text, g.target_text, "gold_core", split="train", quality=1.0)
    for idx, (u_g, t_g) in enumerate(_extra_gold_examples()):
        _add(f"gold-extra-{idx:03d}", u_g, t_g, "gold_core", split="train", quality=1.0)

    # 2. Extract Clean Quantum, Math, and Critical Thinking from v1 Master
    if Path(v1_master_path).exists():
        v1_items = load_jsonl(v1_master_path)
        for ex in v1_items:
            dom = str(ex.state.get("domain", ""))
            src = str(ex.evidence.get("source", ""))
            if dom == "quantum_computing":
                if category_counts["quantum_concise"] < 1800:
                    _add(f"v2-{ex.example_id}", ex.user_text, ex.target_text, "quantum_concise", quality=0.95)
            elif dom == "mathematics":
                # Limit GSM8K word problems to 1,100 clean (non-Socratic, non-<<>>) examples so they don't dominate
                if "gsm8k" in src and category_counts["math_clean"] >= 1100:
                    continue
                if category_counts["math_clean"] < 1800:
                    _add(f"v2-{ex.example_id}", ex.user_text, ex.target_text, "math_clean", quality=0.92)
            elif dom == "critical_thinking":
                if category_counts["critical_thinking"] < 700:
                    _add(f"v2-{ex.example_id}", ex.user_text, ex.target_text, "critical_thinking", quality=0.92)

    # 3. Extract High-Quality Concise General English & STEM from mirror_oasst2_curated_v1.jsonl
    if Path(oasst_path).exists():
        oasst_items = load_jsonl(oasst_path)
        for ex in oasst_items:
            combined = f"{ex.user_text}\n{ex.target_text}"
            if QUANTUM_RE.search(combined) and category_counts["quantum_concise"] < 1900:
                _add(f"v2-oasst-q-{ex.example_id}", ex.user_text, ex.target_text, "quantum_concise", quality=0.94)
            elif MATH_RE.search(combined) and category_counts["math_clean"] < 1900:
                _add(f"v2-oasst-m-{ex.example_id}", ex.user_text, ex.target_text, "math_clean", quality=0.92)
            elif REASONING_RE.search(combined) and category_counts["critical_thinking"] < 900:
                _add(f"v2-oasst-c-{ex.example_id}", ex.user_text, ex.target_text, "critical_thinking", quality=0.92)
            elif category_counts["english_general"] < 2400:
                # High-clarity general English instruction/explanation
                words = ex.target_text.split()
                if len(words) >= 20:
                    _add(f"v2-oasst-e-{ex.example_id}", ex.user_text, ex.target_text, "english_general", quality=0.90)

    validate_split_isolation(curated)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in curated:
            fh.write(json.dumps(_example_to_dict(item), ensure_ascii=True) + "\n")

    if desktop_copy_path:
        desk = Path(desktop_copy_path)
        shutil.copy2(out, desk)

    report = summarize(curated)
    report["category_breakdown"] = category_counts
    report["output_path"] = str(out)
    report["desktop_copy"] = str(desktop_copy_path) if desktop_copy_path else None
    report["file_size_mb"] = round(out.stat().st_size / (1024 * 1024), 2)
    return report


if __name__ == "__main__":
    rep = build_complete_master_v2()
    print(json.dumps(rep, indent=2))
