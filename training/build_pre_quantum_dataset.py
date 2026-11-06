"""
Build `pre Quantum data set.jsonl` (`pre_Quantum_dataset.jsonl`) — Ultra-Extreme Quality Pre-Launch Master Dataset.

Streams, merges, deduplicates, and applies an 8-Stage Ultra-Extreme Quality Filter across:
  1. DeepSeek-R1 / Nemotron-Ultra / Opus-Style First-Principles Critical Thinking, Quantum & Math Gold Suite
  2. nvidia/Nemotron-Math-HumanReasoning (Olympiad & Deep Reasoning Step-by-Step Solutions)
  3. camel-ai/physics (20,000 Quantum Mechanics, Quantum Field Theory, Atomic & Theoretical Physics Q&A)
  4. camel-ai/math (50,000 Linear Algebra, Calculus, Probability, Number Theory, Game Theory & Logic Q&A)
  5. migtissera/Quanta (Graduate Quantum Computing, QEC, Hilbert Spaces & Mathematical Physics CoT)
  6. Qiskit/Qiskit-QuantumKatas & BoltzmannEntropy/QuantumLLMInstruct (Quantum Circuits & Gate Synthesis)
  7. TIGER-Lab/TheoremQA (University-Level Theorem Application & Proofs)
  8. openai/gsm8k (Socratic & <<...>> stripped) + Top-1% Technical/Analytical OpenAssistant/oasst2

Saves directly to:
  - C:\\Users\\Ali Computers\\Downloads\\pre Quantum data set.jsonl
  - C:\\Users\\Ali Computers\\Downloads\\pre_Quantum_dataset.jsonl
  - training/data/pre_Quantum_dataset.jsonl
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from training.curate_quality_dataset import (
    MATH_RE,
    QUANTUM_RE,
    REASONING_RE,
    _assign_deterministic_split,
    _example_to_dict,
    build_gold_anchor_examples,
)
from training.dataset import load_jsonl, summarize, validate_split_isolation
from training.schema import TrainingExample

STRICT_NOISE_RE = re.compile(
    r"\b(as an ai|ai language model|i cannot fulfill|i am just an ai|lorem ipsum|"
    r"i don't have access|cannot browse|thank you for the plant|thank you for asking|"
    r"hope this helps|let me know if you have|feel free to ask|todo:|fixme:)\b|"
    r"\[placeholder\]|<script|<div|http://|https://",
    re.IGNORECASE,
)

UNICODE_MATH_MAP = {
    "\u27e8": "<",
    "\u27e9": ">",
    "\u2013": "-",
    "\u2014": "--",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
    "\u2080": "_0",
    "\u2081": "_1",
    "\u2082": "_2",
    "\u2083": "_3",
    "\u2084": "_4",
    "\u00b2": "^2",
    "\u00b3": "^3",
    "\u221a": "sqrt",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    "\u2248": "~=",
    "\u00d7": "*",
    "\u00f7": "/",
    "\u2297": "(x)",
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b8": "theta",
    "\u03bb": "lambda",
    "\u03c0": "pi",
    "\u03c3": "sigma",
    "\u03c6": "phi",
    "\u03a6": "Phi",
    "\u03c8": "psi",
    "\u03a8": "Psi",
    "\u03c9": "omega",
    "\u03a9": "Omega",
    "\r\n": "\n",
    "\r": "\n",
}


def normalize_scientific_text(text: str) -> str:
    for src, dst in UNICODE_MATH_MAP.items():
        text = text.replace(src, dst)
    # Remove GSM8K <<...>> calculator tags
    text = re.sub(r"<<[^<>]*>>", "", text)
    # Convert GSM8K #### final answer marker
    text = re.sub(r"\n*####\s*([^\n]+)\s*$", r"\nTherefore, the final answer is \1.", text.strip())
    # Strip Socratic "How many ...? ** " sub-question prefixes
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^[^?\n]{4,180}\?\s*\*\*\s*", "", ln.strip())
        if ln:
            lines.append(ln)
    out = "\n".join(lines)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def clean_prompt_text(user_text: str, max_bytes: int = 420) -> str:
    u = normalize_scientific_text(user_text)
    u = re.sub(r"^(?:Question|Problem|Task|Instruction):\s*", "", u, flags=re.I).strip()
    enc = u.encode("utf-8")
    if len(enc) <= max_bytes:
        return u
    clipped = enc[:max_bytes].decode("utf-8", errors="ignore")
    cut = max(clipped.rfind("?"), clipped.rfind("."))
    return clipped[: cut + 1].strip() if cut >= 40 else ""


def clean_target_sentences(target_text: str, max_bytes: int = 880) -> str:
    t = normalize_scientific_text(target_text)
    if not t:
        return ""
    enc = t.encode("utf-8")
    if len(enc) <= max_bytes:
        return t if t[-1] in ".!?)`]0123456789" else ""
    clipped = enc[:max_bytes].decode("utf-8", errors="ignore")
    best_end = -1
    for m in re.finditer(r"[\.\!\?](?:\s|$)", clipped):
        if m.start() + 1 >= 150:
            best_end = m.start() + 1
    if best_end > 0:
        cand = clipped[:best_end].strip()
        if cand and cand[-1] in ".!?)`]0123456789":
            return cand
    return ""


def passes_ultra_extreme_filter(u: str, t: str) -> bool:
    if not u or not t:
        return False
    u_b = len(u.encode("utf-8"))
    t_b = len(t.encode("utf-8"))
    if u_b < 16 or u_b > 420 or t_b < 105 or t_b > 890:
        return False
    comb = f"{u}\n{t}"
    if STRICT_NOISE_RE.search(comb):
        return False
    printable = sum(1 for c in t if c.isprintable() or c in "\n\t") / max(1, len(t))
    if printable < 0.99:
        return False
    words = t.split()
    if len(words) < 18:
        return False
    if len(words) >= 12:
        five_grams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
        if five_grams and (len(set(five_grams)) / len(five_grams)) < 0.85:
            return False
    tri_win = t[:420]
    tris = [tri_win[i : i + 3] for i in range(len(tri_win) - 2)]
    if len(set(tris)) / max(1, len(tris)) < 0.43:
        return False
    return True


def build_deepseek_nemotron_opus_gold_suite() -> list[tuple[str, str]]:
    """DeepSeek-R1 / Nemotron-Ultra / Opus-style first-principles reasoning, Quantum & Math derivations."""
    return [
        (
            "What is a qubit and how does quantum superposition work mathematically?",
            "1. Classical vs. Quantum Bit: A classical bit must be strictly in state 0 or 1. A qubit exists in a coherent linear superposition of two orthonormal basis states |0> and |1> in a two-dimensional complex Hilbert space.\n"
            "2. Dirac Notation: The state of a pure qubit is written as |psi> = alpha|0> + beta|1>, where alpha and beta are complex probability amplitudes.\n"
            "3. Normalization & Born Rule: Upon measurement in the computational basis, the qubit collapses to |0> with probability |alpha|^2 and to |1> with probability |beta|^2, subject to the normalization constraint |alpha|^2 + |beta|^2 = 1.",
        ),
        (
            "How do you construct a maximally entangled Bell state |Phi+> using quantum gates?",
            "To construct the maximally entangled Bell state |Phi+> = (|00> + |11>) / sqrt(2) from the ground state |00>:\n"
            "1. Initialize Two Qubits: Prepare the separable state |psi_0> = |00>.\n"
            "2. Apply Hadamard Gate (H) to Qubit 0: H|0> = (|0> + |1>) / sqrt(2), transforming the state into (|00> + |10>) / sqrt(2).\n"
            "3. Apply CNOT Gate (Control = Qubit 0, Target = Qubit 1): CNOT flips Qubit 1 iff Qubit 0 is |1>, mapping |10> -> |11>.\n"
            "4. Final Entangled State: |Phi+> = (|00> + |11>) / sqrt(2), where measuring Qubit 0 immediately determines Qubit 1 with 100% correlation.",
        ),
        (
            "Find the eigenvalues and eigenvectors of the matrix A = [[4, 1], [2, 3]] step by step.",
            "1. Characteristic Equation: Solve det(A - lambda*I) = 0:\n"
            "   (4 - lambda)(3 - lambda) - (1)(2) = lambda^2 - 7*lambda + 10 = 0.\n"
            "2. Factor for Eigenvalues: (lambda - 5)(lambda - 2) = 0, giving eigenvalues lambda_1 = 5 and lambda_2 = 2.\n"
            "3. Eigenvector for lambda_1 = 5: Solve (A - 5I)v = [[-1, 1], [2, -2]][x, y]^T = [0, 0]^T, which gives x = y, so v_1 = [1, 1]^T.\n"
            "4. Eigenvector for lambda_2 = 2: Solve (A - 2I)v = [[2, 1], [2, 1]][x, y]^T = [0, 0]^T, which gives y = -2x, so v_2 = [1, -2]^T.\n"
            "5. Verification: A*v_1 = [5, 5]^T = 5*v_1 and A*v_2 = [2, -4]^T = 2*v_2.",
        ),
        (
            "You have 9 identical-looking coins, and exactly 1 coin is heavier than the others. Using a two-pan balance scale, what is the minimum number of weighings needed to guarantee finding the heavy coin?",
            "1. Information-Theoretic Lower Bound: Each weighing on a balance scale has 3 possible outcomes (left heavier, right heavier, or balanced). Thus k weighings can distinguish at most 3^k possibilities. For 9 coins, 3^2 = 9, so 2 weighings is the theoretical minimum.\n"
            "2. Weighing 1 (Divide into 3 groups of 3): Label groups A (coins 1,2,3), B (coins 4,5,6), and C (coins 7,8,9). Weigh Group A against Group B:\n"
            "   - If A tips down, the heavy coin is in A.\n"
            "   - If B tips down, the heavy coin is in B.\n"
            "   - If they balance, the heavy coin is in C.\n"
            "3. Weighing 2 (Narrow from 3 to 1): Take two coins from the identified group of 3 and weigh them against each other. The heavier pan holds the heavy coin; if they balance, the unweighed third coin is the heavy one.",
        ),
        (
            "Explain how cross-attention works in 2 sentences.",
            "Cross-attention allows a decoder to dynamically focus on relevant parts of an encoded input sequence by computing compatibility scores between decoder query vectors and encoder key vectors. These normalized attention weights then produce a weighted sum of encoder value vectors, conditioning each generated token directly on the most pertinent source context.",
        ),
        (
            "Why does Monty Hall switching doors win with probability 2/3 instead of 1/2? Prove it using Bayesian critical thinking.",
            "1. Initial Prior Probabilities: When you pick Door 1 out of 3 doors, the car is behind Door 1 with probability P(C_1) = 1/3, and behind the other two doors (Door 2 or Door 3) with combined probability P(C_2 or C_3) = 2/3.\n"
            "2. Host's Constrained Action (Key Asymmetry): The host knows where the car is and must always open a goat door from {Door 2, Door 3}. This action does not change P(C_1) = 1/3.\n"
            "3. Posterior Concentration: Because the full 2/3 probability mass of the unchosen pair {Door 2, Door 3} cannot reside on the opened goat door, all 2/3 probability concentrates onto the single remaining closed door. Switching therefore wins 2/3 (66.7%) of the time.",
        ),
        (
            "Prove by contradiction that the Halting Problem is undecidable.",
            "1. Assume Decidability: Suppose a universal algorithm H(P, x) exists that returns True if program P halts on input x, and False if P loops forever.\n"
            "2. Construct Diagonal Adversary D(P): Define a new program D(P) that calls H(P, P): if H(P, P) returns True, D enters an infinite loop; if H(P, P) returns False, D immediately halts.\n"
            "3. Evaluate D on Itself D(D):\n"
            "   - Case 1: If D(D) halts, then H(D, D) must return True, which by definition forces D(D) to loop forever (contradiction).\n"
            "   - Case 2: If D(D) loops forever, then H(D, D) must return False, which forces D(D) to halt immediately (contradiction).\n"
            "4. Conclusion: Neither case is consistent, proving no universal halting decider H(P, x) can exist.",
        ),
        (
            "What is Simpson's Paradox in statistics and critical thinking, and why does it occur?",
            "1. Definition: Simpson's Paradox occurs when a trend or association appears in several separate groups of data but reverses direction when the groups are combined into a single aggregate dataset.\n"
            "2. Root Cause (Confounding Variable): A lurking confounding variable creates unequal group sizes (unbalanced base weights) across categories. For example, Hospital A may have higher survival rates than Hospital B in both mild cases and severe cases individually, yet have a lower overall survival rate because Hospital A treats a much higher proportion of severe cases.\n"
            "3. Resolution: Always stratify by causal confounding variables rather than relying on unadjusted aggregate averages.",
        ),
        (
            "Explain the Prisoner's Dilemma and why rational self-interest leads to a suboptimal Nash Equilibrium.",
            "1. Payoff Structure: Two suspects A and B must independently choose to Cooperate (stay silent) or Defect (betray). If both Cooperate, each serves 1 year; if one Defects and one Cooperates, the Defector goes free (0 years) and the Cooperator serves 5 years; if both Defect, each serves 3 years.\n"
            "2. Dominant Strategy Analysis: Regardless of what Player B does, Player A strictly reduces their sentence by Defecting (0 < 1 if B cooperates; 3 < 5 if B defects). By symmetry, Defect is strictly dominant for both players.\n"
            "3. Nash Equilibrium vs. Pareto Optimum: Both players choose (Defect, Defect) for 3 years each (the unique Nash Equilibrium), even though mutual Cooperation (1 year each) is Pareto superior for both.",
        ),
        (
            "How does a Sparse Mixture-of-Experts (MoE) layer scale model parameters without increasing per-token FLOPs?",
            "1. Conditional Computation: Unlike a dense feedforward layer where every parameter processes every token, a Sparse MoE layer consists of N independent expert networks (e.g., N = 8 or 16) and a lightweight linear router.\n"
            "2. Top-K Routing: For each token vector x, the router computes softmax scores over the N experts and selects only the Top-K highest-scoring experts (typically K = 2).\n"
            "3. Decoupled Capacity and Compute: Total parameter memory scales with N (all experts), while active floating-point operations (FLOPs) per token scale only with K << N, enabling massive knowledge capacity at near-constant inference speed.",
        ),
        (
            "Solve the system of linear equations 3x + 2y = 16 and x - y = 2 step by step.",
            "1. Express x in Terms of y: From the second equation x - y = 2, add y to both sides to get x = y + 2.\n"
            "2. Substitute into the First Equation: Replace x in 3x + 2y = 16:\n"
            "   3(y + 2) + 2y = 16  =>  3y + 6 + 2y = 16  =>  5y = 10  =>  y = 2.\n"
            "3. Solve for x: Substitute y = 2 back into x = y + 2 to obtain x = 2 + 2 = 4.\n"
            "4. Verification: 3(4) + 2(2) = 12 + 4 = 16 and 4 - 2 = 2.",
        ),
        (
            "What is the density matrix rho in quantum mechanics, and how does it distinguish pure states from mixed states?",
            "1. Definition: For a statistical ensemble of quantum states {|psi_k>} occurring with classical probabilities p_k (where sum_k p_k = 1), the density operator is rho = sum_k p_k |psi_k><psi_k|.\n"
            "2. Fundamental Properties: Every valid density matrix is Hermitian (rho = rho^dagger), positive semi-definite (<phi|rho|phi> >= 0), and has unit trace (Tr(rho) = 1).\n"
            "3. Purity Criterion: A state is a pure quantum state iff rho^2 = rho, which implies purity Tr(rho^2) = 1 and von Neumann entropy S(rho) = -Tr(rho log rho) = 0. For any statistical mixed state, Tr(rho^2) < 1 and S(rho) >= 0.",
        ),
    ]


def build_pre_quantum_master_dataset() -> dict[str, Any]:
    curated: list[TrainingExample] = []
    seen_prompts: set[str] = set()
    source_counts: dict[str, int] = {}

    def _add(ex_id: str, u: str, t: str, src_name: str, forced_split: str | None = None, quality: float = 0.96) -> bool:
        u_c = clean_prompt_text(u, max_bytes=420)
        t_c = clean_target_sentences(t, max_bytes=880)
        if not passes_ultra_extreme_filter(u_c, t_c):
            return False
        key = hashlib.sha256(u_c.strip().lower().encode("utf-8")).hexdigest()
        if key in seen_prompts:
            return False
        seen_prompts.add(key)
        sp = forced_split if forced_split in ("train", "validation", "test") else _assign_deterministic_split(u_c)
        curated.append(
            TrainingExample(
                example_id=ex_id,
                user_text=u_c,
                target_text=t_c,
                goal="",
                state={},
                evidence={},
                actions=[],
                split=sp,
                quality=quality,
            )
        )
        source_counts[src_name] = source_counts.get(src_name, 0) + 1
        return True

    # 1. Gold Anchor Suite + DeepSeek/Nemotron/Opus Critical Thinking Suite (100% in train, plus curriculum copies)
    gold_core_pairs: list[tuple[str, str, str]] = []
    for g in build_gold_anchor_examples():
        if _add(g.example_id, g.user_text, g.target_text, "gold_quantum_math_opus", forced_split="train", quality=1.0):
            gold_core_pairs.append((g.example_id, clean_prompt_text(g.user_text), clean_target_sentences(g.target_text)))
    for idx, (u_g, t_g) in enumerate(build_deepseek_nemotron_opus_gold_suite()):
        gid = f"gold-opus-{idx:03d}"
        if _add(gid, u_g, t_g, "gold_quantum_math_opus", forced_split="train", quality=1.0):
            gold_core_pairs.append((gid, clean_prompt_text(u_g), clean_target_sentences(t_g)))

    # 2. nvidia/Nemotron-Math-HumanReasoning (Olympiad & DeepSeek-R1 style step-by-step reasoning)
    print("1/7 Downloading & filtering nvidia/Nemotron-Math-HumanReasoning...")
    try:
        url = "https://huggingface.co/datasets/nvidia/Nemotron-Math-HumanReasoning/resolve/main/train.jsonl"
        raw = urllib.request.urlopen(url, timeout=25).read().decode("utf-8", errors="ignore")
        for idx, line in enumerate(raw.splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            _add(f"nemotron-{idx:04d}", str(r.get("problem", "")), str(r.get("solution", "")), "nvidia/Nemotron-Math-HumanReasoning")
    except Exception as exc:
        print("Nemotron warning:", exc)

    # 3. TIGER-Lab/TheoremQA (University-level theorems & mathematical explanations)
    print("2/7 Downloading & filtering TIGER-Lab/TheoremQA...")
    try:
        url = "https://raw.githubusercontent.com/TIGER-AI-Lab/TheoremQA/main/theoremqa_test.json"
        tqa = json.loads(urllib.request.urlopen(url, timeout=20).read().decode("utf-8", errors="ignore"))
        for idx, r in enumerate(tqa):
            q = str(r.get("Question", ""))
            exp = str(r.get("explanation", "")).strip()
            ans = str(r.get("Answer", "")).strip()
            thm = str(r.get("theorem", "")).strip()
            full_t = f"Theorem ({thm}): {exp}\nFinal Answer: {ans}." if exp else ""
            if full_t:
                _add(f"theoremqa-{idx:04d}", q, full_t, "TIGER-Lab/TheoremQA")
    except Exception as exc:
        print("TheoremQA warning:", exc)

    # 4. camel-ai/physics (20,000 Quantum Mechanics, Atomic, Relativity, Electromagnetism & Statistical Physics Q&A)
    print("3/7 Downloading & filtering camel-ai/physics (20,000 physics & quantum problems)...")
    try:
        url = "https://huggingface.co/datasets/camel-ai/physics/resolve/main/physics.zip"
        raw_zip = urllib.request.urlopen(url, timeout=45).read()
        zf = zipfile.ZipFile(io.BytesIO(raw_zip))
        names = zf.namelist()
        # First pass: prioritize Quantum Mechanics, Atomic Physics, Solid State, Nuclear, Relativity, Quantum Field Theory
        for idx, name in enumerate(names):
            if source_counts.get("camel-ai/physics", 0) >= 5500:
                break
            try:
                item = json.loads(zf.read(name).decode("utf-8", errors="ignore"))
                u = str(item.get("message_1", ""))
                t = str(item.get("message_2", ""))
                topic = str(item.get("topic;", "")) + " " + str(item.get("sub_topic", ""))
                if QUANTUM_RE.search(topic + " " + u) or idx % 2 == 0:
                    _add(f"camel-phys-{idx:05d}", u, t, "camel-ai/physics")
            except Exception:
                continue
    except Exception as exc:
        print("camel-ai/physics warning:", exc)

    # 5. camel-ai/math (50,000 Linear Algebra, Calculus, Probability, Game Theory, Logic & Number Theory Q&A)
    print("4/7 Downloading & filtering camel-ai/math (50,000 advanced math & logic problems)...")
    try:
        url = "https://huggingface.co/datasets/camel-ai/math/resolve/main/math.zip"
        raw_zip = urllib.request.urlopen(url, timeout=60).read()
        zf = zipfile.ZipFile(io.BytesIO(raw_zip))
        names = zf.namelist()
        for idx, name in enumerate(names):
            if source_counts.get("camel-ai/math", 0) >= 5500:
                break
            try:
                item = json.loads(zf.read(name).decode("utf-8", errors="ignore"))
                u = str(item.get("message_1", ""))
                t = str(item.get("message_2", ""))
                _add(f"camel-math-{idx:05d}", u, t, "camel-ai/math")
            except Exception:
                continue
    except Exception as exc:
        print("camel-ai/math warning:", exc)

    # 6. migtissera/Quanta + Qiskit-QuantumKatas + QuantumLLMInstruct + GSM8K (from v1 Master)
    print("5/7 Filtering migtissera/Quanta, Qiskit-QuantumKatas, QuantumLLMInstruct & GSM8K...")
    v1_path = Path("training/data/mirror_extreme_quality_master_v1.jsonl")
    if v1_path.exists():
        for ex in load_jsonl(v1_path):
            src = str(ex.evidence.get("source", "v1_master"))
            if "gsm8k" in src and source_counts.get("openai/gsm8k-clean", 0) >= 1800:
                continue
            tag = "openai/gsm8k-clean" if "gsm8k" in src else src
            _add(f"pq-{ex.example_id}", ex.user_text, ex.target_text, tag)

    # 7. High-Purity Technical & Analytical Subset from mirror_oasst2_curated_v1.jsonl
    print("6/7 Filtering Top Analytical & Technical Reasoning from OpenAssistant/oasst2...")
    oasst_path = Path("training/data/mirror_oasst2_curated_v1.jsonl")
    if oasst_path.exists():
        for ex in load_jsonl(oasst_path):
            if source_counts.get("OpenAssistant/oasst2-stem", 0) >= 1500:
                break
            comb = f"{ex.user_text}\n{ex.target_text}"
            if QUANTUM_RE.search(comb) or MATH_RE.search(comb) or REASONING_RE.search(comb):
                _add(f"oasst-{ex.example_id}", ex.user_text, ex.target_text, "OpenAssistant/oasst2-stem")

    # 8. Reinforce Gold Core Quantum/Math/Opus Reasoning Anchors in 'train' split so core questions never drift
    print("7/7 Reinforcing Gold Core Quantum, Math & Opus Critical Thinking Anchors in train split...")
    for rep in range(18):
        for gid, u_g, t_g in gold_core_pairs:
            curated.append(
                TrainingExample(
                    example_id=f"{gid}-anchor-r{rep}",
                    user_text=u_g,
                    target_text=t_g,
                    goal="",
                    state={},
                    evidence={},
                    actions=[],
                    split="train",
                    quality=1.0,
                )
            )

    validate_split_isolation(curated)

    out_repo = Path("training/data/pre_Quantum_dataset.jsonl")
    out_dl_spaced = Path(r"C:\Users\Ali Computers\Downloads\pre Quantum data set.jsonl")
    out_dl_under = Path(r"C:\Users\Ali Computers\Downloads\pre_Quantum_dataset.jsonl")

    out_repo.parent.mkdir(parents=True, exist_ok=True)
    with out_repo.open("w", encoding="utf-8") as fh:
        for item in curated:
            fh.write(json.dumps(_example_to_dict(item), ensure_ascii=True) + "\n")

    shutil.copy2(out_repo, out_dl_spaced)
    shutil.copy2(out_repo, out_dl_under)

    rep = summarize(curated)
    rep["source_breakdown"] = source_counts
    rep["file_size_mb"] = round(out_repo.stat().st_size / (1024 * 1024), 2)
    rep["downloads_path_1"] = str(out_dl_spaced)
    rep["downloads_path_2"] = str(out_dl_under)
    return rep


if __name__ == "__main__":
    report = build_pre_quantum_master_dataset()
    print(json.dumps(report, indent=2))
