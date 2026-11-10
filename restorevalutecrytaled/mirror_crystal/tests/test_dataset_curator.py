from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pytest

from mirror_crystal.dataset_curator import (
    CurationConfig,
    SourceAuditStats,
    _assert_train_split,
    _has_persona_leakage,
    clean_gsm8k_answer,
    curate_all_datasets,
    extract_pennylane_sections,
    extract_qiskit_sections,
    extract_thealgorithms_from_zip_bytes,
)
from mirror_crystal.phase7_qwen_to_opus import SixGateDataHygienePipeline


def _make_sample_algorithms_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "Python-master/graphs/shortest_sample.py",
            (
                "import heapq\n"
                "\n"
                "def _build_adj(n, edges):\n"
                "    adj = [[] for _ in range(n)]\n"
                "    for u, v, w in edges:\n"
                "        adj[u].append((v, w))\n"
                "    return adj\n"
                "\n"
                "def shortest_hop_cost(n: int, edges: list[tuple[int, int, int]], src: int) -> list[int]:\n"
                "    '''\n"
                "    Compute weighted shortest path costs from src.\n"
                "    >>> shortest_hop_cost(3, [(0, 1, 2), (1, 2, 3)], 0)\n"
                "    [0, 2, 5]\n"
                "    '''\n"
                "    adj = _build_adj(n, edges)\n"
                "    dist = [10**9] * n\n"
                "    dist[src] = 0\n"
                "    pq = [(0, src)]\n"
                "    while pq:\n"
                "        d, u = heapq.heappop(pq)\n"
                "        if d > dist[u]:\n"
                "            continue\n"
                "        for v, w in adj[u]:\n"
                "            if d + w < dist[v]:\n"
                "                dist[v] = d + w\n"
                "                heapq.heappush(pq, (dist[v], v))\n"
                "    return dist\n"
            ),
        )
        zf.writestr(
            "Python-master/other/bad_os_script.py",
            (
                "import os\n"
                "def list_cwd() -> str:\n"
                "    '''\n"
                "    Unsafe OS accessor.\n"
                "    >>> list_cwd()\n"
                "    'ok'\n"
                "    '''\n"
                "    return os.getcwd()\n"
            ),
        )
        zf.writestr(
            "Python-master/maths/wrong_doctest.py",
            (
                "def square_val(x: int) -> int:\n"
                "    '''\n"
                "    Intentionally wrong doctest.\n"
                "    >>> square_val(3)\n"
                "    99\n"
                "    '''\n"
                "    return x * x\n"
            ),
        )
    return buf.getvalue()


def test_split_guard_rejects_test_and_eval():
    _assert_train_split("train")
    for bad in ("test", "eval", "evaluation", "eval_holdout"):
        with pytest.raises(ValueError):
            _assert_train_split(bad)


def test_extract_thealgorithms_sandbox_and_helper_closure():
    pipeline = SixGateDataHygienePipeline()
    stats = SourceAuditStats("TheAlgorithms/Python", "MIT", "train", "coding")
    recs = extract_thealgorithms_from_zip_bytes(_make_sample_algorithms_zip(), pipeline=pipeline, stats=stats)

    assert len(recs) == 1
    assert recs[0]["trust_tier"] == 1
    assert recs[0]["spdx_license"] == "MIT"
    assert "_build_adj" in recs[0]["target_text"]
    assert "assert (shortest_hop_cost" in recs[0]["test_code"]
    assert stats.sandbox_verified == 1
    assert stats.sandbox_rejected == 2


def test_extract_qiskit_sections_clean_markdown_and_ipynb():
    nb_obj = {
        "cells": [
            {"cell_type": "markdown", "source": ["---\ntitle: Quantum Teleportation\n---\n\n# Quantum Teleportation\n\n"]},
            {
                "cell_type": "markdown",
                "source": [
                    "## Protocol Overview <a id='overview'></a>\n\n"
                    "Quantum teleportation transmits an unknown qubit state using one shared Bell pair and two classical bits "
                    "without cloning the original quantum state cleanly across noisy links."
                ],
            },
            {
                "cell_type": "code",
                "source": [
                    "!pip install qiskit\n",
                    "from qiskit import QuantumCircuit\n",
                    "qc = QuantumCircuit(3, 2)\n",
                    "qc.h(1)\n",
                    "qc.cx(1, 2)\n",
                ],
            },
        ]
    }
    pairs = extract_qiskit_sections(json.dumps(nb_obj), "learning/courses/quantum-teleportation.ipynb", "Qiskit")
    assert len(pairs) >= 1
    prompt, answer = pairs[0]
    assert "quantum computing" in prompt.lower()
    assert "!pip install" not in answer
    assert "<a id=" not in answer
    assert "QuantumCircuit(3, 2)" in answer


def test_clean_gsm8k_answer_and_offline_end_to_end(tmp_path: Path):
    formatted, ans = clean_gsm8k_answer(
        "Alice buys 4 books at $12 each = $<<4*12=48>>48.\nShe pays with $60 and gets 60 - 48 = <<60-48=12>>12 change.\n#### 12"
    )
    assert ans == "12"
    assert "<<" not in formatted
    assert "Step 1:" in formatted
    assert "Final Answer: 12" in formatted

    cfg = CurationConfig(
        output_jsonl_path=tmp_path / "curated.jsonl",
        report_json_path=tmp_path / "report.json",
        codefeedback_target=0,
        mathinstruct_target=0,
        oasst2_target=0,
        slimorca_target=0,
        cosmopedia_target=0,
        commonsense_target=0,
    )
    bundle = {
        "thealgorithms_zip": _make_sample_algorithms_zip(),
        "qiskit_files": [
            (
                "docs/entanglement.mdx",
                "# Bell States\n\n## Maximally Entangled Pairs\n\n"
                "A Bell state forms a maximally entangled two-qubit superposition where measuring one qubit immediately "
                "determines the computational basis outcome of the partner qubit.",
            )
        ],
        "gsm8k_rows": [
            {
                "question": "A bookstore sells 5 notebooks for $8 each. How much do all 5 notebooks cost?",
                "answer": "Multiply 5 * 8 = $<<5*8=40>>40.\nThe total bill equals $40.\n#### 40",
            },
            # Decontamination probe must be dropped by Gate 6!
            {
                "question": "def has_close_elements(numbers, threshold): check HumanEval leak",
                "answer": "Step 1: Naive pair check.\n#### False",
            },
        ],
    }
    report = curate_all_datasets(config=cfg, offline_bundle=bundle)
    assert report["split_enforced"] == "train"
    assert report["total_curated_records"] == 3
    assert report["total_sandbox_verified"] == 1
    assert cfg.output_jsonl_path.is_file()
    assert cfg.report_json_path.is_file()

    rows = [json.loads(ln) for ln in cfg.output_jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 3
    assert all(r["split"] == "train" for r in rows)
    assert all("has_close_elements" not in r["user_text"] for r in rows)


def test_persona_leakage_and_pennylane_extractor():
    assert _has_persona_leakage("Who are you?", "Hello! I am Open Assistant, trained by LAION.") is True
    assert _has_persona_leakage("Who are you?", "As OpenAssistant model, I help users.") is True
    assert _has_persona_leakage("Hello", "As an AI language model, I have a knowledge cutoff.") is True
    assert _has_persona_leakage("Explain Bell states", "Bell states are maximally entangled two-qubit pairs.") is False

    pl_demo = '''r"""
Variational Quantum Eigensolver
===============================

.. meta::
    :property="og:description": Ground state chemistry

The Variational Quantum Eigensolver uses parameterized quantum circuits to estimate
ground state energies of molecular Hamiltonians using the variational principle.
It alternates a quantum ansatz circuit with a classical optimizer step.
"""
# %%
import pennylane as qml

def run_ansatz(theta):
    return theta * 2
'''
    pairs = extract_pennylane_sections(pl_demo, "demonstrations_v2/tutorial_vqe/demo.py")
    assert len(pairs) == 1
    p_q, p_a = pairs[0]
    assert "Variational Quantum Eigensolver" in p_q
    assert ".. meta::" not in p_a
    assert "def run_ansatz" in p_a


def test_odd_code_fence_and_trailing_output_guard(tmp_path: Path):
    cfg = CurationConfig(
        output_jsonl_path=tmp_path / "fences.jsonl",
        report_json_path=tmp_path / "fences_report.json",
        codefeedback_target=0,
        magicoder_target=0,
        gsm8k_target=0,
        mathinstruct_target=0,
        oasst2_target=0,
        slimorca_target=0,
        cosmopedia_target=0,
        commonsense_target=0,
    )
    long_code = "\n".join(f"qc.h({i % 3})" for i in range(180))
    bundle = {
        "qiskit_files": [
            (
                "docs/broken_unclosed.mdx",
                "# Broken QFT Tutorial\n\n## Broken Block\n\n"
                "Step 1: Unclosed fence should never pass curation:\n"
                "```python\nx = 1 + 2\n"
                + ("line_padding = 123456\n" * 6),
            ),
            (
                "docs/truncated_qpe.mdx",
                "# Quantum Phase Estimation\n\n## Circuit Implementation\n\n"
                "Step 1: Quantum phase estimation extracts eigenvalues of a unitary operator using inverse QFT circuits "
                "and coherent ancilla control registers in fault-tolerant subroutines.\n\n"
                "Output:\n```\n",
            ),
            (
                "docs/long_qiskit.mdx",
                "# Quantum Fourier Transform\n\n## Long Circuit Split\n\n"
                "Theorem: QFT transforms computational basis amplitudes into Fourier basis phases with Hadamard and controlled rotations.\n\n"
                "Step 1: Apply recursive controlled-phase gates cleanly across target register qubits.\n\n"
                f"```python\n{long_code}\n```\n",
            ),
        ]
    }
    report = curate_all_datasets(config=cfg, offline_bundle=bundle)
    assert report["total_curated_records"] == 2
    rows = [json.loads(ln) for ln in cfg.output_jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 2
    assert all(r["target_text"].count("```") % 2 == 0 for r in rows)
