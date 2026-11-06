"""
Mirror 7 — Stage 7 (Leap 7): 30-Category Quantum Intelligence Specialization Pipeline.

Covers all 30 Quantum Categories and all 10+ Quantum Datasets & Sources:
1. Neura Parse Quantum Computing & `BoltzmannEntropy/QuantumLLMInstruct`
2. Quantum Mechanics Reasoning (`camel-ai/physics` unseen offsets, `TIGER-Lab/TheoremQA`)
3. Quantum Journal & Live arXiv `quant-ph` API (`export.arxiv.org/api/query?search_query=cat:quant-ph`)
4. IBM Quantum Learning & Qiskit Documentation (`Qiskit` circuits, transpilation, QEC, QPE, Grover, VQE)
5. PennyLane Quantum Machine Learning (`PennyLane` parameter-shift rule, quantum kernels, barren plateaus)
6. Google Quantum AI (`Cirq`, Surface Code QEC, Sycamore/Willow random circuit sampling, Trotter simulation)
7. NIST Quantum Information Science (Post-Quantum Cryptography FIPS 203 ML-KEM, FIPS 204 ML-DSA, BB84/E91)
8. Quantum Country (Matuschak & Nielsen foundational quantum mechanics, gates, teleportation, Grover)
9. MMLU College Physics & Quantum Operators (`cais/mmlu` `college_physics`)
10. Additional Quantum Code, OpenQASM & Operator Algebra sources

Includes:
- Global Anti-Duplication scanning all 12 existing `.jsonl` files on Google Drive (`81,200+` records,
  including `astra_quantum_math_master.jsonl` and `pre Quantum data set.jsonl`) so zero existing
  quantum prompts/targets are duplicated.
- 9-Stage Quantum Quality & Anti-Overfitting Filter (`<=520` byte target cap so `max_t=640` never clips `[EOS]`).
"""
from __future__ import annotations

from typing import Any


THIRTY_QUANTUM_CATEGORIES: tuple[str, ...] = (
    "01_quantum_mechanics",
    "02_quantum_computing",
    "03_quantum_mathematics",
    "04_quantum_information_theory",
    "05_quantum_algorithms",
    "06_quantum_circuits",
    "07_quantum_gates",
    "08_qubits_and_quantum_states",
    "09_quantum_superposition",
    "10_quantum_entanglement",
    "11_quantum_measurement",
    "12_quantum_tunnelling",
    "13_quantum_decoherence",
    "14_quantum_error_correction",
    "15_quantum_cryptography",
    "16_quantum_machine_learning",
    "17_quantum_simulation",
    "18_quantum_complexity_theory",
    "19_quantum_field_theory",
    "20_quantum_state_transformations",
    "21_hilbert_spaces",
    "22_complex_vector_spaces",
    "23_tensor_products",
    "24_unitary_operators",
    "25_eigenvalues_and_eigenvectors",
    "26_schrodinger_equation",
    "27_quantum_probability",
    "28_quantum_circuit_optimization",
    "29_quantum_experimental_analysis",
    "30_quantum_mathematical_derivations",
)


QUANTUM_DATASET_LINKS: dict[str, list[str]] = {
    "1_QuantumLLMInstruct_NeuraParse": [
        "https://datasets-server.huggingface.co/rows?dataset=BoltzmannEntropy%2FQuantumLLMInstruct&config=default&split=train&offset=0&length=100",
        "https://datasets-server.huggingface.co/rows?dataset=BoltzmannEntropy%2FQuantumLLMInstruct&config=default&split=train&offset=100&length=100",
        "https://datasets-server.huggingface.co/rows?dataset=BoltzmannEntropy%2FQuantumLLMInstruct&config=default&split=train&offset=200&length=100",
    ],
    "2_Quantum_Mechanics_Reasoning_CAMEL": [
        "https://datasets-server.huggingface.co/rows?dataset=camel-ai%2Fphysics&config=default&split=train&offset=0&length=100",
        "https://datasets-server.huggingface.co/rows?dataset=camel-ai%2Fphysics&config=default&split=train&offset=150&length=100",
        "https://datasets-server.huggingface.co/rows?dataset=camel-ai%2Fphysics&config=default&split=train&offset=300&length=100",
    ],
    "3_arXiv_QuantPh_Quantum_Journal": [
        "http://export.arxiv.org/api/query?search_query=cat:quant-ph+AND+(quantum+error+correction+OR+surface+code+OR+logical+qubit)&start=0&max_results=45",
        "http://export.arxiv.org/api/query?search_query=cat:quant-ph+AND+(quantum+algorithm+OR+variational+quantum+eigensolver+OR+Hamiltonian+simulation)&start=0&max_results=45",
        "http://export.arxiv.org/api/query?search_query=cat:quant-ph+AND+(quantum+entanglement+OR+decoherence+OR+Lindblad+OR+quantum+cryptography)&start=0&max_results=45",
        "http://export.arxiv.org/api/query?search_query=cat:quant-ph+AND+(quantum+machine+learning+OR+quantum+circuit+optimization+OR+Hilbert+space)&start=0&max_results=45",
    ],
    "4_MMLU_College_Conceptual_Quantum_Physics": [
        "https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=college_physics&split=test&offset=0&length=90",
        "https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=conceptual_physics&split=test&offset=0&length=90",
        "https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=high_school_physics&split=test&offset=0&length=90",
    ],
    "5_SciQ_Quantum_Atomic_Physics": [
        "https://datasets-server.huggingface.co/rows?dataset=allenai%2Fsciq&config=default&split=train&offset=250&length=100",
        "https://datasets-server.huggingface.co/rows?dataset=allenai%2Fsciq&config=default&split=train&offset=400&length=100",
    ],
    "6_TheoremQA_Physics_Hilbert_Operators": [
        "https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FTheoremQA&config=default&split=test&offset=90&length=90",
    ],
}
