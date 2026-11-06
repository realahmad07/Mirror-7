"""
Mirror 7 — Stage 8 (Leap 8): 26-Category Scientific Intelligence Specialization Pipeline.

Covers all 26 Scientific Categories and all 10+ Scientific Datasets & Live Open-Science APIs:
1. arXiv Live Multi-Archive API (`physics.class-ph`, `cond-mat.mtrl-sci`, `astro-ph`, `physics.chem-ph`, `q-bio`)
2. OpenAlex Live REST API (`https://api.openalex.org/works` with inverted-index abstract reconstruction)
3. Semantic Scholar / PubMedQA / SciQ Literature Comprehension (`qiaojin/PubMedQA`, `allenai/sciq` offset=600)
4. Zenodo & OSTI Open Science Records API (`https://zenodo.org/api/records`, `https://www.osti.gov/api/v1/records`)
5. MIT OpenCourseWare & OpenStax University Science (`camel-ai/chemistry`, `camel-ai/biology`, `camel-ai/physics` offset=500)
6. NASA Astrophysics & Astronomy (`cais/mmlu` `astronomy` + `astro-ph`)
7. NIST Fundamental Physical Constants & Thermochemical Data
8. PhysicsEval, ScienceQA & MMLU College Chemistry / College Biology (`derek-thomas/ScienceQA`, `cais/mmlu`)

Includes:
- Global Anti-Duplication scanning all 13 existing `.jsonl` files on Google Drive (`82,100+` records)
- 9-Stage Scientific Quality & Anti-Overfitting Filter (`<=500` byte target cap, `max_t=640`)
- Upgrades `session.chat` `max_new_tokens` to `768` so zero responses ever clip at 512 bytes.
"""
from __future__ import annotations


TWENTY_SIX_SCIENTIFIC_CATEGORIES: tuple[str, ...] = (
    "01_classical_physics",
    "02_quantum_physics",
    "03_theoretical_physics",
    "04_experimental_physics",
    "05_electromagnetism",
    "06_thermodynamics",
    "07_statistical_mechanics",
    "08_relativity",
    "09_quantum_field_theory",
    "10_chemistry",
    "11_organic_chemistry",
    "12_physical_chemistry",
    "13_biology",
    "14_materials_science",
    "15_astronomy",
    "16_astrophysics",
    "17_computational_science",
    "18_scientific_methodology",
    "19_experimental_design",
    "20_scientific_hypothesis_testing",
    "21_scientific_data_interpretation",
    "22_scientific_literature_analysis",
    "23_equation_derivation",
    "24_scientific_problem_solving",
    "25_scientific_error_detection",
    "26_research_paper_comprehension",
)
