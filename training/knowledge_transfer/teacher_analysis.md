# Teacher Model Selection & Comparative Legal/Technical Analysis

## 1. Overview
Mirror 7 requires a capable open-weight teacher model to generate verified demonstrations across 13 knowledge categories (Phase 3 & 4). In accordance with project requirements, teacher outputs will be used strictly during training; the teacher will **never** be invoked during production inference.

---

## 2. Candidate Model Evaluation Matrix

| Candidate Model | Parameter Size | Primary License | Distillation / Output Use Terms | Mathematics & Reasoning | Code & Tool Use | Hardware Footprint (FP16 / Int4) | Overall Suitability |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** | 7.61B | **Apache 2.0** | **Permissive**: Apache 2.0 explicitly grants reproduction, derivative works, and unrestricted output usage. | **State-of-the-art** in 7B class (MATH ~75%, GSM8K ~88%). | Strong Python, SQL, and tool-call formatting. | 15.2 GB (FP16) / 4.8 GB (4-bit AWQ) — runs on Colab T4. | **Recommended Primary Teacher** |
| **Qwen2.5-Coder-7B-Instruct** | 7.61B | **Apache 2.0** | **Permissive**: Fully open under Apache 2.0. | Moderate general math; exceptional code-logic. | State-of-the-art code generation, debugging, diff repair. | 15.2 GB (FP16) / 4.8 GB (4-bit AWQ) — runs on Colab T4. | **Recommended Coding Specialist** |
| **DeepSeek-R1-Distill-Qwen-7B** | 7.61B | **MIT** | **Permissive**: MIT license allows unrestricted use and fine-tuning. | State-of-the-art chain-of-thought reasoning, multi-step problem solving. | Strong structured reasoning. | 15.2 GB (FP16) / 4.8 GB (4-bit) — runs on Colab T4. | **Recommended Reasoning Specialist** |
| **Llama-3.1-8B-Instruct** | 8.03B | Llama 3.1 Community License | **Conditional**: Permitted for commercial use under 700M monthly active users; outputs may not be used to improve another model that is not Llama derivative unless authorized. | Strong general reasoning, conversational fluency. | Good general coding. | 16.0 GB (FP16) / 5.2 GB (4-bit). | **Secondary Baseline** |
| **Mistral-7B-Instruct-v0.3** | 7.24B | **Apache 2.0** | **Permissive**: Unrestricted Apache 2.0. | Solid conversational and factual reasoning. | Function calling support. | 14.5 GB (FP16) / 4.5 GB (4-bit). | Viable General Fallback |

---

## 3. Legal and Terms of Service Audit

### Why Proprietary API Teachers (OpenAI / Google / Anthropic) Are Constrained
- **OpenAI Terms of Service (Section 2c)**: Prohibits using outputs from OpenAI services to develop models that compete with OpenAI.
- **Google Generative AI Terms**: Similar restrictions on competitive model training.
- **Conclusion**: Relying on proprietary closed-source APIs creates legal ambiguity for open distribution of student checkpoints.

### Why Qwen2.5-7B and DeepSeek-R1-Distill are Superior
- Both models are distributed under standard, recognized open-source licenses (**Apache 2.0** and **MIT** respectively).
- No restrictive clauses prohibiting distillation or training derivative student models.
- Fully reproducible in an open, offline environment on standard GPU hardware (e.g., Google Colab free T4 or A100).

---

## 4. Final Recommendation & Implementation Strategy

### Recommended Multi-Teacher Strategy:
1. **Primary General Teacher**: `Qwen/Qwen2.5-7B-Instruct`
   - Handles: General conversation, language & writing, factual explanation, instruction following, planning, ambiguous requests.
2. **Reasoning & Problem-Solving Teacher**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
   - Handles: Mathematics, logical reasoning, multi-step problem solving, self-correction.
3. **Programming Teacher**: `Qwen/Qwen2.5-Coder-7B-Instruct`
   - Handles: Programming & debugging, tool use, problem decomposition.

All three models share the exact same 7B architecture and quantization profiles, allowing unified inference via Hugging Face `transformers` or `vLLM` on a single Google Colab T4 or A100 instance.
