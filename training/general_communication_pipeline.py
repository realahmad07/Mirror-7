"""
Mirror 7 — 10-Category General Communication Dataset Extractor & 7-Stage Quality Filter.

Covers all 10 General Communication pillars via direct HuggingFace Hub / REST API links:
1. General conversation datasets (`HuggingFaceTB/smoltalk`, `HuggingFaceH4/ultrachat_200k`)
2. Multi-turn dialogue datasets (`OpenAssistant/oasst2`, `smoltalk/smol-magpie-ultra`)
3. Human-written assistant responses (`databricks/databricks-dolly-15k`, `OpenAssistant/oasst1`)
4. Question-answer datasets (`allenai/sciq`, `truthfulqa/truthful_qa`)
5. Instruction-following datasets (`tatsu-lab/alpaca`, `VMware/open-instruct`)
6. Emotional intelligence & sentiment datasets (`Estwld/empathetic_dialogues_llm`, `dair-ai/emotion`)
7. Context-retention datasets (`stanfordnlp/coqa`, multi-turn coreference chains)
8. Clarification & ambiguity datasets (`sewon/ambig_qa`, proactive clarification dialogues)
9. Multilingual conversation datasets (`CohereForAI/aya_dataset`, `OpenAssistant/oasst2` multilingual)
10. Preference-ranking datasets (`argilla/ultrafeedback-binarized-preferences-cleaned`, `Intel/orca_dpo_pairs`)
11. Critical thinking & common-sense reasoning (`allenai/ai2_arc` ARC-Challenge)
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from typing import Any


GENERAL_COMMUNICATION_DATASET_LINKS: dict[str, dict[str, str]] = {
    "1_general_conversation": {
        "name": "HuggingFaceTB/smoltalk (everyday-conversations)",
        "url": "https://huggingface.co/datasets/HuggingFaceTB/smoltalk",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceTB%2Fsmoltalk&config=everyday-conversations&split=train&offset=0&length=100",
    },
    "2_multiturn_dialogue": {
        "name": "OpenAssistant/oasst2 (Multi-Turn Trees)",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst2",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=OpenAssistant%2Foasst2&config=default&split=train&offset=0&length=100",
    },
    "3_human_written_responses": {
        "name": "databricks/databricks-dolly-15k (Human-Written)",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=databricks%2Fdatabricks-dolly-15k&config=default&split=train&offset=0&length=100",
    },
    "4_question_answering": {
        "name": "allenai/sciq & truthfulqa/truthful_qa",
        "url": "https://huggingface.co/datasets/truthfulqa/truthful_qa",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=truthfulqa%2Ftruthful_qa&config=generation&split=validation&offset=0&length=100",
    },
    "5_instruction_following": {
        "name": "tatsu-lab/alpaca",
        "url": "https://huggingface.co/datasets/tatsu-lab/alpaca",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=tatsu-lab%2Falpaca&config=default&split=train&offset=0&length=100",
    },
    "6_emotional_intelligence": {
        "name": "Estwld/empathetic_dialogues_llm",
        "url": "https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=Estwld%2Fempathetic_dialogues_llm&config=default&split=train&offset=0&length=100",
    },
    "7_context_retention": {
        "name": "stanfordnlp/coqa (Conversational Context Retention)",
        "url": "https://huggingface.co/datasets/stanfordnlp/coqa",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=stanfordnlp%2Fcoqa&config=default&split=train&offset=0&length=100",
    },
    "8_clarification_ambiguity": {
        "name": "sewon/ambig_qa (Ambiguity & Clarification)",
        "url": "https://huggingface.co/datasets/sewon/ambig_qa",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=sewon%2Fambig_qa&config=light&split=train&offset=0&length=100",
    },
    "9_multilingual_conversation": {
        "name": "CohereForAI/aya_dataset (Multilingual Human Dialogues)",
        "url": "https://huggingface.co/datasets/CohereForAI/aya_dataset",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=CohereForAI%2Faya_dataset&config=default&split=train&offset=0&length=100",
    },
    "10_preference_ranking": {
        "name": "Intel/orca_dpo_pairs & argilla/ultrafeedback-binarized-preferences-cleaned",
        "url": "https://huggingface.co/datasets/Intel/orca_dpo_pairs",
        "api_url": "https://datasets-server.huggingface.co/rows?dataset=Intel%2Forca_dpo_pairs&config=default&split=train&offset=0&length=100",
    },
}


BANNED_PHRASES = (
    "as an ai",
    "as a large language model",
    "i am chatgpt",
    "openai",
    "anthropic",
    "i cannot browse",
    "i don't have personal",
    "i do not have feelings",
    "i'm just an ai",
)


def score_and_filter_pair(prompt: str, response: str, rejected: str | None = None) -> tuple[bool, float]:
    """7-Stage Quality Filter & Preference-Margin Scorer (0..10 scale)."""
    u = " ".join(prompt.strip().split())
    t = response.strip()
    if not u or not t:
        return False, 0.0

    u_bytes = len(u.encode("utf-8"))
    t_bytes = len(t.encode("utf-8"))
    if u_bytes < 3 or u_bytes > 460 or t_bytes < 25 or t_bytes > 780:
        return False, 0.0

    if t[-1] not in ".!?\"')>]":
        return False, 0.0

    t_low = t.lower()
    if any(b in t_low for b in BANNED_PHRASES):
        return False, 0.0

    words = t.split()
    if len(words) < 5:
        return False, 0.0
    uniq_word_ratio = len(set(w.lower() for w in words)) / len(words)
    if uniq_word_ratio < 0.45:
        return False, 0.0

    trigrams = [t[i : i + 3] for i in range(max(0, len(t) - 2))]
    trigram_div = len(set(trigrams)) / max(1, len(trigrams))
    if trigram_div < 0.50:
        return False, 0.0

    score = 4.0 + 3.0 * trigram_div + 2.0 * uniq_word_ratio
    if rejected:
        r_words = rejected.split()
        r_div = len(set(w.lower() for w in r_words)) / max(1, len(r_words))
        if uniq_word_ratio >= r_div:
            score += 1.0
    return score >= 7.2, round(score, 2)
