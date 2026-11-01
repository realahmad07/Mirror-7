"""
Knowledge categories and prompt definitions for teacher transfer.
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Sequence


class KnowledgeCategory(str, Enum):
    GENERAL_CONVERSATION = "general_conversation"
    LANGUAGE_AND_WRITING = "language_and_writing"
    MATHEMATICS = "mathematics"
    LOGICAL_REASONING = "logical_reasoning"
    PROGRAMMING_AND_DEBUGGING = "programming_and_debugging"
    INSTRUCTION_FOLLOWING = "instruction_following"
    PLANNING = "planning"
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    TOOL_USE = "tool_use"
    SELF_CORRECTION = "self_correction"
    FACTUAL_EXPLANATION = "factual_explanation"
    AMBIGUOUS_REQUESTS = "ambiguous_requests"
    MULTI_STEP_PROBLEM_SOLVING = "multi_step_problem_solving"


ALL_CATEGORIES: tuple[KnowledgeCategory, ...] = tuple(KnowledgeCategory)


@dataclass(frozen=True)
class CategoryMetadata:
    category: KnowledgeCategory
    description: str
    requires_code_validation: bool = False
    requires_math_validation: bool = False
    requires_step_decomposition: bool = False
    sample_seed_prompts: tuple[str, ...] = ()


CATEGORY_REGISTRY: dict[KnowledgeCategory, CategoryMetadata] = {
    KnowledgeCategory.GENERAL_CONVERSATION: CategoryMetadata(
        category=KnowledgeCategory.GENERAL_CONVERSATION,
        description="Everyday natural dialogue, empathetic responses, conversational engagement.",
        sample_seed_prompts=(
            "Hello! How are you doing today?",
            "Can you tell me an interesting fact about space?",
            "What should someone consider before adopting a pet?",
        ),
    ),
    KnowledgeCategory.LANGUAGE_AND_WRITING: CategoryMetadata(
        category=KnowledgeCategory.LANGUAGE_AND_WRITING,
        description="Grammar, tone modulation, text restructuring, poetry, and essay composition.",
        sample_seed_prompts=(
            "Rewrite this paragraph to make it more professional: 'hey man got your email, looks cool.'",
            "Explain the difference between active and passive voice with two examples.",
            "Compose a concise haiku about winter mornings.",
        ),
    ),
    KnowledgeCategory.MATHEMATICS: CategoryMetadata(
        category=KnowledgeCategory.MATHEMATICS,
        description="Arithmetic, algebra, calculus, discrete math, and numeric reasoning.",
        requires_math_validation=True,
        requires_step_decomposition=True,
        sample_seed_prompts=(
            "Calculate 37 * 48 showing each step.",
            "If a car travels at 60 mph for 2.5 hours, how far does it travel?",
            "Solve for x: 3x + 15 = 42.",
        ),
    ),
    KnowledgeCategory.LOGICAL_REASONING: CategoryMetadata(
        category=KnowledgeCategory.LOGICAL_REASONING,
        description="Deductive, inductive, and abductive logic, fallacy identification, syllogisms.",
        requires_step_decomposition=True,
        sample_seed_prompts=(
            "All humans are mortal. Socrates is human. What follows necessarily?",
            "If it rains, the grass gets wet. The grass is wet. Does that prove it rained? Explain.",
            "Three boxes are labeled Apples, Oranges, and Mixed, but all labels are wrong. How many fruits must you draw to label all correctly?",
        ),
    ),
    KnowledgeCategory.PROGRAMMING_AND_DEBUGGING: CategoryMetadata(
        category=KnowledgeCategory.PROGRAMMING_AND_DEBUGGING,
        description="Code generation, algorithmic implementation, bug diagnosis, and refactoring.",
        requires_code_validation=True,
        sample_seed_prompts=(
            "Write a Python function to reverse a string in-place without slicing.",
            "Explain what causes an IndexError in Python and how to guard against it.",
            "Implement a binary search function in Python that returns the index or -1.",
        ),
    ),
    KnowledgeCategory.INSTRUCTION_FOLLOWING: CategoryMetadata(
        category=KnowledgeCategory.INSTRUCTION_FOLLOWING,
        description="Adherence to complex constraints (length, formatting, negative constraints).",
        sample_seed_prompts=(
            "List 5 European capitals in alphabetical order. Do not include any country names.",
            "Explain photosynthesis in exactly three sentences.",
            "Provide a recipe for scrambled eggs using only bullet points and no introductory text.",
        ),
    ),
    KnowledgeCategory.PLANNING: CategoryMetadata(
        category=KnowledgeCategory.PLANNING,
        description="Temporal sequencing, dependency management, resource-aware roadmap construction.",
        requires_step_decomposition=True,
        sample_seed_prompts=(
            "Create a 4-week study plan to prepare for a basic statistics exam.",
            "Outline the steps required to migrate a database from SQLite to PostgreSQL with zero downtime.",
            "Plan a weekend camping trip checklist considering shelter, food, and emergency safety.",
        ),
    ),
    KnowledgeCategory.PROBLEM_DECOMPOSITION: CategoryMetadata(
        category=KnowledgeCategory.PROBLEM_DECOMPOSITION,
        description="Breaking ambiguous or large problems into mutually exclusive, collectively exhaustive subproblems.",
        requires_step_decomposition=True,
        sample_seed_prompts=(
            "Decompose the problem of 'Reducing website load time from 5s to 1s' into 4 technical subproblems.",
            "Break down the challenge of building an autonomous delivery robot into distinct engineering domains.",
            "How would you break down the task of organizing a 500-person tech conference?",
        ),
    ),
    KnowledgeCategory.TOOL_USE: CategoryMetadata(
        category=KnowledgeCategory.TOOL_USE,
        description="Determining when to call external functions/tools, format arguments, and interpret tool output.",
        requires_code_validation=True,
        sample_seed_prompts=(
            "Given tool `search_web(query: str)`, format the call to find the current population of Tokyo.",
            "A user asks: 'What is 4539 * 289?'. Should you call `calculator(expr: str)` or answer directly? Output the tool call.",
            "Given tool `read_file(path: str)`, formulate the call to inspect '/etc/hosts'.",
        ),
    ),
    KnowledgeCategory.SELF_CORRECTION: CategoryMetadata(
        category=KnowledgeCategory.SELF_CORRECTION,
        description="Identifying an error in a preliminary response, explaining the flaw, and correcting it.",
        requires_step_decomposition=True,
        sample_seed_prompts=(
            "A student claims: '7 * 8 = 54'. Critique this calculation, explain the mistake, and provide the correct answer.",
            "The following Python code has a bug: `def is_even(n): return n % 2 == 1`. Identify the bug and fix it.",
            "Review this argument: 'Paris is in France, France is in Europe, so London is in Europe.' Explain the flaw.",
        ),
    ),
    KnowledgeCategory.FACTUAL_EXPLANATION: CategoryMetadata(
        category=KnowledgeCategory.FACTUAL_EXPLANATION,
        description="Clear, verified scientific, historical, and technical explanations.",
        sample_seed_prompts=(
            "Explain how vaccines help the human immune system fight viruses.",
            "What is the difference between nuclear fission and nuclear fusion?",
            "How does HTTPS establish an encrypted connection between client and server?",
        ),
    ),
    KnowledgeCategory.AMBIGUOUS_REQUESTS: CategoryMetadata(
        category=KnowledgeCategory.AMBIGUOUS_REQUESTS,
        description="Recognizing underspecified queries, asking clarifying questions, or presenting structured options.",
        sample_seed_prompts=(
            "The user says: 'It doesn't work.' How should an AI assistant respond productively?",
            "A user requests: 'Recommend a car.' Formulate 3 key clarifying questions before making a recommendation.",
            "The user asks: 'How do I install Python?' Address the ambiguity regarding their operating system.",
        ),
    ),
    KnowledgeCategory.MULTI_STEP_PROBLEM_SOLVING: CategoryMetadata(
        category=KnowledgeCategory.MULTI_STEP_PROBLEM_SOLVING,
        description="Chained reasoning where the output of step N is the required input for step N+1.",
        requires_step_decomposition=True,
        sample_seed_prompts=(
            "A train leaves Station A at 10:00 AM at 50 mph. Another train leaves Station B at 11:00 AM towards A at 70 mph. The distance is 290 miles. At what time do they meet?",
            "Design a data pipeline that ingests CSV logs, filters out null values, calculates hourly averages, and writes to Parquet.",
            "A baker has 10 kg of flour. Each loaf of bread uses 400g of flour. If they bake 15 loaves in the morning, how many loaves can they bake in the afternoon?",
        ),
    ),
}
