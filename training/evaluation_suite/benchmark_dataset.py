"""
Benchmark datasets across the 10 evaluation dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from .dimensions import EvaluationDimension


@dataclass(frozen=True)
class EvalBenchmarkExample:
    example_id: str
    dimension: EvaluationDimension
    prompt: str
    reference_targets: tuple[str, ...]
    keywords_expected: tuple[str, ...]
    keywords_forbidden: tuple[str, ...] = ()
    partition: str = "dev"  # "dev" or "held_out"


BENCHMARK_EXAMPLES: tuple[EvalBenchmarkExample, ...] = (
    # 1. Conversational Quality
    EvalBenchmarkExample(
        example_id="conv_01",
        dimension=EvaluationDimension.CONVERSATIONAL_QUALITY,
        prompt="Hi, how are you doing today?",
        reference_targets=("Hello! I am doing well, thank you for asking. How can I assist you today?",),
        keywords_expected=("hello", "well", "assist", "today"),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="conv_02",
        dimension=EvaluationDimension.CONVERSATIONAL_QUALITY,
        prompt="Thank you for your help!",
        reference_targets=("You are very welcome! If you have any more questions, feel free to ask.",),
        keywords_expected=("welcome", "questions", "help"),
        partition="held_out",
    ),

    # 2. Prompt Understanding
    EvalBenchmarkExample(
        example_id="und_01",
        dimension=EvaluationDimension.PROMPT_UNDERSTANDING,
        prompt="Reply with exactly the word 'CONFIRMED' and nothing else.",
        reference_targets=("CONFIRMED",),
        keywords_expected=("CONFIRMED",),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="und_02",
        dimension=EvaluationDimension.PROMPT_UNDERSTANDING,
        prompt="Name three primary colors. Do not mention any secondary colors.",
        reference_targets=("Red, blue, and yellow.",),
        keywords_expected=("red", "blue", "yellow"),
        keywords_forbidden=("green", "purple", "orange"),
        partition="held_out",
    ),

    # 3. Factual Accuracy
    EvalBenchmarkExample(
        example_id="fact_01",
        dimension=EvaluationDimension.FACTUAL_ACCURACY,
        prompt="What is the chemical formula for water?",
        reference_targets=("The chemical formula for water is H2O.",),
        keywords_expected=("H2O", "water"),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="fact_02",
        dimension=EvaluationDimension.FACTUAL_ACCURACY,
        prompt="What planet is known as the Red Planet?",
        reference_targets=("Mars is known as the Red Planet.",),
        keywords_expected=("Mars",),
        partition="held_out",
    ),

    # 4. Mathematical Reasoning
    EvalBenchmarkExample(
        example_id="math_01",
        dimension=EvaluationDimension.MATHEMATICAL_REASONING,
        prompt="What is 15 multiplied by 8?",
        reference_targets=("15 multiplied by 8 is 120.",),
        keywords_expected=("120",),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="math_02",
        dimension=EvaluationDimension.MATHEMATICAL_REASONING,
        prompt="Solve for x: 4x - 8 = 16.",
        reference_targets=("4x = 24, so x = 6.",),
        keywords_expected=("6", "x"),
        partition="held_out",
    ),

    # 5. Multi-Step Reasoning
    EvalBenchmarkExample(
        example_id="multi_01",
        dimension=EvaluationDimension.MULTI_STEP_REASONING,
        prompt="Alice has 5 apples. She buys 3 more boxes with 4 apples each, then gives 2 apples to Bob. How many does she have left?",
        reference_targets=("Alice starts with 5, adds 12 (3*4 = 12) for 17, then gives away 2, leaving 15 apples.",),
        keywords_expected=("15", "apples"),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="multi_02",
        dimension=EvaluationDimension.MULTI_STEP_REASONING,
        prompt="A car travels at 50 mph for 2 hours, then 60 mph for 3 hours. What is the total distance traveled?",
        reference_targets=("First leg: 50 * 2 = 100 miles. Second leg: 60 * 3 = 180 miles. Total distance = 280 miles.",),
        keywords_expected=("280", "miles"),
        partition="held_out",
    ),

    # 6. Coding Ability
    EvalBenchmarkExample(
        example_id="code_01",
        dimension=EvaluationDimension.CODING_ABILITY,
        prompt="Write a Python function to check if a number n is even.",
        reference_targets=("def is_even(n):\n    return n % 2 == 0",),
        keywords_expected=("def", "return", "%", "2"),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="code_02",
        dimension=EvaluationDimension.CODING_ABILITY,
        prompt="Write a Python function that returns the maximum element of a list lst.",
        reference_targets=("def find_max(lst):\n    return max(lst)",),
        keywords_expected=("def", "return", "max"),
        partition="held_out",
    ),

    # 7. Planning
    EvalBenchmarkExample(
        example_id="plan_01",
        dimension=EvaluationDimension.PLANNING,
        prompt="Provide a 3-step plan for baking chocolate chip cookies.",
        reference_targets=("Step 1: Preheat oven and mix ingredients.\nStep 2: Scoop dough onto baking sheet.\nStep 3: Bake for 10-12 minutes and cool.",),
        keywords_expected=("step 1", "step 2", "step 3", "mix", "bake"),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="plan_02",
        dimension=EvaluationDimension.PLANNING,
        prompt="Outline a 3-phase checklist for releasing a software library.",
        reference_targets=("Phase 1: Run automated tests.\nPhase 2: Build release package and documentation.\nPhase 3: Deploy to package index and announce.",),
        keywords_expected=("phase 1", "phase 2", "phase 3", "test", "deploy"),
        partition="held_out",
    ),

    # 8. Tool Use Correctness
    EvalBenchmarkExample(
        example_id="tool_01",
        dimension=EvaluationDimension.TOOL_USE_CORRECTNESS,
        prompt="Format a tool call to search the web for 'current weather in London'.",
        reference_targets=('search_web(query="current weather in London")',),
        keywords_expected=("search_web", "weather in London"),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="tool_02",
        dimension=EvaluationDimension.TOOL_USE_CORRECTNESS,
        prompt="Format a tool call to read file 'data/report.json'.",
        reference_targets=('read_file(path="data/report.json")',),
        keywords_expected=("read_file", "data/report.json"),
        partition="held_out",
    ),

    # 9. Generalization
    EvalBenchmarkExample(
        example_id="gen_01",
        dimension=EvaluationDimension.GENERALIZATION,
        prompt="Translate the concept of 'thermal expansion' into an analogy suitable for a 10-year-old.",
        reference_targets=("Imagine people crowded in a room getting energized and needing more elbow room.",),
        keywords_expected=("expand", "energy", "space"),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="gen_02",
        dimension=EvaluationDimension.GENERALIZATION,
        prompt="What happens to water molecules when water freezes into ice?",
        reference_targets=("Water molecules slow down and form an open crystalline hexagonal lattice.",),
        keywords_expected=("crystal", "molecules", "lattice"),
        partition="held_out",
    ),

    # 10. Strategy Discovery
    EvalBenchmarkExample(
        example_id="strat_01",
        dimension=EvaluationDimension.STRATEGY_DISCOVERY,
        prompt="Propose two alternative algorithmic strategies for sorting an array of numbers.",
        reference_targets=("Strategy 1: Quicksort (divide and conquer).\nStrategy 2: Mergesort (stable recursive merge).",),
        keywords_expected=("quicksort", "mergesort", "strategy"),
        partition="dev",
    ),
    EvalBenchmarkExample(
        example_id="strat_02",
        dimension=EvaluationDimension.STRATEGY_DISCOVERY,
        prompt="Describe how dynamic programming avoids redundant computation compared to pure recursion.",
        reference_targets=("Dynamic programming stores intermediate subproblem results in a memoization table.",),
        keywords_expected=("memoization", "table", "redundant", "subproblem"),
        partition="held_out",
    ),
)


def load_evaluation_benchmark(partition: str | None = None) -> list[EvalBenchmarkExample]:
    if partition:
        return [ex for ex in BENCHMARK_EXAMPLES if ex.partition == partition]
    return list(BENCHMARK_EXAMPLES)
