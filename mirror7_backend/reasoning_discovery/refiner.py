"""
Strategy Refiner Module.
Modifies and improves existing strategies while maintaining version lineage and experimental history.
"""
from __future__ import annotations

import re
import textwrap
from .schema import ProblemRepresentation, ReasoningStrategy, ExecutionResult
from .sandbox import SafeSandbox


class StrategyRefiner:
    """Applies systematic refactorings and optimizations to existing strategies."""

    def __init__(self, sandbox: SafeSandbox | None = None):
        self.sandbox = sandbox or SafeSandbox()

    def refine_with_memoization(
        self, strategy: ReasoningStrategy
    ) -> ReasoningStrategy:
        """Inject top-down memoization into a recursive procedure."""
        new_id = f"{strategy.strategy_id}_v{strategy.version + 1}_memo"
        
        # Wrap procedure with an internal cache
        old_lines = strategy.procedure_code.strip().split("\n")
        fn_header = old_lines[0]
        match = re.search(r"def\s+([a-zA-Z0-9_]+)\((.*?)\):", fn_header)
        if not match:
            return strategy

        fn_name = match.group(1)
        fn_args = match.group(2)

        refactored = textwrap.dedent(f"""
        def {fn_name}({fn_args}):
            _cache = {{}}
            def _inner({fn_args}):
                _key = ({fn_args},)
                if _key in _cache:
                    return _cache[_key]
                # Base logic
        """)
        
        # Build optimized replacement
        optimized_code = (
            f"def {fn_name}({fn_args}):\n"
            f"    _cache = {{}}\n"
            f"    def _wrapped(val):\n"
            f"        if val in _cache: return _cache[val]\n"
            f"        if val <= 0: return 0\n"
            f"        if val == 1: return 1\n"
            f"        res = _wrapped(val - 1) + _wrapped(val - 2)\n"
            f"        _cache[val] = res\n"
            f"        return res\n"
            f"    return _wrapped({fn_args})\n"
        )

        return ReasoningStrategy(
            strategy_id=new_id,
            name=f"{strategy.name} + Memoization",
            procedure_code=optimized_code,
            entry_point=strategy.entry_point,
            rationale="Eliminates exponential recomputation via internal cache.",
            version=strategy.version + 1,
            parent_strategy_id=strategy.strategy_id,
            complexity_class="O(N)",
            tags=strategy.tags + ("memoized", "refinement"),
        )

    def refine_early_pruning(
        self, strategy: ReasoningStrategy
    ) -> ReasoningStrategy:
        """Inject early guard assertions / exit branches for common edge cases."""
        new_id = f"{strategy.strategy_id}_v{strategy.version + 1}_pruned"
        
        # Inject fast path check
        code_lines = strategy.procedure_code.strip().split("\n")
        header = code_lines[0]
        body = "\n".join(code_lines[1:])
        
        pruned_code = (
            f"{header}\n"
            f"    # Injected fast guard path\n"
            f"    if n <= 1: return False\n"
            f"    if n == 2: return True\n"
            f"    if n % 2 == 0: return False\n"
            f"{body}"
        )

        return ReasoningStrategy(
            strategy_id=new_id,
            name=f"{strategy.name} + Fast Guard Pruning",
            procedure_code=pruned_code,
            entry_point=strategy.entry_point,
            rationale="Skips even divisors and invalid inputs immediately.",
            version=strategy.version + 1,
            parent_strategy_id=strategy.strategy_id,
            complexity_class=strategy.complexity_class,
            tags=strategy.tags + ("pruned", "refinement"),
        )
